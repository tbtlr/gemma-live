/*
 * VoiceSession — the model-owning half of gemma-live.
 *
 *     mic PCM ──► [mtmd audio encoder ──► Gemma 4 LLM ──► token stream]
 *                                                              │
 *                                                              ▼
 *                                                  [VibeVoice streaming TTS]
 *                                                              │
 *                                                              ▼
 *                                                          on_audio
 *
 * Mic capture, speaker playback, terminal, signal handling, AEC, and the
 * wake/EOU/barge-in VADs live in main.cpp. This class takes mic PCM in via
 * push_audio() and hands token text + TTS audio back through the on_token /
 * on_audio / on_done callables.
 *
 * Threading: the methods are NOT thread-safe with respect to one another,
 * except abort_turn() which is safe to call from any thread. on_audio may
 * fire from the TTS worker thread, including briefly after on_done.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

/* All capture audio fed to push_audio must be mono float32 at this rate. */
constexpr int GL_MIC_RATE = 16000;

/* TTS native sample rate (what VibeVoice synthesises at). The ACTUAL
 * on_audio rate differs when the DFN post-filter is enabled — ask
 * tts_sample_rate() for the real one. (DFN bumps it to 48000.) */
constexpr int GL_TTS_RATE = 24000;

struct SessionConfig {
    /* Model paths. All four required. */
    std::string llm_model_path;   /* models/gemma-4-E2B-it-IQ4_XS.gguf        */
    std::string mmproj_path;      /* models/mmproj-...-Q8_0.gguf              */
    std::string tts_model_path;   /* models/vibevoice-realtime-0.5b-q4_k.gguf */
    std::string tts_voice_path;   /* voices/vibevoice-voice-en-Gemma_woman.gguf */

    /* Optional system prompt. Empty for no system role. */
    std::string system_prompt;

    /* Multi-token prediction (MTP) speculative decoding.
     *
     * mtp_model_path must point at the separately-exported MTP head
     * (models/mtp-gemma-4-E2B-it.gguf, arch gemma4-assistant) — our IQ4_XS
     * trunk was converted without nextn tensors, so the head cannot be read
     * off the target weights. Empty path, or enable_mtp = false, decodes one
     * token per llama_decode as before.
     *
     * A load failure is NOT fatal: the session logs and falls back to
     * one-token decoding. Ask mtp_active() for what actually happened.
     *
     * Speculation pays in proportion to how expensive ONE target decode is:
     * you always pay for the draft, and only win back a target step when the
     * guess is accepted. That makes it a property of the MODEL PAIR, not of
     * the knobs. Measured here, M-series, chat prompt, 9 turns per config,
     * vs the same config's one-token-per-decode baseline:
     *
     *   E4B + matching head          E2B-IQ4_XS + Q8_0 head
     *     n=1  +36%  (62% acc)  <--   n=1  -13%  (33-48% acc)
     *     n=2  +10%  (29-75%)         n=2  -20%  (25-56%)
     *     n=3  -16%  (19-67%)         n=3  -35%  (14-41%)
     *
     * The E2B pair loses on both counts: its target step is cheap enough
     * that the draft can't earn it back, and its head came from a different
     * quantisation pipeline than the trunk, so it predicts it less well.
     *
     * Acceptance is NOISY — the same pair measures 71-81% across runs. Do not
     * read a single run as a regression; average at least three.
     *
     * On the shipped QAT pair (trunk UD-Q4_K_XL, head Q4_0, both unsloth),
     * 128-token text decode over 3 runs:
     *     no speculation   88.0 tok/s
     *     n=1             104.4 tok/s   (74% accepted)
     * The previous plain-Q4_0 pair measured 81.2 / 102.8 at 75% — same with
     * speculation on, 8% slower without, and 350 MB larger.
     *
     * n=1 is the setting; n=3 (llama.cpp's own default) is a regression on
     * BOTH pairs, because a rejected draft costs a full target slot and
     * acceptance decays fast with depth. Watch the acceptance rate in
     * TurnStats: below ~50% speculation is losing, and re-measure whenever
     * the trunk, the head, or their quantisation changes. */
    bool        enable_mtp = true;
    std::string mtp_model_path;
    int         mtp_n_draft = 1;

    /* Optional DeepFilterNet3 speech-enhancement post-filter over the
     * VibeVoice output. A valid DFN gguf path routes every on_audio chunk
     * through (upsample 24→48 → DFN → emit) and makes tts_sample_rate()
     * report 48000. Empty to disable. */
    std::string dfn_model_path;

    int   n_ctx            = 8192;
    int   n_predict        = 256;

    /* Sampling. The default temperature is deliberately well below
     * llama.cpp's 0.80, for two reasons that both matter to a voice
     * assistant:
     *   - It decides whether MTP pays. The draft head's top guess only
     *     matches the target's pick often enough to be worth verifying when
     *     the target isn't wandering: ~65-75% accepted at 0.3, ~30-50% at
     *     0.8, and below ~50% speculation costs more than it saves.
     *   - It stabilises reply *shape*. At 0.8 the model flips between a
     *     one-word backchannel and a full paragraph on the same input, which
     *     is jarring when the answer is spoken aloud.
     * Still non-zero, so repeated turns on similar audio don't read back as
     * a canned recording. */
    float temperature      = 0.3f;
    float top_p            = 0.95f;
    int   top_k            = 40;

    /* Compute threads for the LLM and the audio encoder. 0 = size to the
     * machine's performance cores; including efficiency cores measurably
     * hurts latency on M-series, because the batch waits on the slowest
     * thread. */
    int   n_threads        = 0;
    float tts_cfg          = 1.5f;
    int   tts_steps        = 5;

    /* Size of the FIRST TTS chunk, in latent frames (~133 ms of audio each);
     * later chunks double up to a cap. This is the largest single term in
     * time-to-first-audio, because on_audio cannot fire until a whole first
     * chunk has been generated AND sigma-VAE decoded.
     *
     * Measured on M-series (ttfa = end_turn entry -> first audio):
     *     frames   1     2     3     6 (VibeVoice default)
     *     ttfa   238   295   335   447 ms
     * The VAE decode is linear at ~26 ms/frame, which is most of the slope.
     * tts_steps, by contrast, barely moves ttfa at all (445-483 ms across
     * 3-6 steps) — the cost is the chunk, not the diffusion.
     *
     * 3 is a deliberate compromise. Lower speaks sooner but leaves less
     * buffered audio to cover generating the next chunk: at 3 frames the
     * first chunk is ~400 ms of audio against ~228 ms to produce the next
     * (~1.7x margin, ~171 ms of absolute slack). Going to 2 buys another
     * ~40 ms of ttfa but halves that slack, and an underrun here is an
     * audible gap — playback_cb has no recovery, it just emits silence.
     * Raise it if the GPU is contended; 0 = VibeVoice's built-in 6. */
    int   tts_first_chunk_frames = 3;
    float tts_neg_anchor   = 0.2f;
    bool  tts_loudness_norm = true;
    /* Perceived-loudness target for the TTS normaliser. */
    float tts_target_rms   = 0.06f;

    /* 0 = silent, 1 = boot/turn info, 2 = ggml/llama diag. */
    int   verbosity        = 1;
};

/* Per-turn statistics. Timings are wall-clock ms; counts non-negative.
 * Readable via last_stats() once end_turn() (or abort) has returned. */
struct TurnStats {
    int    n_audio_tokens = 0;  /* tokens from the mtmd audio encoder         */
    int    n_llm_tokens   = 0;  /* LLM tokens sampled (kept; excludes EOS)    */
    int    n_tts_samples  = 0;  /* total samples emitted via on_audio         */

    /* MTP speculative decoding. Both stay 0 when MTP is inactive.
     * n_accepted / n_drafted is the acceptance rate — the number that says
     * whether mtp_n_draft is set well. */
    int    n_drafted      = 0;  /* tokens proposed by the MTP head            */
    int    n_accepted     = 0;  /* of those, tokens the target agreed with    */

    double ms_ttft        = 0;  /* end_turn entry → first sampled token       */
    double ms_llm_gen     = 0;  /* wall time inside the sampling loop         */
    double ms_tts_wall    = 0;  /* wall time the TTS stream was open          */
};

class VoiceSession {
public:
    /* Load all models. Returns nullptr on failure with *err set. */
    static std::unique_ptr<VoiceSession> create(const SessionConfig & cfg, std::string * err);

    ~VoiceSession();

    VoiceSession(const VoiceSession &)             = delete;
    VoiceSession & operator=(const VoiceSession &) = delete;

    /* Set before the first begin_turn(). Any may be left empty.
     *   on_token — UTF-8 fragment; pointer valid for the call only
     *   on_audio — mono float32 at tts_sample_rate(); valid for the call only
     *   on_done  — LLM hit EOS or n_predict; TTS may still be synthesising */
    std::function<void(const char * text)>              on_token;
    std::function<void(const float * pcm, size_t n)>    on_audio;
    std::function<void()>                               on_done;

    /* Rate on_audio fires at: GL_TTS_RATE, or 48000 when DFN loaded. */
    int tts_sample_rate() const;

    /* One-shot text -> speech in the loaded voice, at tts_sample_rate().
     * Empty on failure with *err set.
     *
     * For short fixed phrases that must play with no synthesis latency at
     * all — backchannels, in particular, where the streaming path's ~335 ms
     * time-to-first-audio would put the sound in the wrong place. Render
     * once at startup, keep the samples, push them straight at the speaker.
     *
     * Uses the same context as the streaming path, so call it BETWEEN turns
     * only; concurrent use with an open TTS stream is not supported.
     *
     * The DFN post-filter is NOT applied even when loaded — it is a
     * streaming filter, and the output is only resampled to match. On the
     * ~300 ms clips this exists for the difference is inaudible. */
    std::vector<float> synthesize(const std::string & text, std::string * err);

    /* True when the MTP draft head loaded and generation is speculative.
     * False when MTP was disabled, had no path, or failed to load. */
    bool mtp_active() const;

    /* Tokens the system block occupies: <bos> + the turn markers + the prompt
     * itself. This is a STANDING cost — the context guard rewinds to the end
     * of this block and never evicts it — so it is the number worth showing,
     * rather than a character count that maps to nothing actionable.
     * 0 when there is no system prompt. */
    int system_tokens() const;

    /* Turn protocol:
     *   1. begin_turn()
     *   2. push_audio(pcm, n)   — repeatedly, while the user speaks
     *   3. end_turn()           — blocks until the LLM has finished sampling
     *                             AND TTS has emitted every chunk (the synth
     *                             worker joins inside). on_token fires per
     *                             piece; on_audio as each ~0.8 s window
     *                             decodes; on_done at EOS, before TTS ends.
     *   4. abort_turn()         — optional, any thread, any time. Idempotent.
     *
     * The three turn calls return false on failure; last_error() has why. */
    bool begin_turn();
    bool push_audio(const float * pcm, size_t n_samples);
    bool end_turn();
    void abort_turn();

    const TurnStats   & last_stats() const;
    const std::string & last_error() const;

private:
    VoiceSession();

    /* All model state (llama, mtmd, vibevoice, dfn) lives in Impl so that
     * main.cpp — which only ever drives the turn protocol — doesn't pull in
     * llama.h / mtmd.h / vibevoice.h / dfn.h. */
    struct Impl;
    std::unique_ptr<Impl> impl;
};
