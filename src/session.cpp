// VoiceSession implementation. See session.h for the interface; this
// translation unit owns:
//
//   - LLM model + mtmd audio encoder + per-turn KV cursor (Gemma 4 E2B)
//   - VibeVoice streaming TTS + RMS loudness normaliser
//   - Per-turn lifecycle (begin/push_audio/end/abort)
//
// What this file does NOT own (all of it lives in main.cpp):
//   - Mic capture / speaker playback hardware
//   - AEC + post-AEC VAD (needs render PCM from the actual playback path)
//   - Terminal, signal handling, process state
//   - Anything UI-shaped — token text and audio chunks go out via callbacks
//
#include "session.h"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "sampling.h"
#include "speculative.h"

#include "vibevoice.h"     // CrispASR streaming TTS


#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

// Return the number of performance cores on Apple Silicon (via sysctl
// hw.perflevel0.physicalcpu), or std::thread::hardware_concurrency() / fallback
// elsewhere. Used to size the LLM + mtmd compute thread pools: oversubscribing
// across both perf and efficiency cores measurably hurts latency on M-series.
static int detect_perf_cores() {
#if defined(__APPLE__)
    int    n   = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) {
        return n;
    }
    if (sysctlbyname("hw.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) {
        return n;
    }
#endif
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? (int) hc : 4;
}

// ────────────────────────────────────────────────────────────────────────
// Quiet log sink.
// Installed on llama, ggml, and mtmd at session create when verbosity == 0.
// CrispASR's bundled ggml is a separate library; vibevoice_init_from_file
// with verbosity=0 makes vibevoice install the same silent callback on ITS
// ggml, so all four noise sources are quieted through their official APIs.
// ────────────────────────────────────────────────────────────────────────
static void silent_log(ggml_log_level /*level*/, const char * /*text*/, void * /*ud*/) {}

// Errors only. The default verbosity wants a usable console, and the loaders
// are anything but: a single startup emits ~1420 lines of per-tensor debug
// from clip alone, plus hparams dumps, warmup traces, KV-cache layer-sharing
// notes and "control-looking token" warnings that are normal for this model.
// None of it is actionable, and all of it buries the lines that are.
// Verbosity 2 restores the lot.
static void errors_only_log(ggml_log_level level, const char * text, void * /*ud*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

// ────────────────────────────────────────────────────────────────────────
// RMS loudness normaliser for the TTS audio path.
// VibeVoice's per-chunk amplitude drifts noticeably over a long reply;
// this keeps each emitted chunk near a constant perceived loudness across
// the whole utterance.
//
// Two key choices vs naïve per-chunk RMS:
//   - "Voiced-only" RMS: silent samples (|x| < VOICE_THRESH) are excluded
//     from the measurement. A chunk with a long pause would otherwise have
//     artificially-low RMS, leading the normaliser to apply a huge boost
//     that makes the actual voiced parts blare. Skipping silence keeps the
//     measurement on speech only.
//   - Chunks with too few voiced samples (n_voiced < MIN_VOICED, ~8 ms of
//     real speech) don't update the gain at all — inter-word pauses and
//     end-of-sentence breaths shouldn't tug the running estimate.
//
// After the running-gain multiply, a soft tanh peak limiter keeps anything
// that would clip below PEAK_CEIL with perceptually-transparent compression
// on the rare loud peaks.
//
// The target is set from SessionConfig::tts_target_rms; the rest are tuned
// defaults.
// ────────────────────────────────────────────────────────────────────────
struct loudness_filter {
    static constexpr float DEFAULT_TARGET_RMS = 0.06f;   // perceived loudness target
    static constexpr float VOICE_THRESH       = 0.005f;  // ~ -46 dBFS — silence floor
    static constexpr int   MIN_VOICED         = 200;     // ~8 ms @ 24 kHz to update gain
    static constexpr float EMA_ALPHA          = 0.05f;   // slower than per-chunk hard reset
    static constexpr float GAIN_MIN           = 0.4f;
    static constexpr float GAIN_MAX           = 3.0f;
    static constexpr float PEAK_CEIL          = 0.92f;   // soft-clip above this

    float target_rms = DEFAULT_TARGET_RMS;
    float ema_gain   = 1.0f;
    bool  fresh      = true;

    void reset() {
        ema_gain = 1.0f;
        fresh    = true;
    }

    void process(float * pcm, int n_samples) {
        if (n_samples <= 0) return;

        // Voiced-only RMS — ignore samples below the silence floor.
        double sumsq = 0.0;
        int    n_voiced = 0;
        for (int i = 0; i < n_samples; i++) {
            const float a = std::fabs(pcm[i]);
            if (a > VOICE_THRESH) {
                sumsq += (double) pcm[i] * pcm[i];
                n_voiced++;
            }
        }

        // Update the running gain only if we saw enough voiced content.
        // Pure-silence chunks (or chunks that are mostly inter-word pauses)
        // leave the gain alone; we just re-apply the existing estimate.
        if (n_voiced >= MIN_VOICED) {
            const float rms     = std::sqrt((float)(sumsq / n_voiced));
            const float wanted  = target_rms / rms;
            const float clamped = std::max(GAIN_MIN, std::min(GAIN_MAX, wanted));
            if (fresh) { ema_gain = clamped; fresh = false; }
            else        ema_gain = (1.0f - EMA_ALPHA) * ema_gain + EMA_ALPHA * clamped;
        }

        // Apply gain + soft peak limit. tanh-style compression on samples
        // that would exceed PEAK_CEIL — keeps the loudness target aggressive
        // without producing audible clipping on loud syllables.
        const float head = 1.0f - PEAK_CEIL;
        for (int i = 0; i < n_samples; i++) {
            float s = pcm[i] * ema_gain;
            if (s >  PEAK_CEIL)      s =  PEAK_CEIL + head * std::tanh((s - PEAK_CEIL) / head);
            else if (s < -PEAK_CEIL) s = -PEAK_CEIL - head * std::tanh((-s - PEAK_CEIL) / head);
            pcm[i] = s;
        }
    }
};

// The system block exactly as it goes into the KV cache. Built here rather
// than inline because two places need it to agree byte for byte: begin_turn
// decodes it on turn 0, and create() counts its tokens for the boot line. If
// they drifted, the reported context cost would quietly stop matching the
// real one.
static std::string system_block(const std::string & system_prompt) {
    if (system_prompt.empty()) return {};
    return "<|turn>system\n" + system_prompt + "<turn|>\n";
}

// ────────────────────────────────────────────────────────────────────────
// Low-level decode helpers.
// ────────────────────────────────────────────────────────────────────────
static int decode_text_tokens(llama_context * lctx,
                              const std::vector<llama_token> & tokens,
                              llama_pos & n_past,
                              int seq_id,
                              int n_batch,
                              bool logits_last) {
    if (tokens.empty()) return 0;
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int ret = 0;
    for (int i = 0; i < (int) tokens.size(); ) {
        const int n = std::min((int) tokens.size() - i, n_batch);
        common_batch_clear(batch);
        for (int j = 0; j < n; j++) {
            const bool is_last_overall = (i + j == (int) tokens.size() - 1) && logits_last;
            common_batch_add(batch, tokens[i + j], n_past + (llama_pos) j, {seq_id}, is_last_overall);
        }
        if (llama_decode(lctx, batch)) { ret = 1; break; }
        n_past += (llama_pos) n;
        i += n;
    }
    llama_batch_free(batch);
    return ret;
}

static int decode_embeddings(llama_context * lctx,
                             const float * embd,
                             int n_embd_tokens,
                             llama_pos & n_past,
                             int seq_id) {
    if (n_embd_tokens <= 0) return 0;
    const int n_embd = llama_model_n_embd(llama_get_model(lctx));
    llama_batch batch = llama_batch_init(n_embd_tokens, n_embd, 1);
    batch.n_tokens = n_embd_tokens;
    std::memcpy(batch.embd, embd, sizeof(float) * n_embd * n_embd_tokens);
    for (int i = 0; i < n_embd_tokens; i++) {
        batch.pos    [i] = n_past + (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id [i][0] = seq_id;
        batch.logits [i] = 0;
    }
    const int ret = llama_decode(lctx, batch);
    if (!ret) n_past += (llama_pos) n_embd_tokens;
    llama_batch_free(batch);
    return ret;
}

// ────────────────────────────────────────────────────────────────────────
// VoiceSession::Impl — owns all model state across turns.
// ────────────────────────────────────────────────────────────────────────
struct VoiceSession::Impl {
    // Back-pointer, for reaching the owner's on_token / on_audio / on_done.
    VoiceSession             * self = nullptr;

    // TTS audio bridge — vibevoice fires this from its synth worker.
    // Static so it can be passed as a plain C function pointer.
    static void tts_audio_bridge(const float * pcm, int n_samples, void * user);

    // Persistent model state
    common_params              params;
    common_init_result_ptr     llama_init;
    llama_model              * model   = nullptr;
    llama_context            * lctx    = nullptr;
    const llama_vocab        * vocab   = nullptr;
    common_sampler           * smpl    = nullptr;
    mtmd::context_ptr          ctx_mtmd;
    int                        sample_rate = 16000;
    int                        n_batch     = 0;
    int                        n_predict   = 256;

    // MTP speculative decoding. spec_init owns the draft model + its context;
    // spec is the drafter driving them. Declared in this order so `spec` (the
    // consumer) is destroyed before the contexts it points at, and both before
    // llama_init, which is declared above and so outlives them.
    common_speculative_init_result_ptr spec_init;
    common_speculative_ptr             spec;
    bool                       spec_on    = false;
    int                        spec_n_max = 1;   // max drafted tokens per step

    // Text-token history handed to the drafter. Audio positions occupy KV
    // slots on the target but carry no token id, so this is deliberately NOT
    // aligned with n_past — it holds only the tokens that have one. The MTP
    // head drafts from the target's hidden state rather than from token ids,
    // so the gap costs it nothing.
    llama_tokens               prompt_hist;
    size_t                     prompt_hist_before_turn = 0;

    // Persistent TTS state
    vibevoice_context        * tts_ctx = nullptr;
    float                      tts_cfg            = 1.5f;
    int                        tts_steps          = 5;
    float                      tts_neg_anchor     = 0.2f;
    bool                       loudness_norm      = true;
    loudness_filter            loudness;

    int                        tts_output_rate = GL_TTS_RATE;

    // Conversation cursor
    std::string                system_prompt;
    llama_pos                  n_past = 0;
    int                        turn   = 0;

    // End of the <bos>+system block, and the matching point in prompt_hist.
    // The context guard rewinds here: n_ctx is finite and n_past only grows,
    // so without it a long enough conversation eventually fails a decode and
    // every turn after it. Dropping the dialogue but keeping the system block
    // costs the assistant its memory of the conversation and keeps it alive,
    // which is the better failure for something always-on.
    int                        n_system_tokens = 0;
    llama_pos                  n_past_after_system  = -1;
    size_t                     prompt_hist_after_system = 0;
    bool                       history_reset = false;  // next prefix starts fresh

    // Active-turn state (valid between begin_turn and end_turn)
    bool                                 in_turn          = false;
    bool                                 turn_is_text     = false;
    mtmd_audio_stream                  * mtmd_stream      = nullptr;
    std::atomic<vibevoice_tts_stream *>  tts_session{nullptr};
    int                                  n_audio_tokens   = 0;
    int                                  n_text_tokens    = 0;
    llama_pos                            n_past_before_turn = 0;
    std::atomic<bool>                    abort_flag{false};

    // Per-turn TTS sample count. Written by the TTS audio bridge from
    // vibevoice's worker thread; read by end_turn after the worker joins.
    std::atomic<int>                     tts_samples{0};

    // Reusable scratch buffers. Both live across the session and grow once,
    // so the hot paths (one TTS chunk every ~80 ms during synth, one token
    // every few ms during sampling) don't malloc per call.
    //   tts_scratch_pcm: target of the per-chunk memcpy + loudness pass in
    //                    tts_audio_bridge. Only touched from vibevoice's TTS
    //                    worker thread (one in flight at a time).
    //   tts_scratch_txt: target of the whitespace-collapsed token feed in
    //                    push_tts_collapsed. Only touched from the main
    //                    thread inside end_turn.
    std::vector<float>                   tts_scratch_pcm;
    std::string                          tts_scratch_txt;

    // Stats from the most recently completed turn. Filled at the end of
    // end_turn (or abort path) so last_stats() can return them.
    TurnStats                            stats{};

    std::string                          error;
};

// Apply loudness normalisation (in-place on the session scratch buffer to
// avoid a per-chunk allocation), accumulate the per-turn sample count for
// stats, then forward to the owner's on_audio.

std::vector<float> VoiceSession::synthesize(const std::string & text, std::string * err) {
    std::vector<float> out;
    if (!impl || !impl->tts_ctx) {
        if (err) *err = "no TTS context";
        return out;
    }
    int     n   = 0;
    float * pcm = vibevoice_synthesize(impl->tts_ctx, text.c_str(), &n);
    if (!pcm || n <= 0) {
        if (pcm) free(pcm);
        if (err) *err = "vibevoice_synthesize produced nothing for \"" + text + "\"";
        return out;
    }
    // vibevoice synthesises at GL_TTS_RATE and that is now the only output
    // rate — the resample here existed for the DFN post-filter's 48 kHz.
    out.assign(pcm, pcm + n);
    free(pcm);
    return out;
}

void VoiceSession::Impl::tts_audio_bridge(const float * pcm, int n_samples, void * user) {
    auto * s = (VoiceSession::Impl *) user;
    if (!s || n_samples <= 0) return;
    s->tts_samples.fetch_add(n_samples, std::memory_order_relaxed);

    // Loudness normalise into the scratch buffer (no per-chunk malloc) —
    // vibevoice's `pcm` is const and only valid for this call, so the in-place
    // filter needs its own storage. Without loudness norm we forward/resample
    // straight from `pcm`; both remaining readers finish before we return.
    const float * src = pcm;
    int           src_n = n_samples;
    if (s->loudness_norm) {
        if ((int) s->tts_scratch_pcm.size() < n_samples) {
            s->tts_scratch_pcm.resize((size_t) n_samples);
        }
        std::memcpy(s->tts_scratch_pcm.data(), pcm, sizeof(float) * (size_t) n_samples);
        s->loudness.process(s->tts_scratch_pcm.data(), n_samples);
        src = s->tts_scratch_pcm.data();
    }


    if (s->self->on_audio) s->self->on_audio(src, (size_t) src_n);
}

// ────────────────────────────────────────────────────────────────────────
// VoiceSession
// ────────────────────────────────────────────────────────────────────────

VoiceSession::VoiceSession() : impl(new Impl()) { impl->self = this; }

const std::string & VoiceSession::last_error() const { return impl->error; }
const TurnStats   & VoiceSession::last_stats() const { return impl->stats; }
int   VoiceSession::tts_sample_rate()          const { return impl->tts_output_rate; }
bool  VoiceSession::mtp_active()               const { return impl->spec_on; }
int   VoiceSession::system_tokens()            const { return impl->n_system_tokens; }

std::unique_ptr<VoiceSession> VoiceSession::create(const SessionConfig & cfg,
                                                   std::string * err) {
    auto fail = [&](const std::string & msg) {
        if (err) *err = msg;
        return std::unique_ptr<VoiceSession>();
    };

    if (cfg.llm_model_path.empty() || cfg.mmproj_path.empty() ||
        cfg.tts_model_path.empty() || cfg.tts_voice_path.empty()) {
        return fail("VoiceSession::create: all four model paths are required");
    }

    std::unique_ptr<VoiceSession> session(new VoiceSession());
    Impl * s = session->impl.get();

    // common_init() must come FIRST: it ends with
    // llama_log_set(common_log_default_callback), so calling it after the
    // silencing block below would immediately undo llama_log_set(silent_log)
    // and verbosity 0 would still print llama's load/warn output.
    common_init();

    // Quieten the loaders BEFORE anything loads. Three sinks, all of which
    // default to firing everything at stderr:
    //   verbosity 0 — nothing at all
    //   verbosity 1 — errors only (the default; see errors_only_log)
    //   verbosity 2 — untouched, full ggml/llama/clip diagnostics
    if (cfg.verbosity < 2) {
        const auto sink = (cfg.verbosity == 0) ? silent_log : errors_only_log;
        common_log_set_verbosity_thold(cfg.verbosity == 0 ? -1 : 0);
        llama_log_set(sink, nullptr);
        ggml_log_set (sink, nullptr);
        mtmd_log_set (sink, nullptr);
    }
    ggml_backend_load_all();

    // ---- LLM ----
    s->params.model.path  = cfg.llm_model_path;
    s->params.mmproj.path = cfg.mmproj_path;
    s->params.use_jinja   = true;
    s->params.n_predict   = cfg.n_predict;
    s->params.n_ctx       = cfg.n_ctx;

    // Flash Attention: measurably faster on Metal for Gemma's head shape.
    // AUTO leaves it off when the backend's heuristics can't decide; force ON.
    s->params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    // Pin compute to performance cores. The default (-1 → hardware_concurrency)
    // includes efficiency cores on Apple Silicon, which slows the slowest
    // thread and stalls the whole batch. Override with --llm-threads.
    {
        int nt = (cfg.n_threads > 0) ? cfg.n_threads : detect_perf_cores();
        s->params.cpuparams.n_threads       = nt;
        s->params.cpuparams_batch.n_threads = nt;
    }

    s->llama_init = common_init_from_params(s->params);
    s->model = s->llama_init->model();
    s->lctx  = s->llama_init->context();
    if (!s->model || !s->lctx) {
        return fail("common_init_from_params failed");
    }

    // ---- MTP speculative decoding (optional) ----
    // Loads the MTP head as a *draft model* rather than as a self-MTP context
    // on the target weights: our Gemma 4 trunk was exported with --no-mtp, so
    // it carries no nextn tensors for a self-MTP context to read. Pushing the
    // DRAFT_MTP type alongside a draft path gives the head an
    // LLAMA_CONTEXT_TYPE_MTP context of its own, chained to the target via
    // cparams.ctx_other so it can see the target's hidden state.
    //
    // Every failure here is non-fatal — we log and leave spec_on false, and
    // end_turn falls back to one token per decode.
    if (cfg.enable_mtp && !cfg.mtp_model_path.empty()) {
        s->params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        s->params.speculative.draft.mparams.path = cfg.mtp_model_path;
        if (cfg.mtp_n_draft > 0) {
            s->params.speculative.draft.n_max = cfg.mtp_n_draft;
        }

        common_params params_dft = common_base_params_to_speculative(s->params);
        s->spec_init = common_speculative_init_from_params(params_dft, s->model, s->lctx);
        if (s->spec_init && s->spec_init->context()) {
            // common_speculative_init reads the two contexts back out of the
            // params struct (same handoff llama-server and stream-asr use).
            s->params.speculative.draft.ctx_tgt = s->lctx;
            s->params.speculative.draft.ctx_dft = s->spec_init->context();
            s->spec.reset(common_speculative_init(s->params.speculative, /*n_seq=*/ 1));
        }

        if (s->spec) {
            s->spec_on    = true;
            s->spec_n_max = std::max(common_speculative_n_max(&s->params.speculative), 1);
            if (cfg.verbosity >= 1) {
                // Say what a step yields, not just what is drafted: each step
                // decodes the token we already sampled PLUS the draft, so it
                // returns 1 token when the draft is rejected and n+1 when it
                // is accepted. Reporting only "draft n" reads as though that
                // were the whole output.
                fprintf(stderr, "mtp      %s, draft %d/step -> %d-%d tok\n",
                        cfg.mtp_model_path.c_str(), s->spec_n_max,
                        1, s->spec_n_max + 1);
            }
        } else {
            // Drop the half-built speculative state so nothing downstream sees
            // a DRAFT_MTP type with no drafter behind it.
            s->spec_init.reset();
            s->params.speculative.types.clear();
            s->params.speculative.draft.ctx_tgt = nullptr;
            s->params.speculative.draft.ctx_dft = nullptr;
            fprintf(stderr, "mtp      DISABLED — %s failed to load\n",
                    cfg.mtp_model_path.c_str());
        }
    }

    s->vocab     = llama_model_get_vocab(s->model);
    s->params.sampling.temp  = cfg.temperature;
    s->params.sampling.top_p = cfg.top_p;
    s->params.sampling.top_k = cfg.top_k;
    s->smpl      = common_sampler_init(s->model, s->params.sampling);
    s->n_batch   = s->params.n_batch;
    s->n_predict = s->params.n_predict;

    // ---- mtmd (audio mmproj) ----
    mtmd_context_params mparams = mtmd_context_params_default();
    // Force the audio encoder onto Metal — common_params leaves this on the
    // default (CPU) and the audio mmproj is the dominant prefix cost.
    mparams.use_gpu       = true;
    // Audio only. The Gemma 4 mmproj carries both towers, and the vision half
    // is ~215 MiB of weights plus its own ~101 MiB Metal compute buffer plus a
    // 768x768 warmup pass at startup — none of which this app can ever reach,
    // since the only thing it ever feeds mtmd is microphone PCM.
    mparams.load_vision   = false;
    mparams.load_audio    = true;
    mparams.print_timings = false;
    mparams.n_threads     = s->params.cpuparams.n_threads;
    mparams.warmup        = s->params.warmup;
    s->ctx_mtmd.reset(mtmd_init_from_file(s->params.mmproj.path.c_str(), s->model, mparams));
    if (!s->ctx_mtmd) {
        return fail("mtmd_init_from_file failed");
    }
    if (!mtmd_support_audio(s->ctx_mtmd.get())) {
        return fail("mmproj has no audio encoder");
    }
    s->sample_rate = mtmd_get_audio_sample_rate(s->ctx_mtmd.get());

    // ---- VibeVoice TTS ----
    vibevoice_context_params vparams = vibevoice_context_default_params();
    vparams.use_gpu              = cfg.tts_gpu;
    vparams.flash_attn           = true;   // σ-VAE encoder + Qwen2.5 attention on Metal
    vparams.verbosity            = (cfg.verbosity >= 2) ? cfg.verbosity : 0;
    vparams.cfg_scale            = (cfg.tts_cfg        > 0.0f) ? cfg.tts_cfg        : 1.5f;
    vparams.tts_steps            = (cfg.tts_steps      > 0)    ? cfg.tts_steps      : 5;
    vparams.neg_condition_anchor = (cfg.tts_neg_anchor > 0.0f) ? cfg.tts_neg_anchor : 0.2f;
    vparams.stream_first_chunk_frames = cfg.tts_first_chunk_frames;
    s->tts_cfg          = vparams.cfg_scale;
    s->tts_steps        = vparams.tts_steps;
    s->tts_neg_anchor   = vparams.neg_condition_anchor;
    s->loudness_norm    = cfg.tts_loudness_norm;
    if (cfg.tts_target_rms > 0.0f && cfg.tts_target_rms < 1.0f) {
        s->loudness.target_rms = cfg.tts_target_rms;
    }

    s->tts_ctx = vibevoice_init_from_file(cfg.tts_model_path.c_str(), vparams);
    if (!s->tts_ctx) {
        return fail("vibevoice_init_from_file failed: " + cfg.tts_model_path);
    }
    if (vibevoice_load_voice(s->tts_ctx, cfg.tts_voice_path.c_str()) != 0) {
        return fail("vibevoice_load_voice failed: " + cfg.tts_voice_path);
    }

    s->system_prompt = cfg.system_prompt;
    while (!s->system_prompt.empty() &&
           (s->system_prompt.back() == '\n' || s->system_prompt.back() == '\r')) {
        s->system_prompt.pop_back();
    }

    // Count exactly what begin_turn will decode for turn 0, markers and <bos>
    // included, so the reported figure is the real standing context cost.
    {
        s->n_system_tokens = (int) common_tokenize(s->lctx, system_block(s->system_prompt),
                                                   /*add_special=*/ true,
                                                   /*parse_special=*/ true).size();
    }

    return session;
}

VoiceSession::~VoiceSession() {
    Impl * s = impl.get();
    s->abort_flag.store(true);
    if (auto * tts = s->tts_session.load()) vibevoice_tts_stream_abort(tts);
    if (s->in_turn && s->mtmd_stream) {
        mtmd_audio_stream_free(s->mtmd_stream);
        s->mtmd_stream = nullptr;
    }
    // Speculative state first, and in this order: the drafter's destructor
    // detaches backend samplers from the draft context, so that context (owned
    // by spec_init) has to outlive it. Both must go before llama_init releases
    // the target model they were chained to.
    s->spec.reset();
    s->spec_init.reset();
    s->spec_on = false;

    if (s->tts_ctx) vibevoice_free(s->tts_ctx);
    if (s->smpl)    common_sampler_free(s->smpl);
    // llama_init's unique_ptr cleans up model + lctx.
}

bool VoiceSession::begin_turn(VoiceSession::turn_kind kind) {
    Impl * s = impl.get();
    if (s->in_turn) { s->error = "begin_turn: turn already in progress"; return false; }

    // ---- Context guard ----
    // Reserve room for what this turn is about to add: a full 30 s of audio
    // (~23 tokens/s measured), the reply, and the turn markers. Checked before
    // n_past_before_turn is taken, so a rollback lands after the reset rather
    // than restoring the history we just dropped.
    if (s->n_past_after_system >= 0) {
        const int reserve = s->n_predict + 768 + 64;
        if (s->n_past > (llama_pos) std::max(0, s->params.n_ctx - reserve)) {
            llama_memory_seq_rm(llama_get_memory(s->lctx), /*seq=*/ 0,
                                s->n_past_after_system, /*p1=*/ -1);
            if (s->spec_init && s->spec_init->context()) {
                llama_memory_seq_rm(llama_get_memory(s->spec_init->context()), /*seq=*/ 0,
                                    s->n_past_after_system, /*p1=*/ -1);
            }
            s->n_past = s->n_past_after_system;
            s->prompt_hist.resize(s->prompt_hist_after_system);
            common_sampler_reset(s->smpl);
            s->history_reset = true;
            fprintf(stderr, "session: context full — dropped conversation history "
                            "(system prompt kept)\n");
        }
    }

    s->n_audio_tokens  = 0;
    s->n_text_tokens   = 0;
    s->n_past_before_turn = s->n_past;
    s->prompt_hist_before_turn = s->prompt_hist.size();
    s->abort_flag.store(false);
    s->tts_samples.store(0);
    s->stats = TurnStats{};

    // Per-turn prefix:
    //   Turn 0:   <bos>{<|turn>system\n{system}<turn|>\n} then the user block
    //   After a history reset: the user block alone — the system block it
    //   rewound to already ends in <turn|>, so there is no open turn to close
    //   Turn N>0: <turn|>\n then the user block
    //
    // Turn 0 decodes the system block SEPARATELY from the user block purely so
    // that n_past_after_system exists as a rewind point for the guard above.
    if (s->turn == 0) {
        auto sys_toks = common_tokenize(s->lctx, system_block(s->system_prompt),
                                        /*add_special=*/ true,
                                        /*parse_special=*/ true);
        if (decode_text_tokens(s->lctx, sys_toks, s->n_past, /*seq=*/ 0, s->n_batch,
                               /*logits_last=*/ false)) {
            s->error = "begin_turn: system prefix decode failed";
            return false;
        }
        s->prompt_hist.insert(s->prompt_hist.end(), sys_toks.begin(), sys_toks.end());
        s->n_past_after_system      = s->n_past;
        s->prompt_hist_after_system = s->prompt_hist.size();
    }

    s->turn_is_text = (kind == VoiceSession::turn_kind::text);
    // Same turn framing either way; an audio turn additionally opens the
    // <|audio> span that the encoder's embeddings live inside.
    const char * open_tag = s->turn_is_text ? "" : "<|audio>";
    const std::string prefix = (s->turn == 0 || s->history_reset)
        ? std::string("<|turn>user\n") + open_tag
        : std::string("<turn|>\n<|turn>user\n") + open_tag;
    s->history_reset = false;
    auto toks = common_tokenize(s->lctx, prefix, /*add_special=*/ false,
                                                /*parse_special=*/ true);
    if (decode_text_tokens(s->lctx, toks, s->n_past, /*seq=*/ 0, s->n_batch,
                           /*logits_last=*/ false)) {
        s->error = "begin_turn: prefix decode failed";
        return false;
    }
    s->prompt_hist.insert(s->prompt_hist.end(), toks.begin(), toks.end());

    if (!s->turn_is_text) {
        s->mtmd_stream = mtmd_audio_stream_init(s->ctx_mtmd.get());
        if (!s->mtmd_stream) {
            s->error = "begin_turn: mtmd_audio_stream_init failed";
            return false;
        }
    }


    s->in_turn = true;
    return true;
}

bool VoiceSession::push_text(const std::string & text) {
    Impl * s = impl.get();
    if (!s->in_turn)      { s->error = "push_text: no turn in progress"; return false; }
    if (!s->turn_is_text) { s->error = "push_text: this is an audio turn"; return false; }
    if (text.empty())     return true;

    // parse_special=false: user text is DATA. Letting it be parsed would let
    // a typed "<|turn>model" forge a turn boundary and put words in Gemma's
    // mouth for every turn after it.
    auto toks = common_tokenize(s->lctx, text, /*add_special=*/ false,
                                               /*parse_special=*/ false);
    if (decode_text_tokens(s->lctx, toks, s->n_past, /*seq=*/ 0, s->n_batch,
                           /*logits_last=*/ false)) {
        s->error = "push_text: decode failed";
        return false;
    }
    s->prompt_hist.insert(s->prompt_hist.end(), toks.begin(), toks.end());
    s->n_text_tokens += (int) toks.size();
    return true;
}

bool VoiceSession::push_audio(const float * pcm, size_t n_samples) {
    Impl * s = impl.get();
    if (!pcm || n_samples == 0) return false;
    if (!s->in_turn) { s->error = "push_audio: no turn in progress"; return false; }
    mtmd_audio_stream_push_pcm(s->mtmd_stream, pcm, n_samples);
    int n_new = 0;
    const float * embd = mtmd_audio_stream_pull(s->mtmd_stream, &n_new);
    if (n_new > 0) {
        if (decode_embeddings(s->lctx, embd, n_new, s->n_past, /*seq=*/ 0)) {
            s->error = "push_audio: audio embd decode failed";
            return false;
        }
        s->n_audio_tokens += n_new;
    }
    return true;
}

bool VoiceSession::end_turn(bool speak) {
    Impl * s = impl.get();
    if (!s->in_turn) { s->error = "end_turn: no turn in progress"; return false; }

    // t0 = entry to end_turn (≈ end-of-utterance from the session's POV; the
    // caller's actual EOU is a millisecond or two earlier).
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto ms_since = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Undo everything begin_turn/push_audio wrote into the KV cache, putting
    // the conversation back exactly where it was before this turn started.
    //
    // Every failure exit below goes through here. Without it a failed turn
    // leaves the prefix ("<|turn>user\n<|audio>" + the encoded audio) in the
    // cache with no matching model reply, and `turn` un-incremented — so the
    // next begin_turn re-emits <bos> and the system block in the middle of the
    // sequence. One bad turn would corrupt every turn after it.
    auto rollback_turn = [&]() {
        llama_memory_seq_rm(llama_get_memory(s->lctx), /*seq=*/ 0,
                            s->n_past_before_turn, /*p1=*/ -1);
        if (s->spec_init && s->spec_init->context()) {
            llama_memory_seq_rm(llama_get_memory(s->spec_init->context()), /*seq=*/ 0,
                                s->n_past_before_turn, /*p1=*/ -1);
        }
        s->n_past = s->n_past_before_turn;
        s->prompt_hist.resize(s->prompt_hist_before_turn);
        common_sampler_reset(s->smpl);
    };

    // Finalise audio encoder. A text turn never opened one.
    int n_tail = 0;
    const float * tail = s->turn_is_text
        ? nullptr
        : mtmd_audio_stream_finalize(s->mtmd_stream, &n_tail);
    if (n_tail > 0) {
        if (decode_embeddings(s->lctx, tail, n_tail, s->n_past, /*seq=*/ 0)) {
            mtmd_audio_stream_free(s->mtmd_stream);
            s->mtmd_stream = nullptr;
            rollback_turn();
            s->in_turn = false;
            s->error = "end_turn: audio tail decode failed";
            s->stats.n_audio_tokens = s->n_audio_tokens;
            return false;
        }
        s->n_audio_tokens += n_tail;
    }
    if (s->mtmd_stream) {
        mtmd_audio_stream_free(s->mtmd_stream);
        s->mtmd_stream = nullptr;
    }

    // Safety net: nothing came through — roll the turn back so we don't ask
    // the model to answer an empty user turn.
    if ((s->turn_is_text ? s->n_text_tokens : s->n_audio_tokens) == 0) {
        rollback_turn();
        s->in_turn = false;
        if (on_done) on_done();
        // Stats stay zero — nothing happened.
        return true;
    }

    // Per-turn suffix + model role transition.
    llama_token suffix_last = 0;
    {
        const std::string suffix = s->turn_is_text
            ? std::string("<turn|>\n<|turn>model\n")
            : std::string("<audio|><turn|>\n<|turn>model\n");
        auto toks = common_tokenize(s->lctx, suffix, /*add_special=*/ false,
                                                     /*parse_special=*/ true);
        if (decode_text_tokens(s->lctx, toks, s->n_past, /*seq=*/ 0, s->n_batch,
                               /*logits_last=*/ true)) {
            rollback_turn();
            s->in_turn = false;
            s->error = "end_turn: suffix decode failed";
            s->stats.n_audio_tokens = s->n_audio_tokens;
            return false;
        }
        s->prompt_hist.insert(s->prompt_hist.end(), toks.begin(), toks.end());
        if (!toks.empty()) suffix_last = toks.back();
    }

    // Open async TTS stream for this reply, unless the caller only wants
    // text back.
    s->loudness.reset();
    const auto t_tts_begin = clk::now();
    vibevoice_tts_stream * tts = speak
        ? vibevoice_tts_stream_begin(s->tts_ctx, Impl::tts_audio_bridge, s)
        : nullptr;
    if (speak && !tts) {
        rollback_turn();
        s->in_turn = false;
        s->error = "end_turn: vibevoice_tts_stream_begin failed";
        s->stats.n_audio_tokens = s->n_audio_tokens;
        return false;
    }
    s->tts_session.store(tts);

    // Sampling loop with whitespace-collapsing TTS feed (VibeVoice tokenises
    // every space/newline as a small pause; collapse runs to a single space
    // so blank lines don't become multi-second breaths). Uses the session's
    // scratch string so the per-token feed doesn't malloc.
    bool last_was_ws = true;
    auto & scratch = s->tts_scratch_txt;
    auto push_tts_collapsed = [&](const std::string & p) {
        scratch.clear();
        for (char c : p) {
            const bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
            if (is_ws) {
                if (!last_was_ws) { scratch += ' '; last_was_ws = true; }
            } else {
                scratch += c;
                last_was_ws = false;
            }
        }
        if (tts && !scratch.empty()) vibevoice_tts_stream_push_text(tts, scratch.c_str());
    };

    const auto      t_gen_begin = clk::now();
    clk::time_point t_first_tok {};
    int n_generated = 0;
    int n_drafted   = 0;
    int n_accepted  = 0;

    // Hand one sampled token to the caller and the TTS stream. Returns false
    // when it terminates the reply (EOS, or the turn markers the chat format
    // uses to close the model's turn) — in that case nothing is emitted and
    // the token stays out of prompt_hist.
    auto emit_token = [&](llama_token tok) -> bool {
        if (llama_vocab_is_eog(s->vocab, tok)) return false;
        const std::string piece = common_token_to_piece(s->lctx, tok);
        if (piece == "<turn|>" || piece == "<|turn>") return false;
        if (on_token) on_token(piece.c_str());
        push_tts_collapsed(piece);
        s->prompt_hist.push_back(tok);
        n_generated++;
        return true;
    };

    if (s->spec_on) {
        // ── Speculative path ──────────────────────────────────────────────
        // Per step: the MTP head drafts up to spec_n_max tokens, the target
        // decodes [id_last, draft...] in ONE batch, and the sampler tells us
        // how many of the drafts it would have produced itself. Everything
        // past the accepted prefix is trimmed back out of both KV caches.
        llama_context * ctx_dft = s->spec_init->context();
        llama_batch batch_gen = llama_batch_init(s->spec_n_max + 1, 0, 1);

        // Seed the drafter with the target's hidden state at the last suffix
        // position — the position whose logits we are about to sample. The
        // audio prefill never passes through here; the MTP head works off
        // that hidden state, which already encodes the audio.
        common_batch_clear(batch_gen);
        common_batch_add(batch_gen, suffix_last, s->n_past - 1, {0}, true);
        common_speculative_process(s->spec.get(), batch_gen);

        // First token comes from the suffix prefill's logits, before any
        // drafting can happen.
        llama_token id_last = common_sampler_sample(s->smpl, s->lctx, -1);
        t_first_tok = clk::now();
        common_sampler_accept(s->smpl, id_last, true);
        bool go = emit_token(id_last);

        while (go && n_generated < s->n_predict && !s->abort_flag.load()) {
            llama_tokens draft;
            common_speculative_get_draft_params(s->spec.get(), /*seq=*/ 0) = {
                /* .drafting = */ true,
                /* .n_max    = */ -1,
                /* .n_past   = */ s->n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &s->prompt_hist,
                /* .result   = */ &draft,
            };
            common_speculative_draft(s->spec.get());
            n_drafted += (int) draft.size();

            // id_last has been sampled but never decoded, so it takes the
            // first free slot; drafts follow it.
            common_batch_clear(batch_gen);
            common_batch_add(batch_gen, id_last, s->n_past, {0}, true);
            for (size_t i = 0; i < draft.size(); i++) {
                common_batch_add(batch_gen, draft[i], s->n_past + 1 + (llama_pos) i, {0}, true);
            }
            if (llama_decode(s->lctx, batch_gen)) {
                s->error = "end_turn: target decode failed";
                break;
            }
            if (!common_speculative_process(s->spec.get(), batch_gen)) {
                s->error = "end_turn: speculative process failed";
                break;
            }

            // Length 1 + <accepted drafts>; every element is a token we have
            // not emitted yet (id_last went out before the loop / last round).
            auto accepted = common_sampler_sample_and_accept_n(s->smpl, s->lctx, draft);
            if (accepted.empty()) {
                s->error = "end_turn: sampler returned no tokens";
                break;
            }
            const int n_new = (int) accepted.size();
            common_speculative_accept(s->spec.get(), /*seq=*/ 0, n_new - 1);
            n_accepted += n_new - 1;

            // KV advanced by 1 + draft.size(); we keep only n_new of those.
            s->n_past += (llama_pos) n_new;
            id_last = accepted.back();
            llama_memory_seq_rm(llama_get_memory(s->lctx), /*seq=*/ 0, s->n_past, /*p1=*/ -1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft),  /*seq=*/ 0, s->n_past, /*p1=*/ -1);

            for (int i = 0; i < n_new; i++) {
                if (n_generated >= s->n_predict || !emit_token(accepted[i])) { go = false; break; }
                // Barge-in checked per token, not just per batch, so an
                // interrupt stops feeding TTS as promptly as it used to.
                if (s->abort_flag.load()) { go = false; break; }
            }
        }
        llama_batch_free(batch_gen);
    } else {
        // ── One token per decode ──────────────────────────────────────────
        llama_batch one_tok = llama_batch_init(1, 0, 1);
        for (int i = 0; i < s->n_predict && !s->abort_flag.load(); i++) {
            const llama_token tok = common_sampler_sample(s->smpl, s->lctx, -1);
            if (i == 0) t_first_tok = clk::now();
            common_sampler_accept(s->smpl, tok, true);
            if (!emit_token(tok)) break;

            common_batch_clear(one_tok);
            common_batch_add(one_tok, tok, s->n_past, {0}, true);
            if (llama_decode(s->lctx, one_tok)) {
                // Advance n_past only on success — a failed decode put nothing
                // in the cache, and an n_past that runs ahead of the cache
                // would decode every later token at the wrong position.
                s->error = "end_turn: sample decode failed";
                break;
            }
            s->n_past++;
        }
        llama_batch_free(one_tok);
    }
    const auto t_gen_end = clk::now();

    if (on_done) on_done();

    // Signal end-of-text + join the TTS worker (synth flushes its trailing
    // window through on_audio, then returns).
    if (tts) vibevoice_tts_stream_free(tts);
    s->tts_session.store(nullptr);

    const auto t_tts_end = clk::now();

    // Record stats. tts_samples atomic has been written by the worker;
    // we read it now that the worker has joined.
    s->stats.n_audio_tokens = s->n_audio_tokens;
    s->stats.n_llm_tokens   = n_generated;
    s->stats.n_tts_samples  = s->tts_samples.load();
    s->stats.n_drafted      = n_drafted;
    s->stats.n_accepted     = n_accepted;
    s->stats.ms_ttft =
        (t_first_tok.time_since_epoch().count() == 0) ? 0.0 : ms_since(t0, t_first_tok);
    s->stats.ms_llm_gen  = ms_since(t_gen_begin, t_gen_end);
    s->stats.ms_tts_wall = ms_since(t_tts_begin, t_tts_end);

    // An aborted turn (barge-in) leaves a reply that stops mid-sentence, and
    // begin_turn would then close it with <turn|> as though the model had
    // chosen to stop there. Repeat that a few times and the context is a
    // worked example of "assistant turns are truncated" — measured here, ten
    // interrupted turns shrink the next clean reply from a full sentence to a
    // clipped fragment, which reads as the model stuttering.
    //
    // So drop the whole exchange instead: the user interrupted, meaning they
    // did not want that answer, and their interrupting speech becomes the next
    // turn's audio via the caller's lookback. The cost is that a follow-up
    // which refers back to the abandoned question ("no, I meant Berlin") loses
    // its antecedent — judged the smaller harm than teaching the model to
    // truncate itself.
    if (s->abort_flag.load()) {
        rollback_turn();
        s->in_turn = false;
        return true;          // deliberately NOT s->turn++ — the turn did not happen
    }

    s->in_turn = false;
    s->turn++;
    return true;
}

void VoiceSession::abort_turn() {
    Impl * s = impl.get();
    if (s->abort_flag.exchange(true)) return;  // already requested
    if (auto * tts = s->tts_session.load()) {
        vibevoice_tts_stream_abort(tts);
    }
}

