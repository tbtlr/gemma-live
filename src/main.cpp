// gemma-live — the app: audio frontend + turn state machine.
//
// Interaction model (hands-free):
//   1. Say the wake phrase ("hey gemma") → begin listening
//   2. Speak; the EOU VAD (firered_vad on the live mic) flips us to REPLY
//      after ~500 ms of silence following at least one voiced hop
//   3. The model replies; start talking during the reply to barge in (the
//      reply aborts and the loop re-enters listening on the same speech)
//   4. After the reply, a follow-up window keeps the conversation open
//      without re-saying the wake phrase; it times out back to idle
//
// This file owns everything VoiceSession deliberately does NOT:
//   - miniaudio capture (mic, 16 kHz mono f32) + playback (at the TTS rate)
//   - LocalVQE neural AEC, and the detectors layered on top of it
//   - SIGINT handler for graceful shutdown
//
#include "session.h"
#include "barge.h"
#include "transcript.h"
#include "vqe.h"

#include "firered_vad.h"
#include "moonshine_streaming.h"          // CrispASR Moonshine — voice-command ASR

// miniaudio: include the implementation, but keep its symbols TU-local so
// they don't clash with the copy already linked inside libmtmd. MA_API=static
// makes every public miniaudio function internal-linkage; MA_IMPLEMENTATION
// pulls in the bodies.
#define MA_IMPLEMENTATION
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_API static
#include "miniaudio/miniaudio.h"

#include "arg.h"
#include "common.h"

#include <algorithm>
#include <atomic>
#include <cctype>    // isalpha, isdigit, tolower (normalise)
#include <chrono>
#include <cmath>     // sqrt, log10, pow (RMS gates)
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#  include <unistd.h>     // write (used by sigint_handler)
#endif

// ────────────────────────────────────────────────────────────────────────
// PCM queue — raw mic samples, used only by the startup liveness probe.
// Everything downstream reads the AEC'd rings instead.
// ────────────────────────────────────────────────────────────────────────
struct pcm_queue {
    std::mutex         m;
    std::vector<float> buf;

    void push(const float * s, size_t n) {
        std::lock_guard<std::mutex> lk(m);
        buf.insert(buf.end(), s, s + n);
    }
    size_t drain(std::vector<float> & out, size_t max_samples) {
        std::lock_guard<std::mutex> lk(m);
        const size_t take = std::min(buf.size(), max_samples);
        out.assign(buf.begin(), buf.begin() + take);
        buf.erase(buf.begin(), buf.begin() + take);
        return take;
    }
    void clear() {
        std::lock_guard<std::mutex> lk(m);
        buf.clear();
    }
};

// ────────────────────────────────────────────────────────────────────────
// End-of-utterance VAD — runs firered_vad during LISTENING over the same
// AEC'd chunks that get pushed to the model (see the pump loop in main).
// Time-based: once we've heard at least one voiced VAD verdict, fire EOU
// after SILENCE_MS_FOR_EOU milliseconds of wall time without another
// voiced verdict.
// ────────────────────────────────────────────────────────────────────────
struct cli_eou_vad {
    firered_vad_context * vad = nullptr;

    static constexpr int WINDOW_SAMPLES      = 12800;  // 800 ms at 16 kHz
    static constexpr int HOP_MS              = 100;    // VAD eval cadence
    static constexpr int DEFAULT_SILENCE_MS  = 500;    // default EOU threshold

    // Silence threshold, extended while the user is audibly mid-thought.
    // Written by the keyword-detector worker (which is already transcribing
    // continuously), read here on the main thread — hence atomic.
    //
    // It only ever goes UP. Shortening it on a completed sentence was the
    // other half of the original idea, and moonshine's rolling transcript
    // cannot support it: measured against a 3.9 s utterance, the text ends
    // in a period at 3.0 s ("...stays up in the area.") and again at 3.2 s
    // ("3.") while the speaker is still going. Acting on that cuts people
    // off mid-sentence, which costs far more than the ~200 ms it would save.
    // A hanging conjunction is the safe direction: guessing wrong just adds
    // a little latency.
    std::atomic<int> silence_threshold_ms{DEFAULT_SILENCE_MS};
    static constexpr int HANGING_SILENCE_MS = 900;

    std::vector<float> window;
    bool   ever_voiced = false;
    std::chrono::steady_clock::time_point last_check;
    std::chrono::steady_clock::time_point last_voiced_at;
    bool   debug = false;

    bool init(const char * model_path) {
        debug = std::getenv("GEMMA_LIVE_VAD_DEBUG") != nullptr;
        vad = firered_vad_init(model_path);
        return vad != nullptr;
    }
    void shutdown() {
        if (vad) { firered_vad_free(vad); vad = nullptr; }
    }
    void reset() {
        window.clear();
        ever_voiced    = false;
        last_check     = std::chrono::steady_clock::now();
        last_voiced_at = std::chrono::steady_clock::now();
        silence_threshold_ms.store(DEFAULT_SILENCE_MS);
    }
    // True once feed() has seen at least one voiced VAD verdict.

    // Feed a chunk of recently-pumped audio. Returns true the first time
    // SILENCE_MS_FOR_EOU has elapsed since the most recent voiced verdict,
    // provided we've heard at least one voiced verdict already.
    bool feed(const float * pcm, size_t n) {
        if (!vad) return false;
        // Apply a fixed pre-gain so the audio sits in the level range that
        // firered_vad was trained on (~-26 dBFS). Even after the AEC's AGC2
        // the macOS capture path lands well below that, low enough that
        // speech VAD models get unreliable. 8x is +18 dB; combined with the
        // (post-gain) threshold of 0.3 below it gives a wide-but-not-trigger-
        // happy detection band. Samples are clamped, so the loud case just
        // saturates rather than wrapping.
        constexpr float GAIN = 8.0f;
        const size_t off = window.size();
        window.resize(off + n);
        for (size_t i = 0; i < n; i++) {
            const float v = pcm[i] * GAIN;
            window[off + i] = std::max(-1.0f, std::min(1.0f, v));
        }
        if ((int) window.size() > WINDOW_SAMPLES) {
            window.erase(window.begin(),
                         window.begin() + (window.size() - WINDOW_SAMPLES));
        }
        const auto now = std::chrono::steady_clock::now();
        const auto since_check = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_check).count();
        if (since_check < HOP_MS) return false;
        if ((int) window.size() < WINDOW_SAMPLES) return false;
        last_check = now;

        firered_vad_segment * segs = nullptr;
        int n_segs = 0;
        firered_vad_detect(vad, window.data(), (int) window.size(),
                           &segs, &n_segs,
                           /*threshold=*/        0.3f,
                           /*min_speech_sec=*/   0.25f,
                           /*min_silence_sec=*/  0.10f);
        const bool voiced = (n_segs > 0);
        if (segs) std::free(segs);

        if (debug) {
            // RMS of the current 800 ms window after pre-gain, in dBFS.
            double sumsq = 0.0;
            for (float s : window) sumsq += (double) s * s;
            const float rms = (float) std::sqrt(sumsq / (double) window.size());
            const float db  = (rms > 1e-6f) ? 20.0f * std::log10(rms) : -120.0f;
            fprintf(stderr, "  [vad: %s | rms %.3f (%.0f dBFS) | win %zu]\n",
                    voiced ? "VOICE " : "silent",
                    rms, db, window.size());
        }

        if (voiced) {
            if (!ever_voiced) {
                if (debug) fprintf(stderr, "  [vad: speech onset]\n");
                ever_voiced = true;
            }
            last_voiced_at = now;
        }
        if (ever_voiced) {
            const auto silence_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_voiced_at).count();
            const int thresh = silence_threshold_ms.load();
            if (silence_ms >= thresh) {
                if (debug) fprintf(stderr, "  [vad: EOU after %lld ms (thresh=%d)]\n",
                                   (long long) silence_ms, thresh);
                return true;
            }
        }
        return false;
    }
};

// ────────────────────────────────────────────────────────────────────────
// Barge-in / followup VAD, on the AEC'd mic. Mandatory — main aborts if
// cli_aec::init fails, since every mic consumer reads the AEC output.
//
// Pipeline:  mic (16 kHz)          ─┐
//                                    ├─► LocalVQE AEC (16 ms hops) ─► rings
//   speaker (tts_rate → 16 kHz)    ─┘
//
// Independent of cli_eou_vad above — both read AEC'd audio, but that one
// gates the LISTENING → REPLYING transition off end-of-utterance silence,
// while this one watches during REPLYING / AWAITING_FOLLOWUP for the user
// starting to talk.
// ────────────────────────────────────────────────────────────────────────

// Phase + the flags the detector worker raises. Defined here, ahead of the
// detectors that reference them, so the rest of the file needn't be
// rearranged.
enum gl_phase : int {
    GL_IDLE              = 0,   // waiting for wake word
    GL_LISTENING         = 1,   // capturing audio for a user turn
    GL_REPLYING          = 2,   // TTS playing back; barge_detector armed
    GL_AWAITING_FOLLOWUP = 3,   // post-reply window; followup VAD armed;
                                // timeout returns us to GL_IDLE
};
static std::atomic<int>   g_phase          {GL_IDLE};
static std::atomic<bool>  g_interrupted    {false};   // barge fired during REPLYING
static std::atomic<bool>  g_voice_activity {false};   // barge OR followup VAD fired
static VoiceSession     * g_session = nullptr;
static int64_t mono_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ────────────────────────────────────────────────────────────────────────
// Followup voice-start detector — fed from inside the keyword detector
// worker so we don't need a second AEC consumer for it.
//
// Armed ONLY during GL_AWAITING_FOLLOWUP, where the assistant has stopped
// speaking: no echo, so a VAD reads the mic honestly and is the right tool.
// Sustained voice sets g_voice_activity; no abort is needed because no turn
// is in flight. The main loop polls it and transitions to LISTENING, keeping
// the AEC session_buf so the user's leading words aren't dropped.
//
// It is deliberately NOT armed during GL_REPLYING. There the AEC has
// suppressed the user's voice along with the echo and a VAD cannot see the
// interruption at all — barge_detector (barge.h) owns that phase instead.
//
// Two gates filter false positives in REPLYING (AEC residual would
// otherwise tug the detector):
//   1) RMS gate (default -30 dBFS). Real user speech ≥ -30; AEC residual
//      typically ≤ -40. ~10 dB margin.
//   2) Sustained-voice gate. Single noisy hop doesn't fire; we require
//      N consecutive 100 ms hops (default 3 = 300 ms) of voiced verdict.
// ────────────────────────────────────────────────────────────────────────
struct cli_voice_vad {
    firered_vad_context * vad = nullptr;
    std::atomic<bool>     active{false};

    // Re-arm request, posted by set_active() on the main thread and consumed
    // at the top of feed() on the detector worker. Deferring it this way keeps
    // `window` and `voiced_hops` owned by the worker alone — clearing them
    // from the main thread could land while feed() is mid-iteration.
    std::atomic<bool>     reset_req{false};

    static constexpr int WINDOW_SAMPLES = 12800;  // 800 ms @ 16 kHz, the cap
    // Smallest window we will run the VAD over. Arming clears the window (it
    // holds the reply we were not watching), so waiting for the full 800 ms
    // before the FIRST verdict put a floor of 800 + 300 = 1.1 s under every
    // followup. firered_vad takes any length and needs 250 ms of speech to
    // call a segment, so 400 ms is enough to ask — and the window keeps
    // growing to 800 ms for the verdicts after it.
    static constexpr int MIN_EVAL_SAMPLES = 6400;   // 400 ms @ 16 kHz
    static constexpr int HOP_MS         = 100;
    int   required_consecutive_hops = 3;          // 300 ms sustained → fire
    float rms_gate_dbfs             = -30.0f;     // AEC residual ≤ ~-40 dBFS

    std::vector<float>    window;
    int                   voiced_hops = 0;
    std::chrono::steady_clock::time_point last_check;
    bool                  debug = false;

    bool init(const char * model_path) {
        debug = std::getenv("GEMMA_LIVE_VAD_DEBUG") != nullptr;
        if (const char * v = std::getenv("GEMMA_LIVE_VAD_HOPS")) {
            required_consecutive_hops = std::max(1, atoi(v));
        }
        if (const char * v = std::getenv("GEMMA_LIVE_VAD_RMS_DBFS")) {
            rms_gate_dbfs = (float) std::atof(v);
        }
        vad = firered_vad_init(model_path);
        return vad != nullptr;
    }
    void shutdown() {
        if (vad) { firered_vad_free(vad); vad = nullptr; }
    }
    // Arming always starts from scratch: the window holds audio from phases
    // this VAD wasn't watching (the whole reply), which must not count toward
    // a followup.
    void set_active(bool a) {
        if (a) reset_req.store(true);
        active.store(a);
    }
    // Called from cli_keyword_detector worker with each AEC'd chunk
    // (16 kHz mono f32). When inactive this is a near no-op (just the
    // atomic-bool load).
    void feed(const float * pcm, size_t n) {
        if (!active.load() || !vad || n == 0) return;
        if (reset_req.exchange(false)) {
            voiced_hops = 0;
            window.clear();
            last_check = std::chrono::steady_clock::now();
        }
        const size_t off = window.size();
        window.resize(off + n);
        std::memcpy(window.data() + off, pcm, n * sizeof(float));
        if ((int) window.size() > WINDOW_SAMPLES) {
            window.erase(window.begin(),
                         window.begin() + (window.size() - WINDOW_SAMPLES));
        }
        const auto now = std::chrono::steady_clock::now();
        const auto since_check = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_check).count();
        if (since_check < HOP_MS) return;
        if ((int) window.size() < MIN_EVAL_SAMPLES) return;
        last_check = now;

        // RMS gate first — cheap, kills most AEC-residual false positives
        // before we pay for the VAD inference.
        double sumsq = 0.0;
        for (float s : window) sumsq += (double) s * s;
        const float rms = (float) std::sqrt(sumsq / (double) window.size());
        const float rms_gate = std::pow(10.0f, rms_gate_dbfs / 20.0f);
        if (rms < rms_gate) {
            voiced_hops = 0;
            if (debug) {
                const float db = (rms > 1e-6f) ? 20.0f * std::log10(rms) : -120.0f;
                fprintf(stderr, "  [vad: gated %.0f dBFS < %.0f]\n", db, rms_gate_dbfs);
            }
            return;
        }

        firered_vad_segment * segs = nullptr;
        int n_segs = 0;
        firered_vad_detect(vad, window.data(), (int) window.size(),
                           &segs, &n_segs,
                           /*threshold=*/        0.3f,
                           /*min_speech_sec=*/   0.25f,
                           /*min_silence_sec=*/  0.10f);
        const bool voiced = (n_segs > 0);
        if (segs) std::free(segs);

        if (voiced) {
            voiced_hops++;
            if (debug) {
                fprintf(stderr, "  [vad: voiced hop %d/%d]\n",
                        voiced_hops, required_consecutive_hops);
            }
            if (voiced_hops >= required_consecutive_hops) {
                if (!g_voice_activity.exchange(true)) {
                    const int phase = g_phase.load();
                    if (phase == GL_REPLYING) {
                        g_interrupted.store(true);
                        fprintf(stderr, "  [↩ voice bargein]\n");
                        fflush(stderr);
                        if (g_session) g_session->abort_turn();
                    } else if (phase == GL_AWAITING_FOLLOWUP) {
                        // Main loop will see g_voice_activity and continue.
                    }
                }
            }
        } else {
            voiced_hops = 0;
        }
    }
};

struct cli_aec {
    vqe_aec            vqe;
    std::vector<float> vqe_out;   // reused across iterations; wrapper appends

    // LocalVQE is the largest fixed cost in this app: ~73% of a core,
    // continuously, measured subtractively in situ (a standalone harness
    // says ~20% — do not trust it, it does not reproduce the thread config).
    //
    // An idle bypass to reclaim that has been tried TWICE and is not shipped.
    //
    // The naive version forwards the raw mic while the room is quiet. It made
    // things dramatically worse: LocalVQE's noise gate takes a quiet room to
    // exactly zero, and that zero is what keeps the wake detector's own gate
    // shut. Feeding raw ambient instead re-opens it and runs moonshine flat
    // out. The bypass fires precisely when the room is quiet, so it did the
    // most damage exactly when it engaged.
    //
    // The corrected version publishes zeros rather than raw mic, at a
    // threshold matching LocalVQE's own noise gate. That reasoning is sound
    // but unproven: the benefit only appears in a genuinely silent room,
    // which is hard to measure in, and it puts a hard gate on the audio path
    // — speech below the threshold would be zeroed where the model might
    // have recovered it. Unmeasurable upside plus a real downside is not a
    // trade worth making on the path the whole app depends on.
    //
    // If you revisit it: measure in a silent room, and verify wake-word
    // recall at low speech levels before trusting it.

    std::mutex         cap_m, ren_m;
    std::vector<float> cap_buf;
    std::vector<float> ren_buf;
    double             ren_phase = 0.0;
    std::atomic<bool>  running{false};
    std::thread        worker;

    // AEC'd output rings (16 kHz mono f32). Populated by the worker after
    // every successful 10 ms ProcessStream call. Bounded — drops oldest
    // on overflow so a slow/absent consumer never grows it without bound.
    //
    // Two parallel rings, one per consumer, so each consumer can drain
    // independently without racing:
    //   out_buf      → keyword detector (wake-word matching)
    //   session_buf  → main loop (audio fed to Gemma via push_audio)
    // Same data, different read positions.
    std::mutex                  out_m;
    std::vector<float>          out_buf;
    std::mutex                  session_m;
    std::vector<float>          session_buf;
    static constexpr size_t     OUT_CAP_SAMPLES = 16000 * 8;   // 8 s of AEC'd audio

    // Smoothed RMS of the render reference the worker just consumed, i.e.
    // how loud the speaker is RIGHT NOW. The barge detector needs it to tell
    // "assistant audible" from "assistant between chunks" — it may only learn
    // its echo baseline while there is actually echo to learn from.
    // One EMA over 10 ms frames, tau ~50 ms.
    std::atomic<float>          far_rms{0.0f};

    bool init(const std::string & model_path) {
        // Threads: LocalVQE is ~5M params on CPU and has to keep up with a
        // 16 ms hop while the LLM owns the performance cores. 2 is plenty and
        // leaves the cores where the latency actually is.
        int threads = 2;
        if (const char * v = std::getenv("GEMMA_LIVE_VQE_THREADS")) {
            threads = std::max(1, std::atoi(v));
        }
        float gate_dbfs = -45.0f;
        if (const char * v = std::getenv("GEMMA_LIVE_VQE_GATE_DBFS")) {
            gate_dbfs = (float) std::atof(v);
        }
        if (!vqe.init(model_path, threads, gate_dbfs)) return false;
        vqe_out.reserve(4096);
        return true;
    }
    void start() {
        {
            std::lock_guard<std::mutex> a(cap_m);
            std::lock_guard<std::mutex> b(ren_m);
            cap_buf.clear();
            ren_buf.clear();
            ren_phase = 0.0;
        }
        {
            std::lock_guard<std::mutex> lk(out_m);
            out_buf.clear();
        }
        {
            std::lock_guard<std::mutex> lk(session_m);
            session_buf.clear();
        }
        running.store(true);
        worker = std::thread([this]() { run(); });
    }
    void stop() {
        running.store(false);
        if (worker.joinable()) worker.join();
    }
    void push_capture(const float * p, size_t n) {
        std::lock_guard<std::mutex> lk(cap_m);
        cap_buf.insert(cap_buf.end(), p, p + n);
    }
    // Push render-side PCM at the configured source rate. The AEC runs at
    // 16 kHz internally, so we decimate by the ratio src/16000 using a
    // simple phase accumulator (same shape as the original push_render_24k,
    // generalised). Examples:
    //   src=24000 → 2/3 decimation (every 3rd input → 2 outputs)
    //   src=48000 → 1/3 decimation (every 3rd input → 1 output)
    //   src=16000 → identity (every input → 1 output)
    // The 16k internal rate is hard-coded to match StreamConfig(16000, 1).
    int  render_src_rate = 24000;
    void set_render_src_rate(int hz) {
        std::lock_guard<std::mutex> lk(ren_m);
        if (hz <= 0 || hz == render_src_rate) return;
        render_src_rate = hz;
        ren_phase = 0.0;
        ren_buf.clear();
    }
    void push_render(const float * p, size_t n) {
        std::lock_guard<std::mutex> lk(ren_m);
        const double step = 16000.0 / (double) render_src_rate;
        for (size_t i = 0; i < n; i++) {
            ren_phase += step;
            while (ren_phase >= 1.0) {
                ren_phase -= 1.0;
                ren_buf.push_back(p[i]);
            }
        }
        // Backstop only. Fed from the playback callback, this queue drains at
        // the DAC's own rate and should sit near-empty; it can only build up
        // if the AEC worker stalls, and a reference that far behind the mic
        // cancels nothing anyway. Drop the oldest rather than let the lag —
        // and the memory — grow without bound.
        constexpr size_t REN_CAP = 16000;  // 1 s at the AEC's internal rate
        if (ren_buf.size() > REN_CAP) {
            ren_buf.erase(ren_buf.begin(), ren_buf.begin() + (ren_buf.size() - REN_CAP));
        }
    }
    // Note: there is deliberately no flush_render(). Dropping queued playback
    // (barge-in, LISTENING entry) must NOT drop the queued reference — what
    // is still in here corresponds to sound the speaker already emitted and
    // the room is still echoing. Discarding it would misalign the very
    // streams the playback tap exists to keep locked.
    // Pull at least one AEC'd 16 kHz mono frame's worth of samples (if any)
    // into dst. Returns count taken. Caps the take at max_samples.
    size_t drain_out(std::vector<float> & dst, size_t max_samples) {
        std::lock_guard<std::mutex> lk(out_m);
        const size_t take = std::min(out_buf.size(), max_samples);
        if (take == 0) { dst.clear(); return 0; }
        dst.assign(out_buf.begin(), out_buf.begin() + take);
        out_buf.erase(out_buf.begin(), out_buf.begin() + take);
        return take;
    }
    // Parallel drain for the model-input audio path. The same 10 ms frames
    // the keyword detector sees on out_buf land here too — each consumer
    // has its own read position, so neither starves the other.
    size_t drain_session(std::vector<float> & dst, size_t max_samples) {
        std::lock_guard<std::mutex> lk(session_m);
        const size_t take = std::min(session_buf.size(), max_samples);
        if (take == 0) { dst.clear(); return 0; }
        dst.assign(session_buf.begin(), session_buf.begin() + take);
        session_buf.erase(session_buf.begin(), session_buf.begin() + take);
        return take;
    }
    size_t session_size() {
        std::lock_guard<std::mutex> lk(session_m);
        return session_buf.size();
    }
    // Trim the session ring down to keep at most `keep_samples` of the
    // most recent audio. Used at each LISTENING entry so Gemma only sees
    // a bounded lookback (wake-phrase / interrupt / first-followup-word)
    // instead of the full 8 s ring. AEC has been running continuously
    // throughout, so what's preserved is post-AEC (echo-cancelled).
    void trim_session_to(size_t keep_samples) {
        std::lock_guard<std::mutex> lk(session_m);
        if (session_buf.size() > keep_samples) {
            session_buf.erase(session_buf.begin(),
                              session_buf.begin() + (session_buf.size() - keep_samples));
        }
    }
    void run() {
        // One LocalVQE hop per iteration. Unlike a classic AEC this does not
        // need the mic and reference handed over in lock-step — the model
        // estimates the alignment itself — but it does want equal-length
        // pairs, so we still take them a hop at a time.
        const size_t F = (size_t) std::max(1, vqe.hop());
        std::vector<float> cap(F), ren(F);
        while (running.load()) {
            bool have_cap = false;
            {
                std::unique_lock<std::mutex> a(cap_m, std::defer_lock);
                std::unique_lock<std::mutex> b(ren_m, std::defer_lock);
                std::lock(a, b);
                if (cap_buf.size() >= F) {
                    std::memcpy(cap.data(), cap_buf.data(), F * sizeof(float));
                    cap_buf.erase(cap_buf.begin(), cap_buf.begin() + F);
                    if (ren_buf.size() >= F) {
                        std::memcpy(ren.data(), ren_buf.data(), F * sizeof(float));
                        ren_buf.erase(ren_buf.begin(), ren_buf.begin() + F);
                    } else {
                        // Speaker idle (or the tap is behind): silence is the
                        // truthful reference, and keeps the two timelines
                        // advancing together.
                        std::memset(ren.data(), 0, F * sizeof(float));
                    }
                    have_cap = true;
                }
            }
            if (!have_cap) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            // How loud the speaker is this hop — barge.h needs it to know
            // whether there is any echo to learn a floor from.
            {
                double sumsq = 0.0;
                for (size_t i = 0; i < F; i++) sumsq += (double) ren[i] * ren[i];
                const float rms  = (float) std::sqrt(sumsq / (double) F);
                const float prev = far_rms.load(std::memory_order_relaxed);
                far_rms.store(0.8f * prev + 0.2f * rms, std::memory_order_relaxed);
            }

            vqe_out.clear();
            vqe.process(cap.data(), ren.data(), F, vqe_out);
            if (vqe_out.empty()) continue;

            // Publish to both consumer rings. Each is drained independently —
            // out_buf by the detector worker, session_buf by the main loop.
            {
                std::lock_guard<std::mutex> lk(out_m);
                out_buf.insert(out_buf.end(), vqe_out.begin(), vqe_out.end());
                if (out_buf.size() > OUT_CAP_SAMPLES) {
                    out_buf.erase(out_buf.begin(),
                                  out_buf.begin() + (out_buf.size() - OUT_CAP_SAMPLES));
                }
            }
            {
                std::lock_guard<std::mutex> lk(session_m);
                session_buf.insert(session_buf.end(), vqe_out.begin(), vqe_out.end());
                if (session_buf.size() > OUT_CAP_SAMPLES) {
                    session_buf.erase(session_buf.begin(),
                                      session_buf.begin() + (session_buf.size() - OUT_CAP_SAMPLES));
                }
            }
        }
    }
};

// cli_aec instance — defined just below the detector class for state-init
// reasons, but the detector's worker thread needs to call into it. Forward
// declare here so cli_keyword_detector::run() can reference it.
static struct cli_aec g_aec;

// ────────────────────────────────────────────────────────────────────────
// Wake detector — moonshine streaming ASR + substring match on the rolling
// transcript. Acts on hits only when phase == GL_IDLE; the streaming model
// still runs in other phases (it needs continuous audio), output is ignored.
// ────────────────────────────────────────────────────────────────────────
static std::atomic<bool> g_wake_fired{false};

struct cli_keyword_detector {
    moonshine_streaming_context * ctx    = nullptr;
    moonshine_streaming_stream  * stream = nullptr;
    std::thread                   worker;
    std::atomic<bool>             running{false};
    bool                          debug = false;  // GEMMA_LIVE_KWD_DEBUG=1

    // Track moonshine's emission counter so we only re-evaluate the
    // transcript on actual new emissions (not on stale repeats).
    int64_t                       last_counter = -1;

    // Energy gate — only active during REPLYING (AEC-residual suppression).
    // During IDLE the gate would silence quiet user speech.
    float                         gate_rms = 0.0f;
    float                         gate_dbfs = -35.0f;

    // Streaming params. step_ms = how often the transcript is recomputed,
    // length_ms = the rolling window each inference encodes. Cost per second
    // of audio scales as length_ms / step_ms, and this model runs CONTINUOUSLY
    // for as long as the app is open, so it dominates idle power.
    //
    // Measured (cores burned to keep up with realtime, M-series):
    //     step=100 len=2000   226%   <- old default: could not keep up at all
    //     step=200 len=2000   115%
    //     step=200 len=1500    85%   <- default now
    //     step=300 len=1500    58%
    //     step=300 len=1000    40%
    //
    // 200/1500 keeps a comfortable margin over the ~1 s wake phrase while
    // costing a third of what it used to. Raise step_ms for more savings at
    // the cost of wake latency; shorten length_ms only if the phrase still
    // fits inside it.
    int                           step_ms   = 200;
    int                           length_ms = 1500;

    // Skip inference entirely while the room is quiet. The wake detector only
    // has to react to speech, and at idle — the state this app spends nearly
    // all its time in — there is none. A short pre-roll is replayed on the
    // silence→sound transition so the phrase onset is not clipped.
    //
    // The threshold is LEARNED, not fixed. A constant has to sit above the
    // room's noise floor and below quiet speech, and those are only ~10 dB
    // apart and different in every room: measured here, 0.01 left the gate
    // permanently open (37% of a core burned on silence) while 0.03 closed it
    // completely (0.9%) — the floor was sitting between the two. So track the
    // floor with an EMA while the gate is closed and open at a multiple of it,
    // the same shape barge.h uses against the echo floor.
    //
    // Learning only while closed is what keeps it honest: speech can never be
    // absorbed into the floor and raise the bar against itself.
    float                         gate_ratio     = 3.0f;    // open at N x floor
    float                         gate_abs_floor = 0.008f;  // never open below this
    int                           gate_hangover_ms = 700;
    float                         noise_floor    = 0.0f;    // learned, EMA
    int                           floor_hops     = 0;
    bool                          gate_off       = false;
    int                           gate_hops_total = 0, gate_hops_open = 0;
    int64_t                       gate_log_ms    = 0;
    std::vector<float>            preroll;                  // ~300 ms
    int64_t                       loud_until_ms = 0;

    // Loaded wake-phrase list. Stored pre-wrapped with single leading +
    // trailing spaces (shape normalise() produces) so substring matching
    // works on word boundaries.
    std::vector<std::string>      wake_phrases;

    // Both detectors ride this worker's AEC'd chunks so we don't need extra
    // consumers on the ring. They are never active at the same time: the
    // energy detector owns REPLYING (double-talk), the VAD owns
    // AWAITING_FOLLOWUP (assistant silent, so a plain VAD is the better tool).
    cli_voice_vad               * voice_vad = nullptr;
    barge_detector              * barge     = nullptr;
    cli_eou_vad                 * eou_vad   = nullptr;   // threshold target

    bool init(const char * model_path, const char * wake_path) {
        debug = std::getenv("GEMMA_LIVE_KWD_DEBUG") != nullptr;
        if (const char * v = std::getenv("GEMMA_LIVE_KWD_GATE_DBFS")) {
            gate_dbfs = (float) std::atof(v);
        }
        gate_rms = std::pow(10.0f, gate_dbfs / 20.0f);
        if (const char * v = std::getenv("GEMMA_LIVE_KWD_STEP_MS"))   step_ms   = std::max(50, atoi(v));
        if (const char * v = std::getenv("GEMMA_LIVE_KWD_LENGTH_MS")) length_ms = std::max(1000, atoi(v));
        if (const char * v = std::getenv("GEMMA_LIVE_KWD_GATE_RATIO")) gate_ratio     = (float) std::atof(v);
        // Escape hatch: feed every chunk, as before the gate existed. Useful
        // for A/B-ing what the gate is actually worth on a given machine.
        if (const char * v = std::getenv("GEMMA_LIVE_KWD_GATE_OFF")) gate_off = (atoi(v) != 0);
        if (const char * v = std::getenv("GEMMA_LIVE_KWD_GATE_FLOOR")) gate_abs_floor = (float) std::atof(v);
        preroll.reserve(16000);

        static const char * builtin_wake[] = {
            "hey gemma",  "hey jemma",  "hey jemmah", "hey gemmer",
            "hey gema",   "hey jema",   "hey jenna",
            "hi gemma",   "hi jemma",
            "ok gemma",   "okay gemma", "okay jemma",
        };
        wake_phrases = load_phrases(wake_path, builtin_wake,
                                    sizeof(builtin_wake)/sizeof(builtin_wake[0]),
                                    "wake");

        auto params = moonshine_streaming_context_default_params();
        params.n_threads = 4;
        // CPU by default: measured identical to Metal for this model
        // (57.8% vs 58.1% of a core), and keeping it off the GPU leaves that
        // for the LLM and the TTS diffusion, which are the latency-critical
        // consumers. Set GEMMA_LIVE_KWD_USE_GPU=1 to put it back.
        params.use_gpu   = false;
        if (const char * v = std::getenv("GEMMA_LIVE_KWD_USE_GPU")) {
            params.use_gpu = (atoi(v) != 0);
        }
        params.verbosity = 0;
        ctx = moonshine_streaming_init_from_file(model_path, params);
        if (!ctx) return false;
        stream = moonshine_streaming_stream_open(ctx, step_ms, length_ms);
        if (!stream) {
            moonshine_streaming_free(ctx); ctx = nullptr;
            return false;
        }
        return true;
    }
    void shutdown() {
        if (stream) { moonshine_streaming_stream_close(stream); stream = nullptr; }
        if (ctx)    { moonshine_streaming_free(ctx);            ctx    = nullptr; }
    }
    void start() {
        running.store(true);
        worker = std::thread([this]() { run(); });
    }
    void stop() {
        running.store(false);
        if (worker.joinable()) worker.join();
    }

    static std::string normalise(const std::string & s) { return transcript::normalise(s); }
    static size_t n_loaded;   // phrases read from file; 0 = using built-ins
    static std::vector<std::string> load_phrases(const char * path,
                                                 const char * const * builtins,
                                                 size_t n_builtins,
                                                 const char * label) {
        std::vector<std::string> out;
        if (path && *path) {
            std::ifstream f(path);
            if (f.is_open()) {
                std::string line;
                while (std::getline(f, line)) {
                    const size_t s = line.find_first_not_of(" \t\r\n");
                    if (s == std::string::npos) continue;
                    if (line[s] == '#') continue;
                    std::string norm = normalise(line);
                    if (norm.size() > 2) out.push_back(std::move(norm));
                }
                n_loaded = out.size();
                if (!out.empty()) return out;
                fprintf(stderr, "asr      %s had no usable phrases — using built-ins\n", path);
            } else {
                fprintf(stderr, "asr      %s not found — using built-ins\n", path);
            }
        }
        for (size_t i = 0; i < n_builtins; i++) {
            std::string w = " ";
            w += builtins[i];
            w += " ";
            out.push_back(std::move(w));
        }
        return out;
    }
    bool contains_wake(const std::string & norm) const {
        for (const auto & v : wake_phrases) {
            if (norm.find(v) != std::string::npos) return true;
        }
        return false;
    }
    void run() {
        std::vector<float> chunk;
        std::string        text;
        text.reserve(2048);
        while (running.load()) {
            const size_t got = g_aec.drain_out(chunk, 16000);
            if (got > 0) {
                if (voice_vad) voice_vad->feed(chunk.data(), got);
                // Read the far-end level before the energy gate below can
                // zero parts of `chunk` — the detector needs the residual as
                // the AEC produced it.
                if (barge) barge->feed(chunk.data(), got,
                                       g_aec.far_rms.load(std::memory_order_relaxed));

                if (g_phase.load() == GL_REPLYING) {
                    constexpr size_t GATE_WINDOW = 1600;
                    const float gate_rms_sq = gate_rms * gate_rms;
                    for (size_t i = 0; i < got; i += GATE_WINDOW) {
                        const size_t w = std::min(GATE_WINDOW, got - i);
                        double sumsq = 0.0;
                        for (size_t j = 0; j < w; j++) {
                            sumsq += (double) chunk[i + j] * chunk[i + j];
                        }
                        if ((sumsq / (double) w) < gate_rms_sq) {
                            std::fill(chunk.begin() + i, chunk.begin() + i + w, 0.0f);
                        }
                    }
                }
                // Silence gate. Encoding a 1.5 s window several times a
                // second is the single largest standing cost in this app, and
                // at idle in a quiet room every one of those inferences
                // transcribes nothing. Skip the feed until the mic has energy,
                // then hold the gate open for `gate_hangover_ms` past the last
                // loud chunk so a pause mid-phrase does not chop it up.
                //
                // The pre-roll matters: without it the gate opens on the first
                // loud sample and the model never sees the attack of "hey",
                // which is exactly the part it needs.
                double sumsq = 0.0;
                for (size_t i = 0; i < got; i++) sumsq += (double) chunk[i] * chunk[i];
                const float rms = (float) std::sqrt(sumsq / (double) got);
                const int64_t now_ms = mono_ms();

                const float trigger = std::max(gate_abs_floor, gate_ratio * noise_floor);
                if (floor_hops > 8 && rms > trigger) loud_until_ms = now_ms + gate_hangover_ms;
                const bool gate_open = gate_off || (now_ms < loud_until_ms);
                if (!gate_open) {
                    // Quiet: this is what the room sounds like. tau ~2 s.
                    if (floor_hops == 0) noise_floor = rms;
                    else                 noise_floor = 0.95f * noise_floor + 0.05f * rms;
                    floor_hops++;
                }
                // Periodic gate report. Without this the only observable is
                // total CPU, which moves with room noise and cannot tell
                // "gate correctly open because someone is talking" from
                // "gate stuck open and burning a core on nothing".
                gate_hops_total++;
                if (gate_open) gate_hops_open++;
                if (debug && now_ms - gate_log_ms > 5000) {
                    gate_log_ms = now_ms;
                    fprintf(stderr, "  [kwd gate: open %.0f%% of last window | rms %.4f "
                                    "trigger %.4f floor %.4f]\n",
                            gate_hops_total ? 100.0 * gate_hops_open / gate_hops_total : 0.0,
                            rms, trigger, noise_floor);
                    gate_hops_total = gate_hops_open = 0;
                }

                if (gate_open) {
                    if (!preroll.empty()) {
                        moonshine_streaming_stream_feed(stream, preroll.data(), (int) preroll.size());
                        preroll.clear();
                    }
                    moonshine_streaming_stream_feed(stream, chunk.data(), (int) got);
                } else {
                    // Keep the most recent ~300 ms so the next opening has an
                    // onset to replay.
                    constexpr size_t PREROLL_SAMPLES = 16000 * 300 / 1000;
                    preroll.insert(preroll.end(), chunk.begin(), chunk.begin() + got);
                    if (preroll.size() > PREROLL_SAMPLES) {
                        preroll.erase(preroll.begin(),
                                      preroll.begin() + (preroll.size() - PREROLL_SAMPLES));
                    }
                }
            }
            char     buf[2048];
            double   t0 = 0, t1 = 0;
            int64_t  counter = 0;
            const int n = moonshine_streaming_stream_get_text(stream, buf, sizeof(buf),
                                                              &t0, &t1, &counter);
            if (n > 0 && counter != last_counter) {
                last_counter = counter;
                text.assign(buf, (size_t) n);
                const std::string norm = normalise(text);
                if (debug) {
                    fprintf(stderr, "  [kwd: \"%s\"]\n", text.c_str());
                }
                const int phase = g_phase.load();
                if (phase == GL_IDLE && contains_wake(norm)) {
                    if (!g_wake_fired.exchange(true)) {
                        fprintf(stderr, "  [↩ wake]\n");
                        fflush(stderr);
                    }
                } else if (phase == GL_LISTENING && eou_vad) {
                    // Give the user longer to finish when they have clearly not
                    // finished. Costs nothing when we are wrong; saves a whole
                    // wasted turn when we are right, because a truncated
                    // question means a useless answer and a restart.
                    const int ms = transcript::ends_mid_thought(norm)
                        ? cli_eou_vad::HANGING_SILENCE_MS
                        : cli_eou_vad::DEFAULT_SILENCE_MS;
                    if (eou_vad->silence_threshold_ms.exchange(ms) != ms && debug) {
                        fprintf(stderr, "  [kwd: eou threshold -> %d ms]\n", ms);
                    }
                }
            }
            if (got == 0 && n <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
    }
};

// ────────────────────────────────────────────────────────────────────────
// Globals (CLI-owned)
// Note: gl_phase enum + g_phase, g_interrupted, g_session, mono_ms() are
// defined further up so cli_aec can reference them. The rest live here.
// ────────────────────────────────────────────────────────────────────────
static std::atomic<bool>  g_shutdown     {false};

// True only during the startup mic-liveness probe. capture_cb feeds the raw
// pcm_queue while set; afterwards the mic path is AEC-only and the queue is
// left alone. See the probe just after ma_device_start below.
static std::atomic<bool>  g_mic_probe    {true};

// g_aec is declared above (so cli_keyword_detector can reference it).

static cli_eou_vad  g_eou_vad;

size_t cli_keyword_detector::n_loaded = 0;
static cli_keyword_detector g_kwd;

static cli_voice_vad        g_voice_vad;
static bool                 g_voice_vad_inited = false;

// Barge-in during REPLYING. Always available — unlike the VADs it needs no
// model, just the AEC residual.
static barge_detector       g_barge;

// ────────────────────────────────────────────────────────────────────────
// Mic capture — feeds the AEC capture buffer, and the raw PCM queue during
// the startup liveness probe only.
// ────────────────────────────────────────────────────────────────────────
static void capture_cb(ma_device * dev, void * /*output*/, const void * input,
                       ma_uint32 frame_count) {
    g_aec.push_capture((const float *) input, (size_t) frame_count);
    if (g_mic_probe.load(std::memory_order_relaxed)) {
        auto * q = (pcm_queue *) dev->pUserData;
        q->push((const float *) input, frame_count);
    }
}

// ────────────────────────────────────────────────────────────────────────
// Playback ring + fade
// ────────────────────────────────────────────────────────────────────────
struct audio_ring {
    std::mutex         m;
    std::vector<float> buf;
    // ~12 ms of ramp. Set from the real output rate at startup, because DFN
    // moves playback to 48 kHz and a constant sized for 24 kHz would halve
    // the ramp exactly when it is doing the most work.
    size_t             fade_length = 12 * 24;
    std::atomic<int>   fade_remaining{-1};
    void reset_for_session() {
        std::lock_guard<std::mutex> lk(m);
        buf.clear();
        fade_remaining.store(-1);
    }
    void push(const float * pcm, size_t n) {
        std::lock_guard<std::mutex> lk(m);
        buf.insert(buf.end(), pcm, pcm + n);
    }
    size_t size() {
        std::lock_guard<std::mutex> lk(m);
        return buf.size();
    }
    void start_fade_then_clear() {
        std::lock_guard<std::mutex> lk(m);
        if (!buf.empty()) fade_remaining.store((int) fade_length);
        else              fade_remaining.store(-1);
    }
    // Block until playback_cb has walked the ramp to zero and dropped the rest
    // of the queued reply (fade_remaining back to -1), or until the timeout.
    //
    // Needed because every start_fade_then_clear() caller recycles straight
    // into the next LISTENING, whose reset_for_session() clears the ring —
    // it would win the race against the ~12 ms ramp every time, and barge-in
    // would cut the reply off with an audible click instead of a fade.
    void wait_fade_done(int timeout_ms = 60) {
        const auto t0 = std::chrono::steady_clock::now();
        while (fade_remaining.load() >= 0) {
            const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (waited >= timeout_ms) break;   // device stalled — don't hang
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
};

// CLI-side timing for ttfa (time from EOU to first speaker audio). The
// library tracks token counts + tts sample counts itself — see
// VoiceSession::last_stats. The two combine in the per-turn metric line.
static std::atomic<bool>    g_first_audio_fired{false};
static std::atomic<int64_t> g_first_audio_ms   {0};

// Playback underruns: the ring ran dry after the reply had started speaking
// but before it finished. Each one is an audible gap — the speaker stutters.
// Counted per contiguous dry stretch, not per callback.
static std::atomic<int>     g_underruns   {0};
static std::atomic<bool>    g_playback_dry{false};

// mono_ms() is defined further up alongside the cli_aec forward decls.

static void playback_cb(ma_device * dev, void * out, const void * /*in*/,
                        ma_uint32 frames) {
    auto * ring = (audio_ring *) dev->pUserData;
    float * outf = (float *) out;
    const size_t need = frames;

    std::lock_guard<std::mutex> lk(ring->m);
    if (ring->buf.empty()) {
        std::memset(outf, 0, need * sizeof(float));
        // Silence is still a far-end reference and MUST be published, or the
        // render timeline stops advancing while the mic's keeps going and the
        // AEC loses the alignment this tap exists to preserve.
        g_aec.push_render(outf, need);
        // Dry after we started speaking = the synthesiser fell behind the DAC.
        if (g_first_audio_fired.load() && g_phase.load() == GL_REPLYING) {
            if (!g_playback_dry.exchange(true)) {
                g_underruns.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return;
    }
    g_playback_dry.store(false);

    const size_t take = std::min(need, ring->buf.size());
    int fade = ring->fade_remaining.load();
    if (fade < 0) {
        // No fade in progress — copy directly.
        std::memcpy(outf, ring->buf.data(), take * sizeof(float));
    } else {
        // Linear fade-out: ramp 1.0 → 0.0 over fade_length samples, then
        // silence. At fade-completion (fade <= 0) we drop everything still
        // queued — that's how barge-in cuts off the rest of the reply audio.
        for (size_t i = 0; i < take; i++) {
            if (fade > 0) {
                const float g = (float) fade / (float) ring->fade_length;
                outf[i] = ring->buf[i] * g;
                --fade;
            } else {
                outf[i] = 0.0f;
            }
        }
        if (fade <= 0) {
            ring->buf.clear();
            ring->fade_remaining.store(-1);
        } else {
            ring->fade_remaining.store(fade);
        }
    }
    if (!ring->buf.empty()) {
        ring->buf.erase(ring->buf.begin(),
                        ring->buf.begin() + std::min(take, ring->buf.size()));
    }
    if (take < need) {
        std::memset(outf + take, 0, (need - take) * sizeof(float));
    }

    // AEC far-end reference, tapped HERE rather than where TTS produces the
    // audio. This is the only place that knows what the speaker actually
    // emits, and the AEC can only subtract a signal it has correctly aligned
    // in time with the mic.
    //
    // Taking it at the producer instead is wrong twice over. TTS synthesises
    // faster than realtime, so it runs ahead of the DAC by however much is
    // queued; and every barge-in throws that queue away, leaving the AEC with
    // seconds of reference for sound nobody heard — an offset it never
    // recovers from, since nothing re-syncs the two. Here the reference is
    // produced at exactly the rate the DAC consumes it, silence included, so
    // the two streams stay locked by construction.
    g_aec.push_render(outf, need);
    if (take > 0 && !g_first_audio_fired.exchange(true)) {
        g_first_audio_ms.store(mono_ms());
    }
}

// ────────────────────────────────────────────────────────────────────────
// SIGINT handler — set shutdown, abort any in-flight turn.
// ────────────────────────────────────────────────────────────────────────
static void sigint_handler(int) {
    g_shutdown.store(true);
    if (g_session) g_session->abort_turn();
    static const char msg[] = "  [↩ shutting down…]\n";
    (void) write(2, msg, sizeof(msg) - 1);
}

// ────────────────────────────────────────────────────────────────────────
// main
// ────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv) {
    ggml_time_init();
    common_params params;
    common_init();
    // Gemma 4 E4B, and the mmproj + MTP head from the SAME ggml-org build.
    // Keeping all three on one quantisation pipeline is what makes the MTP
    // head predict the trunk well enough for speculative decoding to pay —
    // a trunk and head from different pipelines drop acceptance badly.
    // The mmproj is not interchangeable across model sizes: an E2B mmproj
    // with an E4B trunk produces garbage embeddings, so override them
    // together or not at all.
    params.model.path  = "models/gemma-4-E4B-it-Q4_0.gguf";
    params.mmproj.path = "models/mmproj-gemma-4-E4B-it-Q8_0.gguf";
    if (const char * v = std::getenv("GEMMA_LIVE_LLM_MODEL")) params.model.path  = v;
    if (const char * v = std::getenv("GEMMA_LIVE_MMPROJ"))    params.mmproj.path = v;
    params.use_jinja   = true;
    params.n_predict   = 256;
    params.n_ctx       = 8192;

    std::signal(SIGINT, sigint_handler);

    const char * tts_model_env = std::getenv("GEMMA_LIVE_TTS_MODEL");
    const char * tts_voice_env = std::getenv("GEMMA_LIVE_TTS_VOICE");
    const std::string tts_model_path = tts_model_env
        ? std::string(tts_model_env)
        : std::string("models/vibevoice-realtime-0.5b-q4_k.gguf");
    const std::string tts_voice_path = tts_voice_env
        ? std::string(tts_voice_env)
        : std::string("voices/vibevoice-voice-en-Gemma_woman.gguf");

    std::string system_prompt;
    std::string system_prompt_path;
    {
        const char * prompt_env  = std::getenv("GEMMA_LIVE_SYSTEM_PROMPT");
        const std::string prompt_path = prompt_env ? prompt_env : "prompts/chat.txt";
        std::ifstream f(prompt_path);
        if (f) {
            std::stringstream ss; ss << f.rdbuf();
            system_prompt = ss.str();
            system_prompt_path = prompt_path;
        } else {
            fprintf(stderr, "system   none (%s not found)\n", prompt_path.c_str());
        }
    }

    SessionConfig cfg;
    cfg.llm_model_path = params.model.path;
    cfg.mmproj_path    = params.mmproj.path;
    cfg.tts_model_path = tts_model_path;
    cfg.tts_voice_path = tts_voice_path;
    cfg.system_prompt  = system_prompt;
    cfg.n_ctx          = params.n_ctx;
    cfg.n_predict      = params.n_predict;
    // MTP speculative decoding — ON, worth +36% decode on the E4B trio above
    // (62% of drafts accepted). The head must match the trunk: swap the LLM
    // and you must swap this too, or acceptance collapses and it turns into a
    // slowdown. See SessionConfig for the measurements behind n_draft=1.
    //
    //   GEMMA_LIVE_MTP=0          disable
    //   GEMMA_LIVE_MTP_MODEL=...  draft head path
    //   GEMMA_LIVE_MTP_DRAFT=N    tokens drafted per step (1 measured best)
    //
    // A missing or unloadable head is not fatal — the session logs and falls
    // back to one token per decode.
    cfg.enable_mtp     = true;
    cfg.mtp_model_path = "models/mtp-gemma-4-E4B-it-Q4_0.gguf";
    if (const char * v = std::getenv("GEMMA_LIVE_MTP_MODEL")) cfg.mtp_model_path = v;
    if (const char * v = std::getenv("GEMMA_LIVE_MTP"))       cfg.enable_mtp     = (atoi(v) != 0);
    if (const char * v = std::getenv("GEMMA_LIVE_MTP_DRAFT")) cfg.mtp_n_draft    = std::max(1, atoi(v));
    if (const char * v = std::getenv("GEMMA_LIVE_TTS_CFG"))    cfg.tts_cfg        = (float) atof(v);
    if (const char * v = std::getenv("GEMMA_LIVE_TTS_STEPS"))  cfg.tts_steps      = atoi(v);
    if (const char * v = std::getenv("GEMMA_LIVE_TTS_ANCHOR")) cfg.tts_neg_anchor = (float) atof(v);
    // Latency knob: smaller first chunk speaks sooner. See SessionConfig.
    if (const char * v = std::getenv("GEMMA_LIVE_TTS_FIRST_CHUNK")) {
        cfg.tts_first_chunk_frames = std::max(1, atoi(v));
    }

    // DFN post-filter for TTS is disabled by default — we're going to
    // handle vibevoice's music-artifact openers via prompt engineering
    // instead of scrubbing them after the fact. Set GEMMA_LIVE_DFN_MODEL
    // to a valid path to re-enable.
    if (const char * v = std::getenv("GEMMA_LIVE_DFN_MODEL")) {
        cfg.dfn_model_path = v;
    }

    std::string session_err;
    auto session = VoiceSession::create(cfg, &session_err);
    if (!session) {
        fprintf(stderr, "ERR: VoiceSession::create: %s\n", session_err.c_str());
        return 1;
    }
    // Raw alias for the SIGINT handler and the barge-in VAD thread, both of
    // which need to call abort_turn() without owning the session.
    g_session = session.get();

    // Resolve the actual TTS output rate. With DFN enabled this is 48000;
    // without DFN it stays at GL_TTS_RATE (24000). Everything downstream —
    // speaker, AEC render, stats — must use this value, not the constant.
    // Reported here rather than at read time: tokenising needs the model's
    // vocab, which only exists once the session is up.
    if (!system_prompt_path.empty()) {
        fprintf(stderr, "system   %s (%d tokens)\n",
                system_prompt_path.c_str(), session->system_tokens());
    }

    const int tts_rate = session->tts_sample_rate();
    fprintf(stderr, "tts      %s @ %d Hz%s\n"
                    "         voice %s, cfg %.2f, steps %d, anchor %.2f\n",
            tts_model_path.c_str(), tts_rate,
            tts_rate == GL_TTS_RATE ? "" : " (dfn post-filter active)",
            tts_voice_path.c_str(), cfg.tts_cfg, cfg.tts_steps, cfg.tts_neg_anchor);

    // ---- End-of-utterance VAD (required for the listen→reply transition) ----
    {
        const char * vad_env = std::getenv("GEMMA_LIVE_VAD_MODEL");
        const std::string vad_path = vad_env ? vad_env : "models/firered-vad.gguf";
        if (g_eou_vad.init(vad_path.c_str())) {
            fprintf(stderr, "eou      firered-vad, %d ms silence\n",
                    cli_eou_vad::DEFAULT_SILENCE_MS);
        } else {
            fprintf(stderr,
                    "ERR: EOU VAD init failed (%s). Without it there's no way to know\n"
                    "     when the user has stopped speaking. Set GEMMA_LIVE_VAD_MODEL\n"
                    "     to a valid firered-vad GGUF file.\n", vad_path.c_str());
            return 1;
        }
    }

    // ---- AEC (echo cancellation only — VAD lives in cli_keyword_detector) ----
    // Always on: started once after mic+speaker init below, stopped at
    // shutdown. Its output is the ONLY mic path downstream — both the keyword
    // detector and the audio fed to the model read AEC'd rings. Mandatory
    // because without it the detectors hear the model's own reply bleeding
    // through the mic and fire barge-in on Gemma's voice.
    //
    // Note this replaced WebRTC's AudioProcessing, which also ran an AGC and
    // a high-pass filter. LocalVQE does echo cancellation and noise
    // suppression only, so mic levels downstream are no longer normalised —
    // relevant to cli_eou_vad's fixed pre-gain if levels ever look off.
    {
        const char * vqe_env = std::getenv("GEMMA_LIVE_VQE_MODEL");
        const std::string vqe_path = vqe_env ? vqe_env : "models/localvqe.gguf";
        if (!g_aec.init(vqe_path)) {
            fprintf(stderr,
                    "ERR: AEC init failed (%s). Every mic consumer reads the AEC\n"
                    "     output, so there is nothing to fall back to. Fetch the\n"
                    "     model with:\n"
                    "       curl -L -o models/localvqe.gguf \\\n"
                    "         https://huggingface.co/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.3-4.8M-f32.gguf\n",
                    vqe_path.c_str());
            return 1;
        }
        fprintf(stderr, "aec      localvqe, %.0f ms hop\n",
                1000.0 * g_aec.vqe.hop() / (double) GL_MIC_RATE);
    }

    // ---- Mic device (16 kHz mono f32) ----
    pcm_queue queue;
    ma_device_config dev_cfg = ma_device_config_init(ma_device_type_capture);
    dev_cfg.capture.format   = ma_format_f32;
    dev_cfg.capture.channels = 1;
    dev_cfg.sampleRate       = GL_MIC_RATE;
    dev_cfg.dataCallback     = capture_cb;
    dev_cfg.pUserData        = &queue;
    ma_device device;
    if (ma_device_init(nullptr, &dev_cfg, &device) != MA_SUCCESS) {
        fprintf(stderr, "ERR: capture device init failed\n");
        return 1;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
        fprintf(stderr, "ERR: capture device start failed\n");
        return 1;
    }
    fprintf(stderr, "mic      %s @ %u Hz\n", device.capture.name, device.sampleRate);
    // Sanity check: macOS Microphone permission is keyed by binary identity,
    // and rebuilt binaries silently produce pure-zero audio when the grant is
    // missing. Sample the mic briefly and warn if it's dead. The model would
    // otherwise just respond to silence with generic greetings every turn.
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::vector<float> probe;
        queue.drain(probe, 8000);  // up to ~500 ms
        if (!probe.empty()) {
            double sumsq = 0.0;
            for (float s : probe) sumsq += (double) s * s;
            const float rms = (float) std::sqrt(sumsq / probe.size());
            if (rms < 1e-5f) {
                fprintf(stderr,
                        "WARN: mic is producing pure silence (rms=%.6f over %.2fs).\n"
                        "      Almost certainly a missing macOS Microphone permission.\n"
                        "      System Settings → Privacy & Security → Microphone\n"
                        "      (remove gemma-live if listed, then re-launch).\n",
                        rms, (double) probe.size() / GL_MIC_RATE);
            }
        }
        g_mic_probe.store(false);
        queue.clear();
    }

    // ---- Playback device (mono f32, at tts_rate — 24 kHz native or 48 kHz with DFN) ----
    audio_ring playback_ring;
    ma_device_config pb_cfg          = ma_device_config_init(ma_device_type_playback);
    pb_cfg.playback.format           = ma_format_f32;
    pb_cfg.playback.channels         = 1;
    pb_cfg.sampleRate                = (ma_uint32) tts_rate;
    pb_cfg.dataCallback              = playback_cb;
    pb_cfg.pUserData                 = &playback_ring;
    ma_device playback_device;
    {
        ma_result r = ma_device_init(nullptr, &pb_cfg, &playback_device);
        if (r != MA_SUCCESS) {
            fprintf(stderr, "ERR: playback init failed: %s (%d)\n",
                    ma_result_description(r), (int) r);
            return 1;
        }
        // Device-negotiation detail: only interesting when the audio path is
        // misbehaving, so keep it out of a normal boot.
        if (cfg.verbosity >= 2) {
            fprintf(stderr, "playback negotiated: native sr=%u ch=%u format=%d (asked sr=%d ch=1 f32)\n",
                    playback_device.playback.internalSampleRate,
                    playback_device.playback.internalChannels,
                    (int) playback_device.playback.internalFormat, tts_rate);
        }
        r = ma_device_start(&playback_device);
        if (r != MA_SUCCESS) {
            fprintf(stderr, "ERR: playback start failed: %s (%d)\n",
                    ma_result_description(r), (int) r);
            return 1;
        }
    }
    fprintf(stderr, "speaker  %s @ %u Hz\n",
            playback_device.playback.name, playback_device.sampleRate);

    // ---- Wake-word detector (moonshine streaming on AEC'd mic) ----
    // Always-on. Streaming ASR transcribes the AEC'd mic; substring match
    // against the wake-phrase list fires when phase == GL_IDLE.
    {
        // NOTE: moonshine needs a `tokenizer.bin` sitting NEXT TO the model
        // file — it is resolved as dirname(model) + "/tokenizer.bin", never
        // named anywhere in this repo. Point KWD_MODEL at a GGUF without that
        // sibling and init fails with a message about the model, not the
        // tokenizer. It ships alongside the model in the same HF repo.
        const char * kwd_env = std::getenv("GEMMA_LIVE_KWD_MODEL");
        const std::string kwd_path = kwd_env
            ? std::string(kwd_env)
            : std::string("models/moonshine-streaming-tiny-q4_k.gguf");
        const char * wake_env = std::getenv("--kwd-wake");
        const std::string wake_path = wake_env ? wake_env : std::string("keywords/wake.txt");
        if (g_kwd.init(kwd_path.c_str(), wake_path.c_str())) {
            fprintf(stderr, "asr      moonshine %d/%d ms, %zu wake phrases\n",
                    g_kwd.step_ms, g_kwd.length_ms, cli_keyword_detector::n_loaded);
        } else {
            fprintf(stderr,
                    "ERR: keyword detector init failed (model at %s?).\n"
                    "     Without it there's no wake word.\n",
                    kwd_path.c_str());
            return 1;
        }
    }

    // ---- Followup voice-start detector (VAD, assistant silent) ----
    {
        const char * vad_env = std::getenv("GEMMA_LIVE_VAD_MODEL");
        const std::string vad_path = vad_env ? vad_env : "models/firered-vad.gguf";
        if (g_voice_vad.init(vad_path.c_str())) {
            g_voice_vad_inited = true;
            g_kwd.voice_vad = &g_voice_vad;
            fprintf(stderr, "followup %d hops sustained, gate %.0f dBFS\n",
                    g_voice_vad.required_consecutive_hops, g_voice_vad.rms_gate_dbfs);
        } else {
            fprintf(stderr, "followup DISABLED — %s failed to load\n", vad_path.c_str());
        }
    }

    // ---- Barge-in detector (AEC residual energy, assistant audible) ----
    g_barge.init();
    g_barge.on_fire = []() {
        if (g_voice_activity.exchange(true)) return;
        g_interrupted.store(true);
        fprintf(stderr, "  [↩ voice bargein]\n");
        fflush(stderr);
        if (g_session) g_session->abort_turn();
    };
    g_kwd.barge   = &g_barge;
    g_kwd.eou_vad = &g_eou_vad;
    fprintf(stderr, "bargein  residual energy, %.1fx floor, %d ms sustained\n",
            g_barge.ratio, g_barge.sustain_hops * 10);

    // Start the always-on workers now that everything they depend on exists.
    // Tell AEC how to decimate the render-side reference: 24 kHz normally,
    // 48 kHz when DFN is on.
    playback_ring.fade_length = (size_t) (12 * tts_rate / 1000);   // ~12 ms
    g_aec.set_render_src_rate(tts_rate);
    g_aec.start();
    g_kwd.start();

    // How long the followup window stays open after a reply before we fall
    // back to needing the wake word. Read before the banner because the
    // banner quotes it — a hardcoded number there goes stale the moment
    // anyone sets the override.
    constexpr int AWAIT_TIMEOUT_MS_DEFAULT = 5000;
    int await_timeout_ms = AWAIT_TIMEOUT_MS_DEFAULT;
    if (const char * v = std::getenv("GEMMA_LIVE_AWAIT_TIMEOUT_MS")) {
        await_timeout_ms = std::max(1000, atoi(v));
    }

    fprintf(stderr,
        "\n"
        "  gemma-live ready.\n"
        "\n"
        "  Say \"hey gemma\" to start a conversation, then keep talking — no\n"
        "  need to re-say the wake word between turns. Just start talking\n"
        "  during a reply to interrupt it. After %.0f s of silence the\n"
        "  conversation ends and we go back to waiting for the wake word.\n"
        "\n"
        "  Ctrl+C to quit.\n"
        "\n", await_timeout_ms / 1000.0);

    // ── Session callbacks ───────────────────────────────────────────────
    // marker_printed drives the cyan [♪] shown inline in the token stream
    // at the moment the first audio chunk reaches the speaker.
    bool marker_printed = false;

    session->on_token = [&](const char * text) {
        printf("%s", text);
        if (!marker_printed && g_first_audio_fired.load()) {
            printf("\033[36m[♪]\033[0m");
            marker_printed = true;
        }
        fflush(stdout);
    };
    // Queue for playback only. The AEC reference and the first-audio
    // timestamp are both taken in playback_cb, where the audio actually
    // reaches the speaker — TTS runs ahead of realtime, so this callback is
    // the wrong clock for either.
    session->on_audio = [&](const float * samples, size_t n_samples) {
        playback_ring.push(samples, n_samples);
    };
    // on_done (LLM hit EOS, TTS may still be synthesising) needs no action.

    // Pump cadence: drain in 100 ms slices so the EOU VAD evaluates at
    // ~10 Hz wall-clock. mtmd_audio_stream buffers internally — passing
    // it small chunks is fine.
    const int    chunk_ms      = 100;
    const size_t chunk_samples = (size_t) GL_MIC_RATE * (size_t) chunk_ms / 1000;
    std::vector<float> pcm;

    // Audio fed to Gemma comes from g_aec.session_buf — already echo-
    // cancelled, so transitions out of REPLYING don't carry the model's
    // own voice into Gemma's next prefix. trim_session_to() is called at
    // each transition INTO LISTENING so Gemma sees ~2 s of recent context
    // (wake phrase + lead-in on the first turn; the user's first words on
    // bargein / followup) instead of the ring's full 8 s.
    constexpr size_t SESSION_KEEP_SAMPLES = (size_t) GL_MIC_RATE * 2;  // 2 s lookback

    while (!g_shutdown.load()) {
        // ──────────────────────────────────────────────────────────────
        // Outer wake gate — wait for the wake word, then run the inner
        // conversation loop until either a 20 s silence timeout or
        // ─ Ctrl+C ─ kicks us back out here.
        // ──────────────────────────────────────────────────────────────
        g_phase.store(GL_IDLE);
        if (g_voice_vad_inited) g_voice_vad.set_active(false);
        g_barge.set_active(false);
        g_wake_fired.store(false);
        g_voice_activity.store(false);
        g_interrupted.store(false);
        fprintf(stderr, "[idle — say \"hey gemma\" to start]\n");
        while (!g_wake_fired.load() && !g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (g_shutdown.load()) break;

        // First turn of the conversation — trim AEC session audio to keep
        // just the recent 2 s (wake phrase + lead-in). The inner loop
        // re-trims at each subsequent LISTENING entry, so this is just
        // the entry condition.
        g_aec.trim_session_to(SESSION_KEEP_SAMPLES);

        // ──────────────────────────────────────────────────────────────
        // Inner conversation loop. One iteration = one user turn + one
        // reply + one followup window. We break out when the followup
        // window times out (→ back to outer wake gate) or on shutdown.
        // ──────────────────────────────────────────────────────────────
        bool conversation_active = true;
        while (conversation_active && !g_shutdown.load()) {
            // ── LISTENING ─────────────────────────────────────────────
            marker_printed = false;
            playback_ring.reset_for_session();
            g_first_audio_fired.store(false);
            g_first_audio_ms.store(0);
            g_interrupted.store(false);
            g_voice_activity.store(false);
            g_eou_vad.reset();

            if (!session->begin_turn()) {
                fprintf(stderr, "ERR: begin_turn: %s\n", session->last_error().c_str());
                conversation_active = false;
                break;
            }
            g_phase.store(GL_LISTENING);
            if (g_voice_vad_inited) g_voice_vad.set_active(false);
            g_barge.set_active(false);
            fprintf(stderr, "[listening — pause ~500 ms to send]\n");

            // 28 s wall-clock max + a hard cap on cumulative audio pushed
            // to the mtmd encoder. mtmd_audio_stream_preproc_gemma4a asserts
            // at >30 s per turn, and our 2 s lookback already eats into
            // that budget — without a per-sample cap, a user who talks
            // continuously past ~28 s would crash the process. We cap at
            // (30 s - one chunk) of headroom so the final push can't trip
            // the assert.
            constexpr int MAX_LISTENING_MS = 28000;
            const size_t  MAX_AUDIO_SAMPLES =
                (size_t) (30 * GL_MIC_RATE) - chunk_samples;
            size_t        pushed_samples = 0;
            bool          push_failed    = false;
            const auto    t_listening_start = std::chrono::steady_clock::now();
            auto          listening_elapsed_ms = [&]() {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t_listening_start).count();
            };

            while (!g_shutdown.load()) {
                if (listening_elapsed_ms() >= MAX_LISTENING_MS) {
                    fprintf(stderr, "  [max listening %d ms reached — sending]\n", MAX_LISTENING_MS);
                    break;
                }
                if (pushed_samples >= MAX_AUDIO_SAMPLES) {
                    fprintf(stderr, "  [max audio %d s reached — sending]\n",
                            (int) (MAX_AUDIO_SAMPLES / GL_MIC_RATE));
                    break;
                }
                if (g_aec.session_size() >= chunk_samples) {
                    const size_t headroom = MAX_AUDIO_SAMPLES - pushed_samples;
                    const size_t want = std::min(chunk_samples, headroom);
                    const size_t got = g_aec.drain_session(pcm, want);
                    if (got > 0) {
                        // A failed push means the audio embeddings never made
                        // it into the KV cache (context full, backend error).
                        // Stop feeding and let end_turn roll the turn back —
                        // pushing more would just pile up further failures.
                        if (!session->push_audio(pcm.data(), got)) {
                            push_failed = true;
                            break;
                        }
                        pushed_samples += got;
                        if (g_eou_vad.feed(pcm.data(), got)) break;
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            if (g_shutdown.load()) break;

            // Drain whatever is left at EOU, respecting the same cap.
            while (!push_failed && g_aec.session_size() > 0 && !g_shutdown.load()) {
                if (pushed_samples >= MAX_AUDIO_SAMPLES) break;
                const size_t headroom = MAX_AUDIO_SAMPLES - pushed_samples;
                const size_t want = std::min({g_aec.session_size(), chunk_samples, headroom});
                if (want == 0) break;
                const size_t got = g_aec.drain_session(pcm, want);
                if (got > 0) {
                    if (!session->push_audio(pcm.data(), got)) { push_failed = true; break; }
                    pushed_samples += got;
                }
            }
            if (push_failed) {
                fprintf(stderr, "\nERR: push_audio: %s\n", session->last_error().c_str());
            }

            const auto t_eou = std::chrono::steady_clock::now();

            // ── REPLYING ──────────────────────────────────────────────
            g_phase.store(GL_REPLYING);
            // Double-talk: the VAD would be reading a signal the AEC has
            // already suppressed the user out of, so the energy detector
            // takes over here and the VAD stands down.
            if (g_voice_vad_inited) g_voice_vad.set_active(false);
            g_barge.set_active(true);
            fprintf(stderr, "[gemma] ");
            fflush(stderr);
            const bool turn_ok = session->end_turn();
            if (!marker_printed && g_first_audio_fired.load()) {
                printf("\033[36m[♪]\033[0m");
                marker_printed = true;
            }
            printf("\n"); fflush(stdout);
            if (!turn_ok) {
                // end_turn has already rolled the turn out of the KV cache, so
                // the session is consistent — but whatever broke (context
                // full, backend error) will break the next turn too. Drop back
                // to the wake gate instead of falling through to the followup
                // window and failing again every 20 s.
                const std::string & e = session->last_error();
                if (!e.empty()) fprintf(stderr, "ERR: end_turn: %s\n", e.c_str());
                playback_ring.start_fade_then_clear();
                playback_ring.wait_fade_done();
                g_interrupted.store(false);
                g_voice_activity.store(false);
                conversation_active = false;
                break;
            }

            // Voice barge-in mid-reply: cli_voice_vad set g_interrupted
            // and aborted end_turn. Fade playback and recycle straight
            // into the next LISTENING — keep the AEC session_buf because
            // AEC already cancelled out the model's voice, so the user's
            // interrupt audio is preserved cleanly.
            if (g_interrupted.exchange(false)) {
                playback_ring.start_fade_then_clear();
                playback_ring.wait_fade_done();
                fprintf(stderr, "[interrupted]\n");
                g_voice_activity.store(false);
                g_aec.trim_session_to(SESSION_KEEP_SAMPLES);
                continue;   // next inner iteration → LISTENING
            }

            // Stats for a normally-completed reply.
            const int64_t eou_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_eou.time_since_epoch()).count();
            const int64_t first_audio = g_first_audio_ms.load();
            const double  ms_ttfa = (first_audio > 0) ? (double)(first_audio - eou_ms) : -1.0;
            const TurnStats & st = session->last_stats();
            const double tok_per_s = (st.ms_llm_gen > 0.0)
                ? 1000.0 * (double) st.n_llm_tokens / st.ms_llm_gen
                : 0.0;
            const double audio_s = (double) st.n_tts_samples / (double) GL_TTS_RATE;
            const double rtf = (audio_s > 0.0) ? (st.ms_tts_wall / 1000.0) / audio_s : 0.0;
            // Acceptance rate is the number to watch when tuning
            // GEMMA_LIVE_MTP_DRAFT; omitted entirely when MTP is off.
            const int underruns = g_underruns.exchange(0);
            char ur_note[32] = "";
            if (underruns > 0) snprintf(ur_note, sizeof(ur_note), " | UNDERRUN x%d", underruns);
            char mtp_note[64] = "";
            if (st.n_drafted > 0) {
                snprintf(mtp_note, sizeof(mtp_note), " | mtp %d/%d acc %.0f%%",
                         st.n_accepted, st.n_drafted,
                         100.0 * (double) st.n_accepted / (double) st.n_drafted);
            }
            fprintf(stderr,
                "[enc %d tok | llm %d tok @ %.1f tok/s | ttft %.0f ms"
                " | tts %.2f s | ttfa %.0f ms | rtf %.2f%s%s]\n",
                st.n_audio_tokens,
                st.n_llm_tokens, tok_per_s,
                st.ms_ttft,
                audio_s, ms_ttfa, rtf, mtp_note, ur_note);

            // Drain the playback ring (replay tail of the reply). Voice
            // VAD stays active — a bargein during drain ends the playback
            // and recycles into the next LISTENING, exactly like a mid-
            // reply bargein.
            while (!g_shutdown.load() && playback_ring.size() > 0) {
                if (g_interrupted.load()) {
                    playback_ring.start_fade_then_clear();
                    playback_ring.wait_fade_done();
                    fprintf(stderr, "[interrupted]\n");
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            if (g_interrupted.exchange(false)) {
                g_voice_activity.store(false);
                g_aec.trim_session_to(SESSION_KEEP_SAMPLES);
                continue;   // next inner iteration → LISTENING
            }
            if (g_shutdown.load()) break;

            // ── AWAITING_FOLLOWUP ─────────────────────────────────────
            // Reply finished cleanly. Stay in conversation mode for a
            // short window: any voice activity → followup turn; timeout
            // → back to outer wake gate. No log line per the spec.
            g_phase.store(GL_AWAITING_FOLLOWUP);
            // Clear any voice_activity that fired during the playback
            // drain after end_turn already returned (race-free reset).
            g_voice_activity.store(false);
            // The VAD stays on across the REPLYING → AWAITING_FOLLOWUP
            // transition; set_active clears the carried-over hop count so the
            // followup needs a fresh 300 ms of sustained voice, not whatever
            // partial count the reply left behind.
            // Speaker is quiet now, so there is no echo to confuse a VAD and
            // no residual floor for the energy detector to measure against.
            g_barge.set_active(false);
            if (g_voice_vad_inited) g_voice_vad.set_active(true);
            // Trim session_buf so when followup voice fires we don't
            // feed Gemma the tail of the just-played reply audio.
            g_aec.trim_session_to(SESSION_KEEP_SAMPLES);

            const auto t_await = std::chrono::steady_clock::now();
            auto await_elapsed_ms = [&]() {
                return std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t_await).count();
            };
            bool got_followup = false;
            while (!g_shutdown.load()) {
                if (g_voice_activity.exchange(false)) {
                    got_followup = true;
                    g_aec.trim_session_to(SESSION_KEEP_SAMPLES);
                    break;
                }
                if ((int) await_elapsed_ms() >= await_timeout_ms) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (g_shutdown.load()) break;
            if (!got_followup) {
                conversation_active = false;   // → outer wake gate
            }
            // got_followup → fall through, next inner iteration is the
            // followup turn.
        }
    }

    g_shutdown.store(true);

    // Fast shutdown. Audio device release + flush is sub-100ms; everything
    // else (unloading Gemma weights, freeing TTS Metal buffers, freeing
    // moonshine/firered_vad models) is 1-2 s of GPU resource release that
    // the OS redoes for us in microseconds when the process exits. Skip it
    // via _exit so Ctrl+C returns to the shell promptly.
    g_aec.stop();
    ma_device_uninit(&device);
    ma_device_uninit(&playback_device);
    std::fflush(stdout);
    std::fflush(stderr);
    _exit(0);
}
