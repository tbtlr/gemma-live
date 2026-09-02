// Voice-activity detection shared between the microphone front-end
// (main.cpp) and the Realtime server (realtime.cpp).
//
// The two differ in ONE respect, and it matters: main.cpp measures silence
// against the wall clock, because its audio arrives from a sound card in
// real time and wall time is sample time. A socket client has no such
// guarantee — it may upload ten seconds of audio in a single frame, or
// stall for a minute mid-utterance — so the server counts SAMPLES instead.
// Using wall time there would end a turn the instant a slow client paused,
// and never end one for a client uploading faster than real time.
//
// The detection parameters below are the tuned part and are shared, so the
// two front-ends cannot drift apart on the numbers that decide what counts
// as speech.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "firered_vad.h"

namespace gl_vad {

// The audio fed to firered_vad is pre-gained so it sits in the level range
// the model was trained on (~-26 dBFS); the macOS capture path lands well
// below that. 8x is +18 dB. Samples are clamped, so loud input saturates
// rather than wrapping.
constexpr float GAIN            = 8.0f;
constexpr int   WINDOW_SAMPLES  = 12800;   // 800 ms at 16 kHz
constexpr int   HOP_MS          = 100;     // evaluation cadence
constexpr float THRESHOLD       = 0.3f;    // post-gain
constexpr float MIN_SPEECH_SEC  = 0.25f;
constexpr float MIN_SILENCE_SEC = 0.10f;

// Append `n` samples to `window`, pre-gained and clamped, keeping only the
// most recent WINDOW_SAMPLES.
inline void push_window(std::vector<float> & window, const float * pcm, size_t n) {
    const size_t off = window.size();
    window.resize(off + n);
    for (size_t i = 0; i < n; i++) {
        window[off + i] = std::max(-1.0f, std::min(1.0f, pcm[i] * GAIN));
    }
    if ((int) window.size() > WINDOW_SAMPLES) {
        window.erase(window.begin(), window.begin() + (window.size() - WINDOW_SAMPLES));
    }
}

// One firered_vad evaluation over a full window.
inline bool window_is_voiced(firered_vad_context * vad, const std::vector<float> & window) {
    firered_vad_segment * segs = nullptr;
    int n_segs = 0;
    firered_vad_detect(vad, window.data(), (int) window.size(), &segs, &n_segs,
                       THRESHOLD, MIN_SPEECH_SEC, MIN_SILENCE_SEC);
    if (segs) std::free(segs);
    return n_segs > 0;
}

// ── sample-clocked end-of-utterance detector (gl-serve) ─────────────────
//
// Turn detection for a socket session: report the onset on the first voiced
// window, and end of utterance once silence_ms of audio has arrived without
// another voiced verdict. Clocked on samples appended rather than wall time,
// so a client that sends faster or slower than real time still gets the same
// turn boundaries.
struct eou {
    firered_vad_context * vad = nullptr;
    int  sample_rate  = 16000;
    int  silence_ms   = 500;
    int  prefix_pad_ms = 300;   // audio kept before the onset

    std::vector<float> window;
    bool     speaking      = false;   // inside an utterance right now
    bool     ever_voiced   = false;
    uint64_t samples_total = 0;       // sample clock for this buffer
    uint64_t last_voiced   = 0;       // sample index of the last voiced verdict
    uint64_t onset_sample  = 0;
    uint64_t next_eval     = 0;

    bool init(const char * model_path) {
        vad = firered_vad_init(model_path);
        return vad != nullptr;
    }
    void shutdown() { if (vad) { firered_vad_free(vad); vad = nullptr; } }

    void reset() {
        window.clear();
        speaking = ever_voiced = false;
        samples_total = last_voiced = onset_sample = 0;
        next_eval = (uint64_t) WINDOW_SAMPLES;
    }

    uint64_t ms_to_samples(int ms) const {
        return (uint64_t) ms * (uint64_t) sample_rate / 1000;
    }

    enum class event { none, speech_started, speech_stopped };

    // Feed appended audio. At most one event per call; callers push in
    // client-sized chunks, and a chunk large enough to contain both an
    // onset and an end is pathological (it would have to be a single
    // append spanning a whole utterance plus the trailing silence).
    event feed(const float * pcm, size_t n) {
        if (!vad || n == 0) return event::none;
        push_window(window, pcm, n);
        samples_total += n;

        if (samples_total < next_eval)          return event::none;
        if ((int) window.size() < WINDOW_SAMPLES) return event::none;
        next_eval = samples_total + ms_to_samples(HOP_MS);

        const bool voiced = window_is_voiced(vad, window);
        if (voiced) {
            last_voiced = samples_total;
            if (!speaking) {
                speaking    = true;
                ever_voiced = true;
                // The window is 800 ms wide, so the onset is already that
                // far behind us; back off by the padding the API documents
                // rather than by the whole window.
                const uint64_t back = ms_to_samples(prefix_pad_ms);
                onset_sample = samples_total > back ? samples_total - back : 0;
                return event::speech_started;
            }
            return event::none;
        }
        if (speaking && samples_total - last_voiced >= ms_to_samples(silence_ms)) {
            speaking = false;
            return event::speech_stopped;
        }
        return event::none;
    }
};

} // namespace gl_vad
