// Barge-in detector — double-talk only, i.e. active while the assistant is
// audibly speaking. Deliberately NOT a VAD.
//
// A VAD cannot do this job. Echo cancellation suppresses whatever correlates
// with the reference, and during double-talk it takes the user's voice down
// with the echo — mangled enough that no speech classifier will call it
// speech. (MetaMensch measured <=0.06 post-AEC speech probability while the
// user was plainly talking over the assistant.) Raw-mic energy fails the
// other way: echo IS speech, so the assistant's own loud syllables blow past
// any fixed threshold and self-trigger.
//
// What survives is a level comparison against the echo floor. Echo-only
// residual after a working AEC is very quiet and fairly steady; real
// near-end speech leaves noticeably more energy behind even when it is too
// distorted to classify. So: learn the residual floor with an EMA, and fire
// when the residual runs several times above it for long enough to not be a
// transient.
//
// Two rules keep the baseline honest:
//   - It only learns while the speaker is actually audible (far_rms above a
//     floor). Silence between TTS chunks is not evidence about echo level.
//   - It never learns while hot, or the user's own speech would be absorbed
//     into the baseline and un-trigger the detector.
//
// Threading: set_active() is called from the main thread, feed() from the
// detector worker. Everything else is owned by the worker.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <vector>

struct barge_detector {
    std::atomic<bool>  active{false};
    std::atomic<bool>  reset_req{false};

    static constexpr size_t HOP_SAMPLES    = 160;     // 10 ms @ 16 kHz
    static constexpr float  FAR_ACTIVE_RMS = 0.001f;  // speaker audible at all

    float ratio        = 3.0f;    // fire at N x the learned echo floor
    float abs_floor    = 0.010f;  // never fire below this, however quiet the echo
    int   sustain_hops = 15;      // 150 ms hot before firing
    int   warmup_hops  = 30;      // 300 ms of audible echo before trusting the EMA

    // Invoked from the worker thread, at most once per active window.
    std::function<void()> on_fire;

    std::vector<float> pending;   // sub-hop remainder between feeds
    float ema      = 0.0f;        // learned echo-residual floor
    int   ema_hops = 0;
    int   hot_hops = 0;
    bool  fired    = false;
    bool  debug    = false;

    void init() {
        debug = std::getenv("GEMMA_LIVE_BARGE_DEBUG") != nullptr;
        if (const char * v = std::getenv("GEMMA_LIVE_BARGE_RATIO")) ratio     = (float) std::atof(v);
        if (const char * v = std::getenv("GEMMA_LIVE_BARGE_FLOOR")) abs_floor = (float) std::atof(v);
        if (const char * v = std::getenv("GEMMA_LIVE_BARGE_SUSTAIN_MS")) {
            sustain_hops = std::max(1, std::atoi(v) / 10);
        }
        pending.reserve(HOP_SAMPLES * 4);
    }

    void set_active(bool a) {
        if (a) reset_req.store(true);
        active.store(a);
    }

    // Feed AEC'd 16 kHz mono, plus the far-end level for those samples.
    // Near no-op when inactive.
    //
    // far_rms_now is one value for the whole call, so a chunk spanning a gap
    // between TTS phrases gets a single verdict on whether the speaker was
    // audible. It is an approximation: exact would mean carrying a per-hop
    // far level alongside the samples. It is tolerable because a misjudged
    // gap contributes low-energy hops that only nudge the floor, and firing
    // still needs `sustain_hops` consecutive hops above `ratio` x floor. If
    // barge-in ever misbehaves in real use, tightening this to a per-hop far
    // level is the first thing to try.
    void feed(const float * pcm, size_t n, float far_rms_now) {
        if (!active.load() || n == 0) return;
        if (reset_req.exchange(false)) {
            pending.clear();
            ema = 0.0f; ema_hops = 0; hot_hops = 0; fired = false;
        }
        pending.insert(pending.end(), pcm, pcm + n);

        size_t off = 0;
        for (; off + HOP_SAMPLES <= pending.size(); off += HOP_SAMPLES) {
            double sumsq = 0.0;
            for (size_t i = 0; i < HOP_SAMPLES; i++) {
                const float s = pending[off + i];
                sumsq += (double) s * s;
            }
            const float clean_rms = (float) std::sqrt(sumsq / (double) HOP_SAMPLES);

            // Speaker silent this hop — hold the learned floor, but don't
            // count it as warmup and don't let heat carry across the gap.
            if (far_rms_now <= FAR_ACTIVE_RMS) { hot_hops = 0; continue; }

            if (ema_hops == 0) ema = clean_rms;
            ema_hops++;

            const float trigger = std::max(abs_floor, ratio * ema);
            const bool  hot     = (ema_hops > warmup_hops) && (clean_rms > trigger);
            if (hot) {
                if (++hot_hops >= sustain_hops && !fired) {
                    fired = true;
                    if (debug) {
                        fprintf(stderr, "  [barge: residual %.4f > %.4f (floor %.4f)]\n",
                                clean_rms, trigger, ema);
                    }
                    if (on_fire) on_fire();
                }
            } else {
                hot_hops = 0;
                // tau ~0.5 s at 10 ms hops. Learn only while not hot, so the
                // user's own speech can never be absorbed into the floor.
                ema = 0.98f * ema + 0.02f * clean_rms;
            }
        }
        if (off > 0) pending.erase(pending.begin(), pending.begin() + off);
    }
};
