// Behavioural test for barge_detector. The whole reason this thing exists is
// that it must fire on the user talking over the assistant WITHOUT firing on
// the assistant's own echo, so those are the two cases that matter most.
#include "barge.h"

#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

// Feed `ms` of noise at the given RMS, in 10 ms hops.
static void feed_ms(barge_detector & d, int ms, float rms, float far_rms,
                    std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, rms);
    std::vector<float> buf(barge_detector::HOP_SAMPLES);
    for (int i = 0; i < ms / 10; i++) {
        for (auto & s : buf) s = dist(rng);
        d.feed(buf.data(), buf.size(), far_rms);
    }
}

static void check(const std::string & name, bool got, bool want) {
    const bool ok = (got == want);
    if (!ok) g_fail++;
    printf("%-58s %s (fired=%d, want=%d)\n", name.c_str(),
           ok ? "PASS" : "FAIL", (int) got, (int) want);
}

int main() {
    std::mt19937 rng(1234);

    // Levels chosen from the ranges the MetaMensch notes cite: echo-only
    // residual after a working AEC is very quiet; near-end speech survives
    // double-talk suppression with real energy even when a VAD can't read it.
    constexpr float ECHO_RESIDUAL = 0.002f;  // AEC doing its job
    constexpr float USER_SPEECH   = 0.050f;  // user talking over it
    constexpr float FAR_LOUD      = 0.200f;  // speaker audible
    constexpr float FAR_SILENT    = 0.000f;  // speaker between chunks

    auto make = [&]() {
        auto d = std::make_unique<barge_detector>();
        d->pending.reserve(barge_detector::HOP_SAMPLES * 4);
        d->on_fire = []() {};
        d->set_active(true);
        return d;
    };

    // 1. Echo only, for a long time. Must never fire — this is the failure
    //    mode that makes raw-mic energy unusable.
    {
        auto d = make();
        feed_ms(*d, 5000, ECHO_RESIDUAL, FAR_LOUD, rng);
        check("echo-only for 5 s does not self-trigger", d->fired, false);
    }

    // 2. Echo, then the user talks over it. Must fire.
    {
        auto d = make();
        feed_ms(*d, 1000, ECHO_RESIDUAL, FAR_LOUD, rng);
        feed_ms(*d,  300, USER_SPEECH,   FAR_LOUD, rng);
        check("user speech over echo fires", d->fired, true);
    }

    // 3. Same, but the burst is shorter than the sustain window. Must not
    //    fire — a door slam or a single loud syllable of echo is not a barge.
    {
        auto d = make();
        feed_ms(*d, 1000, ECHO_RESIDUAL, FAR_LOUD, rng);
        feed_ms(*d,   60, USER_SPEECH,   FAR_LOUD, rng);
        feed_ms(*d,  500, ECHO_RESIDUAL, FAR_LOUD, rng);
        check("60 ms transient does not fire (sustain gate)", d->fired, false);
    }

    // 4. Loud near-end while the speaker is SILENT. Must not fire: that is not
    //    double-talk, and the followup VAD owns that case.
    {
        auto d = make();
        feed_ms(*d, 1000, ECHO_RESIDUAL, FAR_LOUD,   rng);
        feed_ms(*d,  500, USER_SPEECH,   FAR_SILENT, rng);
        check("loud mic while speaker silent does not fire", d->fired, false);
    }

    // 5. Firing during warmup must be impossible, however loud.
    {
        auto d = make();
        feed_ms(*d, 200, USER_SPEECH, FAR_LOUD, rng);
        check("cannot fire inside the 300 ms warmup", d->fired, false);
    }

    // 6. A LOUD but steady echo floor must still not trigger — the baseline
    //    adapts to it. This is the case a fixed absolute threshold gets wrong.
    {
        auto d = make();
        feed_ms(*d, 5000, 0.030f, FAR_LOUD, rng);
        check("loud-but-steady echo floor adapts, no trigger", d->fired, false);
    }

    // 7. Re-arming clears state so the next reply can be interrupted too.
    {
        auto d = make();
        feed_ms(*d, 1000, ECHO_RESIDUAL, FAR_LOUD, rng);
        feed_ms(*d,  300, USER_SPEECH,   FAR_LOUD, rng);
        const bool first = d->fired;
        d->set_active(false);
        d->set_active(true);
        feed_ms(*d, 1000, ECHO_RESIDUAL, FAR_LOUD, rng);
        const bool quiet_after_rearm = d->fired;
        feed_ms(*d,  300, USER_SPEECH,   FAR_LOUD, rng);
        check("fires on first reply", first, true);
        check("re-arm clears fired flag", quiet_after_rearm, false);
        check("fires again on the next reply", d->fired, true);
    }

    // 8. Inactive detector ignores everything.
    {
        auto d = make();
        d->set_active(false);
        feed_ms(*d, 2000, USER_SPEECH, FAR_LOUD, rng);
        check("inactive detector never fires", d->fired, false);
    }

    printf("\n%s\n", g_fail == 0 ? "all passed" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
