// Behavioural test for nod_detector. A backchannel is only ever right in a
// narrow window, and every guard here exists because the failure outside it
// is audible: too early clips a word, too late delays the real answer,
// too often is unbearable.
#include "nod.h"

#include <cstdio>
#include <string>

static int g_fail = 0;

static void check(const std::string & name, nod_detector::verdict got,
                  nod_detector::verdict want) {
    const bool ok = (got == want);
    if (!ok) g_fail++;
    printf("%-56s %s  (%s, want %s)\n", name.c_str(), ok ? "PASS" : "FAIL",
           nod_detector::name(got), nod_detector::name(want));
}

static nod_detector::state st(int64_t now, int utt, int sil, bool mid) {
    nod_detector::state s;
    s.now_ms = now; s.utterance_ms = utt; s.silence_ms = sil; s.mid_thought = mid;
    return s;
}

int main() {
    using v = nod_detector::verdict;

    // ── the two tiers ───────────────────────────────────────────────────
    {
        nod_detector d;
        check("mid-thought pause in a long enough utterance",
              d.evaluate(st(10000, 4000, 250, true)), v::fire);
        check("same pause, no transcript evidence, utterance still short",
              d.evaluate(st(10000, 4000, 250, false)), v::no_evidence);
        check("no evidence but past the monologue threshold",
              d.evaluate(st(10000, 7000, 250, false)), v::fire);
    }

    // ── the pause window ────────────────────────────────────────────────
    // Below it we are inside a word; above it the turn is probably over and
    // a nod is pure latency in front of the answer.
    {
        nod_detector d;
        check("pause of 80 ms is a gap between words, not a pause",
              d.evaluate(st(10000, 4000, 80, true)), v::pause_too_short);
        check("pause of 600 ms — let EOU have it",
              d.evaluate(st(10000, 4000, 600, true)), v::pause_too_long);
        check("pause at the low edge fires",
              d.evaluate(st(10000, 4000, d.min_pause_ms, true)), v::fire);
        check("pause at the high edge fires",
              d.evaluate(st(10000, 4000, d.max_pause_ms, true)), v::fire);
    }

    // ── short utterances never get one ──────────────────────────────────
    // "What time is it?" answered with "mm-hm" is the worst case in the
    // whole feature: it reads as not having understood.
    {
        nod_detector d;
        check("a two-second question gets no backchannel",
              d.evaluate(st(10000, 2000, 250, true)), v::too_soon_in_turn);
    }

    // ── spacing and budget ──────────────────────────────────────────────
    {
        nod_detector d;
        check("first nod fires", d.evaluate(st(10000, 4000, 250, true)), v::fire);
        d.note_fired(10000, 350);
        check("still audible — no overlap",
              d.evaluate(st(10200, 4200, 250, true)), v::busy);
        check("clip done but inside the spacing gap",
              d.evaluate(st(11000, 5000, 250, true)), v::too_soon_after_last);
        check("past the gap, fires again",
              d.evaluate(st(14000, 8000, 250, true)), v::fire);
    }
    {
        nod_detector d;
        int64_t t = 10000;
        for (int i = 0; i < d.max_per_turn; i++) {
            d.note_fired(t, 350);
            t += d.gap_ms + 100;
        }
        check("budget spent within one turn",
              d.evaluate(st(t, 30000, 250, true)), v::spent);
        d.reset();
        check("next turn starts fresh",
              d.evaluate(st(t, 4000, 250, true)), v::fire);
    }

    // ── off switch ──────────────────────────────────────────────────────
    {
        nod_detector d;
        d.enabled = false;
        check("disabled never fires",
              d.evaluate(st(10000, 9000, 250, true)), v::disabled);
    }

    // ── pool: no immediate repeats ──────────────────────────────────────
    {
        nod_pool p;
        p.clips.resize(4);
        for (size_t i = 0; i < p.clips.size(); i++) p.clips[i].assign(8, (float) i);
        size_t prev = SIZE_MAX;
        int    repeats = 0;
        for (int i = 0; i < 200; i++) {
            p.pick();
            if (p.last == prev) repeats++;
            prev = p.last;
        }
        const bool ok = (repeats == 0);
        if (!ok) g_fail++;
        printf("%-56s %s  (%d immediate repeats in 200 picks)\n",
               "pool never repeats a clip back to back", ok ? "PASS" : "FAIL", repeats);
    }

    // ── silence trimming ────────────────────────────────────────────────
    {
        const int rate = 24000;
        std::vector<float> clip((size_t) rate, 0.0f);          // 1 s of silence
        for (int i = 0; i < rate / 10; i++) {                  // 100 ms of tone
            clip[(size_t) (rate / 2 + i)] = (i % 2) ? 0.4f : -0.4f;
        }
        const size_t before = clip.size();
        trim_silence(clip, rate);
        const int ms = (int) (clip.size() * 1000 / (size_t) rate);
        const bool ok = ms >= 100 && ms <= 160;                // tone + 2x margin
        if (!ok) g_fail++;
        printf("%-56s %s  (%zu -> %d ms)\n",
               "trim keeps the tone and drops the padding",
               ok ? "PASS" : "FAIL", before, ms);

        std::vector<float> quiet((size_t) rate, 0.0f);
        trim_silence(quiet, rate);
        const bool ok2 = quiet.size() == (size_t) rate;
        if (!ok2) g_fail++;
        printf("%-56s %s  (%zu samples)\n",
               "an all-quiet clip is left alone, not emptied",
               ok2 ? "PASS" : "FAIL", quiet.size());
    }

    // ── length cap ──────────────────────────────────────────────────────
    {
        const int rate = 24000;
        std::vector<float> clip((size_t) rate, 0.5f);          // 1 s, full level
        cap_length(clip, rate, 450);
        const int ms = (int) (clip.size() * 1000 / (size_t) rate);
        const bool ok = (ms == 450);
        if (!ok) g_fail++;
        printf("%-56s %s  (1000 -> %d ms)\n",
               "long clip is capped", ok ? "PASS" : "FAIL", ms);

        const bool faded = clip.back() < 0.05f && clip[clip.size() / 2] > 0.4f;
        if (!faded) g_fail++;
        printf("%-56s %s  (tail %.3f, middle %.3f)\n",
               "the cut is faded, not a click", faded ? "PASS" : "FAIL",
               clip.back(), clip[clip.size() / 2]);

        std::vector<float> shortclip((size_t) rate / 10, 0.5f);   // 100 ms
        cap_length(shortclip, rate, 450);
        const bool ok3 = shortclip.size() == (size_t) rate / 10 && shortclip.back() == 0.5f;
        if (!ok3) g_fail++;
        printf("%-56s %s\n", "a clip under the cap is untouched", ok3 ? "PASS" : "FAIL");
    }

    printf("\n%s\n", g_fail ? "FAILURES" : "all passed");
    return g_fail ? 1 : 0;
}
