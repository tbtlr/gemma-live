// Backchannels — the short "mm-hm" a listener makes to signal "still with
// you, keep going" without taking the floor.
//
// The decision is a pure function of four numbers and one boolean, kept
// here so it can be tested without models, a microphone, or a clock (see
// tests/nod_test.cpp). main.cpp owns the audio; this owns the timing.
//
// ── Why the trigger looks like this ─────────────────────────────────────
//
// People backchannel at transition-relevance places: points where the turn
// COULD end but doesn't. Those are exactly the points where a turn also
// DOES end, and nothing in the signal distinguishes the two in advance. So
// the gate is built out of evidence that continuation is likely, in two
// tiers:
//
//   confident  the rolling transcript ends on a word an English sentence
//              essentially cannot end on ("and", "because", "the"). The
//              keyword worker already computes this to stretch the EOU
//              threshold from 500 to 900 ms — the same evidence, already
//              tuned, that says "they are not finished".
//
//   monologue  no transcript evidence, but the user has been talking for
//              long enough that saying nothing has itself become unnatural.
//
// Firing only on the confident tier is safe but so rare it is invisible on
// short exchanges; the monologue tier is what makes the feature audible.
//
// ── The pause window ────────────────────────────────────────────────────
//
// A nod goes in EARLY in a pause or not at all. Before min_pause_ms we risk
// clipping a word the speaker is still finishing; after max_pause_ms the
// user is probably done, and a nod there is both hesitant-sounding and a
// pure delay in front of the answer they are waiting for. Let EOU have it.
//
// ── Cost of being wrong ─────────────────────────────────────────────────
//
// Low, and worth stating because it is what makes the feature safe to ship:
// a nod fired at a real turn end produces "Mm-hm. The capital of France is
// Paris." — which is how people actually talk. The genuinely bad failure is
// firing mid-word, which reads as an interruption, and the pause window is
// what guards against it.
#pragma once

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

struct nod_detector {
    bool enabled          = true;
    int  min_utterance_ms = 3000;   // never nod at a short question
    int  min_pause_ms     = 200;    // a real pause, not a gap between words
    int  max_pause_ms     = 450;    // past this, the turn is probably over
    int  gap_ms           = 3500;   // minimum spacing between nods
    int  monologue_ms     = 6000;   // fire without transcript evidence past this
    int  max_per_turn     = 3;
    bool debug            = false;

    // Per-utterance state. reset() at every LISTENING entry.
    int     fired_this_turn = 0;
    int64_t last_fire_ms    = 0;
    int64_t busy_until_ms   = 0;    // a nod is audible until here

    void reset() {
        fired_this_turn = 0;
        last_fire_ms    = 0;
        busy_until_ms   = 0;
    }

    struct state {
        int64_t now_ms       = 0;
        int     utterance_ms = 0;   // since speech onset in this turn
        int     silence_ms   = 0;   // since the last voiced VAD verdict
        bool    mid_thought  = false;
    };

    // Reason codes exist so --nod-debug can say why a nod did NOT happen;
    // "it never fires" is otherwise impossible to diagnose from outside.
    enum class verdict {
        fire,
        disabled, busy, spent, too_soon_in_turn,
        pause_too_short, pause_too_long, too_soon_after_last, no_evidence,
    };

    static const char * name(verdict v) {
        switch (v) {
            case verdict::fire:                return "fire";
            case verdict::disabled:            return "disabled";
            case verdict::busy:                return "already nodding";
            case verdict::spent:               return "max per turn";
            case verdict::too_soon_in_turn:    return "utterance too short";
            case verdict::pause_too_short:     return "pause too short";
            case verdict::pause_too_long:      return "pause too long";
            case verdict::too_soon_after_last: return "too soon after last";
            case verdict::no_evidence:         return "no continuation evidence";
        }
        return "?";
    }

    verdict evaluate(const state & s) const {
        if (!enabled)                              return verdict::disabled;
        if (s.now_ms < busy_until_ms)              return verdict::busy;
        if (fired_this_turn >= max_per_turn)       return verdict::spent;
        if (s.utterance_ms < min_utterance_ms)     return verdict::too_soon_in_turn;
        if (s.silence_ms   < min_pause_ms)         return verdict::pause_too_short;
        if (s.silence_ms   > max_pause_ms)         return verdict::pause_too_long;
        if (last_fire_ms && s.now_ms - last_fire_ms < gap_ms)
                                                   return verdict::too_soon_after_last;
        // Tier gate: transcript evidence, or a long enough monologue that
        // silence from the listener has itself become the odd choice.
        if (!s.mid_thought && s.utterance_ms < monologue_ms)
                                                   return verdict::no_evidence;
        return verdict::fire;
    }

    bool should_fire(const state & s) const { return evaluate(s) == verdict::fire; }

    // Call immediately after the clip is queued.
    void note_fired(int64_t now_ms, int clip_ms) {
        fired_this_turn++;
        last_fire_ms  = now_ms;
        busy_until_ms = now_ms + clip_ms;
    }
};

// Strip near-silence from both ends of a rendered clip.
//
// VibeVoice pads what it synthesises, and on a 300 ms phrase the padding is
// most of the file: measured here, "Uh-huh." came back 674 ms long. A nod
// that runs two thirds of a second stops reading as a listener signal and
// starts reading as a turn, so the padding has to go. A small margin is
// kept either side so the consonant onset is not clipped.
inline void trim_silence(std::vector<float> & pcm, int rate,
                         float floor_rms = 0.005f, int margin_ms = 15) {
    if (pcm.empty()) return;
    const size_t win = (size_t) std::max(1, rate / 200);   // 5 ms
    auto loud = [&](size_t at) {
        const size_t end = std::min(pcm.size(), at + win);
        double sumsq = 0.0;
        for (size_t i = at; i < end; i++) sumsq += (double) pcm[i] * pcm[i];
        return std::sqrt(sumsq / (double) (end - at)) > floor_rms;
    };
    // One forward pass recording the first and last window above the floor.
    // Walking in from both ends instead leaves first < last even when the
    // clip is entirely silent, and shreds it to a couple of frames.
    size_t first = SIZE_MAX, last = 0;
    for (size_t at = 0; at + win <= pcm.size(); at += win) {
        if (!loud(at)) continue;
        if (first == SIZE_MAX) first = at;
        last = at + win;
    }
    if (first == SIZE_MAX) return;                         // all quiet: leave it

    const size_t margin = (size_t) margin_ms * (size_t) rate / 1000;
    first = first > margin ? first - margin : 0;
    last  = std::min(pcm.size(), last + margin);
    pcm.assign(pcm.begin() + (ptrdiff_t) first, pcm.begin() + (ptrdiff_t) last);
}

// Cap a clip's length, fading the tail so the cut is not a click.
//
// VibeVoice draws these out — measured here, "Mm." came back 610 ms for a
// single syllable, which reads as hesitation rather than acknowledgment. A
// hum that simply stops is what people actually do, so a hard cut under a
// short fade is faithful as well as convenient.
inline void cap_length(std::vector<float> & pcm, int rate, int max_ms,
                       int fade_ms = 40) {
    const size_t cap = (size_t) max_ms * (size_t) rate / 1000;
    if (max_ms <= 0 || pcm.size() <= cap) return;
    pcm.resize(cap);
    const size_t fade = std::min(cap, (size_t) fade_ms * (size_t) rate / 1000);
    for (size_t i = 0; i < fade; i++) {
        pcm[cap - fade + i] *= 1.0f - (float) i / (float) fade;
    }
}

// ── clip pool ───────────────────────────────────────────────────────────
//
// Backchannels are pre-rendered. A nod that arrives 400 ms late lands in
// the wrong place and is worse than none at all, and VibeVoice's ~335 ms
// time-to-first-audio cannot meet that; playing cached PCM costs only the
// device buffer.
//
// Repeating one clip is the tell that gives the whole thing away, so the
// pool is sampled without immediate repeats.
struct nod_pool {
    std::vector<std::vector<float>> clips;
    std::mt19937                    rng{0x6E6F64u};   // fixed: reproducible logs
    size_t                          last = SIZE_MAX;

    bool empty() const { return clips.empty(); }

    const std::vector<float> & pick() {
        if (clips.size() == 1) { last = 0; return clips[0]; }
        std::uniform_int_distribution<size_t> d(0, clips.size() - 1);
        size_t i = d(rng);
        if (i == last) i = (i + 1) % clips.size();
        last = i;
        return clips[i];
    }
};
