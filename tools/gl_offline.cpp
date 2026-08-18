// gl-offline — drive VoiceSession from WAV files instead of a microphone.
//
// Everything in main.cpp (audio hardware, AEC, the detectors, the turn state
// machine) is bypassed; this exercises the model half only — the audio
// encoder, the LLM sampling loop, MTP speculative decoding, and TTS — through
// the real begin_turn / push_audio / end_turn protocol.
//
// Pass one WAV per turn so the model sees a different utterance each time.
// That matters: feeding the same clip repeatedly makes MTP look far better
// than it is, because the second and third replies become trivially
// predictable and acceptance jumps to 100%.
//
// Multi-turn is also the point. A speculative-decoding bug that corrupts the
// KV cache does not show up on turn 1 — it shows up as turn 2 degrading into
// nonsense, which is exactly what a single-shot test would miss.
//
// Run from the repo root (paths below are relative). Requires 16 kHz mono
// 16-bit WAVs; make one with:
//   say -v Samantha -o q.aiff "your text here"
//   afconvert -f WAVE -d LEI16@16000 -c 1 q.aiff q.wav
//
// Usage:  gl-offline <turn1.wav> [turn2.wav ...]
// Env:    GL_LLM, GL_MMPROJ, GL_MTP_MODEL, GL_MTP (0/1), GL_MTP_DRAFT,
//         GL_TEMP, GL_PROMPT, GL_NPREDICT, GL_NCTX, GL_TTS_STEPS, GL_TTS_CFG
//         GL_ABORT_MS  — abort every turn this many ms into end_turn, to
//                        reproduce what repeated barge-in does to the
//                        conversation state. The LAST turn is never aborted,
//                        so its reply shows the damage the earlier ones did.
#include "session.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * env_or(const char * key, const char * fallback) {
    const char * v = std::getenv(key);
    return (v && *v) ? v : fallback;
}

// Minimal 16-bit PCM WAV reader. Walks the chunk list rather than assuming a
// 44-byte header — afconvert writes more than that.
static bool load_wav16(const char * path, std::vector<float> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    std::vector<unsigned char> raw;
    unsigned char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) raw.insert(raw.end(), buf, buf + n);
    std::fclose(f);

    if (raw.size() < 44 || std::memcmp(raw.data(), "RIFF", 4) ||
        std::memcmp(raw.data() + 8, "WAVE", 4)) {
        std::fprintf(stderr, "%s: not a RIFF/WAVE file\n", path);
        return false;
    }
    for (size_t p = 12; p + 8 <= raw.size(); ) {
        uint32_t sz;
        std::memcpy(&sz, raw.data() + p + 4, 4);
        if (!std::memcmp(raw.data() + p, "data", 4)) {
            const size_t avail = std::min((size_t) sz, raw.size() - (p + 8));
            const int16_t * s  = (const int16_t *) (raw.data() + p + 8);
            out.resize(avail / 2);
            for (size_t i = 0; i < out.size(); i++) out[i] = (float) s[i] / 32768.0f;
            return true;
        }
        p += 8 + sz + (sz & 1);
    }
    std::fprintf(stderr, "%s: no data chunk\n", path);
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gl-offline <turn1.wav> [turn2.wav ...]\n"
                             "       one 16 kHz mono WAV per conversation turn\n");
        return 2;
    }

    std::vector<std::vector<float>> clips;
    for (int i = 1; i < argc; i++) {
        std::vector<float> c;
        if (!load_wav16(argv[i], c)) return 1;
        std::fprintf(stderr, "turn %d: %s (%.2f s)\n", i - 1, argv[i],
                     (double) c.size() / GL_MIC_RATE);
        clips.push_back(std::move(c));
    }

    SessionConfig cfg;
    cfg.llm_model_path = env_or("GL_LLM",       "models/gemma-4-E4B-it-Q4_0.gguf");
    cfg.mmproj_path    = env_or("GL_MMPROJ",    "models/mmproj-gemma-4-E4B-it-Q8_0.gguf");
    cfg.tts_model_path = env_or("GL_TTS",       "models/vibevoice-realtime-0.5b-q4_k.gguf");
    cfg.tts_voice_path = env_or("GL_TTS_VOICE", "voices/vibevoice-voice-en-Gemma_woman.gguf");
    cfg.mtp_model_path = env_or("GL_MTP_MODEL", "models/mtp-gemma-4-E4B-it-Q4_0.gguf");
    cfg.n_predict      = std::atoi(env_or("GL_NPREDICT", "128"));
    cfg.n_ctx          = std::atoi(env_or("GL_NCTX", "8192"));
    cfg.verbosity      = 1;
    if (const char * v = std::getenv("GL_MTP"))       cfg.enable_mtp  = std::atoi(v) != 0;
    if (const char * v = std::getenv("GL_MTP_DRAFT")) cfg.mtp_n_draft = std::max(1, std::atoi(v));
    if (const char * v = std::getenv("GL_TEMP"))      cfg.temperature = (float) std::atof(v);
    if (const char * v = std::getenv("GL_TTS_STEPS"))  cfg.tts_steps   = std::max(1, std::atoi(v));
    if (const char * v = std::getenv("GL_TTS_CFG"))    cfg.tts_cfg     = (float) std::atof(v);
    if (const char * v = std::getenv("GL_TTS_FIRST_CHUNK")) cfg.tts_first_chunk_frames = std::max(1, std::atoi(v));
    {
        const char * pp = env_or("GL_PROMPT", "prompts/chat.txt");
        if (FILE * pf = std::fopen(pp, "rb")) {
            char b[8192];
            const size_t n = std::fread(b, 1, sizeof(b) - 1, pf);
            b[n] = 0;
            cfg.system_prompt = b;
            std::fclose(pf);
        } else {
            std::fprintf(stderr, "warning: no system prompt at %s\n", pp);
        }
    }

    std::string err;
    auto session = VoiceSession::create(cfg, &err);
    if (!session) { std::fprintf(stderr, "create failed: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "mtp: %s\n", session->mtp_active() ? "active" : "off");

    // TTS audio is synthesised for real (it is part of what we are timing) but
    // discarded — there is no speaker here.
    // Time to first audio chunk, measured from end_turn entry. This is the
    // dominant term in what a user actually feels after they stop speaking —
    // the EOU wait and this, with token sampling a distant third.
    size_t tts_samples = 0;
    std::chrono::steady_clock::time_point t_turn_start{};
    double ms_first_audio = -1.0;
    session->on_audio = [&](const float *, size_t n) {
        if (ms_first_audio < 0.0) {
            ms_first_audio = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_turn_start).count();
        }
        tts_samples += n;
    };
    session->on_token = [&](const char * t) { std::printf("%s", t); std::fflush(stdout); };

    const size_t chunk = GL_MIC_RATE / 10;   // 100 ms, the cadence main.cpp uses
    int failures = 0;

    for (size_t turn = 0; turn < clips.size(); turn++) {
        const std::vector<float> & pcm = clips[turn];
        std::printf("\n===== turn %zu =====\n", turn);
        std::fflush(stdout);

        if (!session->begin_turn()) {
            std::fprintf(stderr, "begin_turn: %s\n", session->last_error().c_str());
            return 1;
        }
        for (size_t off = 0; off < pcm.size(); off += chunk) {
            const size_t take = std::min(chunk, pcm.size() - off);
            if (!session->push_audio(pcm.data() + off, take)) {
                std::fprintf(stderr, "push_audio: %s\n", session->last_error().c_str());
                return 1;
            }
        }
        ms_first_audio = -1.0;
        t_turn_start = std::chrono::steady_clock::now();
        const auto t0 = t_turn_start;
        // Simulated barge-in: interrupt mid-reply exactly as the detector does.
        std::thread aborter;
        const int abort_ms = std::atoi(env_or("GL_ABORT_MS", "0"));
        if (abort_ms > 0 && turn + 1 < clips.size()) {
            aborter = std::thread([&, abort_ms]{
                std::this_thread::sleep_for(std::chrono::milliseconds(abort_ms));
                session->abort_turn();
            });
        }
        const bool ok = session->end_turn();
        if (aborter.joinable()) aborter.join();
        const double wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (!ok) {
            std::fprintf(stderr, "end_turn: %s\n", session->last_error().c_str());
            failures++;
            continue;
        }

        const TurnStats & st = session->last_stats();
        std::printf("\n[enc %d tok | llm %d tok | ttft %.0f ms | ttfa %.0f ms | gen %.0f ms | %.1f tok/s",
                    st.n_audio_tokens, st.n_llm_tokens, st.ms_ttft, ms_first_audio, st.ms_llm_gen,
                    st.ms_llm_gen > 0 ? 1000.0 * st.n_llm_tokens / st.ms_llm_gen : 0.0);
        if (st.n_drafted > 0) {
            std::printf(" | mtp %d/%d acc %.0f%%", st.n_accepted, st.n_drafted,
                        100.0 * st.n_accepted / st.n_drafted);
        }
        std::printf(" | turn %.0f ms]\n", wall);
        std::fflush(stdout);
    }

    std::fprintf(stderr, "\n%zu turns, %d failed, %.1f s of TTS audio\n",
                 clips.size(), failures, (double) tts_samples / GL_TTS_RATE);
    return failures == 0 ? 0 : 1;
}
