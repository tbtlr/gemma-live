// gl-bench-stt — audio straight into Gemma, versus transcribe-then-text.
//
// Both paths answer the same utterance with the same model and prompt:
//
//   audio   begin_turn(audio) -> push_audio -> end_turn
//   text    moonshine -> begin_turn(text) -> push_text -> end_turn
//
// The interesting term is the prefix. A second of speech is ~25 audio
// tokens through the mtmd encoder; the same second transcribed is a handful
// of text tokens. The text path therefore prefills far less — but pays for
// a whole ASR pass first, and throws away everything the audio carried that
// words do not.
//
// Turns alternate A, B, A, B so both paths see the same conversation depth;
// the context is shared and grows for both alike.
//
// Usage: gl-bench-stt <stt-model.gguf> <clip.wav> [clip.wav ...]
//        16 kHz or 24 kHz mono 16-bit WAVs.
#include "session.h"

extern "C" {
#include "moonshine_streaming.h"
#include "moonshine.h"
#include "parakeet.h"
#include "kyutai_stt.h"
}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static std::string slurp(const char * p) {
    FILE * f = fopen(p, "rb");
    std::string s;
    char b[4096];
    size_t n;
    while (f && (n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    if (f) fclose(f);
    return s;
}

// Minimal 16-bit PCM WAV reader; resamples to 16 kHz for both consumers.
static bool read_wav_16k(const char * path, std::vector<float> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) || memcmp(riff + 8, "WAVE", 4)) {
        fclose(f); return false;
    }
    int rate = 0, chans = 1, bits = 16;
    std::vector<int16_t> pcm;
    for (;;) {
        char id[4]; uint32_t sz;
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) break;
        if (!memcmp(id, "fmt ", 4)) {
            uint16_t fmt, ch, bps; uint32_t sr, br; uint16_t align;
            fread(&fmt, 2, 1, f); fread(&ch, 2, 1, f); fread(&sr, 4, 1, f);
            fread(&br, 4, 1, f);  fread(&align, 2, 1, f); fread(&bps, 2, 1, f);
            rate = (int) sr; chans = ch; bits = bps;
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            pcm.resize(sz / 2);
            fread(pcm.data(), 1, sz, f);
            break;
        } else {
            fseek(f, (long) sz, SEEK_CUR);
        }
    }
    fclose(f);
    if (pcm.empty() || rate == 0 || bits != 16) return false;

    std::vector<float> mono(pcm.size() / chans);
    for (size_t i = 0; i < mono.size(); i++) mono[i] = pcm[i * chans] / 32768.0f;
    if (rate == GL_MIC_RATE) { out = mono; return true; }
    const double r = (double) GL_MIC_RATE / rate;
    out.resize((size_t) (mono.size() * r));
    for (size_t i = 0; i < out.size(); i++) {
        const double src = i / r;
        const size_t i0 = (size_t) src, i1 = i0 + 1 < mono.size() ? i0 + 1 : i0;
        const float  fr = (float) (src - i0);
        out[i] = mono[i0] * (1 - fr) + mono[i1] * fr;
    }
    return true;
}

struct stt {
    moonshine_streaming_context * s = nullptr;
    moonshine_context           * p = nullptr;
    parakeet_context            * k = nullptr;
    kyutai_stt_context          * y = nullptr;
    bool init(const char * path) {
        auto pr = moonshine_streaming_context_default_params();
        pr.n_threads = 2; pr.verbosity = 0; pr.use_gpu = false; pr.temperature = 0.0f;
        s = moonshine_streaming_init_from_file(path, pr);
        if (s) return true;
        moonshine_init_params ip{};
        ip.model_path = path; ip.tokenizer_path = nullptr; ip.n_threads = 2;
        p = moonshine_init_with_params(ip);
        if (p) return true;
        auto kp = parakeet_context_default_params();
        kp.n_threads = 2; kp.verbosity = 0; kp.use_gpu = false;
        k = parakeet_init_from_file(path, kp);
        if (k) return true;
        y = kyutai_stt_init_from_file(path, kyutai_stt_context_default_params());
        return y != nullptr;
    }
    std::string run(const std::vector<float> & a) {
        if (s) { char * t = moonshine_streaming_transcribe(s, a.data(), (int) a.size());
                 std::string r = t ? t : ""; if (t) free(t); return r; }
        if (p) { const char * t = moonshine_transcribe(p, a.data(), (int) a.size());
                 return t ? std::string(t) : std::string(); }
        if (k) { char * t = parakeet_transcribe(k, a.data(), (int) a.size());
                 std::string r = t ? t : ""; if (t) free(t); return r; }
        if (y) { char * t = kyutai_stt_transcribe(y, a.data(), (int) a.size());
                 std::string r = t ? t : ""; if (t) free(t); return r; }
        return {};
    }
    const char * kind() const { return s ? "moonshine/streaming" : p ? "moonshine"
                                     : k ? "parakeet" : y ? "kyutai" : "none"; }
};

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <stt-model.gguf> <clip.wav> [clip.wav ...]\n", argv[0]);
        return 2;
    }
    stt asr;
    if (!asr.init(argv[1])) { fprintf(stderr, "ERR: could not load %s\n", argv[1]); return 1; }

    SessionConfig cfg;
    cfg.llm_model_path = "models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf";
    cfg.mmproj_path    = "models/mmproj-gemma-4-E4B-it-Q8_0.gguf";
    cfg.tts_model_path = "models/vibevoice-realtime-0.5b-q4_k.gguf";
    cfg.tts_voice_path = "voices/vibevoice-voice-en-Gemma_woman.gguf";
    cfg.mtp_model_path = "models/mtp-gemma-4-E4B-it-qat-Q4_0.gguf";
    cfg.system_prompt  = slurp("prompts/voice.txt");
    cfg.n_predict      = 96;
    cfg.verbosity      = 0;

    std::string err;
    auto vs = VoiceSession::create(cfg, &err);
    if (!vs) { fprintf(stderr, "ERR: %s\n", err.c_str()); return 1; }

    std::string reply;
    vs->on_token = [&](const char * t) { reply += t; };

    printf("stt: %s (%s)\n\n", argv[1], asr.kind());
    printf("%-14s %-6s %8s %8s %8s %7s %s\n",
           "clip", "path", "stt ms", "ttft ms", "turn ms", "in tok", "reply");
    printf("%s\n", std::string(104, '-').c_str());

    double sum_a_turn = 0, sum_b_turn = 0, sum_a_ttft = 0, sum_b_ttft = 0, sum_stt = 0;
    int n = 0;

    for (int i = 2; i < argc; i++) {
        std::vector<float> pcm;
        if (!read_wav_16k(argv[i], pcm)) { fprintf(stderr, "skip %s\n", argv[i]); continue; }
        const char * name = strrchr(argv[i], '/') ? strrchr(argv[i], '/') + 1 : argv[i];

        // ── A: audio straight in
        reply.clear();
        auto t0 = clk::now();
        vs->begin_turn(VoiceSession::turn_kind::audio);
        vs->push_audio(pcm.data(), pcm.size());
        vs->end_turn(/*speak=*/ false);
        const double a_turn = ms_since(t0);
        const TurnStats a = vs->last_stats();
        printf("%-14s %-6s %8s %8.0f %8.0f %7d  %s\n",
               name, "audio", "—", a.ms_ttft, a_turn, a.n_audio_tokens, reply.c_str());

        // ── B: transcribe, then the same question as text
        reply.clear();
        t0 = clk::now();
        const std::string text = asr.run(pcm);
        const double stt_ms = ms_since(t0);
        auto t1 = clk::now();
        vs->begin_turn(VoiceSession::turn_kind::text);
        vs->push_text(text);
        vs->end_turn(/*speak=*/ false);
        const double b_turn = ms_since(t1) + stt_ms;
        const TurnStats b = vs->last_stats();
        printf("%-14s %-6s %8.0f %8.0f %8.0f %7s  %s\n",
               "", "text", stt_ms, b.ms_ttft, b_turn, "—", reply.c_str());
        printf("%-14s %-6s %8s %8s %8s %7s  \"%s\"\n", "", "", "", "", "", "", text.c_str());

        sum_a_turn += a_turn; sum_b_turn += b_turn;
        sum_a_ttft += a.ms_ttft; sum_b_ttft += b.ms_ttft;
        sum_stt += stt_ms;
        n++;
    }

    if (n) {
        printf("\nmean over %d clips\n", n);
        printf("  audio   ttft %6.0f ms   turn %6.0f ms\n", sum_a_ttft / n, sum_a_turn / n);
        printf("  text    ttft %6.0f ms   turn %6.0f ms   (of which stt %.0f ms)\n",
               sum_b_ttft / n, sum_b_turn / n, sum_stt / n);
        printf("  audio is %.0f ms %s end to end\n",
               std::fabs(sum_a_turn - sum_b_turn) / n,
               sum_a_turn < sum_b_turn ? "FASTER" : "SLOWER");

        // The above is the BATCH case: the whole utterance handed over at
        // once. gl-serve does not do that any more — it opens the turn on
        // speech onset, so the encode and its prefill happen while the user
        // is still talking and are already paid when the turn ends. The
        // transcribe path has no such luxury: ASR cannot start until the
        // utterance is complete, so its cost is serial in front of the reply.
        //
        // What each path actually owes at end-of-turn, live:
        const double live_audio = sum_a_ttft / n;
        const double live_text  = (sum_stt + sum_b_ttft) / n;
        printf("\nat end-of-turn in the live server (audio encoded during speech)\n");
        printf("  audio   %6.0f ms   (ttft; prefill already done)\n", live_audio);
        printf("  text    %6.0f ms   (stt %.0f + ttft %.0f, both serial)\n",
               live_text, sum_stt / n, sum_b_ttft / n);
        printf("  audio is %.0f ms %s\n", std::fabs(live_audio - live_text),
               live_audio < live_text ? "FASTER" : "SLOWER");
    }
    return 0;
}
