// Command-line settings for gemma-live.
//
// One table drives parsing AND the usage message, so --help cannot drift away
// from what the program accepts, and every default shown is the value actually
// compiled in rather than a number someone retyped into a string.
//
// Flags are grouped by the same three-letter prefixes the boot block prints
// (llm, sys, mtp, tts, aec, kwd, vad, nod, fup, brg), so a line of startup output
// tells you which flags tune it.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

struct gl_opts {
    // llm — the model itself. Swap trunk and mmproj together: the mmproj is
    // tied to model SIZE, and an E2B one on an E4B trunk yields garbage.
    std::string llm_model   = "models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf";
    std::string llm_mmproj  = "models/mmproj-gemma-4-E4B-it-Q8_0.gguf";
    int         llm_threads = 0;          // 0 = performance cores
    int         llm_ctx     = 8192;
    int         llm_predict = 256;
    float       llm_temp    = 0.3f;

    // sys
    std::string sys_prompt  = "prompts/chat.txt";

    // mtp — speculative decoding. The head must match the trunk; see README.
    bool        mtp_on      = true;
    std::string mtp_model   = "models/mtp-gemma-4-E4B-it-qat-Q4_0.gguf";
    int         mtp_draft   = 1;

    // tts
    std::string tts_model   = "models/vibevoice-realtime-0.5b-q4_k.gguf";
    std::string tts_voice   = "voices/vibevoice-voice-en-Gemma_woman.gguf";
    float       tts_cfg     = 1.5f;
    int         tts_steps   = 5;
    float       tts_anchor  = 0.2f;
    int         tts_chunk   = 3;          // latent frames before first audio
    float       tts_rms     = 0.06f;
    std::string dfn_model   = "";         // empty = post-filter off

    // aec
    std::string aec_model   = "models/localvqe.gguf";
    int         aec_threads = 2;
    float       aec_gate    = -45.0f;

    // kwd — wake detector
    std::string kwd_model   = "models/moonshine-streaming-tiny-q4_k.gguf";
    std::string kwd_wake    = "keywords/wake.txt";
    int         kwd_step    = 200;
    int         kwd_window  = 1500;
    bool        kwd_gpu     = false;
    float       kwd_ratio   = 3.0f;
    float       kwd_floor   = 0.008f;
    bool        kwd_nogate  = false;
    float       kwd_duck    = -35.0f;     // echo gate while the assistant talks
    bool        kwd_debug   = false;

    // vad — voice activity; shared by the end-of-turn and follow-up detectors
    std::string vad_model   = "models/firered-vad.gguf";
    int         vad_silence = 500;
    int         vad_empty   = 1200;
    bool        vad_debug   = false;

    // fup — followup window
    // nod — backchannels ("mm-hm") while the user is still talking
    bool        nod_on        = true;
    std::string nod_phrases   = "Mm-hm.,Mm.,Uh-huh.,Right.";
    int         nod_after     = 3000;
    int         nod_gap       = 3500;
    int         nod_monologue = 6000;
    int         nod_per_turn  = 3;
    int         nod_len       = 450;
    float       nod_gain      = 0.45f;
    bool        nod_debug     = false;
    std::string nod_dump;
    // stt — dictation: speech in, text back, no conversation turn
    std::string stt_model   = "models/moonshine-streaming-tiny-q4_k.gguf";
    int         stt_threads = 2;

    // rt — Realtime API server (gl-serve only)
    std::string rt_host  = "127.0.0.1";
    int         rt_port  = 8927;

    int         fup_timeout = 5000;
    int         fup_hops    = 3;
    float       fup_gate    = -30.0f;

    // brg — barge-in
    float       brg_ratio   = 3.0f;
    float       brg_floor   = 0.010f;
    int         brg_sustain = 150;
    bool        brg_debug   = false;

    int         verbosity   = 1;
};

struct gl_opt_def {
    const char * flag;
    char         type;      // s=string i=int f=float b=bool flag
    void       * p;
    const char * meta;      // argument placeholder, or nullptr for a flag
    const char * help;
};

// Built against a live gl_opts so the table stores real addresses; call with a
// default-constructed instance to render defaults in the usage message.
inline std::vector<gl_opt_def> gl_option_table(gl_opts & o) {
    return {
      {"--llm-model",   's', &o.llm_model,   "PATH", "Gemma 4 trunk (GGUF)"},
      {"--llm-mmproj",  's', &o.llm_mmproj,  "PATH", "audio/vision projector; must match model size"},
      {"--llm-threads", 'i', &o.llm_threads, "N",    "compute threads (0 = performance cores)"},
      {"--llm-ctx",     'i', &o.llm_ctx,     "N",    "context window"},
      {"--llm-predict", 'i', &o.llm_predict, "N",    "max tokens per reply"},
      {"--llm-temp",    'f', &o.llm_temp,    "X",    "sampling temperature"},

      {"--sys-prompt",  's', &o.sys_prompt,  "PATH", "system prompt file"},

      {"--mtp-off",     'o', &o.mtp_on,      nullptr,"disable speculative decoding"},
      {"--mtp-model",   's', &o.mtp_model,   "PATH", "MTP draft head; must match the trunk"},
      {"--mtp-draft",   'i', &o.mtp_draft,   "N",    "tokens drafted per step (1 measures best)"},

      {"--tts-model",   's', &o.tts_model,   "PATH", "VibeVoice model"},
      {"--tts-voice",   's', &o.tts_voice,   "PATH", "voice prompt"},
      {"--tts-cfg",     'f', &o.tts_cfg,     "X",    "classifier-free guidance scale"},
      {"--tts-steps",   'i', &o.tts_steps,   "N",    "diffusion steps (barely affects latency)"},
      {"--tts-anchor",  'f', &o.tts_anchor,  "X",    "negative-condition anchor blend"},
      {"--tts-chunk",   'i', &o.tts_chunk,   "N",    "latent frames in the first chunk; drives ttfa"},
      {"--tts-rms",     'f', &o.tts_rms,     "X",    "loudness target"},
      {"--dfn-model",   's', &o.dfn_model,   "PATH", "DeepFilterNet3 post-filter (off if unset)"},

      {"--aec-model",   's', &o.aec_model,   "PATH", "LocalVQE model"},
      {"--aec-threads", 'i', &o.aec_threads, "N",    "LocalVQE threads"},
      {"--aec-gate",    'f', &o.aec_gate,    "dBFS", "residual noise gate"},

      {"--kwd-model",   's', &o.kwd_model,   "PATH", "moonshine streaming model"},
      {"--kwd-wake",    's', &o.kwd_wake,    "PATH", "wake phrase list"},
      {"--kwd-step",    'i', &o.kwd_step,    "MS",   "transcript cadence; cost scales as window/step"},
      {"--kwd-window",  'i', &o.kwd_window,  "MS",   "rolling window each inference encodes"},
      {"--kwd-gpu",     'b', &o.kwd_gpu,     nullptr,"run on Metal (measured no faster; contends with TTS)"},
      {"--kwd-ratio",   'f', &o.kwd_ratio,   "X",    "silence gate opens at N x the learned noise floor"},
      {"--kwd-floor",   'f', &o.kwd_floor,   "X",    "silence gate never opens below this RMS"},
      {"--kwd-nogate",  'b', &o.kwd_nogate,  nullptr,"always run inference (A/B the gate)"},
      {"--kwd-duck",    'f', &o.kwd_duck,    "dBFS", "echo gate while the assistant is speaking"},
      {"--kwd-debug",   'b', &o.kwd_debug,   nullptr,"log transcripts and gate state"},

      {"--vad-model",   's', &o.vad_model,   "PATH", "firered-vad model (end-of-turn detection)"},
      {"--vad-silence", 'i', &o.vad_silence, "MS",   "silence before a turn is sent"},
      {"--vad-empty",   'i', &o.vad_empty,   "MS",   "send a turn the VAD heard no speech in after this"},
      {"--vad-debug",   'b', &o.vad_debug,   nullptr,"log VAD verdicts"},

      {"--nod-off",     'o', &o.nod_on,     nullptr,"disable backchannels"},
      {"--nod-phrases", 's', &o.nod_phrases, "LIST", "comma-separated clips to pre-render"},
      {"--nod-after",   'i', &o.nod_after,   "MS",   "never nod before the turn is this long"},
      {"--nod-gap",     'i', &o.nod_gap,     "MS",   "minimum spacing between nods"},
      {"--nod-mono",    'i', &o.nod_monologue,"MS",  "nod without transcript evidence past this"},
      {"--nod-per-turn",'i', &o.nod_per_turn,"N",    "most nods in one turn"},
      {"--nod-len",     'i', &o.nod_len,     "MS",   "cap on clip length (0 = uncapped)"},
      {"--nod-gain",    'f', &o.nod_gain,    "X",    "level relative to speech"},
      {"--nod-debug",   'b', &o.nod_debug,   nullptr,"log every nod and every near miss"},
      {"--nod-dump",    's', &o.nod_dump,    "DIR",  "write the rendered clips there as WAV and exit"},
      {"--stt-model",   's', &o.stt_model,   "PATH", "moonshine model for /api/transcribe (empty to disable)"},
      {"--stt-threads", 'i', &o.stt_threads, "N",    "moonshine threads"},

      {"--rt-host",     's', &o.rt_host,     "ADDR", "address to bind"},
      {"--rt-port",     'i', &o.rt_port,     "N",    "port to bind"},

      {"--fup-timeout", 'i', &o.fup_timeout, "MS",   "how long the follow-up window stays open"},
      {"--fup-hops",    'i', &o.fup_hops,    "N",    "100 ms hops of voice needed to continue"},
      {"--fup-gate",    'f', &o.fup_gate,    "dBFS", "RMS gate for follow-up voice"},

      {"--brg-ratio",   'f', &o.brg_ratio,   "X",    "fire at N x the learned echo-residual floor"},
      {"--brg-floor",   'f', &o.brg_floor,   "X",    "never fire below this residual RMS"},
      {"--brg-sustain", 'i', &o.brg_sustain, "MS",   "how long the residual must stay hot"},
      {"--brg-debug",   'b', &o.brg_debug,   nullptr,"log residual vs trigger on every fire"},

      {"--verbosity",   'i', &o.verbosity,   "0-2",  "0 silent, 1 normal, 2 full ggml/llama diagnostics"},
    };
}

// Spelled-out heading for each flag prefix. The prefixes themselves are the
// three-letter labels the startup block prints; the words are what they stand
// for, so the usage message reads without a decoder ring.
inline const char * gl_group_title(const std::string & prefix) {
    static const struct { const char * prefix, * title; } names[] = {
        { "llm",     "language model"      },
        { "sys",     "system prompt"       },
        { "mtp",     "speculative decoding"},
        { "tts",     "text to speech"      },
        { "dfn",     "speech enhancement"  },
        { "aec",     "echo cancellation"   },
        { "kwd",     "wake word"           },
        { "vad",     "voice activity"      },
        { "fup",     "follow-up"           },
        { "nod",     "backchannels"        },
        { "brg",     "barge-in"            },
        { "stt",     "dictation"           },
        { "rt",      "realtime server"     },
        { "general", "general"             },
    };
    for (const auto & n : names) if (prefix == n.prefix) return n.title;
    return prefix.c_str();
}

// `groups` is the whitelist of flag prefixes this binary accepts — gl-serve
// has no microphone, so --kwd-* would be a lie there, and gemma-live has no
// socket, so --rt-* would be a lie here. Restricting both the usage text and
// the parser from one list keeps a flag from being documented by a binary
// that would ignore it.
inline bool gl_group_enabled(const std::vector<std::string> & groups, const std::string & g) {
    if (groups.empty()) return true;
    if (g == "general") return true;
    return std::find(groups.begin(), groups.end(), g) != groups.end();
}

inline std::string gl_group_of(const char * flag) {
    const char * head = flag + 2;
    const char * dash = strchr(head, '-');
    return dash ? std::string(head, dash) : std::string("general");
}

inline void gl_usage(FILE * f, const char * argv0,
                     const std::vector<std::string> & groups = {}) {
    gl_opts d;                       // defaults, rendered from the real values
    const auto tbl = gl_option_table(d);
    fprintf(f,
        "gemma-live — local voice assistant (wake word, speech in, speech out)\n"
        "\n"
        "usage: %s [options]\n"
        "\n"
        "Flags are prefixed with the same three-letter labels the startup block\n"
        "prints, so the line describing a subsystem tells you which flags tune it.\n", argv0);
    std::string group;
    for (const auto & o : tbl) {
        // Group by the prefix before the first '-'; flags without one (e.g.
        // --verbosity) fall into "general" rather than constructing a string
        // from a null terminator.
        const std::string g = gl_group_of(o.flag);
        if (!gl_group_enabled(groups, g)) continue;
        if (g != group) { group = g; fprintf(f, "\n  %s\n", gl_group_title(g)); }
        char left[64];
        snprintf(left, sizeof(left), "%s%s%s", o.flag, o.meta ? " " : "", o.meta ? o.meta : "");
        char def[128] = "";
        switch (o.type) {
            case 's': { const auto & v = *(std::string*)o.p;
                        if (!v.empty()) snprintf(def, sizeof(def), "  [%s]", v.c_str()); break; }
            case 'i': snprintf(def, sizeof(def), "  [%d]", *(int*)o.p);   break;
            case 'f': snprintf(def, sizeof(def), "  [%g]", *(float*)o.p); break;
            case 'b':
            case 'o': break;   // a switch has no value to show
        }
        fprintf(f, "    %-24s %s%s\n", left, o.help, def);
    }
    fprintf(f, "\n    %-24s %s\n", "-h, --help", "show this message");
}

// Returns false on a bad argument, with *err describing it.
inline bool gl_parse_args(int argc, char ** argv, gl_opts & o, std::string * err,
                          const std::vector<std::string> & groups = {}) {
    const auto tbl = gl_option_table(o);
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") { gl_usage(stdout, argv[0], groups); exit(0); }
        bool matched = false;
        for (const auto & d : tbl) {
            if (a != d.flag) continue;
            if (!gl_group_enabled(groups, gl_group_of(d.flag))) continue;
            matched = true;
            // 'b' sets its flag, 'o' clears it. Spelled as a type rather
            // than special-cased by name: the previous version hardcoded
            // --mtp-off, so every later off-switch silently turned its
            // feature ON instead.
            if (d.type == 'b' || d.type == 'o') {
                *(bool *) d.p = (d.type == 'b');
                break;
            }
            if (i + 1 >= argc) { *err = a + " needs a value"; return false; }
            const char * v = argv[++i];
            switch (d.type) {
                case 's': *(std::string*)d.p = v;              break;
                case 'i': *(int*)d.p         = atoi(v);        break;
                case 'f': *(float*)d.p       = (float) atof(v);break;
            }
            break;
        }
        if (!matched) { *err = "unknown option: " + a; return false; }
    }
    return true;
}
