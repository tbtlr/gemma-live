// gl-serve — an OpenAI Realtime-compatible WebSocket front-end for
// VoiceSession.
//
//   client ──ws──► input_audio_buffer.append (base64 pcm16)
//                        │
//                        ├─► firered VAD (server_vad turn detection)
//                        ▼
//                  VoiceSession: begin_turn / push_audio / end_turn
//                        │
//         ┌──────────────┴───────────────┐
//         ▼                              ▼
//   on_token ──► response.output_audio_transcript.delta
//   on_audio ──► response.output_audio.delta (base64 pcm16)
//
// Why this protocol: gemma-live already implements every hard part of a
// realtime voice session — server-side turn detection, barge-in, and a KV
// rollback on abort that matches conversation.item.truncate's semantics
// exactly (truncate the assistant turn at the point playback stopped, so
// the model's next turn is conditioned on what the user actually heard).
// Speaking the standard protocol makes that reusable by anything already
// written against OpenAI's API, instead of only by src/main.cpp.
//
// ── Deliberate limits ───────────────────────────────────────────────────
//
// ONE SESSION AT A TIME. VoiceSession owns a single llama context and is
// documented not to be thread-safe; a second concurrent conversation would
// need a second copy of every model. A client arriving while another is
// connected is closed with 1013 (try again later) rather than queued
// behind an unbounded wait.
//
// NO INPUT TRANSCRIPTION. `conversation.item.input_audio_transcription` is
// reported unsupported, because Gemma consumes audio as tokens through the
// mtmd encoder and never produces a transcript of the user's speech.
// Output transcription IS supported — those are the sampled tokens.
//
// SESSION-FIXED FIELDS. instructions, voice, temperature and
// max_response_output_tokens are properties of the loaded VoiceSession, set
// by the command-line flags at startup. session.update accepts them so that
// clients which always send them keep working, but ignores the values and
// echoes the real ones back in session.updated — a client can therefore see
// what is actually in effect rather than being told its request was applied.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"

extern "C" {
#include "moonshine_streaming.h"
}

#include "opts.h"
#include "session.h"
#include "vad.h"
#include "ws.h"

using json = nlohmann::ordered_json;

// ── audio helpers ───────────────────────────────────────────────────────
//
// The wire format is mono little-endian PCM16. VoiceSession wants float at
// GL_MIC_RATE going in, and hands float at tts_sample_rate() coming out.

static void pcm16_to_float(const std::vector<uint8_t> & in, std::vector<float> & out) {
    const size_t n = in.size() / 2;
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        const auto lo = (uint16_t) in[2 * i];
        const auto hi = (uint16_t) in[2 * i + 1];
        out[i] = (float) (int16_t) (uint16_t) (lo | (hi << 8)) / 32768.0f;
    }
}

static void float_to_pcm16(const float * in, size_t n, std::vector<uint8_t> & out) {
    out.resize(n * 2);
    for (size_t i = 0; i < n; i++) {
        float v = in[i];
        v = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
        const auto s = (int16_t) lrintf(v * 32767.0f);
        out[2 * i]     = (uint8_t) ((uint16_t) s & 0xFF);
        out[2 * i + 1] = (uint8_t) (((uint16_t) s >> 8) & 0xFF);
    }
}

// Linear resampling. Good enough here because both conversions are between
// close, related rates (24k<->16k on the way in, 48k->24k on the way out
// when the DFN post-filter is loaded) on speech that a 0.5B TTS produced;
// a polyphase filter would cost more than the artefacts are worth.
static void resample_linear(const std::vector<float> & in, int sr_in, int sr_out,
                            std::vector<float> & out) {
    if (sr_in == sr_out || in.empty()) { out = in; return; }
    const double ratio = (double) sr_out / (double) sr_in;
    const size_t n_out = (size_t) ((double) in.size() * ratio);
    out.resize(n_out);
    for (size_t i = 0; i < n_out; i++) {
        const double src = (double) i / ratio;
        const size_t i0  = (size_t) src;
        const size_t i1  = i0 + 1 < in.size() ? i0 + 1 : i0;
        const float  f   = (float) (src - (double) i0);
        out[i] = in[i0] * (1.0f - f) + in[i1] * f;
    }
}

// ── event ids ───────────────────────────────────────────────────────────
// Opaque to the client; a counter is enough and keeps logs readable.
static std::atomic<uint64_t> g_id_seq{0};
static std::string new_id(const char * prefix) {
    char b[64];
    snprintf(b, sizeof(b), "%s_%06llu", prefix,
             (unsigned long long) g_id_seq.fetch_add(1));
    return b;
}

// VoiceSession owns one llama context and is not thread-safe, and it has two
// front ends: the WebSocket voice session and the text chat endpoint. This is
// what keeps a typed message from being decoded into the middle of a spoken
// turn.
static std::mutex g_turn_mu;

// One line per turn, in the same shape gemma-live prints, so numbers from
// the two front ends can be compared directly.
//
//   wait  blocked on the turn lock — the other front end was mid-turn
//   ttft  end of input to the first sampled token
//   ttfa  end of input to the first audio byte written to the socket. This
//         is the server's half of what the browser feels; the page logs its
//         own, and the difference is transport plus playback scheduling.
//   rtf   synthesis wall time over audio produced; >1 cannot keep up
static void log_turn(const char * kind, const TurnStats & st, int tts_rate,
                     double ms_wait, double ms_total, double ms_ttfa) {
    const double tok_s   = st.ms_llm_gen > 0.0
                         ? 1000.0 * (double) st.n_llm_tokens / st.ms_llm_gen : 0.0;
    const double audio_s = tts_rate > 0 ? (double) st.n_tts_samples / tts_rate : 0.0;
    const double rtf     = audio_s > 0.0 ? (st.ms_tts_wall / 1000.0) / audio_s : 0.0;

    char mtp[64] = "";
    if (st.n_drafted > 0) {
        snprintf(mtp, sizeof(mtp), " | mtp %d/%d acc %.0f%%", st.n_accepted, st.n_drafted,
                 100.0 * (double) st.n_accepted / (double) st.n_drafted);
    }
    char wait[32] = "";
    if (ms_wait >= 1.0) snprintf(wait, sizeof(wait), " | wait %.0f ms", ms_wait);
    char ttfa[32] = "";
    if (ms_ttfa >= 0.0) snprintf(ttfa, sizeof(ttfa), " | ttfa %.0f ms", ms_ttfa);
    char aud[48] = "";
    if (audio_s > 0.0) snprintf(aud, sizeof(aud), " | tts %.2f s | rtf %.2f", audio_s, rtf);

    fprintf(stderr, "[%s | enc %d tok | llm %d tok @ %.1f tok/s | ttft %.0f ms%s%s%s%s"
                    " | turn %.0f ms]\n",
            kind, st.n_audio_tokens, st.n_llm_tokens, tok_s, st.ms_ttft,
            ttfa, aud, mtp, wait, ms_total);
}

// ── connection ──────────────────────────────────────────────────────────
struct rt_conn {
    int        fd = -1;
    std::mutex send_mu;         // on_audio fires from the TTS worker thread
    std::atomic<bool> dead{false};

    // Set once a session exists, so a write that fails can stop the turn
    // rather than synthesising the rest of a reply nobody will hear.
    std::function<void()> on_dead;

    void send(const json & j) {
        const std::string s = j.dump();
        bool just_died = false;
        {
            std::lock_guard<std::mutex> lk(send_mu);
            if (dead.load()) return;
            if (!ws::send_text(fd, s)) { dead.store(true); just_died = true; }
        }
        if (just_died && on_dead) on_dead();
    }

    void send_error(const std::string & type, const std::string & code,
                    const std::string & message, const std::string & event_id = "") {
        json e = {{"event_id", new_id("event")},
                  {"type", "error"},
                  {"error", {{"type", type}, {"code", code}, {"message", message}}}};
        if (!event_id.empty()) e["error"]["event_id"] = event_id;
        send(e);
    }
};

// ── the session ─────────────────────────────────────────────────────────
struct rt_session {
    rt_conn      & conn;
    VoiceSession & vs;
    const gl_opts & O;

    // Turn detection. `server_vad` mirrors OpenAI's: the server decides when
    // the user stopped talking. `none` hands that to the client, which must
    // send input_audio_buffer.commit and response.create itself.
    bool        server_vad          = true;
    bool        vad_create_response = true;
    gl_vad::eou vad;

    int  in_rate  = 24000;      // pcm16 rate the client sends (OpenAI default)
    int  out_rate = 24000;      // pcm16 rate we send back

    std::vector<float> buf;     // input not yet handed to the model

    // A turn is opened as soon as speech starts, not when it ends, so the
    // audio encoder runs WHILE the user talks instead of in front of the
    // reply. Measured before this: encode cost ~30ms per second of speech,
    // all of it serial after end-of-turn — 296ms for a 10s utterance, plus
    // the tail finalise pushing ttft from 140 to 307ms. The desktop app
    // never paid it because it pushes every 100ms during LISTENING.
    //
    // The turn lock is therefore held from speech onset to response.done. A
    // typed message arriving mid-utterance waits, which the stats line
    // reports as `wait` — the same serialisation the desktop has by nature,
    // and the price of taking the encode off the critical path.
    std::unique_lock<std::mutex> turn_lock;
    bool turn_open = false;
    std::string        session_id = new_id("sess");

    // Live response state. `active` gates cancel; both are read from the
    // reader thread, hence atomic.
    std::atomic<bool> active{false};
    std::atomic<bool> cancelled{false};
    std::string       resp_id, item_id;
    std::string       transcript;
    uint64_t          audio_samples_sent = 0;

    rt_session(rt_conn & c, VoiceSession & v, const gl_opts & o) : conn(c), vs(v), O(o) {}

    // Begin a turn and hand it everything buffered so far. Safe to call more
    // than once; only the first call does anything.
    bool open_turn() {
        if (turn_open) return true;
        turn_lock = std::unique_lock<std::mutex>(g_turn_mu);
        if (!vs.begin_turn()) {
            turn_lock.unlock();
            conn.send_error("server_error", "session_error", vs.last_error());
            return false;
        }
        turn_open = true;
        return feed_model();
    }

    // Push buffered audio into the open turn. Encoding happens here, on the
    // reader's cadence, rather than all at once at end-of-turn.
    bool feed_model() {
        if (!turn_open || buf.empty()) return true;
        const bool ok = vs.push_audio(buf.data(), buf.size());
        pushed_samples += buf.size();
        buf.clear();
        if (!ok) conn.send_error("server_error", "session_error", vs.last_error());
        return ok;
    }

    // Throw away a turn that was opened but will not be answered.
    void discard_turn() {
        if (!turn_open) return;
        vs.abort_turn();
        vs.end_turn(/*speak=*/ false);   // rolls the KV back
        turn_open = false;
        pushed_samples = 0;
        if (turn_lock.owns_lock()) turn_lock.unlock();
    }

    size_t pushed_samples = 0;

    // ── outbound ────────────────────────────────────────────────────────
    json session_object() const {
        return {
            {"id", session_id},
            {"object", "realtime.session"},
            {"model", "gemma-live"},
            {"modalities", json::array({"audio", "text"})},
            {"instructions", O.sys_prompt},
            {"voice", O.tts_voice},
            {"input_audio_format",  "pcm16"},
            {"output_audio_format", "pcm16"},
            // Null rather than an object: we cannot transcribe the user's
            // audio, and claiming a transcriber that never emits would leave
            // a client waiting for conversation.item.input_audio_transcription
            // .completed forever.
            {"input_audio_transcription", nullptr},
            {"turn_detection", server_vad
                ? json{{"type", "server_vad"},
                       {"threshold", gl_vad::THRESHOLD},
                       {"prefix_padding_ms", vad.prefix_pad_ms},
                       {"silence_duration_ms", vad.silence_ms},
                       {"create_response", vad_create_response}}
                : json(nullptr)},
            {"temperature", O.llm_temp},
            {"max_response_output_tokens", O.llm_predict},
        };
    }

    void send_event(const char * type, json extra = json::object()) {
        json e = {{"event_id", new_id("event")}, {"type", type}};
        for (auto & kv : extra.items()) e[kv.key()] = kv.value();
        conn.send(e);
    }

    // ── turn execution ──────────────────────────────────────────────────
    //
    // Runs on the session thread and BLOCKS for the whole reply: end_turn()
    // returns only once the LLM has stopped sampling and TTS has emitted its
    // last chunk. That is exactly why the socket is read on a separate
    // thread — response.cancel has to be able to arrive during this call.
    void run_turn() {
        const auto t_enter = std::chrono::steady_clock::now();
        if (buf.empty() && pushed_samples == 0) {
            conn.send_error("invalid_request_error", "input_audio_buffer_commit_empty",
                            "no audio in the buffer to respond to");
            return;
        }
        if (active.load()) {
            conn.send_error("invalid_request_error", "conversation_already_has_active_response",
                            "a response is already in progress");
            return;
        }

        resp_id    = new_id("resp");
        item_id    = new_id("item");
        transcript.clear();
        audio_samples_sent = 0;
        cancelled.store(false);
        active.store(true);

        send_event("response.created", {{"response",
            {{"id", resp_id}, {"object", "realtime.response"},
             {"status", "in_progress"}, {"output", json::array()}}}});
        send_event("response.output_item.added", {
            {"response_id", resp_id}, {"output_index", 0},
            {"item", {{"id", item_id}, {"object", "realtime.item"}, {"type", "message"},
                      {"status", "in_progress"}, {"role", "assistant"},
                      {"content", json::array()}}}});
        send_event("response.content_part.added", {
            {"response_id", resp_id}, {"item_id", item_id},
            {"output_index", 0}, {"content_index", 0},
            {"part", {{"type", "audio"}, {"transcript", ""}}}});

        // Manual turn detection never saw a speech onset, so the turn opens
        // here and takes the whole encode at once — the old behaviour.
        const double tail_s = (double) buf.size() / GL_MIC_RATE;
        const auto t_enc0 = std::chrono::steady_clock::now();
        if (!open_turn()) { active.store(false); return; }
        ms_encode = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_enc0).count();
        const auto t_go = std::chrono::steady_clock::now();
        first_audio_at  = {};

        vs.end_turn();
        turn_open = false;
        pushed_samples = 0;
        if (turn_lock.owns_lock()) turn_lock.unlock();
        active.store(false);

        const bool was_cancelled = cancelled.load();
        if (!was_cancelled) {
            send_event("response.output_audio.done", {
                {"response_id", resp_id}, {"item_id", item_id},
                {"output_index", 0}, {"content_index", 0}});
            send_event("response.output_audio_transcript.done", {
                {"response_id", resp_id}, {"item_id", item_id},
                {"output_index", 0}, {"content_index", 0},
                {"transcript", transcript}});
            send_event("response.content_part.done", {
                {"response_id", resp_id}, {"item_id", item_id},
                {"output_index", 0}, {"content_index", 0},
                {"part", {{"type", "audio"}, {"transcript", transcript}}}});
        }
        send_event("response.output_item.done", {
            {"response_id", resp_id}, {"output_index", 0},
            {"item", {{"id", item_id}, {"object", "realtime.item"}, {"type", "message"},
                      {"status", was_cancelled ? "incomplete" : "completed"},
                      {"role", "assistant"},
                      {"content", json::array({json{{"type", "audio"},
                                                    {"transcript", transcript}}})}}}});

        const TurnStats & st = vs.last_stats();
        if (O.verbosity > 0) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            const auto now = std::chrono::steady_clock::now();
            if (ms_encode >= 1.0) {
                fprintf(stderr, "  [encode %.0f ms of tail (%.1f s not yet fed)]\n",
                        ms_encode, tail_s);
            }
            log_turn(was_cancelled ? "voice interrupted" : "voice", st, out_rate,
                     ms(t_enter, t_go), ms(t_enter, now),
                     first_audio_at.time_since_epoch().count()
                         ? ms(t_go, first_audio_at) : -1.0);
        }
        send_event("response.done", {{"response",
            {{"id", resp_id}, {"object", "realtime.response"},
             {"status", was_cancelled ? "cancelled" : "completed"},
             {"output", json::array({json{{"id", item_id}, {"type", "message"},
                                          {"role", "assistant"},
                                          {"content", json::array({json{
                                              {"type", "audio"},
                                              {"transcript", transcript}}})}}})},
             {"usage", {{"input_tokens",  st.n_audio_tokens},
                        {"output_tokens", st.n_llm_tokens},
                        {"total_tokens",  st.n_audio_tokens + st.n_llm_tokens}}}}}});
    }

    // ── model callbacks ─────────────────────────────────────────────────
    void on_token(const char * text) {
        transcript += text;
        send_event("response.output_audio_transcript.delta", {
            {"response_id", resp_id}, {"item_id", item_id},
            {"output_index", 0}, {"content_index", 0},
            {"delta", text}});
    }

    // Fires on the TTS worker thread, and may fire briefly after the LLM is
    // done — rt_conn::send is mutexed for exactly this reason.
    std::chrono::steady_clock::time_point first_audio_at{};
    double ms_encode = 0;

    void on_audio(const float * pcm, size_t n) {
        if (cancelled.load()) return;         // client already stopped playback
        if (first_audio_at.time_since_epoch().count() == 0) {
            first_audio_at = std::chrono::steady_clock::now();
        }
        std::vector<float> src(pcm, pcm + n), rs;
        resample_linear(src, vs.tts_sample_rate(), out_rate, rs);
        std::vector<uint8_t> bytes;
        float_to_pcm16(rs.data(), rs.size(), bytes);
        audio_samples_sent += rs.size();
        send_event("response.output_audio.delta", {
            {"response_id", resp_id}, {"item_id", item_id},
            {"output_index", 0}, {"content_index", 0},
            {"delta", ws::b64_encode(bytes.data(), bytes.size())}});
    }

    // ── inbound events ──────────────────────────────────────────────────
    void handle(const json & ev) {
        const std::string type = ev.value("type", "");
        const std::string eid  = ev.value("event_id", "");

        if (type == "session.update") {
            const auto & s = ev.value("session", json::object());
            if (s.contains("turn_detection")) {
                const auto & td = s["turn_detection"];
                if (td.is_null()) {
                    server_vad = false;
                } else {
                    const std::string t = td.value("type", "server_vad");
                    if (t != "server_vad") {
                        conn.send_error("invalid_request_error", "unsupported_turn_detection",
                                        "only server_vad and null are supported (got \"" + t + "\")", eid);
                        return;
                    }
                    server_vad          = true;
                    vad.silence_ms      = td.value("silence_duration_ms", vad.silence_ms);
                    vad.prefix_pad_ms   = td.value("prefix_padding_ms",   vad.prefix_pad_ms);
                    vad_create_response = td.value("create_response",     vad_create_response);
                }
            }
            for (const char * f : {"input_audio_format", "output_audio_format"}) {
                if (s.contains(f) && s[f].is_string() && s[f] != "pcm16") {
                    conn.send_error("invalid_request_error", "unsupported_audio_format",
                                    std::string(f) + " must be pcm16 (g711 is not supported)", eid);
                    return;
                }
            }
            send_event("session.updated", {{"session", session_object()}});
            return;
        }

        if (type == "input_audio_buffer.append") {
            const std::string b64 = ev.value("audio", "");
            std::vector<uint8_t> bytes;
            if (b64.empty() || !ws::b64_decode(b64, bytes)) {
                conn.send_error("invalid_request_error", "invalid_value",
                                "audio must be base64-encoded pcm16", eid);
                return;
            }
            std::vector<float> f, rs;
            pcm16_to_float(bytes, f);
            resample_linear(f, in_rate, GL_MIC_RATE, rs);
            buf.insert(buf.end(), rs.begin(), rs.end());
            if (turn_open) feed_model();   // encode as it arrives

            if (server_vad && !active.load()) {
                switch (vad.feed(rs.data(), rs.size())) {
                    case gl_vad::eou::event::speech_started:
                        send_event("input_audio_buffer.speech_started", {
                            {"audio_start_ms", (int) (vad.onset_sample * 1000 / GL_MIC_RATE)},
                            {"item_id", new_id("item")}});
                        // Start encoding now, while they are still speaking.
                        open_turn();
                        break;
                    case gl_vad::eou::event::speech_stopped: {
                        send_event("input_audio_buffer.speech_stopped", {
                            {"audio_end_ms", (int) (vad.samples_total * 1000 / GL_MIC_RATE)},
                            {"item_id", new_id("item")}});
                        commit();
                        if (vad_create_response) run_turn();
                        break;
                    }
                    case gl_vad::eou::event::none: break;
                }
            }
            return;
        }

        if (type == "input_audio_buffer.commit")  { commit(); return; }

        if (type == "input_audio_buffer.clear") {
            discard_turn();      // a turn opened on speech onset is unwound
            buf.clear();
            vad.reset();
            send_event("input_audio_buffer.cleared");
            return;
        }

        if (type == "response.create") { run_turn(); return; }

        if (type == "response.cancel") {
            // Handled on the reader thread so it can land mid-turn; reaching
            // here means there was no turn to cancel.
            conn.send_error("invalid_request_error", "response_cancel_not_active",
                            "no active response to cancel", eid);
            return;
        }

        if (type == "conversation.item.truncate") {
            conn.send_error("invalid_request_error", "item_truncate_invalid",
                            "truncate applies to the in-progress response only", eid);
            return;
        }

        conn.send_error("invalid_request_error", "unknown_type",
                        "unsupported event type \"" + type + "\"", eid);
    }

    void commit() {
        if (buf.empty()) {
            conn.send_error("invalid_request_error", "input_audio_buffer_commit_empty",
                            "buffer is empty");
            return;
        }
        const std::string id = new_id("item");
        send_event("input_audio_buffer.committed", {{"item_id", id}});
        send_event("conversation.item.created", {
            {"item", {{"id", id}, {"object", "realtime.item"}, {"type", "message"},
                      {"status", "completed"}, {"role", "user"},
                      {"content", json::array({json{{"type", "input_audio"}}})}}}});
        vad.reset();
    }
};

// ── event queue between the reader thread and the session thread ────────
struct evq {
    std::mutex              m;
    std::condition_variable cv;
    std::deque<json>        q;
    bool                    closed = false;

    void push(json j) {
        { std::lock_guard<std::mutex> lk(m); q.push_back(std::move(j)); }
        cv.notify_one();
    }
    void close() {
        { std::lock_guard<std::mutex> lk(m); closed = true; }
        cv.notify_all();
    }
    bool pop(json * out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return closed || !q.empty(); });
        if (q.empty()) return false;
        *out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

// Serve one client to completion.
static void serve_client(int fd, VoiceSession & vs, const gl_opts & O) {
    rt_conn conn;
    conn.fd = fd;

    rt_session S(conn, vs, O);
    S.in_rate = S.out_rate = 24000;
    S.vad.sample_rate = GL_MIC_RATE;
    S.vad.silence_ms  = O.vad_silence;
    if (!S.vad.init(O.vad_model.c_str())) {
        conn.send_error("server_error", "vad_load_failed",
                        "could not load " + O.vad_model);
        ws::send_close(fd, 1011, "vad load failed");
        return;
    }
    S.vad.reset();

    conn.on_dead = [&S, &vs] {
        if (S.active.load()) { S.cancelled.store(true); vs.abort_turn(); }
    };
    {
        // Under the turn lock: a text request may be mid-turn with its own
        // callbacks installed, and it restores what it found on the way out.
        std::lock_guard<std::mutex> lk(g_turn_mu);
        vs.on_token = [&S](const char * t)            { S.on_token(t); };
        vs.on_audio = [&S](const float * p, size_t n) { S.on_audio(p, n); };
        vs.on_done  = nullptr;
    }

    S.send_event("session.created", {{"session", S.session_object()}});

    evq        q;
    std::thread reader([&] {
        for (;;) {
            ws::op   o;
            std::string payload;
            const int r = ws::recv_message(fd, &o, &payload, 30000);
            if (r < 0) break;   // peer gone; the abort below stops any live turn
            if (r == 0) continue;                     // idle; keep waiting
            if (o != ws::op::text) continue;          // audio rides inside JSON

            json ev;
            try {
                ev = json::parse(payload);
            } catch (const std::exception & e) {
                conn.send_error("invalid_request_error", "invalid_json", e.what());
                continue;
            }
            const std::string type = ev.value("type", "");

            // Interrupts must not queue behind the audio they are meant to
            // stop. abort_turn is documented safe from any thread, and rolls
            // the KV cache back so the next turn is conditioned on what the
            // user actually heard rather than on a reply that was cut off.
            if ((type == "response.cancel" || type == "conversation.item.truncate")
                && S.active.load()) {
                S.cancelled.store(true);
                vs.abort_turn();
                continue;
            }
            q.push(std::move(ev));
        }
        conn.dead.store(true);
        if (S.active.load()) { S.cancelled.store(true); vs.abort_turn(); }
        q.close();
    });

    json ev;
    while (q.pop(&ev)) {
        if (conn.dead.load()) break;
        S.handle(ev);
    }

    conn.dead.store(true);
    ::shutdown(fd, SHUT_RDWR);
    reader.join();
    // A turn opened on speech onset but never finished would otherwise keep
    // the turn lock for the life of the process.
    S.discard_turn();
    conn.on_dead = nullptr;
    S.vad.shutdown();
    {
        std::lock_guard<std::mutex> lk(g_turn_mu);
        vs.on_token = nullptr;
        vs.on_audio = nullptr;
    }
}

// ── dictation ───────────────────────────────────────────────────────────
//
// POST /api/transcribe  {"audio": "<base64 pcm16>", "rate": 24000}
//                    -> {"text": "..."}
//
// Speech to text with NO conversation turn: the audio is transcribed and
// thrown away, nothing enters the KV cache, and the model never sees it.
// That separation is the whole point — dictation fills a text box that the
// user then edits and may never send, so it must not be able to change what
// the assistant thinks was said.
//
// It is a different model for the same reason. Gemma consumes audio as
// tokens through the mtmd encoder and never emits a transcript of it, so
// asking Gemma would mean running a real turn and rolling it back. Moonshine
// is small, fast, and completely outside the conversation.
//
// Single-shot by design. A UI that wants ChatGPT's live partials calls this
// repeatedly on the growing buffer — re-transcribing from the start each
// time, which is what makes the text stable rather than jittering as a
// rolling window slides. Cost grows with the recording, so poll at ~1 Hz,
// not per frame.
struct stt_engine {
    moonshine_streaming_context * ctx = nullptr;
    std::mutex mu;                 // one context, and requests can overlap

    bool init(const std::string & path, int threads, int verbosity) {
        if (path.empty()) return false;
        auto params = moonshine_streaming_context_default_params();
        params.n_threads   = threads > 0 ? threads : 2;
        params.verbosity   = verbosity >= 2 ? 1 : 0;
        params.use_gpu     = false;
        params.temperature = 0.0f;     // greedy: dictation wants repeatable
        ctx = moonshine_streaming_init_from_file(path.c_str(), params);
        return ctx != nullptr;
    }
    void shutdown() {
        if (ctx) { moonshine_streaming_free(ctx); ctx = nullptr; }
    }
    // Longest audio handed to the model in one call.
    //
    // Measured, not guessed: moonshine-streaming-tiny transcribes cleanly up
    // to ~7.6 s, and past that it SILENTLY TRUNCATES — a 12 s clip came back
    // with exactly the transcript of its first 7.6 s, no error, no marker.
    // That is the worst failure a dictation box can have, so anything longer
    // is split rather than passed through and quietly shortened.
    static constexpr double MAX_CHUNK_S = 6.0;
    // How far back from the cap to hunt for a quiet frame to cut on.
    static constexpr double SEARCH_S    = 1.5;

    std::string transcribe_one(const float * pcm, size_t n) {
        if (!ctx || n == 0) return {};
        char * t = moonshine_streaming_transcribe(ctx, pcm, (int) n);
        if (!t) return {};
        std::string out(t);
        free(t);
        return out;
    }

    // Split point for a chunk starting at `from`: the quietest 20 ms frame in
    // the window before the cap, so the cut lands in a pause rather than
    // through the middle of a word.
    static size_t split_at(const std::vector<float> & pcm, size_t from, int rate) {
        const size_t cap = from + (size_t) (MAX_CHUNK_S * rate);
        if (cap >= pcm.size()) return pcm.size();
        const size_t search_from = cap - (size_t) (SEARCH_S * rate);
        const size_t frame = (size_t) (0.02 * rate);
        size_t best = cap;
        double best_rms = 1e9;
        for (size_t at = search_from; at + frame <= cap; at += frame) {
            double sum = 0;
            for (size_t i = at; i < at + frame; i++) sum += (double) pcm[i] * pcm[i];
            const double rms = std::sqrt(sum / (double) frame);
            if (rms < best_rms) { best_rms = rms; best = at + frame / 2; }
        }
        return best;
    }

    // 16 kHz mono float in, UTF-8 out. Returns the number of model calls in
    // *n_chunks so the caller can report it.
    std::string transcribe(const std::vector<float> & pcm, int rate, int * n_chunks) {
        std::lock_guard<std::mutex> lk(mu);
        *n_chunks = 0;
        if (!ctx || pcm.empty()) return {};

        std::string out;
        size_t from = 0;
        while (from < pcm.size()) {
            const size_t to = split_at(pcm, from, rate);
            const std::string part = transcribe_one(pcm.data() + from, to - from);
            (*n_chunks)++;
            if (!part.empty()) {
                if (!out.empty() && out.back() != ' ') out += ' ';
                out += part;
            }
            if (to <= from) break;            // no forward progress: bail
            from = to;
        }
        return out;
    }
};

static stt_engine g_stt;

static void serve_transcribe(int fd, const std::string & body) {
    if (!g_stt.ctx) {
        ws::send_http(fd, "503 Service Unavailable", "application/json",
                      json({{"error", "dictation is not available: no --stt-model loaded"}}).dump());
        return;
    }
    std::string b64;
    int rate = 24000;
    try {
        const auto j = json::parse(body);
        b64  = j.value("audio", "");
        rate = j.value("rate", 24000);
    } catch (const std::exception & e) {
        ws::send_http(fd, "400 Bad Request", "application/json",
                      json({{"error", std::string("bad json: ") + e.what()}}).dump());
        return;
    }
    std::vector<uint8_t> bytes;
    if (b64.empty() || !ws::b64_decode(b64, bytes) || bytes.size() < 2 || rate <= 0) {
        ws::send_http(fd, "400 Bad Request", "application/json",
                      json({{"error", "audio must be base64 pcm16 and rate must be positive"}}).dump());
        return;
    }

    std::vector<float> f, pcm16k;
    pcm16_to_float(bytes, f);
    resample_linear(f, rate, GL_MIC_RATE, pcm16k);   // moonshine wants 16 kHz

    int n_chunks = 0;
    const std::string text = g_stt.transcribe(pcm16k, GL_MIC_RATE, &n_chunks);
    ws::send_http(fd, "200 OK", "application/json",
                  json({{"text", text},
                        {"seconds", (double) pcm16k.size() / GL_MIC_RATE},
                        {"chunks", n_chunks}}).dump());
}

// Read the UI page off disk on every request. It is one small file on a
// local dev tool, and re-reading means editing the HTML and hitting reload
// works without rebuilding the binary.
static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}


// POST /api/chat  {"message": "..."}  ->  text/event-stream
//
// One message per request rather than the whole transcript: the model keeps
// the conversation in its KV cache, so re-sending history would decode it
// again every turn and cost more the longer you talk. That is also why this
// is our own shape and not /v1/chat/completions — the OpenAI schema is
// stateless by definition and would invite exactly the wrong thing.
static void serve_chat(int fd, VoiceSession & vs, const std::string & body) {
    std::string msg;
    // "speak": true synthesises the reply and streams it back on the same
    // event stream. A client in voice mode wants a typed aside answered out
    // loud like everything else in that conversation; a plain text client
    // does not, and synthesis is by far the longest part of a turn.
    bool speak = false;
    try {
        const auto j = json::parse(body);
        msg   = j.value("message", "");
        speak = j.value("speak", false);
    } catch (const std::exception & e) {
        ws::send_http(fd, "400 Bad Request", "application/json",
                      json({{"error", std::string("bad json: ") + e.what()}}).dump());
        return;
    }
    if (msg.empty()) {
        ws::send_http(fd, "400 Bad Request", "application/json",
                      json({{"error", "message is empty"}}).dump());
        return;
    }

    const auto t_enter = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_turn_mu);
    const auto t_go = std::chrono::steady_clock::now();

    const char * head =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    if (!ws::send_all(fd, head, strlen(head))) return;

    bool broken = false;
    // on_token arrives on this thread, on_audio on the TTS worker, and with
    // speak=true both are writing to the same socket.
    std::mutex emit_mu;
    auto emit = [&](const json & j) {
        const std::string line = "data: " + j.dump() + "\n\n";
        std::lock_guard<std::mutex> lk(emit_mu);
        if (broken) return;
        if (!ws::send_all(fd, line.data(), line.size())) broken = true;
    };

    // Save and restore rather than clear. These callbacks are shared state:
    // a live voice session installs its own at connect time, and a typed
    // message arriving mid-session must hand them back exactly as it found
    // them — clearing them would leave the socket's spoken replies silent.
    //
    // Restoring is also what keeps this lambda from outliving the frame it
    // captures. Leaving it installed once serve_chat returns points on_audio
    // at a dead stack, and the next turn to produce audio locks a mutex that
    // is no longer a mutex.
    auto prev_token = vs.on_token;
    auto prev_audio = vs.on_audio;
    auto prev_done  = vs.on_done;
    struct restore {
        VoiceSession & vs;
        std::function<void(const char *)> t;
        std::function<void(const float *, size_t)> a;
        std::function<void()> d;
        ~restore() { vs.on_token = t; vs.on_audio = a; vs.on_done = d; }
    } restore_guard{vs, prev_token, prev_audio, prev_done};

    vs.on_token = [&](const char * t) { emit({{"delta", t}}); };
    vs.on_done  = nullptr;
    vs.on_audio = nullptr;
    std::chrono::steady_clock::time_point first_audio{};
    if (speak) {
        vs.on_audio = [&](const float * pcm, size_t n) {
            if (first_audio.time_since_epoch().count() == 0) {
                first_audio = std::chrono::steady_clock::now();
            }
            std::vector<float> src(pcm, pcm + n), rs;
            resample_linear(src, vs.tts_sample_rate(), 24000, rs);
            std::vector<uint8_t> bytes;
            float_to_pcm16(rs.data(), rs.size(), bytes);
            emit({{"audio", ws::b64_encode(bytes.data(), bytes.size())}});
        };
    }

    bool ok = vs.begin_turn(VoiceSession::turn_kind::text) && vs.push_text(msg);
    if (ok) ok = vs.end_turn(speak);

    if (!ok) emit({{"error", vs.last_error()}});
    else {
        const TurnStats & st = vs.last_stats();
        emit({{"done", true},
              {"tokens", st.n_llm_tokens},
              {"ms", (int) (st.ms_llm_gen + 0.5)}});
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const auto now = std::chrono::steady_clock::now();
        log_turn(speak ? "text spoken" : "text", st, 24000,
                 ms(t_enter, t_go), ms(t_enter, now),
                 first_audio.time_since_epoch().count() ? ms(t_go, first_audio) : -1.0);
    }
    // restore_guard puts the previous callbacks back.
}

static void serve_page(int fd, const std::string & ui_path, const std::string & req_path) {
    if (req_path != "/" && req_path != "/index.html") {
        ws::send_http(fd, "404 Not Found", "text/plain", "not found\n");
        return;
    }
    const std::string body = read_file(ui_path);
    if (body.empty()) {
        ws::send_http(fd, "500 Internal Server Error", "text/plain",
                      "cannot read " + ui_path + "\n"
                      "Run gl-serve from the repo root, or pass --rt-ui.\n");
        return;
    }
    ws::send_http(fd, "200 OK", "text/html; charset=utf-8", body);
}

int main(int argc, char ** argv) {
    // Belt and braces with SO_NOSIGPIPE in ws.h: any write path that slips
    // past the socket option still must not take the process down.
    signal(SIGPIPE, SIG_IGN);

    gl_opts O;
    const std::vector<std::string> groups = {"llm", "sys", "mtp", "tts", "dfn", "vad", "stt", "rt"};
    {
        std::string err;
        if (!gl_parse_args(argc, argv, O, &err, groups)) {
            fprintf(stderr, "%s\n\n", err.c_str());
            gl_usage(stderr, argv[0], groups);
            return 2;
        }
    }

    SessionConfig cfg;
    cfg.llm_model_path = O.llm_model;
    cfg.mmproj_path    = O.llm_mmproj;
    cfg.tts_model_path = O.tts_model;
    cfg.tts_voice_path = O.tts_voice;
    cfg.mtp_model_path = O.mtp_model;
    cfg.dfn_model_path = O.dfn_model;
    cfg.enable_mtp     = O.mtp_on;
    cfg.mtp_n_draft    = O.mtp_draft;
    cfg.n_ctx          = O.llm_ctx;
    cfg.n_predict      = O.llm_predict;
    cfg.n_threads      = O.llm_threads;
    cfg.temperature    = O.llm_temp;
    cfg.tts_cfg        = O.tts_cfg;
    cfg.tts_steps      = O.tts_steps;
    cfg.tts_neg_anchor = O.tts_anchor;
    cfg.tts_first_chunk_frames = O.tts_chunk;
    cfg.tts_target_rms = O.tts_rms;
    cfg.verbosity      = O.verbosity;

    if (FILE * pf = fopen(O.sys_prompt.c_str(), "rb")) {
        std::string p;
        char b[4096];
        size_t n;
        while ((n = fread(b, 1, sizeof(b), pf)) > 0) p.append(b, n);
        fclose(pf);
        cfg.system_prompt = p;
    } else if (!O.sys_prompt.empty()) {
        fprintf(stderr, "ERR: no system prompt at %s\n", O.sys_prompt.c_str());
        return 1;
    }

    std::string err;
    auto vs = VoiceSession::create(cfg, &err);
    if (!vs) {
        fprintf(stderr, "ERR: %s\n", err.c_str());
        return 1;
    }

    // What actually got loaded. The session prints its own mtp line from
    // inside create(); everything else the server owns has to say so here,
    // or the only evidence of a wrong --tts-voice or a missing system prompt
    // is the behaviour hours later.
    const bool stt_ok = g_stt.init(O.stt_model, O.stt_threads, O.verbosity);
    if (O.verbosity > 0) {
        fprintf(stderr, "llm      %s\n", O.llm_model.c_str());
        if (cfg.system_prompt.empty()) {
            fprintf(stderr, "sys      none\n");
        } else {
            fprintf(stderr, "sys      %s (%d tokens)\n",
                    O.sys_prompt.c_str(), vs->system_tokens());
        }
        const int tts_rate = vs->tts_sample_rate();
        fprintf(stderr, "tts      %s @ %d Hz%s\n"
                        "         voice %s, cfg %.2f, steps %d, anchor %.2f\n",
                O.tts_model.c_str(), tts_rate,
                tts_rate == GL_TTS_RATE ? "" : " (dfn post-filter active)",
                O.tts_voice.c_str(), cfg.tts_cfg, cfg.tts_steps, cfg.tts_neg_anchor);
        fprintf(stderr, "vad      firered-vad, %d ms silence\n", O.vad_silence);
        if (stt_ok) fprintf(stderr, "stt      %s\n", O.stt_model.c_str());
        else        fprintf(stderr, "stt      DISABLED — could not load %s\n",
                            O.stt_model.empty() ? "(no model set)" : O.stt_model.c_str());
    }

    const int lfd = ws::listen_on(O.rt_host.c_str(), O.rt_port, &err);
    if (lfd < 0) {
        fprintf(stderr, "ERR: %s\n", err.c_str());
        // Overwhelmingly this is another gl-serve still running, which is
        // easy to do and annoying to diagnose from errno alone.
        if (err.find("Address already in use") != std::string::npos) {
            fprintf(stderr,
                "     Something is already listening there. Find it with:\n"
                "       lsof -nP -iTCP:%d -sTCP:LISTEN\n"
                "     then stop it, or use a different --rt-port.\n", O.rt_port);
        }
        return 1;
    }

    if (O.verbosity > 0) {
        fprintf(stderr, "\ngl-serve ready.\n");
        fprintf(stderr, "  http://%s:%d/                 web ui\n", O.rt_host.c_str(), O.rt_port);
        fprintf(stderr, "  ws://%s:%d/v1/realtime        voice session\n", O.rt_host.c_str(), O.rt_port);
        fprintf(stderr, "  http://%s:%d/api/chat         text chat\n", O.rt_host.c_str(), O.rt_port);
        if (g_stt.ctx) {
            fprintf(stderr, "  http://%s:%d/api/transcribe   dictation\n",
                    O.rt_host.c_str(), O.rt_port);
        }
        fprintf(stderr, "  pcm16 mono 24 kHz in and out, server_vad at %d ms\n\n",
                O.vad_silence);
        fprintf(stderr, "Ctrl+C to quit.\n\n");
    }

    // The accept loop must keep running WHILE a session is live, or a second
    // client just sits in the kernel backlog until it times out — it never
    // gets the 1013 telling it to come back later, and a plain HTTP probe
    // never gets its 426 either. So the conversation runs on a worker thread
    // and this loop stays free to answer the door.
    std::atomic<bool> busy{false};
    std::thread       worker;
    for (;;) {
        const int fd = ::accept(lfd, nullptr, nullptr);
        if (fd < 0) { if (errno == EINTR) continue; break; }

        ws::request req;
        std::string herr;
        const ws::hs r = ws::handshake(fd, &req, &herr);
        if (r == ws::hs::plain_http) {
            if (req.method == "OPTIONS") {
                // CORS preflight for a cross-origin /api/chat fetch from a web UI.
                const char * pf =
                    "HTTP/1.1 204 No Content\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Methods: POST, OPTIONS\r\n"
                    "Access-Control-Allow-Headers: content-type\r\n"
                    "Access-Control-Max-Age: 86400\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n";
                ws::send_all(fd, pf, strlen(pf));
            } else if (req.method == "POST" && req.path == "/api/chat") {
                serve_chat(fd, *vs, req.body);
            } else if (req.method == "POST" && req.path == "/api/transcribe") {
                // Deliberately NOT under g_turn_mu: dictation runs a different
                // model and touches no conversation state, so it must not
                // queue behind a spoken reply.
                serve_transcribe(fd, req.body);
            } else {
                serve_page(fd, O.rt_ui, req.path);
            }
            close(fd);
            continue;
        }
        if (r != ws::hs::upgraded) {
            if (O.verbosity > 0) fprintf(stderr, "  [rejected: %s]\n", herr.c_str());
            close(fd);
            continue;
        }
        if (busy.load()) {
            ws::send_close(fd, 1013, "session already in use");
            close(fd);
            if (O.verbosity > 0) fprintf(stderr, "  [turned away: session in use]\n");
            continue;
        }
        if (worker.joinable()) worker.join();   // previous one has already finished
        busy.store(true);
        if (O.verbosity > 0) fprintf(stderr, "  [client connected %s]\n", req.path.c_str());
        worker = std::thread([fd, &vs, &O, &busy] {
            serve_client(fd, *vs, O);
            close(fd);
            busy.store(false);
            if (O.verbosity > 0) fprintf(stderr, "  [client disconnected]\n");
        });
    }
    if (worker.joinable()) worker.join();
    close(lfd);
    return 0;
}
