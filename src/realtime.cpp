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
#include <climits>
#include <cstdio>
#include <sys/stat.h>
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
#include "moonshine.h"
#include "parakeet.h"
#include "kyutai_stt.h"
}

#include "gguf.h"

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
// on speech that a 0.5B TTS produced;
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
    // Only when there was one. An image is a few hundred tokens of prefix the
    // turn waits on with nothing to overlap it, so when ttft looks wrong this
    // is the first number to check.
    char img[32] = "";
    if (st.n_image_tokens > 0) snprintf(img, sizeof(img), " | img %d tok", st.n_image_tokens);

    fprintf(stderr, "[%s | enc %d tok%s | llm %d tok @ %.1f tok/s | ttft %.0f ms%s%s%s%s"
                    " | turn %.0f ms]\n",
            kind, st.n_audio_tokens, img, st.n_llm_tokens, tok_s, st.ms_ttft,
            ttfa, aud, mtp, wait, ms_total);
}

static long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// ── connection ──────────────────────────────────────────────────────────
struct rt_conn {
    int        fd = -1;
    std::mutex send_mu;         // on_audio fires from the TTS worker thread
    std::atomic<bool> dead{false};

    // Set once a session exists, so a write that fails can stop the turn
    // rather than synthesising the rest of a reply nobody will hear.
    std::function<void()> on_dead;

    void send_pcm(const void * data, size_t n) {
        bool just_died = false;
        {
            std::lock_guard<std::mutex> lk(send_mu);
            if (dead.load()) return;
            if (!ws::send_binary(fd, data, n)) { dead.store(true); just_died = true; }
        }
        if (just_died && on_dead) on_dead();
    }

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
    // Last time this session heard SPEECH — not the last packet. Audio arrives
    // continuously for as long as the socket is open, so packets say only that
    // a tab exists, which is exactly the case worth reclaiming.
    std::atomic<long long> last_speech_ms{0};
    // True from the first byte of a reply until the client says it finished
    // playing. Only the client can close that window; the server's own view
    // ends when generation does, which is far too early.
    std::atomic<bool> playing{false};
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
        // Clamped to what is left of the turn's budget. The cap below ends an
        // over-long turn, but it only gets to look after this returns — and a
        // single oversized append would reach the encoder's limit inside this
        // one call. Whatever does not fit stays buffered and becomes the next
        // turn's pre-roll.
        size_t n = buf.size();
        if (pushed_samples + n > MAX_TURN_SAMPLES) n = MAX_TURN_SAMPLES - pushed_samples;
        if (n == 0) return true;
        const bool ok = vs.push_audio(buf.data(), n);
        pushed_samples += n;
        buf.erase(buf.begin(), buf.begin() + n);
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

    // mtmd's gemma4a streaming preprocessor asserts above 30 s in a single
    // stream, and an assert is a crash, not an error the client can see. A
    // turn opens on speech onset and closes on speech offset, so anything
    // that keeps the VAD in "speech" indefinitely — an unbroken monologue,
    // or room noise it has latched on to — walks straight into it. Ended at
    // 28 s instead, which answers what was said rather than losing it.
    static constexpr size_t MAX_TURN_SAMPLES = 28 * (size_t) GL_MIC_RATE;

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

    // Which protocol this connection speaks. The OpenAI surface stays on
    // /v1/realtime; /v1/live is ours. One session, two vocabularies — the
    // turn machinery below does not know the difference.
    bool native = false;

    // Short keys and no envelope: a native event carries what it means and
    // nothing else. The OpenAI events wrap every delta in event_id,
    // response_id, item_id, output_index and content_index, which is five
    // fields of bookkeeping per fragment of a reply nobody correlates.
    void nev(const char * t, json extra = json::object()) {
        json e = {{"t", t}};
        for (auto & kv : extra.items()) e[kv.key()] = kv.value();
        conn.send(e);
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
    // text == nullptr: answer the audio that has been streaming in. Otherwise
    // answer these words instead, on the same socket and the same session, so
    // a typed aside is part of the conversation rather than a separate
    // transport that has to hand the callbacks back afterwards.
    void run_turn(const std::string * text = nullptr) {
        const auto t_enter = std::chrono::steady_clock::now();
        if (!text && buf.empty() && pushed_samples == 0) {
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
        playing.store(true);

        if (native) nev("start");
        else {
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
        }

        // Manual turn detection never saw a speech onset, so the turn opens
        // here and takes the whole encode at once — the old behaviour.
        const double tail_s = text ? 0.0 : (double) buf.size() / GL_MIC_RATE;
        const auto t_enc0 = std::chrono::steady_clock::now();
        if (text) {
            // A text turn discards audio buffered behind it: those samples
            // belong to a question that is no longer being asked.
            discard_turn();
            buf.clear();
            vad.reset();
            turn_lock = std::unique_lock<std::mutex>(g_turn_mu);
            if (!vs.begin_turn(VoiceSession::turn_kind::text) || !vs.push_text(*text)) {
                turn_lock.unlock();
                active.store(false);
                playing.store(false);
                if (native) nev("err", {{"s", vs.last_error()}});
                else conn.send_error("server_error", "session_error", vs.last_error());
                return;
            }
            turn_open = true;
        } else if (!open_turn()) { active.store(false); playing.store(false); return; }
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
        if (!native) {
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
        }

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
            log_turn(text ? (was_cancelled ? "text interrupted" : "text")
                          : (was_cancelled ? "voice interrupted" : "voice"), st, out_rate,
                     ms(t_enter, t_go), ms(t_enter, now),
                     first_audio_at.time_since_epoch().count()
                         ? ms(t_go, first_audio_at) : -1.0);
        }
        if (native) {
            // One event, with the numbers the client would otherwise have to
            // time for itself. Cancelled says whether it was interrupted, so
            // there is no status string to parse.
            nev("end", {{"cancelled", was_cancelled},
                        {"text",   transcript},
                        {"in_tok", st.n_audio_tokens},
                        {"out_tok", st.n_llm_tokens},
                        {"ttft_ms", (int) (st.ms_ttft + 0.5)},
                        {"ms",     (int) (st.ms_llm_gen + 0.5)}});
            return;
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
        if (native) { nev("txt", {{"s", text}}); return; }
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
        // Straight out as bytes. The base64 below costs a third more on the
        // wire and an allocation per chunk, on the path that decides how soon
        // the reply is heard.
        if (native) { conn.send_pcm(bytes.data(), bytes.size()); return; }
        send_event("response.output_audio.delta", {
            {"response_id", resp_id}, {"item_id", item_id},
            {"output_index", 0}, {"content_index", 0},
            {"delta", ws::b64_encode(bytes.data(), bytes.size())}});
    }

    // ── inbound events ──────────────────────────────────────────────────
    void handle(const json & ev) {
        if (native && ev.contains("t")) { handle_native(ev); return; }
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
            push_pcm(bytes.data(), bytes.size());
            return;
        }

        handle_rest(ev, type, eid);
    }

    // Five verbs. Everything the OpenAI surface expresses through
    // session.update, input_audio_buffer.* and response.* that this actually
    // needs, without the parts that describe items and content parts nothing
    // here has.
    void handle_native(const json & ev) {
        const std::string t = ev.value("t", "");
        if (t == "end") {                 // force end of turn
            if (turn_open || !buf.empty()) { commit(); run_turn(); }
            return;
        }
        if (t == "cancel") {
            if (active.load()) { cancelled.store(true); vs.abort_turn(); }
            return;
        }
        if (t == "text") {                // a typed turn, spoken back
            const std::string body = ev.value("s", "");
            if (body.empty()) { nev("err", {{"s", "text: empty"}}); return; }
            run_turn(&body);
            return;
        }
        if (t == "played") {
            vad.reset();          // whatever it half-heard through the reply
            // The client is the only party that knows when the reply stopped
            // being audible: the server finished generating seconds earlier.
            // Without this the turn detector re-arms while she is still
            // talking and hears her as a user.
            playing.store(false);
            return;
        }
        if (t == "barge") {               // heard the user over the reply
            playing.store(false);
            vad.reset();
            if (active.load()) { cancelled.store(true); vs.abort_turn(); }
            return;
        }
        nev("err", {{"s", "unknown command: " + t}});
    }

    // Input audio, however it arrived. A native client hands us the bytes
    // from a binary frame; an OpenAI one hands us the same bytes after a
    // base64 decode. Everything past this point is identical, which is the
    // point — there is one turn machine, not two.
    void push_pcm(const uint8_t * data, size_t n) {
            std::vector<float> f, rs;
            pcm16_to_float(std::vector<uint8_t>(data, data + n), f);
            resample_linear(f, in_rate, GL_MIC_RATE, rs);
            buf.insert(buf.end(), rs.begin(), rs.end());

            // Nothing recorded before a turn opens is ever wanted beyond the
            // prefix padding the API promises, so keep that and drop the
            // rest. Unbounded, a connection that is merely quiet — a client
            // sending digital silence between turns, or zeroing frames during
            // playback — accumulates every second of it and hands the whole
            // lot to the encoder the instant speech starts. Forty seconds of
            // nothing then arrives as a >30 s stream and mtmd asserts, which
            // takes the process with it.
            if (!turn_open) {
                const size_t keep = (size_t) (vad.prefix_pad_ms + 700)
                                  * (size_t) GL_MIC_RATE / 1000;
                if (buf.size() > keep) buf.erase(buf.begin(), buf.end() - keep);
            }
            // Encode as it arrives. Server VAD opens the turn from
            // speech_started below, but a client driving commit by hand
            // (turn_detection: null) has nothing to open it — the whole
            // utterance would then be encoded in one stall inside run_turn,
            // ahead of the first token. Open on the first append instead so
            // both modes ride the same incremental path. Guarded on !active
            // for the same reason speech_started is: never open a turn on top
            // of a response that is still generating.
            if (turn_open)                          feed_model();
            else if (!server_vad && !active.load()) open_turn();

            // Cut a turn loose before the encoder's limit rather than after.
            // Runs whatever the turn-detection mode is: in manual mode the
            // client would otherwise have to notice this itself, and the cost
            // of it not noticing is the process dying.
            if (turn_open && pushed_samples >= MAX_TURN_SAMPLES) {
                if (O.verbosity > 0) {
                    fprintf(stderr, "  [turn capped at %zu s of audio]\n",
                            MAX_TURN_SAMPLES / (size_t) GL_MIC_RATE);
                }
                if (native) nev("eou", {{"capped", true}});
                else send_event("input_audio_buffer.speech_stopped", {
                    {"audio_end_ms", (int) (pushed_samples * 1000 / GL_MIC_RATE)},
                    {"item_id", new_id("item")}});
                vad.reset();
                run_turn();
                return;
            }

            // Deaf while the reply is still audible. Not just to the onset:
            // gating that alone still left speech_stopped to fire and run a
            // turn on her own voice, which is the same bug wearing a hat.
            // The client says "played" when the sound has actually stopped,
            // or "barge" if that really was someone talking over her.
            if (native && playing.load()) return;

            if (server_vad && !active.load()) {
                switch (vad.feed(rs.data(), rs.size())) {
                    case gl_vad::eou::event::speech_started:
                        last_speech_ms.store(now_ms());
                        if (native) nev("speech");
                        else send_event("input_audio_buffer.speech_started", {
                            {"audio_start_ms", (int) (vad.onset_sample * 1000 / GL_MIC_RATE)},
                            {"item_id", new_id("item")}});
                        // Start encoding now, while they are still speaking.
                        open_turn();
                        break;
                    case gl_vad::eou::event::speech_stopped: {
                        if (native) nev("eou");
                        else send_event("input_audio_buffer.speech_stopped", {
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

    void handle_rest(const json & ev, const std::string & type, const std::string & eid) {
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
        // Empty means NOTHING WAS CAPTURED, not "the buffer happens to be
        // drained". Audio is fed to the model as it arrives, so by the time
        // the VAD reports end of speech feed_model has already emptied buf —
        // testing it alone made every healthy turn emit a commit_empty error
        // that clients had to learn to ignore, and buried the real one.
        if (buf.empty() && pushed_samples == 0) {
            conn.send_error("invalid_request_error", "input_audio_buffer_commit_empty",
                            "buffer is empty");
            return;
        }
        // Native mode already said "eou"; committed and item.created describe
        // a conversation-item model it does not have.
        if (native) { vad.reset(); return; }
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
// One item off the socket: either a control event or a block of audio. Audio
// travels as bytes rather than being wrapped in json, so the native path
// never builds a string for the thing it sends twenty times a second.
struct msg {
    json                 ev;
    std::string          pcm;   // non-empty => raw pcm16 from a binary frame
};

struct evq {
    std::mutex              m;
    std::condition_variable cv;
    std::deque<msg>         q;
    bool                    closed = false;

    void push(msg j) {
        { std::lock_guard<std::mutex> lk(m); q.push_back(std::move(j)); }
        cv.notify_one();
    }
    void close() {
        { std::lock_guard<std::mutex> lk(m); closed = true; }
        cv.notify_all();
    }
    bool pop(msg * out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return closed || !q.empty(); });
        if (q.empty()) return false;
        *out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

// Serve one client to completion.
static void serve_client(int fd, VoiceSession & vs, const gl_opts & O, bool native) {
    rt_conn conn;
    conn.fd = fd;

    rt_session S(conn, vs, O);
    S.native = native;
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

    S.last_speech_ms.store(now_ms());
    if (S.native) {
        // Everything a client needs to start, in one message: the rates it
        // must speak, whether images are accepted, and how long a silence
        // ends a turn. The OpenAI surface makes a client discover the first
        // by assumption and the second from a separate HTTP endpoint.
        S.nev("ready", {{"in_rate",  S.in_rate},
                        {"out_rate", S.out_rate},
                        {"vad_ms",   O.vad_silence},
                        {"max_turn_s", (int) (rt_session::MAX_TURN_SAMPLES / GL_MIC_RATE)},
                        {"vision",   O.llm_vision}});
    } else {
        S.send_event("session.created", {{"session", S.session_object()}});
    }

    // How often the reader wakes with nothing to read. That tick is also when
    // the idle check runs, so it has to be well under the idle window — at the
    // old flat 30 s, --web-idle 8 could never fire before 30.
    const int poll_ms = (O.web_idle > 0)
                      ? std::max(250, std::min(30000, O.web_idle * 1000 / 4))
                      : 30000;

    evq        q;
    std::thread reader([&] {
        for (;;) {
            ws::op   o;
            std::string payload;
            const int r = ws::recv_message(fd, &o, &payload, poll_ms);
            if (r < 0) break;   // peer gone; the abort below stops any live turn
            if (r == 0) {                             // recv timed out
                // Reclaim a session nobody is using. Only while no reply is in
                // flight: a long answer is not an idle connection, and cutting
                // one off mid-sentence would be worse than holding the slot.
                if (O.web_idle > 0 && !S.active.load()
                    && now_ms() - S.last_speech_ms.load() > (long long) O.web_idle * 1000) {
                    if (O.verbosity > 0) fprintf(stderr, "  [reclaimed: idle]\n");
                    ws::send_close(fd, 1000, "idle");
                    break;
                }
                continue;
            }
            // Native audio: straight from the frame to the encoder. No JSON
            // to parse, no base64 to decode, no string to allocate — at 20
            // frames a second in a latency-critical path, that is the whole
            // reason this protocol exists.
            if (o == ws::op::binary) {
                if (!S.native) continue;             // OpenAI mode has no binary input
                q.push(msg{{}, std::move(payload)});
                continue;
            }
            if (o != ws::op::text) continue;

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
            if (type == "response.create" || type == "input_audio_buffer.commit") {
                S.last_speech_ms.store(now_ms());   // a hand-driven turn is use
            }

            if ((type == "response.cancel" || type == "conversation.item.truncate")
                && S.active.load()) {
                S.cancelled.store(true);
                vs.abort_turn();
                continue;
            }
            q.push(msg{std::move(ev), {}});
        }
        conn.dead.store(true);
        if (S.active.load()) { S.cancelled.store(true); vs.abort_turn(); }
        q.close();
    });

    msg m;
    while (q.pop(&m)) {
        if (conn.dead.load()) break;
        if (!m.pcm.empty()) {
            S.push_pcm((const uint8_t *) m.pcm.data(), m.pcm.size());
            continue;
        }
        S.handle(m.ev);
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
    // Moonshine ships two architectures and CrispASR has an API for each: the
    // streaming models (sliding-window encoder) and the plain ones. They are
    // not interchangeable — handing moonshine-base to the streaming loader
    // gets "invalid model metadata" — so --stt-model tries one and then the
    // other rather than making the caller know which they have.
    moonshine_streaming_context * sctx = nullptr;   // moonshine, streaming
    moonshine_context           * pctx = nullptr;   // moonshine, plain
    parakeet_context            * kctx = nullptr;   // parakeet TDT/CTC/RNNT
    kyutai_stt_context          * yctx = nullptr;   // kyutai stt
    std::mutex mu;                 // one context, and requests can overlap

    bool loaded() const { return sctx || pctx || kctx || yctx; }
    const char * kind() const {
        return sctx ? "moonshine/streaming" : pctx ? "moonshine"
             : kctx ? "parakeet"            : yctx ? "kyutai" : "none";
    }

    // Which runtime a file belongs to, from the GGUF metadata rather than by
    // trying loaders until one accepts. It is not a hypothetical: handed a
    // kyutai model, parakeet_init_from_file SUCCEEDS and then transcribes
    // nothing — a silent wrong answer, which is the worst kind.
    static std::string arch_of(const std::string & path) {
        gguf_init_params gp{};
        gp.no_alloc = true;
        gp.ctx      = nullptr;
        gguf_context * g = gguf_init_from_file(path.c_str(), gp);
        if (!g) return {};
        const int64_t k = gguf_find_key(g, "general.architecture");
        std::string a = k >= 0 ? gguf_get_val_str(g, k) : "";
        gguf_free(g);
        return a;
    }

    bool init(const std::string & path, int threads, int verbosity) {
        if (path.empty()) return false;
        const std::string arch = arch_of(path);
        if (arch.empty()) {
            fprintf(stderr, "stt: %s is not a readable GGUF\n", path.c_str());
            return false;
        }
        if (arch == "parakeet") {
            auto kp = parakeet_context_default_params();
            kp.n_threads = threads > 0 ? threads : 2;
            kp.verbosity = verbosity >= 2 ? 1 : 0;
            kp.use_gpu   = false;
            kctx = parakeet_init_from_file(path.c_str(), kp);
            return kctx != nullptr;
        }
        if (arch == "kyutai-stt") {
            yctx = kyutai_stt_init_from_file(path.c_str(), kyutai_stt_context_default_params());
            return yctx != nullptr;
        }
        if (arch != "moonshine" && arch != "moonshine_streaming") {
            fprintf(stderr, "stt: no runtime for architecture \"%s\"\n", arch.c_str());
            return false;
        }
        auto params = moonshine_streaming_context_default_params();
        params.n_threads   = threads > 0 ? threads : 2;
        params.verbosity   = verbosity >= 2 ? 1 : 0;
        params.use_gpu     = false;
        params.temperature = 0.0f;     // greedy: dictation wants repeatable
        sctx = moonshine_streaming_init_from_file(path.c_str(), params);
        if (sctx) return true;

        moonshine_init_params pp{};
        pp.model_path     = path.c_str();
        pp.tokenizer_path = nullptr;   // resolved next to the model
        pp.n_threads      = threads > 0 ? threads : 2;
        pctx = moonshine_init_with_params(pp);
        return pctx != nullptr;
    }
    void shutdown() {
        if (sctx) { moonshine_streaming_free(sctx); sctx = nullptr; }
        if (pctx) { moonshine_free(pctx);           pctx = nullptr; }
        if (kctx) { parakeet_free(kctx);            kctx = nullptr; }
        if (yctx) { kyutai_stt_free(yctx);          yctx = nullptr; }
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
        if (n == 0) return {};
        if (sctx) {
            char * t = moonshine_streaming_transcribe(sctx, pcm, (int) n);
            if (!t) return {};
            std::string out(t);
            free(t);            // streaming API hands over ownership
            return out;
        }
        if (pctx) {
            // The plain API returns a pointer it owns; copy, do not free.
            const char * t = moonshine_transcribe(pctx, pcm, (int) n);
            return t ? std::string(t) : std::string();
        }
        if (kctx) {
            char * t = parakeet_transcribe(kctx, pcm, (int) n);
            if (!t) return {};
            std::string out(t); free(t); return out;
        }
        if (yctx) {
            char * t = kyutai_stt_transcribe(yctx, pcm, (int) n);
            if (!t) return {};
            std::string out(t); free(t); return out;
        }
        return {};
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
        if (!loaded() || pcm.empty()) return {};

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
    if (!g_stt.loaded()) {
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
    // Each entry is one encoded image (png, jpeg, ...) as base64, with or
    // without a "data:image/...;base64," prefix — the browser hands us data
    // URLs and stripping it here beats making every client do it.
    std::vector<std::vector<uint8_t>> images;
    try {
        const auto j = json::parse(body);
        msg   = j.value("message", "");
        speak = j.value("speak", false);
        if (j.contains("images") && j["images"].is_array()) {
            for (const auto & item : j["images"]) {
                if (!item.is_string()) continue;
                std::string b64 = item.get<std::string>();
                const auto comma = b64.find(',');
                if (b64.rfind("data:", 0) == 0 && comma != std::string::npos) {
                    b64 = b64.substr(comma + 1);
                }
                std::vector<uint8_t> bytes;
                if (!ws::b64_decode(b64, bytes) || bytes.empty()) {
                    ws::send_http(fd, "400 Bad Request", "application/json",
                                  json({{"error", "images must be base64"}}).dump());
                    return;
                }
                images.push_back(std::move(bytes));
            }
        }
    } catch (const std::exception & e) {
        ws::send_http(fd, "400 Bad Request", "application/json",
                      json({{"error", std::string("bad json: ") + e.what()}}).dump());
        return;
    }
    // An image on its own is a message. Give it the question the user would
    // otherwise have had to type, rather than sending a turn with a picture
    // and no instruction and hoping the model volunteers something.
    if (msg.empty() && !images.empty()) msg = "What is in this image?";
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
        if (!ws::send_all(fd, line.data(), line.size())) {
            broken = true;
            // The client hung up — pressed stop, closed the tab, or lost the
            // socket. Nobody will read the rest of this, and generating it
            // anyway holds the turn lock and the TTS stream that the next
            // request is waiting on. Safe from either caller: this runs on the
            // generating thread as on_token and on the TTS worker as
            // on_audio, and abort_turn only trips an atomic and the TTS
            // stream's own abort.
            vs.abort_turn();
        }
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

    // Images first, then the words about them: the model reads the turn in
    // order, and "what is in this picture" only means anything once the
    // picture is already in the context.
    const bool opened = vs.begin_turn(VoiceSession::turn_kind::text);
    bool ok = opened;
    for (const auto & img : images) {
        if (!ok) break;
        ok = vs.push_image(img.data(), img.size());
    }
    if (ok) ok = vs.push_text(msg);

    if (ok) {
        ok = vs.end_turn(speak);
        if (!ok) emit({{"error", vs.last_error()}});
    } else {
        // A push that fails leaves the turn OPEN, with its half-decoded prefix
        // sitting in the KV cache for the next request to build on top of. An
        // undecodable upload is an ordinary thing for a client to send, so
        // unwind it the way the socket unwinds a discarded utterance. Read the
        // error first: end_turn overwrites it.
        const std::string why = vs.last_error();
        if (opened) { vs.abort_turn(); vs.end_turn(/*speak=*/ false); }
        emit({{"error", why}});
    }

    if (ok) {
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

// Content type from the extension. Only what a page here plausibly loads —
// an unknown type is served as octet-stream rather than guessed at, because
// guessing text/html for something that is not is how a stored file becomes
// a script.
static const char * mime_for(const std::string & path) {
    const auto dot = path.rfind('.');
    if (dot == std::string::npos) return "application/octet-stream";
    const std::string e = path.substr(dot + 1);
    if (e == "html" || e == "htm") return "text/html; charset=utf-8";
    if (e == "css")   return "text/css; charset=utf-8";
    if (e == "js" || e == "mjs") return "text/javascript; charset=utf-8";
    if (e == "json" || e == "map") return "application/json; charset=utf-8";
    if (e == "svg")   return "image/svg+xml";
    if (e == "png")   return "image/png";
    if (e == "jpg" || e == "jpeg") return "image/jpeg";
    if (e == "gif")   return "image/gif";
    if (e == "webp")  return "image/webp";
    if (e == "ico")   return "image/x-icon";
    if (e == "woff2") return "font/woff2";
    if (e == "woff")  return "font/woff";
    if (e == "wasm")  return "application/wasm";
    if (e == "txt")   return "text/plain; charset=utf-8";
    if (e == "wav")   return "audio/wav";
    return "application/octet-stream";
}

// The document root, which may be a directory or a single file.
//
// A file is the whole site: it is served at / and nothing else exists. That
// is what web/index.html is — everything inlined, so there is one thing to
// edit and no build step. A directory serves its tree, for when a page has
// grown past that.
//
// Escaping the root is the thing to get right. The path is resolved and the
// result must still be inside the root: a check on the string alone is not
// enough, because a symlink inside the root can point anywhere and "%2e%2e"
// is not ".." until it is.
static void serve_page(int fd, const std::string & root, const std::string & req_path) {
    struct stat rst {};
    const bool root_is_dir = (stat(root.c_str(), &rst) == 0) && S_ISDIR(rst.st_mode);

    if (!root_is_dir) {
        if (req_path != "/" && req_path != "/index.html") {
            ws::send_http(fd, "404 Not Found", "text/plain", "not found\n");
            return;
        }
        const std::string body = read_file(root);
        if (body.empty()) {
            ws::send_http(fd, "500 Internal Server Error", "text/plain",
                          "cannot read " + root + "\n"
                          "Run gl-serve from the repo root, or pass --web-root.\n");
            return;
        }
        ws::send_http(fd, "200 OK", "text/html; charset=utf-8", body);
        return;
    }

    std::string rel = (req_path == "/") ? "/index.html" : req_path;
    if (rel.find('\0') != std::string::npos) {
        ws::send_http(fd, "400 Bad Request", "text/plain", "bad path\n");
        return;
    }

    char real_root[PATH_MAX], real_file[PATH_MAX];
    if (!realpath(root.c_str(), real_root)) {
        ws::send_http(fd, "500 Internal Server Error", "text/plain",
                      "cannot resolve " + root + "\n");
        return;
    }
    const std::string want = std::string(real_root) + rel;
    if (!realpath(want.c_str(), real_file)) {
        ws::send_http(fd, "404 Not Found", "text/plain", "not found\n");
        return;
    }

    // Inside the root, and a regular file. The trailing slash matters: it
    // stops "/srv/wwwroot-secrets" passing a prefix test against "/srv/www".
    std::string base = real_root;
    if (base.empty() || base.back() != '/') base += '/';
    if (std::string(real_file).compare(0, base.size(), base) != 0) {
        ws::send_http(fd, "403 Forbidden", "text/plain", "outside the web root\n");
        return;
    }
    struct stat fst {};
    if (stat(real_file, &fst) != 0 || !S_ISREG(fst.st_mode)) {
        ws::send_http(fd, "404 Not Found", "text/plain", "not found\n");
        return;
    }

    const std::string body = read_file(real_file);
    ws::send_http(fd, "200 OK", mime_for(real_file), body);
}

int main(int argc, char ** argv) {
    // Belt and braces with SO_NOSIGPIPE in ws.h: any write path that slips
    // past the socket option still must not take the process down.
    signal(SIGPIPE, SIG_IGN);

    gl_opts O;
    // The desktop app is voice-only and its prompt says so on every line.
    // Here the same session answers typed turns as well, so it needs a
    // prompt that tells the two apart. --sys-prompt still overrides.
    O.sys_prompt = "prompts/chat.txt";
    // Vision on by default here, unlike the CLI: this is the binary an image
    // can actually reach, through /api/chat or the page's attach button.
    // --llm-vision-off gives back the ~215 MiB of weights and the ~101 MiB
    // Metal compute buffer for a text- and speech-only server.
    O.llm_vision = true;
    const std::vector<std::string> groups = {"llm", "sys", "mtp", "tts", "vad", "stt", "web"};
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
    cfg.vision         = O.llm_vision;
    cfg.tts_model_path = O.tts_model;
    cfg.tts_voice_path = O.tts_voice;
    cfg.mtp_model_path = O.mtp_model;
    cfg.enable_mtp     = O.mtp_on;
    cfg.mtp_n_draft    = O.mtp_draft;
    cfg.n_ctx          = O.llm_ctx;
    cfg.n_predict      = O.llm_predict;
    cfg.n_threads      = O.llm_threads;
    cfg.temperature    = O.llm_temp;
    cfg.tts_cfg        = O.tts_cfg;
    cfg.tts_steps      = O.tts_steps;
    cfg.tts_gpu        = !O.tts_cpu;
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
        fprintf(stderr, "tts      %s @ %d Hz\n"
                        "         voice %s, cfg %.2f, steps %d, anchor %.2f\n",
                O.tts_model.c_str(), tts_rate,
                O.tts_voice.c_str(), cfg.tts_cfg, cfg.tts_steps, cfg.tts_neg_anchor);
        fprintf(stderr, "vad      firered-vad, %d ms silence\n", O.vad_silence);
        if (stt_ok) fprintf(stderr, "stt      %s (%s)\n", O.stt_model.c_str(), g_stt.kind());
        else        fprintf(stderr, "stt      DISABLED — could not load %s\n",
                            O.stt_model.empty() ? "(no model set)" : O.stt_model.c_str());
    }

    const int lfd = ws::listen_on(O.web_host.c_str(), O.web_port, &err);
    if (lfd < 0) {
        fprintf(stderr, "ERR: %s\n", err.c_str());
        // Overwhelmingly this is another gl-serve still running, which is
        // easy to do and annoying to diagnose from errno alone.
        if (err.find("Address already in use") != std::string::npos) {
            fprintf(stderr,
                "     Something is already listening there. Find it with:\n"
                "       lsof -nP -iTCP:%d -sTCP:LISTEN\n"
                "     then stop it, or use a different --web-port.\n", O.web_port);
        }
        return 1;
    }

    if (O.verbosity > 0) {
        fprintf(stderr, "\ngl-serve ready.\n");
        fprintf(stderr, "  http://%s:%d/                 web ui\n", O.web_host.c_str(), O.web_port);
        fprintf(stderr, "  ws://%s:%d/v1/realtime        voice session\n", O.web_host.c_str(), O.web_port);
        fprintf(stderr, "  http://%s:%d/api/chat         text chat\n", O.web_host.c_str(), O.web_port);
        if (g_stt.loaded()) {
            fprintf(stderr, "  http://%s:%d/api/transcribe   dictation\n",
                    O.web_host.c_str(), O.web_port);
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
        // The token guards every route, not just the socket: locking the
        // front door and leaving /api/chat open would be no lock at all.
        const ws::hs r = ws::handshake(fd, &req, &herr, O.web_token);
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
            } else if (req.method == "GET" && req.path == "/api/config") {
                // What this server can actually do, so the page stops
                // offering what it cannot. Without this the UI showed an
                // attach button on a build with no vision tower, and the
                // only way to find out was the error that came back.
                ws::send_http(fd, "200 OK", "application/json",
                              json({{"vision", O.llm_vision}}).dump());
            } else if (req.method == "POST" && req.path == "/api/chat") {
                serve_chat(fd, *vs, req.body);
            } else if (req.method == "POST" && req.path == "/api/transcribe") {
                // Deliberately NOT under g_turn_mu: dictation runs a different
                // model and touches no conversation state, so it must not
                // queue behind a spoken reply.
                serve_transcribe(fd, req.body);
            } else {
                serve_page(fd, O.web_root, req.path);
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
        const bool native = (req.path == "/v1/live");
        worker = std::thread([fd, &vs, &O, &busy, native] {
            serve_client(fd, *vs, O, native);
            close(fd);
            busy.store(false);
            if (O.verbosity > 0) fprintf(stderr, "  [client disconnected]\n");
        });
    }
    if (worker.joinable()) worker.join();
    close(lfd);
    return 0;
}
