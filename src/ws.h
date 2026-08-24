// Minimal RFC 6455 WebSocket server — just enough to carry the Realtime
// event stream, with no new third-party dependency.
//
// Scope is deliberate: one client at a time (VoiceSession owns a single
// llama context and is not thread-safe, so a second concurrent session
// could not be served anyway), text and binary data frames, continuation
// frames, and the ping/pong/close control frames. No permessage-deflate,
// no extensions, no client role.
//
// The socket is blocking. recv_message() takes a timeout so the caller can
// interleave "is there a client event?" with "is there TTS audio to flush?"
// on one thread without a second thread or a poll abstraction.
#pragma once

#include <CommonCrypto/CommonDigest.h>   // CC_SHA1 — libSystem, no link flag
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace ws {

// ── base64 ──────────────────────────────────────────────────────────────
// Needed for the handshake accept token and, far more heavily, for every
// audio delta the Realtime protocol carries as base64 PCM16.
inline std::string b64_encode(const void * data, size_t n) {
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto * p = (const unsigned char *) data;
    std::string out;
    out.reserve(((n + 2) / 3) * 4);
    for (size_t i = 0; i < n; i += 3) {
        const unsigned v = (unsigned) p[i] << 16
                         | (unsigned) (i + 1 < n ? p[i + 1] : 0) << 8
                         | (unsigned) (i + 2 < n ? p[i + 2] : 0);
        out += T[(v >> 18) & 63];
        out += T[(v >> 12) & 63];
        out += (i + 1 < n) ? T[(v >> 6) & 63] : '=';
        out += (i + 2 < n) ? T[v & 63]        : '=';
    }
    return out;
}

inline bool b64_decode(const std::string & in, std::vector<uint8_t> & out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;                        // '=' and whitespace handled below
    };
    out.clear();
    out.reserve(in.size() * 3 / 4);
    int acc = 0, bits = 0;
    for (char c : in) {
        if (c == '=' ) break;
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int v = val(c);
        if (v < 0) return false;          // reject rather than silently skip
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((acc >> bits) & 0xFF));
        }
    }
    return true;
}

// ── opcodes ─────────────────────────────────────────────────────────────
enum class op : uint8_t { cont = 0x0, text = 0x1, binary = 0x2,
                          close = 0x8, ping = 0x9, pong = 0xA };

// ── listening socket ────────────────────────────────────────────────────
inline int listen_on(const char * host, int port, std::string * err) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { *err = "socket: " + std::string(strerror(errno)); return -1; }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) {
        *err = std::string("bad host: ") + host; close(fd); return -1;
    }
    if (bind(fd, (sockaddr *) &a, sizeof(a)) != 0) {
        *err = "bind " + std::string(host) + ":" + std::to_string(port) + ": " + strerror(errno);
        close(fd); return -1;
    }
    if (listen(fd, 4) != 0) {
        *err = "listen: " + std::string(strerror(errno)); close(fd); return -1;
    }
    return fd;
}

// ── framing ─────────────────────────────────────────────────────────────
inline bool send_all(int fd, const void * data, size_t n) {
    const auto * p = (const uint8_t *) data;
    while (n) {
        const ssize_t k = ::send(fd, p, n, 0);
        if (k <= 0) { if (errno == EINTR) continue; return false; }
        p += k; n -= (size_t) k;
    }
    return true;
}

// Server-to-client frames are never masked (RFC 6455 §5.1).
inline bool send_frame(int fd, op o, const void * data, size_t n) {
    uint8_t h[10];
    size_t  hn = 0;
    h[hn++] = (uint8_t) (0x80 | (uint8_t) o);          // FIN + opcode
    if (n < 126)             { h[hn++] = (uint8_t) n; }
    else if (n <= 0xFFFF)    { h[hn++] = 126;
                               h[hn++] = (uint8_t) (n >> 8);  h[hn++] = (uint8_t) n; }
    else                     { h[hn++] = 127;
                               for (int i = 7; i >= 0; i--) h[hn++] = (uint8_t) (n >> (8 * i)); }
    if (!send_all(fd, h, hn)) return false;
    return n == 0 || send_all(fd, data, n);
}

inline bool send_text(int fd, const std::string & s) {
    return send_frame(fd, op::text, s.data(), s.size());
}

inline void send_close(int fd, uint16_t code, const std::string & reason) {
    std::string p;
    p += (char) (code >> 8);
    p += (char) (code & 0xFF);
    p += reason;
    send_frame(fd, op::close, p.data(), p.size());
}

// Blocking read of exactly n bytes, with a deadline. Returns:
//   1 ok, 0 timed out (nothing consumed), -1 closed or error.
inline int read_exact(int fd, void * dst, size_t n, int timeout_ms) {
    auto * p = (uint8_t *) dst;
    size_t got = 0;
    while (got < n) {
        pollfd pf{fd, POLLIN, 0};
        const int pr = ::poll(&pf, 1, got == 0 ? timeout_ms : 5000);
        if (pr == 0)  return got == 0 ? 0 : -1;   // mid-frame stall is fatal
        if (pr < 0)   { if (errno == EINTR) continue; return -1; }
        const ssize_t k = ::recv(fd, p + got, n - got, 0);
        if (k <= 0)   { if (k < 0 && errno == EINTR) continue; return -1; }
        got += (size_t) k;
    }
    return 1;
}

// Read one complete application message, reassembling continuation frames
// and answering ping/close inline. *o is text or binary.
// Returns 1 message, 0 timeout, -1 peer closed or protocol error.
inline int recv_message(int fd, op * o, std::string * payload, int timeout_ms,
                        size_t max_bytes = 64u << 20) {
    payload->clear();
    bool started = false;
    for (;;) {
        uint8_t h[2];
        const int r = read_exact(fd, h, 2, started ? 5000 : timeout_ms);
        if (r <= 0) return r;
        const bool    fin    = (h[0] & 0x80) != 0;
        const auto    opcode = (op) (h[0] & 0x0F);
        const bool    masked = (h[1] & 0x80) != 0;
        uint64_t      len    = h[1] & 0x7F;
        if (len == 126) {
            uint8_t e[2];
            if (read_exact(fd, e, 2, 5000) != 1) return -1;
            len = ((uint64_t) e[0] << 8) | e[1];
        } else if (len == 127) {
            uint8_t e[8];
            if (read_exact(fd, e, 8, 5000) != 1) return -1;
            len = 0;
            for (uint8_t b : e) len = (len << 8) | b;
        }
        // A client that omits the mask is violating the spec; a huge length
        // is either a bug or an attempt to make us allocate the heap away.
        if (!masked) return -1;
        if (len > max_bytes || payload->size() + len > max_bytes) return -1;

        uint8_t key[4];
        if (read_exact(fd, key, 4, 5000) != 1) return -1;

        std::string body;
        body.resize((size_t) len);
        if (len && read_exact(fd, &body[0], (size_t) len, 5000) != 1) return -1;
        for (size_t i = 0; i < body.size(); i++) body[i] = (char) (body[i] ^ key[i & 3]);

        switch (opcode) {
            case op::ping:  send_frame(fd, op::pong, body.data(), body.size()); continue;
            case op::pong:  continue;
            case op::close: send_frame(fd, op::close, body.data(), body.size()); return -1;
            case op::cont:
                if (!started) return -1;      // continuation with nothing to continue
                *payload += body;
                break;
            case op::text:
            case op::binary:
                if (started) return -1;       // new data frame mid-message
                started  = true;
                *o       = opcode;
                *payload = body;
                break;
            default: return -1;
        }
        if (fin && started) return 1;
    }
}

// Send a complete plain HTTP response and be done with the connection.
inline void send_http(int fd, const char * status, const char * content_type,
                      const std::string & body) {
    char head[256];
    const int n = snprintf(head, sizeof(head),
        "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        status, content_type, body.size());
    if (send_all(fd, head, (size_t) n)) send_all(fd, body.data(), body.size());
}

// ── handshake ───────────────────────────────────────────────────────────
// Reads the client's request. `upgraded` means the 101 has been sent and the
// connection is now a WebSocket. `plain_http` means it is an ordinary GET —
// the caller decides what to serve, which is how one port can hand out the
// web UI and carry the audio session.
enum class hs { upgraded, plain_http, failed };

// What a plain HTTP request turned out to be, when handshake() reports one.
struct request {
    std::string method;
    std::string path;
    std::string body;
};

inline hs handshake(int fd, request * out, std::string * err) {
    std::string req;
    char buf[1024];
    while (req.find("\r\n\r\n") == std::string::npos) {
        if (req.size() > 32768) { *err = "header too large"; return hs::failed; }
        pollfd pf{fd, POLLIN, 0};
        if (::poll(&pf, 1, 5000) <= 0) { *err = "handshake timeout"; return hs::failed; }
        const ssize_t k = ::recv(fd, buf, sizeof(buf), 0);
        if (k <= 0) { *err = "peer closed during handshake"; return hs::failed; }
        req.append(buf, (size_t) k);
    }

    // Header names are case-insensitive; fold a copy for matching and keep
    // the original for the one value we echo back.
    std::string low = req;
    for (char & c : low) c = (char) tolower((unsigned char) c);
    auto value_of = [&](const char * lname) -> std::string {
        const size_t at = low.find(lname);
        if (at == std::string::npos) return {};
        size_t s = low.find(':', at);
        if (s == std::string::npos) return {};
        s++;
        const size_t e = low.find("\r\n", s);
        std::string v = req.substr(s, e - s);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
        while (!v.empty() && (v.back()  == ' ' || v.back()  == '\t' || v.back() == '\r')) v.pop_back();
        return v;
    };

    if (out) {
        const size_t sp1 = req.find(' ');
        const size_t sp2 = (sp1 == std::string::npos) ? std::string::npos : req.find(' ', sp1 + 1);
        if (sp1 != std::string::npos && sp2 != std::string::npos) {
            out->method = req.substr(0, sp1);
            out->path   = req.substr(sp1 + 1, sp2 - sp1 - 1);
        }
    }

    const std::string key = value_of("sec-websocket-key");
    if (low.find("upgrade: websocket") == std::string::npos || key.empty()) {
        // Ordinary request. Pull in the body too, so a POST arrives whole
        // and the caller never has to touch the socket to finish reading it.
        if (out) {
            const size_t hdr_end = req.find("\r\n\r\n") + 4;
            out->body = req.substr(hdr_end);
            const std::string cl = value_of("content-length");
            const size_t want = cl.empty() ? 0 : (size_t) strtoul(cl.c_str(), nullptr, 10);
            if (want > (16u << 20)) { *err = "body too large"; return hs::failed; }
            while (out->body.size() < want) {
                pollfd pf{fd, POLLIN, 0};
                if (::poll(&pf, 1, 5000) <= 0) { *err = "body timeout"; return hs::failed; }
                const ssize_t k = ::recv(fd, buf, sizeof(buf), 0);
                if (k <= 0) { *err = "peer closed reading body"; return hs::failed; }
                out->body.append(buf, (size_t) k);
            }
        }
        return hs::plain_http;      // caller serves it
    }

    static const char MAGIC[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    const std::string concat = key + MAGIC;
    unsigned char digest[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1(concat.data(), (CC_LONG) concat.size(), digest);

    const std::string accept = b64_encode(digest, sizeof(digest));
    char resp[512];
    const int n = snprintf(resp, sizeof(resp),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n\r\n", accept.c_str());
    if (!send_all(fd, resp, (size_t) n)) { *err = "handshake write failed"; return hs::failed; }

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // audio deltas are latency-critical
    // Without this, writing to a peer that vanished raises SIGPIPE and the
    // default disposition kills the process — a client pressing Ctrl+C
    // mid-reply would take the server down with it. send() returns EPIPE
    // instead, which the framing layer reports as a dead connection.
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
    return hs::upgraded;
}

} // namespace ws
