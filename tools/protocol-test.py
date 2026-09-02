#!/usr/bin/env python3
"""End-to-end check of the /api/live wire protocol against a running gl-serve.

The unit tests under tests/ are pure logic and never open a socket, so
nothing else notices when an event stops being emitted, an audio frame
changes shape, or a turn stops ending.

    ./build/gl-serve &
    tools/protocol-test.py                    # default port 8927
    tools/protocol-test.py --port 8928 --token s3cret

Synthesises its own speech-like audio, so it needs no fixtures — but it
does need the models loaded, which is why this is not a ctest target.
"""
import argparse, base64, json, math, os, socket, struct, sys, time

RATE  = 24000
FRAME = RATE // 20            # 50 ms, the cadence a browser sends at
fails = []

def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)

class Sock:
    def __init__(self, host, port, path, token):
        if token:
            path += ("&" if "?" in path else "?") + "t=" + token
        self.s = socket.create_connection((host, port), timeout=180)
        k = base64.b64encode(os.urandom(16)).decode()
        self.s.send((f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
                     "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                     f"Sec-WebSocket-Key: {k}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        # Read only as far as the end of the HTTP header. The server sends its
        # greeting immediately after, and a plain recv() swallows that frame
        # along with the header — which looks exactly like a server that
        # never greets.
        head = b""
        self.s.settimeout(10)
        while b"\r\n\r\n" not in head:
            d = self.s.recv(4096)
            if not d:
                break
            head += d
        cut = head.find(b"\r\n\r\n") + 4
        self.status = head.split(b"\r\n")[0].decode()
        self.buf = head[cut:]          # anything past the header is already a frame

    def frame(self, op, payload):
        m = os.urandom(4); h = bytes([0x80 | op]); n = len(payload)
        if n < 126:     h += bytes([0x80 | n])
        elif n < 65536: h += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:           h += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.s.sendall(h + m + bytes(c ^ m[i % 4] for i, c in enumerate(payload)))

    def text(self, obj):   self.frame(0x1, json.dumps(obj).encode())
    def binary(self, pcm): self.frame(0x2, pcm)

    def collect(self, seconds, until=None):
        """Read for a while. Returns (json events, bytes of binary audio)."""
        evs, audio = [], 0
        self.s.settimeout(seconds)
        t0 = time.time()
        try:
            while time.time() - t0 < seconds:
                d = self.s.recv(65536)
                if not d:
                    break
                self.buf += d
                while len(self.buf) >= 2:
                    op = self.buf[0] & 0x0F; ln = self.buf[1] & 0x7F; hdr = 2
                    if ln == 126:   ln = struct.unpack(">H", self.buf[2:4])[0]; hdr = 4
                    elif ln == 127: ln = struct.unpack(">Q", self.buf[2:10])[0]; hdr = 10
                    if len(self.buf) < hdr + ln:
                        break
                    body, self.buf = self.buf[hdr:hdr+ln], self.buf[hdr+ln:]
                    if op == 0x2:
                        audio += len(body)
                    elif op == 0x1:
                        j = json.loads(body.decode())
                        evs.append(j)
                        if until and (j.get("t") == until or j.get("type") == until):
                            raise StopIteration
                    elif op == 0x8:
                        evs.append({"t": "__close"})
                        raise StopIteration
        except (socket.timeout, StopIteration):
            pass
        return evs, audio

    def close(self):
        try: self.s.close()
        except OSError: pass

def speech(seconds, t0):
    """Something the VAD will call speech, and then stop calling speech.

    A single voiced tone. A richer harmonic stack reads as speech too, but
    the detector then stays latched through the trailing silence and never
    reports end of utterance — which looks exactly like a server that has
    stopped emitting events. Keep this simple.
    """
    out = bytearray()
    for i in range(int(seconds * RATE)):
        v = math.sin(2 * math.pi * 150 * (t0 + i / RATE)) * 0.6
        out += struct.pack("<h", int(v * 11000))
    return bytes(out)

SILENCE = b"\x00" * (FRAME * 2)

def say(sock, seconds, t0):
    """A whole utterance: speech, then enough silence to end the turn."""
    sent = 0.0
    while sent < seconds:
        sock.binary(speech(0.05, t0 + sent))
        sent += 0.05
        time.sleep(0.004)
    for _ in range(30):
        sock.binary(SILENCE)
        time.sleep(0.004)
    return t0 + sent

def types(evs):
    return [e.get("t") for e in evs]

def test_live(host, port, token):
    print("\n/api/live")
    s = Sock(host, port, "/api/live", token)
    ok("handshake upgrades", "101" in s.status, s.status)
    hello, _ = s.collect(3)
    ready = next((e for e in hello if e.get("t") == "ready"), None)
    ok("greets with ready", ready is not None)
    if ready:
        ok("ready states the rates", ready.get("in_rate") and ready.get("out_rate"),
           f"in={ready.get('in_rate')} out={ready.get('out_rate')}")
        ok("ready states the turn cap", bool(ready.get("max_turn_s")),
           f"{ready.get('max_turn_s')} s")

    t0 = say(s, 4.0, 0.0)
    evs, bin_audio = s.collect(45, until="end")
    t = types(evs)
    for want in ("speech", "eou", "start", "txt", "end"):
        ok(f"emits {want}", want in t)
    ok("reply audio arrives as binary frames", bin_audio > 0, f"{bin_audio} bytes")
    end = next((e for e in evs if e.get("t") == "end"), {})
    ok("end carries the numbers",
       "out_tok" in end and "ttft_ms" in end and "text" in end,
       f"out_tok={end.get('out_tok')} ttft={end.get('ttft_ms')}ms")

    # The turn detector must stay shut until the client says playback ended.
    print("  -- the played gate --")
    t0 = say(s, 4.0, t0)
    during, _ = s.collect(6)
    ok("no turn opens while the reply is still audible", not types(during),
       ",".join(types(during)))
    s.text({"t": "played"}); time.sleep(0.2)
    t0 = say(s, 4.0, t0)
    after, _ = s.collect(45, until="end")
    ok("a turn opens once played is sent", "end" in types(after))

    # barge overrides the gate mid-reply.
    print("  -- barge --")
    t0 = say(s, 2.0, t0)
    s.text({"t": "barge"}); time.sleep(0.2)
    t0 = say(s, 4.0, t0)
    barged, _ = s.collect(45, until="end")
    ok("barge lets an interruption through", "end" in types(barged))
    s.text({"t": "played"}); time.sleep(0.3)

    # typed turns on the same socket.
    print("  -- the text verb --")
    s.text({"t": "text", "s": "Say hello in three words."})
    tevs, taudio = s.collect(60, until="end")
    tt = types(tevs)
    ok("a typed turn replies", "end" in tt and "txt" in tt)
    ok("and is spoken back", taudio > 0, f"{taudio} bytes")
    s.text({"t": "played"}); time.sleep(0.2)
    s.text({"t": "text", "s": ""})
    eevs, _ = s.collect(5, until="err")
    ok("empty text is refused", "err" in types(eevs))
    s.close()

    # There is one endpoint. Anything else upgrades and is then closed,
    # rather than being handed a session it cannot drive.
    print("  -- routing --")
    time.sleep(0.5)
    o = Sock(host, port, "/v1/live", token)
    gone, _ = o.collect(3)
    ok("an unknown endpoint gets no session", not types(gone),
       ",".join(str(x) for x in types(gone)))
    o.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8927)
    ap.add_argument("--token", default="")
    a = ap.parse_args()
    print(f"gl-serve at {a.host}:{a.port}")
    test_live(a.host, a.port, a.token)
    print()
    if fails:
        print(f"FAILED: {len(fails)} — " + ", ".join(fails))
        return 1
    print("all checks passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
