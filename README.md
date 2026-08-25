# gemma-live

A hands-free local voice assistant. Say "hey gemma", talk, and it answers out
loud — wake word, speech in, speech out, and barge-in, all on-device.

Everything runs locally: Gemma 4 E4B for the reply, its audio encoder for
understanding speech (no separate ASR in the reply path), VibeVoice for the
voice, LocalVQE for echo cancellation.

Apple Silicon / macOS only today.

## What a turn looks like

```
[idle — say "hey gemma" to start]
  [↩ wake]
[listening — pause ~500 ms to send]
[gemma] The capital of France is Paris, and it is famous for its art and history.
[enc 60 tok | llm 17 tok | ttft 89 ms | ttfa 331 ms | tts 4.7 s | rtf 0.42]
```

Once it has replied, the conversation stays open for 5 seconds — keep talking
and it answers again without the wake word. Talk over a reply and it stops.

## Setup

Three dependencies. Two are hand-cloned rather than submodules because they are
~2 GB together, and both need a specific branch:

```bash
git clone --branch gemma-live https://github.com/tbtlr/llama.cpp.git vendor/llama.cpp
git clone --branch gemma-live https://github.com/tbtlr/CrispASR.git vendor/CrispASR
git submodule update --init --recursive vendor/localvqe
```

Upstream will not work for either fork. The streaming `mtmd_audio_stream_*` API
and the `load_vision`/`load_audio` params live only on that llama.cpp branch;
the streaming VibeVoice API, the DFN post-filter and the `crispasr-tts` target
live only on that CrispASR branch.

Build the two out-of-tree pieces, fetch the weights (~5.2 GB), then build:

```bash
cd vendor/CrispASR && cmake -B build -S . && cmake --build build --target crispasr-tts && cd ../..
tools/build_localvqe.sh
tools/fetch-models.sh          # --list to see what it would download
cmake -S . -B build && cmake --build build --target gemma-live
./build/gemma-live
```

## How it fits together

Two halves, deliberately kept apart:

- **`src/session.{h,cpp}`** — the model half. Audio in via `push_audio`, tokens
  and synthesised audio out via callbacks. No audio device, no terminal. Owns
  the LLM, the mtmd audio encoder, MTP speculative decoding and the TTS stream.
- **`src/main.cpp`** — the app. miniaudio capture and playback, the AEC chain,
  three detectors, and the IDLE → LISTENING → REPLYING → AWAITING_FOLLOWUP
  state machine.

Plus three header-only pieces: `vqe.h` (LocalVQE wrapper), `barge.h`
(double-talk barge-in), `transcript.h` (rolling-ASR text helpers).

Three detectors, each owning one phase:

| | when | what it watches |
|---|---|---|
| moonshine | IDLE | rolling transcript, matched against `keywords/wake.txt` |
| firered-vad (eou) | LISTENING | silence after speech → send the turn |
| `barge.h` | REPLYING | AEC residual energy → user is talking over the reply |
| firered-vad (fup) | AWAITING_FOLLOWUP | voice start → another turn, no wake word |

## Things that will surprise you

Written down because each one cost real time to find.

**Barge-in is not a VAD.** Echo cancellation suppresses whatever correlates
with the reference, and during double-talk it takes the user's voice down with
the echo — too mangled for any speech classifier. Raw-mic energy fails the
other way, because echo *is* speech. `barge.h` instead learns the echo-residual
floor and fires at 3x it. Measured: echo alone leaves ~0.002 RMS, real
double-talk ~0.16 — a ~95x gap.

**The AEC reference is tapped at the DAC, not where TTS produces audio.** TTS
synthesises faster than realtime, so at the producer it runs ahead of the
speaker, and a barge-in throws that queue away — leaving the canceller aligned
against sound nobody heard. Silence must be published too, or the render
timeline stops advancing while the mic's keeps going.

**MTP is a property of the model pair, not a knob.** The draft head only
predicts a trunk it matches; a head from a different quantisation pipeline
dropped acceptance to 33-48% and made speculation a *slowdown*. Keep trunk and
head from the same repo. `n=1` is the setting — llama.cpp's default of 3 is a
regression here. Acceptance is noisy (71-81% across runs on one pair); average
at least three before concluding anything.

**An interrupted turn is rolled back entirely.** A reply that stops mid-sentence
would otherwise be closed as though the model chose to stop there, and a few of
those in context teach it to truncate itself — measured, ten interrupted turns
shrink the next clean reply to a fragment.

**`models/tokenizer.bin` is a hard dependency with no reference in this repo.**
moonshine resolves it as `dirname(model)/tokenizer.bin`. Without it, startup
fails complaining about the *model*.

**LocalVQE's noise gate takes a quiet room to exactly zero**, and that silence
is what keeps the wake detector's own gate shut. Do not "optimise" by bypassing
LocalVQE while idle — it re-opens that gate and costs far more than it saves.

## Tuning

Everything is a command-line flag, grouped by the same three-letter prefixes the
startup block prints — so the line describing a subsystem tells you which flags
tune it. `--help` lists all of them with their defaults.

```bash
./build/gemma-live --help
./build/gemma-live --mtp-off --vad-silence 350 --tts-chunk 2
```

```
  llm   --llm-model --llm-mmproj --llm-threads --llm-ctx --llm-predict --llm-temp
  sys   --sys-prompt
  mtp   --mtp-off --mtp-model --mtp-draft
  tts   --tts-model --tts-voice --tts-cfg --tts-steps --tts-anchor --tts-chunk --tts-rms
  dfn   --dfn-model
  aec   --aec-model --aec-threads --aec-gate
  kwd   --kwd-model --kwd-wake --kwd-step --kwd-window --kwd-gpu
        --kwd-ratio --kwd-floor --kwd-nogate --kwd-duck --kwd-debug
  vad   --vad-model --vad-silence --vad-empty --vad-debug
  nod   --nod-off --nod-phrases --nod-after --nod-gap --nod-mono
        --nod-per-turn --nod-len --nod-gain --nod-debug --nod-dump
  stt   --stt-model --stt-threads                  (gl-serve only)
  rt    --rt-host --rt-port --rt-ui                (gl-serve only)
  rt    --rt-host --rt-port --rt-ui                (gl-serve only)
  fup   --fup-timeout --fup-hops --fup-gate
  brg   --brg-ratio --brg-floor --brg-sustain --brg-debug
```

The defaults are measured rather than guessed; the reasoning behind the
non-obvious ones is in `src/session.h` and `src/barge.h`.

`--kwd-debug`, `--vad-debug` and `--brg-debug` print what each detector is
seeing, which is the fastest way to tell "correctly quiet" from "stuck".
`--verbosity 2` restores the full ggml/llama diagnostics that are filtered out
by default.

## Backchannels

Short "mm-hm" sounds while you are still talking, so a long turn does not
happen into total silence. They never take the floor: no LLM call, nothing
added to the KV cache, no state change — the turn carries on exactly as it
would have.

The trigger reuses evidence the system already computes. The keyword worker
runs `ends_mid_thought()` on the rolling transcript to stretch the
end-of-turn threshold from 500 ms to 900 ms when you pause on a word an
English sentence cannot end on ("and", "because", "the"). That stretch is a
window where the system has already decided you are not finished, which is
exactly where a nod belongs. Two tiers fire:

```
confident   the transcript ends mid-thought
monologue   no transcript evidence, but you have been talking past --nod-mono
```

Only the confident tier is safe enough to fire on short exchanges, and on
its own it is so rare you would not notice the feature; the monologue tier
is what makes it audible.

A nod goes in early in a pause or not at all — before ~200 ms it clips a
word you are still finishing, and after ~450 ms you are probably done and it
is pure delay in front of the answer. Firing one also holds end-of-turn off
until the clip finishes plus a beat, so the reply cannot land on top of the
nod, and so you get the moment the nod just offered you.

Clips are pre-rendered at startup in the assistant's own voice (~1.5 s of
boot) and played from memory. The streaming path's ~335 ms time-to-first-
audio is far too slow: a backchannel that late has missed the moment it was
reacting to. They are trimmed, capped at `--nod-len`, and played at
`--nod-gain` — at full level a nod does not read as a listener signal, it
reads as an interruption.

Getting a nod wrong is cheap. Fired at a real turn end it produces "Mm-hm.
The capital of France is Paris.", which is how people talk. The failure that
matters is firing mid-word, which is what the pause window guards against.

`--nod-debug` logs every nod with its position and the transcript word that
justified it, and every near miss with the reason it was rejected.

Phrases can only really be judged by ear, so `--nod-dump DIR` renders the
current `--nod-phrases` to WAV and exits without opening the microphone.
What lands on disk is what reaches the speaker — trimmed, capped, and at
`--nod-gain`, so the files are deliberately quiet.

```bash
./build/gemma-live --nod-phrases "Mm-hm.,Sure.,Okay." --nod-dump /tmp/nods
```

## Realtime API server

`gl-serve` exposes the same voice loop over OpenAI's Realtime protocol, so
anything already written against that API can drive Gemma locally.

```bash
./build/gl-serve                       # ws://127.0.0.1:8927/v1/realtime
./build/gl-serve --rt-port 9000 --vad-silence 350
```

Audio is mono PCM16 at 24 kHz in both directions, base64 inside the JSON
events, exactly as the protocol specifies. Turn detection defaults to
`server_vad` (the same firered VAD the app uses); send
`session.update` with `turn_detection: null` to drive turns yourself with
`input_audio_buffer.commit` + `response.create`.

Supported events:

```
client -> server   session.update  input_audio_buffer.append/commit/clear
                   response.create  response.cancel  conversation.item.truncate

server -> client   session.created/updated
                   input_audio_buffer.speech_started/speech_stopped
                   input_audio_buffer.committed/cleared
                   conversation.item.created
                   response.created  response.output_item.added
                   response.content_part.added
                   response.output_audio.delta/done
                   response.output_audio_transcript.delta/done
                   response.content_part.done  response.output_item.done
                   response.done  error
```

Event names follow the GA schema (`response.output_audio.delta`), not the
older beta spelling (`response.audio.delta`).

### Web UI

`gl-serve` serves **Gemma Live**, a browser client at `/` on the same port
that carries the audio session — chat and voice in one page, no install.

```bash
./build/gl-serve          # then open http://127.0.0.1:8927/
```

**Type** to chat: `POST /api/chat` streams the reply back over SSE.
Synthesis is skipped by default (`end_turn(speak=false)`), which is most of
a turn's latency — a short reply comes back in ~120 ms. Passing
`"speak": true` synthesises it and streams the audio back on the same event
stream as `{"audio": "<base64 pcm16>"}`, which is what the page does for a
message typed during voice mode: an aside typed into a spoken conversation
is still answered out loud.

**Press the microphone** to dictate. The text field gives way to a live
waveform and the microphone to discard/keep, so the only choices on screen
are the two that exist. Keep transcribes the take and appends it to
whatever was already in the box, editable, sent only when you send it;
discard throws the audio away and leaves the box untouched.

Nothing is written into the box while you talk, and there is no polling
either — a half-transcribed sentence rewriting itself as you speak is
noise, and the waveform already says the microphone is live. One pass over
the whole take on keep.

**Press the wave** for voice mode: a large animated version of that same
icon takes over the conversation area — graphic_eq's row of bars, so the
button expands into the thing you pressed. Both directions are drawn on the
same bars, blue for what the microphone hears and amber for what Gemma is
saying, amber wide behind and blue narrower in front. They are rarely both
live, so usually one colour shows; talking over her shows both. The composer stays live below and
gains mute and end buttons inside it — so a thought that is easier typed
than spoken does not mean leaving voice mode first. Speaking over Gemma cuts
her off.

Voice mode shows no text at all. What Gemma says streams into the chat
thread as it arrives, so closing voice mode leaves the conversation where
you can read it.

Your own spoken turns appear there too, which takes a second model: Gemma
consumes audio as tokens and never emits a transcript of what it heard. The
page posts the turn's audio to `/api/transcribe` at the moment speech stops
and drops a placeholder bubble in straight away, so the row is there before
the words are. Nothing waits on it — moonshine is a different model, the
route deliberately skips the turn lock, and the two run side by side.
Measured over four turns, a reply with transcription running alongside it
reached first audio in 281 and 283 ms against 295 and 286 ms without: no
cost to the voice turn, and the transcription itself absorbs the wait.

Both share one model context, so they are genuinely one conversation — ask
something aloud, then follow up by typing, and the pronoun resolves. Spoken
replies land in the same thread as typed ones, tagged as voice.

The page borrows llama.cpp's design tokens verbatim from
`tools/ui/src/app.css` — shadcn neutral in oklch, one radius, light and
dark — and its message layout: user messages right-aligned in a bubble,
assistant replies full width and unboxed. The orb's two states are llama's
own accent colours (`--chart-1` blue for listening, `--chart-3` gold for
speaking) rather than invented ones.

Type is **Google Sans Flex** and icons are **Google Symbols**, the two
families `ai.google.dev/gemma` itself loads, from Google's CDN. Neither is
open source — see `fonts.google.com/license/googlerestricted` — so they are
linked, never vendored, and the page falls back to the system sans and to
inline SVG icons when they do not load, which matters for a tool meant to
work offline. Google Symbols is subsetted to the four glyphs used here: the
whole font is 2.5 MB, these are 4 KB.

It is one file (`web/index.html`), served from disk so editing it and
reloading needs no rebuild; `--rt-ui` points elsewhere.

The page predates `/api/transcribe` and does not use it yet — the wave
button opens full voice mode, not dictation into the composer.

Two things worth knowing about how it works. The page owns echo
cancellation via `getUserMedia`'s `echoCancellation` constraint — that is
what makes the server's lack of a far-end reference correct rather than a
gap, and it is the division of labour the Realtime API assumes. And
barge-in is client-side, because the server only runs turn detection when
no response is in flight; since the browser's AEC has already removed
Gemma's voice from the mic, anything above the floor while she speaks is
you talking over her.

`/api/chat` deliberately takes ONE message rather than a transcript. The
model keeps the conversation in its KV cache, so re-sending history would
decode it again every turn and cost more the longer you talk — which is
also why this is not `/v1/chat/completions`, whose schema is stateless by
definition and would invite exactly that.

Chrome and Safari treat `http://localhost` as a secure context, so the
microphone works without TLS. Reaching it from another machine needs HTTPS.

### Dictation

`POST /api/transcribe` turns speech into text without taking a conversation
turn — the microphone-into-the-input-box gesture, as distinct from voice
mode.

```bash
curl -X POST http://127.0.0.1:8927/api/transcribe \
     -H 'content-type: application/json' \
     -d '{"audio":"<base64 pcm16>","rate":24000}'
# {"text":"What is the capital of France?","seconds":1.64,"chunks":1}
```

Nothing enters the KV cache and the model never sees it. That separation is
the point: dictation fills a box the user then edits and may never send, so
it must not be able to change what the assistant believes was said. It is a
different model for the same reason — Gemma consumes audio as tokens and
never emits a transcript of it, so asking Gemma would mean running a real
turn and rolling it back. Moonshine sits entirely outside the conversation,
and `/api/transcribe` deliberately does not take the turn lock, so dictating
never queues behind a spoken reply.

Single-shot by design. For ChatGPT's live partials, call it repeatedly on
the growing buffer: re-transcribing from the start each time is what keeps
the text stable instead of jittering as a window slides. Cost is ~0.12x
realtime (1.6 s of audio in ~200 ms), so poll around 1 Hz and expect the
call to lengthen as the recording does.

Audio longer than 6 s is split and transcribed in pieces, cut at the
quietest frame near the boundary so the seam lands in a pause. This is not
tidiness: moonshine-streaming-tiny transcribes cleanly to about 7.6 s and
past that **silently truncates** — a 12 s clip came back with exactly its
first 7.6 s, no error and no marker, which is the worst thing a dictation
box can do. With the split, 12 s of speech with natural pauses transcribes
in full across three chunks. Speech with no pauses at all still seams
imperfectly; the model is 32 MB, and `--stt-model` takes a bigger one.

### Turn stats

`gl-serve` prints a line per turn at the default verbosity, in the same
shape `gemma-live` uses, so numbers from the two front ends compare
directly:

```
[voice       | enc 41 tok | llm 7 tok @ 65.8 tok/s | ttft 140 ms | ttfa 379 ms | tts 2.51 s | rtf 0.44 | mtp 4/4 acc 100% | turn 1279 ms]
[text spoken | enc  0 tok | llm 5 tok @ 60.4 tok/s | ttft  56 ms | ttfa 266 ms | tts 1.61 s | rtf 0.55 | mtp 2/3 acc  67% | turn  950 ms]
[text        | enc  0 tok | llm 5 tok @ 48.1 tok/s | ttft  57 ms                                      | mtp 2/4 acc  50% | turn  633 ms]
```

`ttft` is end of input to first sampled token, `ttfa` to the first audio
byte written to the socket, `rtf` is synthesis wall time over audio produced
(above 1 it cannot keep up), and `wait` appears when a turn blocked on the
lock because the other front end was mid-turn.

The page logs its own half to the browser console — `ttft`, `ttfa` and total
as the browser experienced them. The gap between the two `ttfa` numbers is
transport plus the playback scheduler's lead, which the server cannot see;
measured here it is a few milliseconds, so a turn that still feels slow is
the end-of-turn silence (`--vad-silence`, 500 ms by default) rather than the
pipe.

Under `server_vad` the turn opens on speech onset, not at the end, so the
audio encoder runs while you are still talking. Before that it did not, and
the whole encode sat in front of the reply — 30 ms per second of speech,
plus a tail finalise that pushed `ttft` up with it:

```
speech    ttfa buffered    ttfa streaming
1.6 s        395 ms            288 ms
4.9 s        506 ms            270 ms
9.8 s        777 ms            363 ms
```

The cost is that the turn lock is held from speech onset to `response.done`,
so a message typed mid-utterance waits — visible as `wait` in the stats
line. Manual turn detection (`turn_detection: null`) has no onset to open
on and still takes the encode at commit time.

### What it deliberately does not do

**One session at a time.** `VoiceSession` owns a single llama context and is
not thread-safe, so a second conversation would need a second copy of every
model. A client arriving while another is connected gets close code 1013
rather than an unbounded wait.

**No input transcription in the voice session.** `input_audio_transcription`
is reported as `null`, because Gemma consumes audio as tokens through the
mtmd encoder and never produces a transcript of the user's speech. Output
transcription works — those are the sampled tokens, streamed as
`response.output_audio_transcript.delta`. For a transcript of the *user*,
use `/api/transcribe` above; it is a separate model and a separate request.

**Session-fixed fields.** `instructions`, `voice`, `temperature` and
`max_response_output_tokens` belong to the loaded `VoiceSession` and are set
by the flags above at startup. `session.update` accepts them so clients that
always send them keep working, but ignores the values and echoes the real
ones back in `session.updated`, so a client can see what is actually in
effect.

**Cancel drops the whole turn.** `response.cancel` and
`conversation.item.truncate` both roll the KV cache back past the entire
assistant turn, rather than truncating it at the exact millisecond playback
stopped. That is coarser than the spec, and deliberately so: leaving a
half-finished reply in context teaches the model to truncate itself on later
turns.

**pcm16 only.** No G.711, no Opus.

## Choosing a transcription model

`--stt-model` takes moonshine, parakeet or kyutai weights and dispatches on
the GGUF's `general.architecture` — not by trying loaders until one accepts,
because parakeet's loader *succeeds* on a kyutai model and then transcribes
nothing, which is a silent wrong answer rather than an error.

The wake word and transcription want different models, and the reason is
language. The wake phrase is fixed English, so `kwd` can use the smallest
thing that works. Dictation and the thread's user rows are whatever the
speaker actually said. Measured on the same four clips:

```
                          en          de / fr / es
moonshine-base    45 MB   good        gibberish
parakeet 110m    121 MB   good        gibberish
parakeet 0.6b-v3 399 MB   good        correct, with punctuation
kyutai-stt-1b    636 MB   empty       partial, truncated
```

So `parakeet-tdt-0.6b-v3` is the default despite being 8x the size and 3x
the time of moonshine-base (170-240 ms against 40-80 ms for one to three
seconds). Transcription is off the critical path — the benchmark below puts
it 11 ms from the audio path at end-of-turn — so its cost buys languages
rather than latency.

English-only and short of memory? moonshine-base is a tenth the size and
three times faster, and is not fetched by default:

```bash
curl -L -o models/moonshine-base-q4_k.gguf \
  https://huggingface.co/cstr/moonshine-base-GGUF/resolve/main/moonshine-base-q4_k.gguf
./build/gl-serve --stt-model models/moonshine-base-q4_k.gguf
```

Kyutai is wired up but did poorly here, and the test was not fair to it: it
is a streaming model driven single-shot, and `stt-1b` is the en/fr variant
being asked for German and Spanish.

## Audio in, or transcribe first?

`gl-bench-stt` answers the same utterance both ways with the same model and
prompt — audio straight through the mtmd encoder, versus moonshine then
`push_text` — alternating so both see the same conversation depth.

```bash
./build/gl-bench-stt models/moonshine-base-q4_k.gguf clip1.wav clip2.wav
```

Five short clips, Gemma 4 E4B QAT on an M4 Max:

```
                    ttft      turn
audio              154 ms    432 ms
text (stt 64 ms)    79 ms    268 ms
```

Handed the whole utterance at once, transcribing first wins easily: a
second of speech is ~25 audio tokens, the same second in words is a
handful, and the prefill difference dwarfs the ASR pass.

That is not how the server runs, though. It opens the turn on speech onset,
so the encode and its prefill are paid while the user is still talking; ASR
cannot start until the utterance is finished. What each path owes at
end-of-turn:

```
audio   154 ms   ttft, prefill already done
text    143 ms   stt 64 + ttft 79, both serial
```

Within noise of each other — so the choice is not really about latency. The
audio path keeps everything words throw away, and the transcript is a lossy
retelling: on one clip moonshine rendered a repeated question three times
over, where Gemma heard it once. Transcribing first is worth it when you
need the text anyway, which is why dictation and the thread's user rows use
it and the reply does not.

## Development

```bash
cmake --build build --target gemma-live gl-offline gl-serve barge-test transcript-test nod-test
cd build && ctest
```

`gl-offline` drives the model half from WAV files with no microphone — the only
way to measure any of this reproducibly:

```bash
./build/gl-offline turn1.wav turn2.wav turn3.wav
```

Pass one WAV per turn, not the same one repeated: identical audio makes MTP look
far better than it is. Run several turns, because a speculative-decoding cache
bug shows up on turn 2, not turn 1. `GL_ABORT_MS=250` reproduces repeated
barge-in.

All three binaries share `gl-session`, so build them together — a stale
`gl-offline` silently measures the old code.

## Numbers

Gemma 4 E4B QAT (UD-Q4_K_XL) on an M4 Max, versus Google's own LiteRT-LM
runtime on the same model and machine:

| | gemma-live | LiteRT-LM |
|---|---|---|
| decode, speculative | **104 tok/s** | 93 tok/s |
| decode, plain | **88 tok/s** | 72 tok/s |
| prefill | 1222 tok/s | **1437 tok/s** |
| model load | **0.7 s** | ~3 s |

Per turn: ~90 ms to first token, ~310 ms to first audio, ~5.6 GB resident.
Perceived latency is dominated by the 500 ms end-of-utterance wait and TTS, not
by the LLM.
