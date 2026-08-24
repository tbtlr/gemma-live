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
  rt    --rt-host --rt-port                        (gl-serve only)
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

### Text chat

`POST /api/chat` on the same port takes one message and streams the reply
back over SSE:

```bash
curl -N -X POST http://127.0.0.1:8927/api/chat \
     -H 'content-type: application/json' -d '{"message":"Hello"}'
```

It shares the model context with the voice session, so the two are one
conversation — ask something aloud, follow up in text, and the pronoun
resolves. Synthesis is skipped (`end_turn(speak=false)`), which is most of
a turn's latency: a short reply comes back in about 120 ms.

One message per request, not a transcript. The model keeps the
conversation in its KV cache, so re-sending history would decode it again
every turn and cost more the longer you talk — which is also why this is
not `/v1/chat/completions`, whose schema is stateless by definition and
would invite exactly that.

### What it deliberately does not do

**One session at a time.** `VoiceSession` owns a single llama context and is
not thread-safe, so a second conversation would need a second copy of every
model. A client arriving while another is connected gets close code 1013
rather than an unbounded wait.

**No input transcription.** `input_audio_transcription` is reported as
`null`, because Gemma consumes audio as tokens through the mtmd encoder and
never produces a transcript of the user's speech. Output transcription works
— those are the sampled tokens, streamed as
`response.output_audio_transcript.delta`.

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
