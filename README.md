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

Everything is an environment variable; the defaults are measured, not guessed.

```
GEMMA_LIVE_LLM_MODEL / _MMPROJ / _MTP_MODEL   model paths (swap together)
GEMMA_LIVE_MTP=0                              disable speculative decoding
GEMMA_LIVE_MTP_DRAFT=N                        tokens drafted per step (1 is best)
GEMMA_LIVE_SYSTEM_PROMPT=path                 default prompts/chat.txt
GEMMA_LIVE_AWAIT_TIMEOUT_MS=N                 followup window, default 5000
GEMMA_LIVE_TTS_FIRST_CHUNK=N                  latent frames before first audio
GEMMA_LIVE_KWD_STEP_MS / _LENGTH_MS           wake detector cadence and window
GEMMA_LIVE_KWD_GATE_RATIO / _GATE_FLOOR       wake silence gate
GEMMA_LIVE_BARGE_RATIO / _FLOOR / _SUSTAIN_MS barge-in sensitivity
```

Debugging: `GEMMA_LIVE_KWD_DEBUG`, `GEMMA_LIVE_VAD_DEBUG`,
`GEMMA_LIVE_BARGE_DEBUG` print what each detector is seeing. Verbosity 2
restores the full ggml/llama diagnostics that are filtered out by default.

## Development

```bash
cmake --build build --target gemma-live gl-offline barge-test transcript-test
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

Both binaries share `gl-session`, so build them together — a stale `gl-offline`
silently measures the old code.

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
