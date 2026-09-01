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

Three dependencies, all submodules, ~2 GB together:

```bash
git submodule update --init --recursive
```

Two of them are forks pinned to a branch, and upstream will not work for
either. The streaming `mtmd_audio_stream_*` API and the `load_vision`/
`load_audio` params live only on that llama.cpp branch; the streaming
VibeVoice API and the `crispasr-tts` target live only on that CrispASR
branch.

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

**Press the wave** for voice mode: a ribbon takes over the conversation
area — overlapping travelling waves whose amplitude follows the audio,
tapered to nothing at the ends. Gemma is her own blue, the middle stop of
the logo gradient, and the microphone is the page's foreground: white on
the dark theme, near-black on the light one. Both are drawn every frame and
translucent, so where they coincide the colours mix rather than one hiding
the other. The composer stays live below and
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
linked, never vendored. Text falls back to the system sans when the face
does not arrive; icons cannot fall back, since an unloaded ligature renders
as the literal word "mic_off", so the glyphs stay hidden until a font-load
check passes and the buttons are briefly unlabelled instead. Google Symbols
is subsetted to the glyphs used here: the whole font is 2.5 MB, these are
4 KB.

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

### Two prompts

`gemma-live` is voice-only and `prompts/voice.txt` says so on every line —
one sentence, no markdown, numbers in words. `gl-serve` answers typed turns
through the same session, so it defaults to `prompts/chat.txt` instead;
`--sys-prompt` overrides either.

`prompts/echo.txt` is a third, for diagnosis rather than use: it tells the
model to repeat what it heard verbatim and nothing else, which turns a turn
into a straight read-back of the audio path.

```bash
./build/gl-serve --sys-prompt prompts/echo.txt
```

The web prompt cannot simply be the longer one. A session holds one system
prompt and voice mode shares it, so spoken replies would become paragraphs
read aloud — the exact thing the voice prompt exists to prevent. It splits
on the channel instead, which the model can genuinely see: an audio turn is
wrapped in the model's audio markers and a text turn is not. Same session,
same question:

```
typed    The sky appears blue because of a phenomenon called **Rayleigh
         scattering**, where the Earth's atmosphere scatters shorter
         wavelengths of visible light...

spoken   The sky is blue because of Rayleigh scattering, which is when the
         Earth's atmosphere scatters shorter wavelengths of light, like
         blue, more than longer wavelengths.
```

Typed answers use markdown and take the room they need; spoken ones stay to
a sentence or two with nothing in them that cannot be read aloud.

### The native protocol

`/v1/realtime` speaks OpenAI's Realtime API, which is what lets an existing
client drive this unchanged. `/v1/live` is the same session with a protocol
built for it instead.

Audio travels as **binary WebSocket frames**, raw pcm16, in both directions.
No base64, no JSON envelope, no string allocated per 50 ms of speech. That
is 1.00 bytes on the wire per byte of audio against 1.36, and it is the path
that decides how soon a reply is heard.

Control is JSON on the text opcode, with five events out —

```
{"t":"ready","in_rate":24000,"out_rate":24000,"vad_ms":500,"vision":true}
{"t":"speech"}                        user started talking
{"t":"eou"}                           and stopped; a reply is coming
{"t":"start"}                         reply begins
{"t":"txt","s":"..."}                 transcript, as it is spoken
{"t":"end","cancelled":false,"text":"...","out_tok":8,"ttft_ms":91,"ms":300}
```

— against fourteen for the same turn on `/v1/realtime`, which wraps every
fragment in event_id, response_id, item_id, output_index and content_index
to describe items and content parts this has none of.

Five verbs in:

```
{"t":"end"}            finish the turn now, do not wait for silence
{"t":"cancel"}         stop the reply
{"t":"text","s":"..."} a typed turn, answered and spoken like any other
{"t":"played"}         the reply has finished coming out of the speakers
{"t":"barge"}          that was the user talking over her, not her own echo
```

`text` is why the web UI no longer opens a second transport onto the same
conversation: in voice mode a typed aside goes down the socket that is
already there. `/api/chat` remains for a client with no session open, and
for images — the socket carries pcm, not pictures.

`played` is the one worth having. The server knows when it stopped
*generating*, which is seconds before the client stops *playing* — and in
that gap its turn detector is live and hears the reply through the
microphone as though it were a user. Only the client knows when the audio
actually ended. Until it says so the server will not open a turn on speech,
and `barge` is how a client that means it overrides that. Audio keeps
streaming throughout, so the interrupting words are already on the server
when it does.

### Sharing it

Both off by default, because on loopback neither earns its keep. Together
they are what makes a tunnel safe to hand out.

```bash
./build/gl-serve --rt-token s3cret --rt-idle 300
```

`--rt-token` requires `?t=` on **every** route, checked in the handshake:
an unauthorised peer gets 401 and is never upgraded. The page passes on
whatever token it was opened with, so `https://host/?t=s3cret` is the whole
setup — a browser cannot put a header on a WebSocket, which is why the query
string carries it for the socket and the fetches alike. It is a shared
secret, not per-user auth; Cloudflare Access does the real thing for free if
you want it.

`--rt-idle` reclaims a session after that many seconds **without speech** —
not without traffic, because the microphone streams continuously and a
connection is never quiet while it is open. It waits for any reply in flight
to finish, so a long answer is never cut off. This matters more than the
token for a public link: the server takes one session at a time, so without
it a single forgotten tab locks everyone else out indefinitely.

### Images

On by default in `gl-serve`, off by default in `gemma-live`. The mmproj
already carries a vision tower beside the audio one, but loading it costs
~215 MiB of weights, a ~101 MiB Metal compute buffer and a 768x768 warmup
pass — worth paying where an image can arrive, wasted where one cannot. The
CLI has no way to supply one, so it does not load it unless asked.

```bash
./build/gl-serve                      # images work
./build/gl-serve --llm-vision-off     # text and speech only, smaller
./build/gemma-live --llm-vision      # loads it, though nothing feeds it yet
```

Then `POST /api/chat` takes an `images` array beside the message: base64,
with or without a `data:image/...;base64,` prefix, in any format stb_image
reads. They are decoded into the turn ahead of the text, because a question
about a picture means nothing until the picture is in the context.

```bash
curl -N -X POST http://127.0.0.1:8927/api/chat \
     -H 'content-type: application/json' \
     -d '{"message":"What is in this?","images":["data:image/png;base64,..."]}'
```

The web UI attaches them with the image button, a drag onto the page, or a
paste — and shrinks anything over 1024px before upload, since the encoder
throws the extra pixels away anyway. It asks `GET /api/config` first and
hides the button when the answer is `{"vision": false}`, so a server started
without the flag does not offer something that can only fail.

Two things to expect. An image is a few hundred tokens of prefix (the turn
log reports `img N tok`) and, unlike audio, there is nothing to overlap the
encode with — audio is encoded while you are still speaking, which is where
the low ttft comes from, whereas a turn with a picture in it waits for the
whole thing. And the token count scales with the image, so a small
thumbnail is cheap and correspondingly vague: a 400x300 test image encodes
to 48 tokens and the model reads it about as well as you would expect from
48 tokens.

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

Three unit tests cover the pure logic — backchannel triggers, barge-in and
transcript assembly — and run with the build:

```bash
cd build && ctest
```

Neither wire protocol is reachable from those: they never open a socket, so
nothing notices when an event stops being emitted or an audio frame changes
shape. `/v1/realtime` especially, since every change lands on `/v1/live`
first and a compatibility surface nobody exercises is one that quietly stops
being compatible. So:

```bash
./build/gl-serve &
tools/protocol-test.py                     # or --port / --token
```

It synthesises its own audio, so there are no fixtures — but it needs the
models loaded, which is why it is not a ctest target. It asserts the event
sequence on both protocols, that reply audio is base64 on one and binary
frames on the other, that neither leaks the other's vocabulary, and that the
played gate, barge, and typed turns behave.
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
