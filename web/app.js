// Gemma Live — the whole client.
//
// A plain script, not a module: everything here shares one scope, the way it
// did when this was inline, and module scope would have quietly broken that.
// Loaded at the end of index.html for the same reason it used to live there
// — the markup above it is already parsed by the time this runs.
const $ = s => document.querySelector(s);
const RATE = 24000, CHUNK = 2400;
// Barge-in floor on the mic. Now that playback goes through a media element
// the browser's canceller has a reference for Gemma's voice and removes it,
// so what is left above this floor while she speaks is the user talking over
// her.
const BARGE_RMS = 0.02, BARGE_HOLD = 3;
// ...so barge-in also has to out-shout the echo, not just the floor: the
// residual scales with how loud she is, and the user talking over her is
// much closer to the microphone than the speakers are.
const BARGE_ECHO = 0.4;

// Whatever token this page was opened with, carried on to everything it asks
// for. A browser cannot put a header on a WebSocket, so the query string is
// the only channel that works for both the socket and the fetches — and using
// one channel for both keeps them from drifting apart.
const TOKEN = new URLSearchParams(location.search).get('t') || '';
const url = p => TOKEN ? p + (p.includes('?') ? '&' : '?') + 't=' + encodeURIComponent(TOKEN) : p;

let ws, ctx, mic, inAn, outAn, outGain, sink, sinkEl;
let playCursor = 0, sources = [], loudRun = 0;
let outRecent = [], gated = false;
// How far ahead of now the first chunk of a reply is scheduled. Measured
// against this server: the first chunk is 300 ms of audio and the second
// arrives 284 ms later, so playback starts with about 17 ms of headroom —
// enough that any scheduling hiccup empties the queue. Starting higher would
// cost every reply its time-to-first-audio, so it starts small and grows
// only on a machine that actually runs dry.
let playLead = 0.06;
let speaking = false, voiceOn = false, muted = false, busy = false;
let liveBot = null, wsResponding = false, httpSpeaking = false;
let chatAbort = null;

// Copy of the audio sent for the turn in progress, kept so the user's own
// words can be shown in the thread. Gemma consumes audio as tokens and never
// emits a transcript of it, so the row is filled by /api/transcribe — a
// different model, off the turn lock, running alongside the reply rather
// than in front of it. Nothing waits on it: the row appears immediately and
// the words arrive when they arrive.
let turnAudio = [];

// Per-turn timing as the page experiences it. The server logs its own half;
// the difference between the two ttfa numbers is transport plus the
// playback scheduler's lead, which is exactly what the server cannot see.
let turnT0 = 0, turnFirstTok = 0, turnFirstAudio = 0;
function turnMark(what) {
  const t = performance.now();
  if (what === 'start') { turnT0 = t; turnFirstTok = turnFirstAudio = 0; }
  if (what === 'tok'   && !turnFirstTok)   turnFirstTok   = t;
  if (what === 'audio' && !turnFirstAudio) turnFirstAudio = t;
}
function turnReport(kind) {
  if (!turnT0) return;
  const r = n => n ? Math.round(n - turnT0) + ' ms' : '—';
  console.log(`[${kind}] ttft ${r(turnFirstTok)} | ttfa ${r(turnFirstAudio)}`
            + ` | done ${r(performance.now())}`);
  turnT0 = 0;
}

// ── thread ────────────────────────────────────────────────────────────
// The blank state is derived from the thread, not toggled by hand. It used
// to be switched off the first time a row appeared and never switched back,
// so a voice turn whose row was removed again — nothing intelligible, or a
// reply with no text — left the greeting gone and the composer stranded at
// the bottom of an empty page.
function refreshBlank() {
  document.body.classList.toggle('blank',
    document.querySelectorAll('#thread .msg').length === 0);
}

function addMsg(who, text, shots) {
  const m = document.createElement('div');
  m.className = 'msg ' + who;
  if (shots && shots.length) {
    const g = document.createElement('div');
    g.className = 'shots';
    for (const url of shots) {
      const im = document.createElement('img'); im.src = url; im.alt = 'attached image';
      g.appendChild(im);
    }
    m.appendChild(g);
  }
  const b = document.createElement('div');
  b.className = 'body'; b.textContent = text;
  m.appendChild(b);
  $('#thread').appendChild(m);
  refreshBlank();
  scroll();
  return m;
}
const scroll = () => { const m = document.querySelector('main'); m.scrollTop = m.scrollHeight; };

// ── attachments ───────────────────────────────────────────────────────
//
// Images ride along with the next typed message. They are held as data URLs
// because that is both what the tray renders and what the API takes, so
// there is one representation rather than a blob plus a copy of it.
let pending = [];
// Whether this server can take an image at all. Assumed false until the
// server says otherwise, so a page that loads against a build without the
// vision tower never offers a button that can only fail.
let canSeeImages = false;
fetch(url('/api/config'))
  .then(r => r.json())
  .then(c => { canSeeImages = !!c.vision; document.body.classList.toggle('novision', !canSeeImages); })
  .catch(() => { document.body.classList.add('novision'); });

// The model sees a few hundred pixels a side; a phone photo is 4000. Shrinking
// before upload is not politeness, it is the difference between a turn that
// starts encoding now and one that spends seconds pushing megabytes through a
// socket to be thrown away by the preprocessor.
const IMG_MAX = 1024;

function shrink(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onerror = () => reject(new Error('could not read ' + file.name));
    fr.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error('not an image: ' + file.name));
      img.onload = () => {
        const scale = Math.min(1, IMG_MAX / Math.max(img.width, img.height));
        if (scale === 1 && file.size < 800 * 1024) { resolve(fr.result); return; }
        const c = document.createElement('canvas');
        c.width  = Math.max(1, Math.round(img.width  * scale));
        c.height = Math.max(1, Math.round(img.height * scale));
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        // JPEG even for a PNG source: these are photographs and screenshots
        // being described, not artwork being reproduced, and the size
        // difference is large.
        resolve(c.toDataURL('image/jpeg', 0.85));
      };
      img.src = fr.result;
    };
    fr.readAsDataURL(file);
  });
}

function drawTray() {
  const t = $('#tray');
  t.textContent = '';
  pending.forEach((url, i) => {
    const d = document.createElement('div');
    d.className = 'thumb';
    const im = document.createElement('img'); im.src = url; im.alt = '';
    const x = document.createElement('button');
    x.type = 'button';
    // Same icon font as every other button in the page.
    const g = document.createElement('span'); g.className = 'gs'; g.textContent = 'close';
    x.appendChild(g);
    x.title = 'Remove'; x.setAttribute('aria-label', 'Remove image');
    x.onclick = () => { pending.splice(i, 1); drawTray(); sync(); };
    d.append(im, x);
    t.appendChild(d);
  });
  sync();
}

// Says its piece and gets out of the way; nothing here is worth a dismiss.
let noticeTimer = null;
function notify(msg) {
  $('#notice').textContent = msg;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { $('#notice').textContent = ''; }, 6000);
}

async function addFiles(files) {
  if (!canSeeImages) {
    notify('this server has vision disabled, so it cannot read images');
    return;
  }
  for (const f of files) {
    if (!f.type.startsWith('image/')) continue;
    try { pending.push(await shrink(f)); }
    catch (err) { notify(err.message); }
  }
  drawTray();
}

$('#attach').onclick = () => $('#file').click();
$('#file').onchange = e => { addFiles(e.target.files); e.target.value = ''; };

// Paste is how people actually move a screenshot into a chat.
addEventListener('paste', e => {
  if (voiceOn || dictOn) return;
  const files = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith('image/'));
  if (!files.length) return;
  e.preventDefault();
  addFiles(files);
});
// Drop anywhere on the page, not just on the composer: aiming at a 36px
// button with a dragged file is a needless precision task.
addEventListener('dragover', e => { if (!voiceOn && !dictOn) e.preventDefault(); });
addEventListener('drop', e => {
  if (voiceOn || dictOn) return;
  const files = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith('image/'));
  if (!files.length) return;
  e.preventDefault();
  addFiles(files);
});

// ── text chat ─────────────────────────────────────────────────────────
async function sendText(text, shots) {
  if (busy) return;

  // In voice mode the socket is already open on the same session, so a typed
  // aside goes down it rather than opening a second transport onto the same
  // conversation. The reply arrives through the ordinary start/txt/end path
  // and is spoken like any other, so there is nothing here to stream.
  // Images still need /api/chat: the socket carries pcm, not pictures.
  if (voiceOn && ws && ws.readyState === 1 && !(shots && shots.length)) {
    addMsg('user', text, shots);
    turnMark('start');
    ws.send(JSON.stringify({ t: 'text', s: text }));
    return;
  }

  busy = true; chatAbort = new AbortController(); sync();
  addMsg('user', text, shots);
  turnMark('start');
  const bot = addMsg('bot', '');
  const body = bot.querySelector('.body');
  // A picture with no question takes noticeably longer to first token than a
  // typed one — the whole image encodes before generation starts, with
  // nothing to overlap it. A bare cursor for that long reads as a hang, so
  // say what is happening instead.
  const analysing = !text && shots && shots.length;
  if (analysing) {
    body.textContent = shots.length > 1 ? 'Analyzing images…' : 'Analyzing image…';
    body.classList.add('thinking');
  } else {
    body.classList.add('cursor');
  }

  try {
    const res = await fetch(url('/api/chat'), {
      method: 'POST', headers: { 'content-type': 'application/json' },
      signal: chatAbort.signal,
      // In voice mode a typed aside is still part of a spoken conversation,
      // so ask for it out loud and play it through the same graph.
      body: JSON.stringify({ message: text, speak: voiceOn,
                             ...(shots && shots.length ? { images: shots } : {}) })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!line.startsWith('data: ')) continue;
        const m = JSON.parse(line.slice(6));
        if (m.delta) {
          turnMark('tok');
          // First token: the status line has served its purpose.
          if (body.classList.contains('thinking')) {
            body.textContent = '';
            body.classList.remove('thinking');
            body.classList.add('cursor');
          }
          body.textContent += m.delta; scroll();
        }
        if (m.audio && ctx) { turnMark('audio'); httpSpeaking = true; enqueue(fromB64(m.audio)); }
        if (m.error) { body.textContent += '\n[' + m.error + ']'; }
        if (m.done) {
          const info = document.createElement('div');
          info.className = 'info';
          info.textContent = `${m.tokens} tokens · ${m.ms} ms`;
          bot.appendChild(info);
        }
      }
    }
  } catch (err) {
    // Stopping is a choice, not a failure — the words already on screen are
    // the answer, so leave them and say nothing. Dropping the connection is
    // also what tells the server to stop generating.
    if (err.name !== 'AbortError') {
      body.textContent += (body.textContent ? '\n' : '') + '[' + err.message + ']';
    }
  } finally {
    body.classList.remove('cursor', 'thinking');
    turnReport(voiceOn ? 'text spoken' : 'text');
    busy = false; chatAbort = null; sync(); scroll();
  }
}

// ── dictation ─────────────────────────────────────────────────────────
//
// Speech into the composer, as distinct from voice mode: the words land in
// the box for you to edit, and nothing is sent until you send it.
//
// While recording, the text field is replaced by a live waveform and the
// microphone by discard/keep. Nothing is written into the box until you
// keep it — a half-transcribed sentence appearing and rewriting itself as
// you talk is noise, and the waveform already says the microphone is live.
// So there is no polling either: one pass over the whole take on keep.
//
// Its own microphone and AudioContext rather than voice mode's, because the
// two are mutually exclusive and sharing would mean rebuilding the graph on
// every switch.
let dictCtx = null, dictMic = null, dictNode = null;
let dictChunks = [], dictAmps = [], dictOn = false;

// Amplitude samples per 100 ms chunk. This sets the scroll speed: the strip holds
// ~60 dots, so 5 slices a chunk showed barely a second of audio and raced.
// 2 gives ~20 a second — three seconds across the strip, and each dot still
// spans 50 ms, well inside a syllable, so it stays lively rather than
// averaging speech into a flat line.
const LEVEL_SLICES = 2;

// Room tone, tracked live. A fixed RMS cut cannot separate silence from
// speech here: autoGainControl normalises the signal per microphone and per
// room, so the level of "quiet" after gain is a property of the environment,
// not a constant. Too low and silence draws as speech; too high and the
// first syllable is clipped off.
let noiseFloor = 0;
// How fast the floor chases the level, by case. Drop onto a new quiet level
// in a few slices; follow the room upward briskly while what we are hearing
// is NOT speech; and during speech only creep, so a long steady vowel cannot
// drag the floor up behind it. The creep is not optional -- without it a
// floor left too low by a dropout would call room tone speech forever, and
// never see a non-speech slice to correct itself with.
const NF_FALL = 0.5, NF_TRACK = 0.15, NF_RISE = 0.004;
// Speech has to clear the floor by this much. Room tone wanders by tens of
// percent between slices; a spoken syllable is several times louder.
const SPEECH_RATIO = 2.2;
// ...and by this absolute margin too, or a digitally silent input — floor 0,
// so any ratio of it is also 0 — would count its own dropout as speech.
const SPEECH_MIN = 0.002;

async function dictStart() {
  try {
    dictMic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true,
               autoGainControl: true, channelCount: 1 }
    });
  } catch (err) {
    $('#vstate').textContent = 'microphone denied — ' + err.message;
    return;
  }
  dictCtx = new AudioContext({ sampleRate: RATE });
  await dictCtx.resume();
  const code = `class Cap extends AudioWorkletProcessor {
    constructor(){ super(); this.b = new Float32Array(${CHUNK}); this.n = 0; }
    process(i){ const c = i[0][0]; if(!c) return true;
      for (let k=0;k<c.length;k++){ this.b[this.n++]=c[k];
        if(this.n===${CHUNK}){ this.port.postMessage(this.b.slice()); this.n=0; } }
      return true; } }
    registerProcessor('cap', Cap);`;
  const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
  await dictCtx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  const src = dictCtx.createMediaStreamSource(dictMic);
  dictNode = new AudioWorkletNode(dictCtx, 'cap');
  src.connect(dictNode);
  dictChunks = [];
  dictAmps = [];
  noiseFloor = 0;
  dictNode.port.onmessage = e => {
    dictChunks.push(e.data);
    // A few levels per chunk, so the waveform moves at speech speed rather
    // than in 100 ms steps.
    const f = e.data, per = Math.floor(f.length / LEVEL_SLICES);
    for (let k = 0; k < LEVEL_SLICES; k++) {
      let sum = 0;
      for (let i = k * per; i < (k + 1) * per; i++) sum += f[i] * f[i];
      const lvl = Math.sqrt(sum / per);
      // Seed from the very first slice instead of creeping up from zero,
      // which would take hundreds of slices and call the opening silence
      // speech the whole way.
      if (noiseFloor === 0) noiseFloor = lvl;
      // Classify against the floor as it stood BEFORE this slice, so a loud
      // one cannot raise the bar it is being measured against.
      const thr = Math.max(noiseFloor * SPEECH_RATIO, SPEECH_MIN);
      // Store what to draw, not the raw level: the floor moves, and a dot
      // must not turn into a bar after it has scrolled into view.
      const speech = lvl > thr;
      dictAmps.push(speech ? Math.min(1, Math.sqrt(lvl - thr) * 3.4) : 0);
      noiseFloor += (lvl - noiseFloor) *
                    (lvl < noiseFloor ? NF_FALL : speech ? NF_RISE : NF_TRACK);
    }
  };

  dictOn = true;
  document.body.classList.add('dictating');
  sync();
  drawWave();
}

// Release the microphone; keep whatever was captured for the caller to use
// or throw away.
function dictRelease() {
  dictOn = false;
  if (dictNode) { dictNode.port.onmessage = null; dictNode.disconnect(); dictNode = null; }
  if (dictMic)  { dictMic.getTracks().forEach(t => t.stop()); dictMic = null; }
  if (dictCtx)  { dictCtx.close(); dictCtx = null; }
}

function dictReset() {
  dictChunks = []; dictAmps = [];
  document.body.classList.remove('dictating');
  $('#accept').classList.remove('working');
  sync();
}

function dictCancel() {
  dictRelease();
  dictReset();
  box.focus();
}

async function dictAccept() {
  dictRelease();
  let n = 0;
  for (const c of dictChunks) n += c.length;
  if (!n) { dictReset(); box.focus(); return; }
  const all = new Float32Array(n);
  let at = 0;
  for (const c of dictChunks) { all.set(c, at); at += c.length; }

  // The waveform stays up while this runs — the microphone is closed but
  // the words are not back yet, and dropping to an empty box would read as
  // having lost them.
  $('#accept').classList.add('working');
  try {
    const res = await fetch(url('/api/transcribe'), {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ audio: toB64(all), rate: RATE })
    });
    const text = ((await res.json()).text || '').trim();
    if (text) box.value = (box.value ? box.value.replace(/\s*$/, '') + ' ' : '') + text;
  } catch (_) {
    $('#vstate').textContent = 'could not transcribe';
  }
  dictReset();
  box.focus();
}

// ── waveform ──────────────────────────────────────────────────────────
// Dots across the strip, one per recent level, newest at the right. Dotted
// rather than a filled trace because the composer is a small strip and a
// solid waveform there reads as a progress bar.
//
// Silence is not stillness -- the microphone is open, and the strip has to
// say so. Scrolling alone cannot: every quiet column holds the same value,
// so a row of identical dots slides along looking perfectly frozen. Instead
// a slow swell of brightness and size travels through the dots, leftward
// with the history. They stay ON the midline, which is what "no signal"
// means here; lifting them off it would imply one. Bars carry their own
// motion, so only the dots take this.
const DOT_WAVELEN = 130;   // px between crests — a few of them across the strip
const DOT_SPEED   = 55;    // px a second; a given dot cycles every ~2.4 s
const wave = $('#wave'), wctx = wave.getContext('2d');
function drawWave() {
  if (!document.body.classList.contains('dictating')) return;
  requestAnimationFrame(drawWave);

  const dpr = devicePixelRatio || 1;
  const w = wave.clientWidth, h = wave.clientHeight;
  if (!w || !h) return;
  if (wave.width !== w * dpr) { wave.width = w * dpr; wave.height = h * dpr; }
  wctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  wctx.clearRect(0, 0, w, h);

  const GAP = 10, R = 1.6;   // spaced enough to read as dots, not a bar chart
  const cols = Math.floor(w / GAP);
  const mid  = h / 2;
  const t    = performance.now() / 1000;
  const cs   = getComputedStyle(document.documentElement);
  const fg   = cs.getPropertyValue('--foreground').trim();
  const dim  = cs.getPropertyValue('--muted-foreground').trim();

  for (let i = 0; i < cols; i++) {
    const x = i * GAP + GAP / 2;
    // Newest on the right: walk the tail of the amplitude history.
    const idx = dictAmps.length - cols + i;
    const amp = idx >= 0 ? (dictAmps[idx] || 0) : 0;

    // A bar or a dot, never both stacked: the bars are translucent, so a dot
    // left underneath one shows through as a bright bead at the waist.
    //
    // The dot is the fallback branch rather than the special case, which is
    // what keeps a column from ever coming out blank — `len > R` is false
    // for a NaN or a missing sample too, so those land on the dot as well.
    const len = amp * (mid - R);
    if (len > R) {
      wctx.strokeStyle = fg;
      wctx.globalAlpha = 0.35 + amp * 0.65;
      wctx.lineWidth   = R * 2;
      wctx.lineCap     = 'round';
      wctx.beginPath();
      wctx.moveTo(x, mid - len);
      wctx.lineTo(x, mid + len);
      wctx.stroke();
    } else {
      // An explicit filled circle, NOT a round cap on a zero-length stroke.
      // Blink paints that degenerate subpath, WebKit skips it, so the pauses
      // came out empty in Safari however the classifier had scored them. An
      // arc has no such ambiguity.
      const swell = 0.5 + 0.5 * Math.sin(2 * Math.PI *
                    (x + t * DOT_SPEED) / DOT_WAVELEN);
      wctx.fillStyle   = dim;
      wctx.globalAlpha = 0.5 + swell * 0.4;
      wctx.beginPath();
      wctx.arc(x, mid, R * (0.8 + swell * 0.4), 0, Math.PI * 2);
      wctx.fill();
    }
  }
  wctx.globalAlpha = 1;
}

// ── composer ──────────────────────────────────────────────────────────
const box = $('#box');
// Everything that depends on a reply being in flight. Separate from sync()
// because the audio paths call it per chunk, and sync() rewrites the
// textarea height — a forced reflow ten times a second for nothing.
//
// A reply counts as in flight until it has finished being HEARD, not just
// generated: audio outlives the stream that produced it, and a stop button
// that vanishes while Gemma is still talking stops the wrong thing.
function refreshBusy() {
  const responding = busy || wsResponding || speaking || httpSpeaking;
  document.body.classList.toggle('responding', responding);
  const hasInput = !!box.value.trim() || pending.length > 0;
  $('#send').disabled = busy || !hasInput;
  // Voice is unavailable mid-reply: the session takes one turn at a time, so
  // offering it would only queue behind the answer being written.
  $('#mic').disabled = busy;
  $('#dict').disabled = busy;
  // Dictation owns the composer while it runs; attaching mid-recording would
  // land in a box the user cannot see.
  $('#attach').disabled = busy || dictOn;
}
function sync() {
  box.style.height = 'auto';
  // Never below one line. window.innerHeight can be 0 while the page is
  // laying out or in a detached view, and Math.min against that collapses
  // the textarea to nothing — placeholder clipped, buttons adrift.
  const cap = Math.max(window.innerHeight * 0.4, 36);
  box.style.height = Math.max(36, Math.min(box.scrollHeight, cap)) + 'px';
  // "typing" is really "has something to send" — an attached image counts,
  // or the send button would stay hidden behind the voice one.
  $('#composer').classList.toggle('typing', !!box.value.trim() || pending.length > 0);
  refreshBusy();
}
box.addEventListener('input', sync);
box.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); }
});
$('#composer').addEventListener('submit', e => {
  e.preventDefault();
  const t = box.value.trim();
  // An image on its own is a message: "what is this" is implied, and making
  // someone type a word to send one would be busywork.
  if ((!t && !pending.length) || busy) return;
  const shots = pending;
  pending = []; drawTray();
  box.value = ''; sync();
  sendText(t, shots);
});

// ── voice: audio helpers ──────────────────────────────────────────────
function toB64(f32) {
  const pcm = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const v = Math.max(-1, Math.min(1, f32[i]));
    pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  const b = new Uint8Array(pcm.buffer);
  let s = '';
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s);
}
function fromB64(b64) {
  const bin = atob(b64), b = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i);
  const pcm = new Int16Array(b.buffer), f = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
  return f;
}
function stopAudio() {
  for (const s of sources) { try { s.stop(); } catch (_) {} }
  sources = []; playCursor = 0; speaking = false;
  speechEnded();
  refreshBusy();
}

// The microphone is gated for the whole length of any playback, and this is
// not belt-and-braces — the server cannot do it for us. It suppresses turn
// detection only while a response is ACTIVE, and active means generating.
// TTS is synthesised faster than realtime, so by the time the last chunk is
// on the wire the client still has seconds of it to play, and the server has
// already re-armed its VAD. Feed it the microphone through that window and
// it hears Gemma, calls her a user turn, and she answers herself.
//
// Closing the gate again clears the server's buffer and resets its VAD, so
// whatever leaked in before it shut cannot be mistaken for the start of a
// turn.
function speechEnded() {
  // Idempotent: stopAudio() empties `sources` and every stopped node still
  // fires onended afterwards, so this is reached more than once per reply.
  if (!gated) return;
  gated = false;
  // The server stopped generating seconds ago; only this side knows the
  // sound has actually stopped. Until it hears this it will not open a turn
  // on speech, so the reply cannot be heard as a user. Below the guard, so
  // it is said once per reply rather than once per audio node.
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ t: 'played' }));
  httpSpeaking = false;
  outRecent = []; loudRun = 0;
}

function enqueue(f32) {
  const buf = ctx.createBuffer(1, f32.length, RATE);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf; src.connect(outGain);
  const now = ctx.currentTime;
  if (playCursor < now) playCursor = now + playLead;   // lead on the first chunk
  src.start(playCursor); playCursor += buf.duration;
  sources.push(src);
  src.onended = () => {
    sources = sources.filter(s => s !== src);
    if (sources.length) return;
    // An empty queue means the buffer ran dry, which is only the end of the
    // reply if the server has also said so. Treating the two as the same
    // thing made an underrun look like a finished reply: the ribbon dropped
    // mid-sentence, and "played" went out while she was still talking —
    // which is precisely what un-gates the server's turn detector and lets
    // her answer herself.
    if (wsResponding) {
      // Ran dry with more coming. Buy margin for the next reply rather than
      // paying for it on every one: a machine that keeps up never does.
      playLead = Math.min(0.30, playLead + 0.06);
      console.log('[audio] buffer ran dry mid-reply; lead now',
                  Math.round(playLead * 1000), 'ms');
      return;
    }
    speaking = false; speechEnded(); refreshBusy();
  };
  speaking = true;
  gated = true;
  refreshBusy();
}

// ── voice: session ────────────────────────────────────────────────────
async function openVoice() {
  if (dictOn) dictCancel();   // one microphone at a time
  document.body.classList.add('voice');
  // Earned back below, once there is a microphone and a socket.
  document.body.classList.remove('listening');
  ampIn = ampOut = 0;
  // Talking is the primary act here; the box is the alternative, so it says
  // what it is for rather than repeating the general invitation.
  $('#box').placeholder = 'Type';

  try {
    mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }
    });
  } catch (err) {
    $('#vstate').textContent = 'microphone denied — ' + err.message;
    return;
  }

  // Pinned to the wire rate, so the browser does the 48k -> 24k conversion
  // on capture with a filtered resampler. Doing it ourselves meant linear
  // decimation with no anti-aliasing, folding everything above 12 kHz back
  // into the band — worse audio for no benefit, once the echo canceller
  // turned out to need a media element rather than a matching sample rate.
  ctx = new AudioContext({ sampleRate: RATE });
  await ctx.resume();
  outGain = ctx.createGain();
  outAn = ctx.createAnalyser(); outAn.fftSize = 512;
  inAn  = ctx.createAnalyser(); inAn.fftSize  = 512;
  // Playback goes out through a hidden <audio> element fed by a
  // MediaStreamDestination, NOT ctx.destination. The browser's echo
  // canceller only subtracts audio it can see as a reference, and
  // Web-Audio-direct output is not in that set in Chrome — element playback
  // is. This is the difference between the canceller having a reference for
  // Gemma's voice and not having one, which is why it bled through however
  // clean the rest of the path was.
  sink   = ctx.createMediaStreamDestination();
  sinkEl = new Audio();
  sinkEl.srcObject = sink.stream;
  sinkEl.play().catch(() => {});   // openVoice runs inside the click
  outGain.connect(outAn).connect(sink);
  const srcNode = ctx.createMediaStreamSource(mic);
  srcNode.connect(inAn);

  // Batched in the worklet: at 128 frames a callback this would otherwise be
  // ~190 postMessages a second for no benefit.
  const code = `class Cap extends AudioWorkletProcessor {
    constructor(){ super(); this.b = new Float32Array(${CHUNK}); this.n = 0; }
    process(i){ const c = i[0][0]; if(!c) return true;
      for (let k=0;k<c.length;k++){ this.b[this.n++]=c[k];
        if(this.n===${CHUNK}){ this.port.postMessage(this.b.slice()); this.n=0; } }
      return true; } }
    registerProcessor('cap', Cap);`;
  const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  const node = new AudioWorkletNode(ctx, 'cap');
  srcNode.connect(node);
  node.port.onmessage = e => onChunk(e.data);

  voiceOn = true; muted = false;
  $('#mute').classList.remove('off');
  connectWs();
}

function rms(x) {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  return Math.sqrt(sum / x.length);
}
// True RMS of what is going to the speakers, in the same units as the mic —
// getByteFrequencyData would be a dB-ish scale and not comparable.
function outRms() {
  if (!outAn) return 0;
  const d = new Float32Array(outAn.fftSize);
  outAn.getFloatTimeDomainData(d);
  return rms(d);
}
// pcm16 straight onto the socket. toB64 stays for /api/transcribe, which is
// an ordinary HTTP body and has nowhere else to put bytes.
function toPcm16(f32) {
  const p = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const v = Math.max(-1, Math.min(1, f32[i]));
    p[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return p.buffer;
}
function pcm16ToFloat(ab) {
  const p = new Int16Array(ab), f = new Float32Array(p.length);
  for (let i = 0; i < p.length; i++) f[i] = p[i] / (p[i] < 0 ? 0x8000 : 0x7fff);
  return f;
}

function sendChunk(f32) {
  turnAudio.push(f32.slice());
  ws.send(toPcm16(f32));
}

// The button form of a barge-in. Same three things have to happen — tell
// the server to stop, stop the playback, close off the half-written bubble —
// but reached by hand, so there is no question of whether it was really
// speech.
function stopReply() {
  if (wsResponding && ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ t: 'cancel' }));
  }
  // Dropping the SSE connection is what tells the server to abandon a typed
  // reply; there is no cancel message on that path.
  if (chatAbort) chatAbort.abort();
  stopAudio();
  endVoiceReply(true);
  refreshBusy();
}
// Not bound to voice mode: the button is not offered there, because talking
// over her already does this and does it without reaching for the mouse.
$('#stop').onclick = stopReply;

function onChunk(f32) {
  if (!ws || ws.readyState !== 1 || muted) return;

  if (speaking || httpSpeaking) {
    // The echo in THIS chunk left the speaker a moment ago, so comparing it
    // against the instantaneous output reading would line up the wrong two
    // things. Take the loudest of the last half second instead — erring
    // towards not cutting Gemma off.
    outRecent.push(outRms());
    if (outRecent.length > 5) outRecent.shift();

    const outNow = Math.max(...outRecent);

    // Client-side barge-in: the server runs turn detection only when no
    // response is in flight, so cutting a reply short is the client's job.
    // The echo term is belt and braces now that the canceller has a
    // reference — with a low residual it never beats the plain floor — but
    // it costs nothing and covers a device where that reference does not
    // land.
    const thr = Math.max(BARGE_RMS, outNow * BARGE_ECHO);
    if (rms(f32) > thr) {
      if (++loudRun >= BARGE_HOLD) {
        // "barge" is both the cancel and the permission to open a turn on
        // what is being said over her — the server holds that back until the
        // client says the reply is no longer audible.
        ws.send(JSON.stringify({ t: 'barge' }));
        stopAudio();
        endVoiceReply(true);
        loudRun = 0;
      }
    } else loudRun = 0;
  }

  sendChunk(f32);
}

// A spoken reply is written into the same thread the typed ones are in, as
// it arrives rather than at the end. One model context means one
// conversation; streaming it means a reply cut short is already saved
// instead of living in a buffer that gets discarded.
function endVoiceReply(interrupted) {
  if (!liveBot) return;
  const body = liveBot.querySelector('.body');
  body.classList.remove('cursor');
  if (!body.textContent.trim()) liveBot.remove();        // nothing was ever said
  else if (interrupted) body.textContent += ' …';
  liveBot = null;
  refreshBlank();
}

// Show the user's spoken turn. The row goes in now — before the reply's row,
// which response.created adds next — and its text is filled in later.
async function showSpokenTurn() {
  const chunks = turnAudio;
  turnAudio = [];
  if (!chunks.length) return;

  const row = addMsg('user', '');
  const body = row.querySelector('.body');
  body.classList.add('pending');

  let n = 0;
  for (const c of chunks) n += c.length;
  const all = new Float32Array(n);
  let at = 0;
  for (const c of chunks) { all.set(c, at); at += c.length; }

  try {
    const res = await fetch(url('/api/transcribe'), {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ audio: toB64(all), rate: RATE })
    });
    const j = await res.json();
    const text = (j.text || '').trim();
    if (text) { body.textContent = text; }
    else      { row.remove(); }            // nothing intelligible: no empty bubble
  } catch (_) {
    row.remove();                          // dictation unavailable — say nothing
  } finally {
    refreshBlank();
    body.classList.remove('pending');
    scroll();
  }
}

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
  ws = new WebSocket(proto + location.host + url('/v1/live'));
  // Reply audio arrives as bytes, not as base64 inside an event.
  ws.binaryType = 'arraybuffer';
  ws.onopen  = () => {
    $('#vstate').textContent = '';                        // clears a previous failure
    // Both halves are in place now: the microphone was granted before this
    // was ever called, and the socket the chunks go to is open.
    if (voiceOn) document.body.classList.add('listening');
  };
  ws.onerror = () => { $('#vstate').textContent = 'connection failed'; };
  ws.onclose = () => {
    wsResponding = false; stopAudio(); refreshBusy();
    document.body.classList.remove('listening');
    if (voiceOn) $('#vstate').textContent = 'disconnected';
  };
  ws.onmessage = ev => {
    if (ev.data instanceof ArrayBuffer) {          // reply audio
      turnMark('audio');
      enqueue(pcm16ToFloat(ev.data));
      return;
    }
    const m = JSON.parse(ev.data);
    switch (m.t) {
      case 'ready':
        // The server states its own rates rather than us assuming them.
        if (m.out_rate && m.out_rate !== RATE) {
          notify('server speaks ' + m.out_rate + ' Hz, this page speaks ' + RATE);
        }
        break;
      // Deliberately NOT sharing a body with eou. These two once both did
      // nothing and were folded together; when the turn handling was added to
      // what looked like the stopped case, started inherited it — and since
      // it fires about 800 ms into an utterance, it took the opening words
      // away into their own bubble.
      case 'speech':
        break;                             // the ribbon already shows this
      case 'eou':
        turnMark('start');                 // the user is waiting from here
        showSpokenTurn();
        break;
      case 'start':
        wsResponding = true;
        refreshBusy();
        if (!turnT0) turnMark('start');
        liveBot = addMsg('bot', '');
        liveBot.querySelector('.body').classList.add('cursor');
        break;
      case 'txt':
        turnMark('tok');
        if (liveBot) liveBot.querySelector('.body').textContent += m.s;
        break;
      case 'end':
        wsResponding = false;
        refreshBusy();
        turnReport('voice');
        endVoiceReply(!!m.cancelled);
        break;
      case 'err':
        $('#vstate').textContent = m.s || 'server error';
        break;
    }
  };
}

function closeVoice() {
  voiceOn = false;
  document.body.classList.remove('listening');
  turnAudio = [];
  endVoiceReply(true);
  if (ws) ws.close();
  stopAudio();
  if (mic) mic.getTracks().forEach(t => t.stop());
  if (sinkEl) {
    try { sinkEl.pause(); } catch (_) {}
    sinkEl.srcObject = null;
    sinkEl = null;
  }
  sink = null;
  if (ctx) ctx.close();
  ctx = null; inAn = outAn = null;
  document.body.classList.remove('voice');
  $('#box').placeholder = 'Ask anything';
  refreshBlank();
}

$('#mic').onclick  = () => voiceOn ? closeVoice() : openVoice();
$('#dict').onclick   = dictStart;
$('#cancel').onclick = dictCancel;
$('#accept').onclick = dictAccept;
$('#hang').onclick = closeVoice;
// The button turning red is the feedback; a caption saying so as well is
// one more thing to read.
$('#mute').onclick = () => { muted = !muted; $('#mute').classList.toggle('off', muted); };
addEventListener('keydown', e => { if (e.key === 'Escape' && voiceOn) closeVoice(); });

// ── the voice ribbon ──────────────────────────────────────────────────
//
// Bars were the obvious visual and the wrong one: discrete, quantised, and
// they snap. Speech does not. This draws each side as a few overlapping
// travelling waves instead — continuous curves whose amplitude follows the
// audio, tapered to nothing at the edges so the shape is a lens rather than
// a band with cut ends.
//
// Blue is the microphone, amber is Gemma, both drawn every frame and
// translucent, so where they coincide the colours actually mix instead of
// one hiding the other. Layers within a side run at slightly different
// speeds and wavelengths; that slow drift in and out of phase is what stops
// it looking like a rendered sine and starts it looking alive.
const cv = $('#viz'), g = cv.getContext('2d');
let phase = 0;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// Per-side smoothed energy. Rising fast and falling slow is what reads as a
// voice; chasing the raw FFT reads as noise.
let ampIn = 0, ampOut = 0;

function energy(an) {
  if (!an) return 0;
  const d = new Uint8Array(an.frequencyBinCount);
  an.getByteFrequencyData(d);
  let sum = 0;
  const n = Math.min(48, d.length);            // voice lives low in the band
  for (let i = 1; i < n; i++) sum += d[i];
  return (sum / (n - 1)) / 255;
}

// Three layers, each a travelling wave at its own wavelength and speed.
const LAYERS = [
  { k: 1.6, v: 0.55, a: 1.00, alpha: 0.30 },
  { k: 2.7, v: -0.38, a: 0.62, alpha: 0.26 },
  { k: 4.3, v: 0.83, a: 0.38, alpha: 0.20 },
];

function ribbon(colour, amp, w, h, seed) {
  const mid = h / 2;
  const max = h * 0.42;
  for (const L of LAYERS) {
    g.beginPath();
    const steps = 96;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = t * w;
      // Taper to nothing at both ends, so the ribbon has no cut edges.
      const env = Math.sin(Math.PI * t);
      // A slow secondary swell along the length keeps the envelope from
      // being a perfect lens every frame.
      const swell = 0.75 + 0.25 * Math.sin(phase * 0.6 + t * 2.1 + seed);
      const y = mid + Math.sin(t * Math.PI * 2 * L.k + phase * L.v * 3 + seed)
                      * env * swell * max * L.a * amp;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    // Mirror back along the centre to close the shape into a lens.
    for (let i = steps; i >= 0; i--) {
      const t = i / steps;
      const x = t * w;
      const env = Math.sin(Math.PI * t);
      const swell = 0.75 + 0.25 * Math.sin(phase * 0.6 + t * 2.1 + seed);
      const y = mid - Math.sin(t * Math.PI * 2 * L.k + phase * L.v * 3 + seed)
                      * env * swell * max * L.a * amp;
      g.lineTo(x, y);
    }
    g.closePath();
    g.fillStyle = colour;
    g.globalAlpha = L.alpha;
    g.fill();
  }
  g.globalAlpha = 1;
}

function draw() {
  requestAnimationFrame(draw);
  // Not merely invisible — not running. The envelope followers would
  // otherwise track a dead analyser and the ribbon would arrive mid-swell.
  if (!document.body.classList.contains('listening')) return;

  const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  phase += 0.02;

  const eIn  = muted    ? 0 : energy(inAn);
  const eOut = speaking ? energy(outAn) : 0;
  ampIn  = eIn  > ampIn  ? eIn  : ampIn  * 0.88 + eIn  * 0.12;
  ampOut = eOut > ampOut ? eOut : ampOut * 0.88 + eOut * 0.12;

  // A resting swell so a quiet session breathes rather than going flat.
  const IDLE = 0.06;
  ribbon(css('--speak'),  Math.max(IDLE, ampOut), w, h, 0.0);
  ribbon(css('--listen'), Math.max(IDLE, ampIn),  w, h, 1.7);
}

// Swap in the Google Symbols glyphs only once the font has really loaded.
if (document.fonts && document.fonts.load) {
  document.fonts.load('20px "Google Symbols"').then(() => {
    if (document.fonts.check('20px "Google Symbols"')) document.body.classList.add('gsym');
  }).catch(() => {});
}

// Greeting by local time. Three bands, not four: "good night" is a
// farewell in English, so using it to greet someone who just arrived reads
// as being shown the door. Late hours get "evening" instead.
(() => {
  const h = new Date().getHours();
  const part = h < 5 ? 'evening'
             : h < 12 ? 'morning'
             : h < 18 ? 'afternoon'
             : 'evening';
  $('#greet').textContent = `Good ${part}, Human`;
})();

refreshBlank();
draw();
sync();
