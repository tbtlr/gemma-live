#!/usr/bin/env bash
# Fetch every model gemma-live needs from Hugging Face into models/ and
# voices/. Both directories are gitignored — the files total ~5.6 GB — so this
# is what turns a fresh clone into something that runs.
#
# Safe to re-run: anything already present at the right size is skipped, and
# partial downloads resume rather than restart.
#
# Usage:
#   tools/fetch-models.sh              # everything
#   tools/fetch-models.sh --list       # show what would be fetched, no download
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models voices

HF=https://huggingface.co
FAILED=0

# dest | url
ITEMS=(
  # ── LLM: Gemma 4 E4B. The trio must come from ONE build. The mmproj is not
  # interchangeable across model sizes, and the MTP head only predicts a trunk
  # it matches — a head from a different quantisation pipeline drops
  # speculative-decoding acceptance far enough to make it a slowdown.
  "models/gemma-4-E4B-it-Q4_0.gguf|$HF/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_0.gguf"
  "models/mmproj-gemma-4-E4B-it-Q8_0.gguf|$HF/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mmproj-gemma-4-E4B-it-Q8_0.gguf"
  "models/mtp-gemma-4-E4B-it-Q4_0.gguf|$HF/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mtp-gemma-4-E4B-it-Q4_0.gguf"

  # ── AEC + noise suppression
  "models/localvqe.gguf|$HF/LocalAI-io/LocalVQE/resolve/main/localvqe-v1.3-4.8M-f32.gguf"

  # ── Wake word. tokenizer.bin is NOT optional: moonshine resolves it as
  # dirname(model)/tokenizer.bin, so it has to sit beside the GGUF or startup
  # fails with a message about the model rather than the tokenizer.
  "models/moonshine-streaming-tiny-q4_k.gguf|$HF/cstr/moonshine-streaming-tiny-GGUF/resolve/main/moonshine-streaming-tiny-q4_k.gguf"
  "models/tokenizer.bin|$HF/cstr/moonshine-streaming-tiny-GGUF/resolve/main/tokenizer.bin"

  # ── End-of-utterance / followup VAD
  "models/firered-vad.gguf|$HF/cstr/firered-vad-GGUF/resolve/main/firered-vad.gguf"

  # ── TTS. The voice prompt is ours, not upstream's — it is a locally-created
  # prompt hosted alongside the project. Other voices for this model live in
  # cstr/vibevoice-realtime-0.5b-GGUF; point GEMMA_LIVE_TTS_VOICE at one to
  # switch.
  "models/vibevoice-realtime-0.5b-q4_k.gguf|$HF/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-realtime-0.5b-q4_k.gguf"
  "voices/vibevoice-voice-en-Gemma_woman.gguf|$HF/tbtlr/gemma-live/resolve/main/vibevoice-voice-en-Gemma_woman.gguf"
)

remote_size() {
  curl -sIL "$1" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{print v+0}' | tr -d '\r'
}
human() { awk -v b="$1" 'BEGIN{ if (b>1073741824) printf "%.1f GB", b/1073741824; else printf "%.0f MB", b/1048576 }'; }

if [[ "${1:-}" == "--list" ]]; then
  total=0
  for it in "${ITEMS[@]}"; do
    dest=${it%%|*}; url=${it#*|}
    sz=$(remote_size "$url"); total=$((total + sz))
    printf "  %-46s %8s  %s\n" "$dest" "$(human "$sz")" "$([[ -f $dest ]] && echo present || echo missing)"
  done
  echo "  ---"
  printf "  %-46s %8s\n" "total" "$(human $total)"
  exit 0
fi

for it in "${ITEMS[@]}"; do
  dest=${it%%|*}; url=${it#*|}
  want=$(remote_size "$url")
  if [[ -f "$dest" ]]; then
    have=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
    if [[ "$have" == "$want" && "$want" != "0" ]]; then
      printf "  ✓ %-44s %8s (present)\n" "$(basename "$dest")" "$(human "$want")"
      continue
    fi
    # Size differs. Do NOT resume onto it: we cannot tell a half-finished
    # download from a DIFFERENT build of the same name, and appending the tail
    # of one file onto the body of another produces something that is the
    # right length, still loads (GGUF is header-indexed, so trailing bytes are
    # ignored) and is quietly wrong. Fetch it again from scratch instead.
    printf "  ↻ %-44s replacing (have %s, want %s)\n" \
           "$(basename "$dest")" "$(human "$have")" "$(human "$want")"
    mv -f "$dest" "$dest.superseded"
  else
    printf "  ↓ %-44s %8s\n" "$(basename "$dest")" "$(human "$want")"
  fi
  # Download to .part so a resume can only ever continue OUR OWN partial file,
  # then verify the length before putting it in place.
  if curl -fL -C - --retry 3 --retry-delay 2 -o "$dest.part" "$url"; then
    got=$(stat -f%z "$dest.part" 2>/dev/null || stat -c%s "$dest.part")
    if [[ "$want" != "0" && "$got" != "$want" ]]; then
      echo "  !! $dest: got $got bytes, expected $want — leaving as $dest.part" >&2
      FAILED=1
    else
      mv -f "$dest.part" "$dest"
      rm -f "$dest.superseded"
    fi
  else
    echo "  !! failed: $url" >&2
    FAILED=1
  fi
done

echo
if [[ $FAILED -ne 0 ]]; then
  echo "One or more downloads failed — re-run to resume." >&2
  exit 1
fi

echo "Models ready. Build with:  cmake -S . -B build && cmake --build build --target gemma-live"
