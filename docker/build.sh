#!/usr/bin/env bash
# Build the gl-serve image for aarch64 NVIDIA machines.
#
#   ./docker/build.sh                    # Orin + Spark  (sm_87 + sm_121)
#   CUDA_ARCHS=87  ./docker/build.sh     # Orin only, roughly half the time
#   CUDA_ARCHS=121 ./docker/build.sh     # Spark only
#
# Build time is dominated by nvcc on llama.cpp's fattn template
# instances, and it scales with the number of architectures: two archs is
# about twice one. Memory is the real limit — Modal pins 24 GB for -j4
# and records silent OOM deaths at 39%, 56% and, at 16 GB, 31%. If the
# build dies with no error, that is what happened: give the VM more
# memory before lowering JOBS.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CUDA_ARCHS="${CUDA_ARCHS:-87;121}"
JOBS="${JOBS:-4}"
TAG="${TAG:-gemma-live:$(git -C "$REPO" describe --tags --always --dirty 2>/dev/null || date +%Y%m%d)}"

command -v docker >/dev/null || { echo "error: docker not on PATH" >&2; exit 1; }

# An x86_64 host would emulate every compile step here; on Apple Silicon
# or any aarch64 machine this is native. Refuse rather than run a build
# that would take a day.
host="$(docker info --format '{{.Architecture}}' 2>/dev/null || uname -m)"
case "$host" in
    aarch64|arm64) ;;
    *) echo "error: docker host is $host; this is a native aarch64 build." >&2
       echo "       Building here would emulate every nvcc invocation." >&2
       exit 1 ;;
esac

echo "==> $TAG   archs=$CUDA_ARCHS  jobs=$JOBS"
docker build --platform linux/arm64 \
    --build-arg CUDA_ARCHS="$CUDA_ARCHS" \
    --build-arg JOBS="$JOBS" \
    -f "$HERE/Dockerfile" -t "$TAG" "$REPO"

echo "==> verify"
docker run --rm --entrypoint bash "$TAG" -c '
  set -e
  readelf -h /opt/gemma-live/bin/gl-serve | grep -E "Class|Machine" | sed "s/^/  /"
  echo "  cuda archs in the binary:"
  cuobjdump --list-elf /opt/gemma-live/lib/libggml-cuda.so* 2>/dev/null \
    | grep -oE "sm_[0-9]+" | sort -u | sed "s/^/    /" || echo "    (cuobjdump absent in runtime image)"
  echo "  unresolved (libcuda.so.1 SHOULD be missing — injected by CDI):"
  ldd /opt/gemma-live/bin/gl-serve | grep "not found" | sed "s/^/    /" || echo "    none"
'
echo
echo "Image: $TAG  ($(docker image inspect "$TAG" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1073741824}'))"
echo "Next:  docker save $TAG | zstd | ssh <host> 'zstd -d | docker load'"
