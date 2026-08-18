#!/usr/bin/env bash
# Build LocalVQE (neural AEC + noise suppression) into a single self-contained
# dylib at build/localvqe-libs/liblocalvqe.dylib.
#
# Why a dylib with an explicit export list, rather than static archives:
#
#   LocalVQE carries its own ggml, and it is not interchangeable with ours —
#   LocalVQE patches in a `ggml_gru` op (see ggml/patches/ggml-gru.patch) that
#   llama.cpp's ggml does not have. So we cannot drop LocalVQE's ggml and let
#   its objects resolve against llama.cpp's; the GRU symbols would be missing.
#
#   That would make LocalVQE the THIRD ggml in this build (llama.cpp's,
#   CrispASR's, and this one), and the top two already collide by install name
#   — see the install_name_tool dance in CMakeLists.txt.
#
#   Linking everything into one dylib and exporting ONLY the localvqe_* C API
#   sidesteps all of it: LocalVQE's ggml is statically bound inside the dylib
#   and is invisible to dyld, so it cannot be confused with either of the
#   others. Verify with:  nm -gU liblocalvqe.dylib | grep ggml   (expect none)
#
# Usage: tools/build_localvqe.sh          (from the repo root)
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${HERE}/vendor/localvqe/ggml"
BUILD="${HERE}/build/localvqe-build"
STAGE="${HERE}/build/localvqe-libs"

if [ ! -f "${SRC}/CMakeLists.txt" ] || [ ! -f "${SRC}/vendor/ggml/CMakeLists.txt" ]; then
    echo "==> initialising localvqe submodule (recursive)"
    (cd "${HERE}" && git submodule update --init --recursive vendor/localvqe)
fi

mkdir -p "${STAGE}"

# CPU-only ggml. LocalVQE is ~5M params — a CPU backend is far more than fast
# enough, and keeping Metal out avoids competing with the LLM and TTS for the
# GPU on every 16 ms hop.
echo "==> configuring"
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLOCALVQE_BUILD_SHARED=OFF \
    -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
    -DGGML_HIP=OFF -DGGML_SYCL=OFF \
    -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
    -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON > /dev/null

echo "==> building"
cmake --build "${BUILD}" --target dvqe_graph dvqe_common dvqe_model ggml -j > /dev/null

# localvqe_api.cpp is not part of any static target in LocalVQE's CMake, so
# compile it ourselves. LOCALVQE_BUILD marks the C API symbols as exported.
echo "==> compiling C API"
xcrun clang++ -std=c++17 -O2 -DLOCALVQE_BUILD -fPIC -arch arm64 \
    -I"${SRC}" -I"${SRC}/vendor/ggml/include" \
    -c "${SRC}/localvqe_api.cpp" -o "${BUILD}/localvqe_api.o"

# The export list IS the isolation. Anything not named here — every ggml
# symbol included — becomes local to the dylib.
nm -gU "${BUILD}/localvqe_api.o" | awk '{print $3}' | grep '^_localvqe_' | sort -u \
    > "${STAGE}/exports.txt"

echo "==> linking liblocalvqe.dylib"
# force_load on the dvqe_* archives: their objects are only reached through
# the C API, so without it the linker drops them as unreferenced.
xcrun clang++ -dynamiclib -arch arm64 \
    -install_name @rpath/liblocalvqe.dylib \
    -o "${STAGE}/liblocalvqe.dylib" \
    "${BUILD}/localvqe_api.o" \
    -Wl,-force_load,"${BUILD}/libdvqe_graph.a" \
    -Wl,-force_load,"${BUILD}/libdvqe_model.a" \
    -Wl,-force_load,"${BUILD}/libdvqe_common.a" \
    -Wl,-force_load,"${BUILD}/libdvqe_gtcrn.a" \
    "${BUILD}/vendor/ggml/src/libggml.a" \
    "${BUILD}/vendor/ggml/src/libggml-cpu.a" \
    "${BUILD}/vendor/ggml/src/libggml-base.a" \
    "${BUILD}/vendor/ggml/src/ggml-blas/libggml-blas.a" \
    -Wl,-exported_symbols_list,"${STAGE}/exports.txt" \
    -framework Accelerate -framework Foundation

cp "${SRC}/localvqe_api.h" "${STAGE}/localvqe_api.h"

leaked=$(nm -gU "${STAGE}/liblocalvqe.dylib" | grep -c ggml || true)
if [ "${leaked}" != "0" ]; then
    echo "!! ${leaked} ggml symbols are exported — isolation broken, would collide" >&2
    exit 1
fi

echo
echo "✓ $(ls -lh "${STAGE}/liblocalvqe.dylib" | awk '{print $5}')  ${STAGE}/liblocalvqe.dylib"
echo "  exports $(wc -l < "${STAGE}/exports.txt" | tr -d ' ') localvqe_* symbols, 0 ggml"
