cmake_minimum_required(VERSION 3.20)
project(gemma_live CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────────────────────────────────
# Dependency locations. Both live under vendor/ by default; override on the
# command line (-DGL_LLAMA_DIR=... / -DGL_CRISPASR_DIR=...) to build against
# a checkout elsewhere.
#
#   llama.cpp — our fork, branch gemma-live. Neither the streaming
#               mtmd_audio_stream_* API this project is built on nor the
#               load_vision/load_audio params are upstream.
#               https://github.com/tbtlr/llama.cpp
#   CrispASR  — vibevoice streaming TTS, firered_vad, moonshine streaming.
# ─────────────────────────────────────────────────────────────────────────
set(GL_LLAMA_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp" CACHE PATH "llama.cpp fork checkout")
set(GL_CRISPASR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/CrispASR"  CACHE PATH "CrispASR checkout")

if (NOT EXISTS "${GL_LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp fork not found at ${GL_LLAMA_DIR}.\n"
        "First-time setup (see also tools/fetch-models.sh for the weights):\n"
        "  git clone --branch gemma-live https://github.com/tbtlr/llama.cpp.git vendor/llama.cpp")
endif()

# Pull in the llama.cpp fork. Provides ggml, llama, llama-common,
# mtmd (with the Gemma 4a audio mmproj streaming API) and bundles miniaudio.
# The tools/ subdirectory is gated on LLAMA_BUILD_TOOLS (default OFF when
# llama.cpp is not the top-level project) — force it on so the mtmd target
# is created.
set(LLAMA_BUILD_TOOLS  ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
add_subdirectory(${GL_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)

find_package(Threads REQUIRED)

if (APPLE)   # macOS: dyld two-level-namespace ggml isolation via install_name_tool

# ─────────────────────────────────────────────────────────────────────────
# CrispASR linkage with ggml-symbol isolation
#
# CrispASR ships its own fork of ggml as separate dylibs (libggml.0.dylib,
# libggml-base.0.dylib, libggml-cpu.0.dylib, libggml-blas.0.dylib,
# libggml-metal.0.dylib). llama.cpp ships another fork under the SAME install
# names. macOS dyld can only load one library per install name, so the lib
# that loads first claims all references — including those from the *other*
# fork's dependents — and any symbol that only exists in the other fork
# crashes at load time.
#
# Fix: copy CrispASR's 5 ggml dylibs + libcrispasr-tts.dylib into a private dir
# under the build tree, run install_name_tool to rewrite their install IDs
# and cross-references to use a `-crispasr` suffix, and add that dir to
# any exe linking the lib via rpath.
#
# Used by:
#   - the library (vibevoice symbols from libcrispasr-tts)
#   - the CLI    (firered_vad symbols, also from libcrispasr-tts)
# ─────────────────────────────────────────────────────────────────────────
set(CRISPASR_BUILD_ROOT   ${GL_CRISPASR_DIR}/build)
set(CRISPASR_PATCHED_DIR  ${CMAKE_BINARY_DIR}/crispasr-libs)
file(MAKE_DIRECTORY ${CRISPASR_PATCHED_DIR})

set(GGML_PATCH_LIBS
    "ggml|ggml/src"
    "ggml-base|ggml/src"
    "ggml-cpu|ggml/src"
    "ggml-blas|ggml/src/ggml-blas"
    "ggml-metal|ggml/src/ggml-metal"
)

set(NAME_CHANGE_ARGS)
foreach(entry ${GGML_PATCH_LIBS})
    string(REPLACE "|" ";" parts ${entry})
    list(GET parts 0 dep)
    list(APPEND NAME_CHANGE_ARGS
        -change "@rpath/lib${dep}.0.dylib" "@rpath/lib${dep}-crispasr.0.dylib"
    )
endforeach()

set(PATCHED_OUTPUTS)
foreach(entry ${GGML_PATCH_LIBS})
    string(REPLACE "|" ";" parts ${entry})
    list(GET parts 0 name)
    list(GET parts 1 subdir)
    set(src "${CRISPASR_BUILD_ROOT}/${subdir}/lib${name}.0.dylib")
    set(dst "${CRISPASR_PATCHED_DIR}/lib${name}-crispasr.0.dylib")
    add_custom_command(
        OUTPUT  ${dst}
        DEPENDS ${src}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
        COMMAND install_name_tool -id "@rpath/lib${name}-crispasr.0.dylib" ${dst}
        COMMAND install_name_tool ${NAME_CHANGE_ARGS} ${dst}
        COMMENT "[crispasr-patch] lib${name}-crispasr.0.dylib"
        VERBATIM
    )
    list(APPEND PATCHED_OUTPUTS ${dst})
endforeach()

set(CRISPASR_SRC "${CRISPASR_BUILD_ROOT}/src/libcrispasr-tts.dylib")
set(CRISPASR_DST "${CRISPASR_PATCHED_DIR}/libcrispasr-tts.1.dylib")
add_custom_command(
    OUTPUT  ${CRISPASR_DST}
    DEPENDS ${CRISPASR_SRC} ${PATCHED_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CRISPASR_SRC} ${CRISPASR_DST}
    COMMAND install_name_tool -id "@rpath/libcrispasr-tts.1.dylib" ${CRISPASR_DST}
    COMMAND install_name_tool ${NAME_CHANGE_ARGS} ${CRISPASR_DST}
    COMMENT "[crispasr-patch] libcrispasr-tts.1.dylib"
    VERBATIM
)
list(APPEND PATCHED_OUTPUTS ${CRISPASR_DST})

add_custom_target(crispasr_patched DEPENDS ${PATCHED_OUTPUTS})

if (NOT EXISTS ${CRISPASR_SRC})
    message(FATAL_ERROR
        "libcrispasr-tts not found at ${CRISPASR_SRC}. "
        "Build it first: cd ${GL_CRISPASR_DIR} && cmake -B build -S . && cmake --build build --target crispasr-tts")
endif()

else()   # Linux: libcrispasr-tts.so is self-contained (static ggml + hidden
         # symbols via -Wl,--exclude-libs,ALL), so there is nothing to isolate.
    set(CRISPASR_DST         "${GL_CRISPASR_DIR}/build-static/src/libcrispasr-tts.so")
    set(CRISPASR_PATCHED_DIR "${GL_CRISPASR_DIR}/build-static/src")
    if (NOT EXISTS "${CRISPASR_DST}")
        message(FATAL_ERROR "libcrispasr-tts.so not found at ${CRISPASR_DST}. Build it: "
            "cd ${GL_CRISPASR_DIR} && cmake -B build-static -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON "
            "-DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--exclude-libs,ALL && "
            "cmake --build build-static --target crispasr-tts")
    endif()
    add_custom_target(crispasr_patched)   # no-op
endif()

# ─────────────────────────────────────────────────────────────────────────
# LocalVQE — neural AEC + noise suppression, built out-of-tree by
# tools/build_localvqe.sh into ONE dylib that exports only the localvqe_* C
# API. It carries its own ggml (patched with a ggml_gru op that llama.cpp's
# ggml does not have), which would otherwise be the third colliding ggml in
# this link; the export list keeps it invisible to dyld. See the script.
# ─────────────────────────────────────────────────────────────────────────
set(LOCALVQE_DIR ${CMAKE_BINARY_DIR}/localvqe-libs)
set(LOCALVQE_LIB ${LOCALVQE_DIR}/liblocalvqe.dylib)
if (APPLE AND NOT EXISTS ${LOCALVQE_LIB})
    message(FATAL_ERROR
        "LocalVQE dylib not found at ${LOCALVQE_LIB}.\n"
        "First-time setup:\n"
        "  git submodule update --init --recursive vendor/localvqe\n"
        "  tools/build_localvqe.sh")
endif()


# ─────────────────────────────────────────────────────────────────────────
# gl-session — the model half, as a library.
#
#   src/session.{h,cpp} — LLM + mtmd audio encoder + MTP speculative decoding
#                         + VibeVoice streaming TTS + per-turn lifecycle.
#                         No audio hardware, no UI.
#
# A library rather than a source file in each executable, so the app and the
# offline driver share one build of it and cannot drift apart. Everything a
# consumer needs to compile against session.h is PUBLIC here.
# ─────────────────────────────────────────────────────────────────────────
add_library(gl-session STATIC src/session.cpp)
add_dependencies(gl-session crispasr_patched)

target_include_directories(gl-session PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GL_LLAMA_DIR}/common                                # arg.h, common.h
    ${GL_LLAMA_DIR}/include
    ${GL_LLAMA_DIR}/tools/mtmd
    ${GL_CRISPASR_DIR}/include                            # crispasr.h
    ${GL_CRISPASR_DIR}/src                                # vibevoice.h, dfn.h,
                                                          # firered_vad.h, moonshine,
                                                          # core/audio_resample.h
)

target_link_libraries(gl-session PUBLIC
    mtmd
    llama-common
    llama
    ggml
    Threads::Threads
    ${CRISPASR_DST}
)

target_compile_features(gl-session PUBLIC cxx_std_17)

# DFN post-filter is wired in src/session.cpp under #if defined(CRISPASR_DFN).
# crispasr-tts is built with CRISPASR_DFN=1 (default ON in CrispASR), so the
# symbols are present in libcrispasr-tts.dylib — we just need the matching
# define here to compile against them.
target_compile_definitions(gl-session PRIVATE CRISPASR_DFN=1)

# rpath so the -crispasr-suffixed dylibs resolve at run time. Inherited by
# anything linking gl-session.
set(GL_RPATH "${CRISPASR_PATCHED_DIR}")

# ─────────────────────────────────────────────────────────────────────────
# gemma-live — the app.
#
#   src/vqe.h    — LocalVQE wrapper (neural AEC + noise suppression)
#   src/barge.h  — double-talk barge-in on the AEC residual
#   src/main.cpp — miniaudio mic/speaker, the AEC chain, wake word, EOU VAD,
#                  the turn state machine.
# ─────────────────────────────────────────────────────────────────────────
if (APPLE)   # gemma-live (miniaudio mic/speaker + LocalVQE) is macOS-only
add_executable(gemma-live src/main.cpp)

target_include_directories(gemma-live PRIVATE
    ${GL_LLAMA_DIR}/vendor                                # miniaudio (header-only)
    ${LOCALVQE_DIR}                                       # localvqe_api.h
)

target_link_libraries(gemma-live PRIVATE gl-session ${LOCALVQE_LIB})

# Both are for miniaudio, which main.cpp compiles inline via MA_IMPLEMENTATION:
# it casts away const in ~50 places and, under MA_API=static, leaves most of
# its API unreferenced. Our own sources build clean at -Wall -Wextra -Wcast-qual.
if (NOT MSVC)
    target_compile_options(gemma-live PRIVATE -Wno-cast-qual -Wno-unused-function)
endif()

set_target_properties(gemma-live PROPERTIES
    BUILD_RPATH   "${GL_RPATH};${LOCALVQE_DIR}"
    INSTALL_RPATH "${GL_RPATH};${LOCALVQE_DIR}"
)

if (APPLE)
    target_link_libraries(gemma-live PRIVATE
        "-framework CoreAudio"
        "-framework AudioToolbox"
        "-framework CoreFoundation"
        "-framework AVFoundation"
    )
elseif (UNIX)
    target_link_libraries(gemma-live PRIVATE ${CMAKE_DL_LIBS})
endif()
endif()   # APPLE: gemma-live

# ─────────────────────────────────────────────────────────────────────────
# Development targets.
#
#   gl-offline — drives gl-session from WAV files, no microphone. Use it to
#                measure decode/MTP behaviour reproducibly, and to check that
#                turn 2+ of a conversation stays coherent (where a KV-cache
#                bug shows up, unlike turn 1).
#   *-test     — behavioural tests for the pure-logic headers (src/barge.h,
#                src/transcript.h). No models, build in a second. `ctest`
#                runs them all.
# ─────────────────────────────────────────────────────────────────────────
add_executable(gl-offline tools/gl_offline.cpp)
target_link_libraries(gl-offline PRIVATE gl-session)
set_target_properties(gl-offline PROPERTIES
    BUILD_RPATH   "${GL_RPATH}"
    INSTALL_RPATH "${GL_RPATH}"
)

# ─────────────────────────────────────────────────────────────────────────
# gl-serve — OpenAI Realtime-compatible WebSocket front-end.
#
# A third consumer of gl-session, alongside the app and the offline driver.
# No microphone, no speaker, no AEC: audio arrives and leaves over a socket,
# so none of LocalVQE, miniaudio or the barge-in detector is linked in.
#
# The only extra include is llama.cpp's vendored nlohmann/json; the
# WebSocket layer (src/ws.h) is written against BSD sockets and CommonCrypto
# so that speaking the protocol costs no new third-party dependency.
# ─────────────────────────────────────────────────────────────────────────
add_executable(gl-serve src/realtime.cpp)
target_include_directories(gl-serve PRIVATE ${GL_LLAMA_DIR}/vendor)   # nlohmann/json.hpp
target_link_libraries(gl-serve PRIVATE gl-session)
if (NOT APPLE)
    target_link_libraries(gl-serve PRIVATE crypto)   # OpenSSL SHA1 for the WebSocket handshake (ws.h)
endif()
set_target_properties(gl-serve PROPERTIES
    BUILD_RPATH   "${GL_RPATH}"
    INSTALL_RPATH "${GL_RPATH}"
)

enable_testing()
foreach(t barge transcript nod)
    add_executable(${t}-test tests/${t}_test.cpp)
    target_include_directories(${t}-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(${t}-test PRIVATE cxx_std_17)
    add_test(NAME ${t} COMMAND ${t}-test)
endforeach()
