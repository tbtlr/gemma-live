// Thin wrapper around LocalVQE (github.com/localai-org/LocalVQE, Apache-2.0):
// neural real-time acoustic echo cancellation + noise suppression, with delay
// estimation built into the model, on a CPU ggml backend.
//
// Why this instead of WebRTC's AudioProcessing:
//   * It estimates mic/reference alignment itself, so the caller does not have
//     to hand it lock-step frame pairs. A classic AEC that is handed a
//     reference even slightly out of step with the mic cancels nothing.
//   * It tolerates the clock drift a real capture path produces between the
//     mic and the playback reference.
//   * Echo suppression is strong enough that the residual doubles as a
//     double-talk signal — measured here at ~0.002 RMS for echo alone versus
//     ~0.16 during real double-talk, a ~95x gap. barge.h relies on that gap.
//
// Contract: 16 kHz mono float32 in [-1, 1] on both directions. Not
// thread-safe; the caller serialises.
#pragma once

#include <cstdio>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
#include "localvqe_api.h"
}

class vqe_aec {
public:
    static constexpr int SAMPLE_RATE = 16000;

    ~vqe_aec() { if (ctx_) localvqe_free(ctx_); }

    // noise_gate_dbfs gates the model's own residual (roughly -60 to -80 dBFS)
    // while leaving speech (-30 to -10 dBFS) untouched.
    // `quiet` captures LocalVQE's own stderr during load. The library has no
    // verbosity switch and unconditionally prints its backend banner plus the
    // absolute path it resolved — noise on every start. It is captured rather
    // than discarded, and replayed if the load fails, so a real diagnostic is
    // never the thing we threw away.
    bool init(const std::string & model_path, int threads = 0,
              float noise_gate_dbfs = -45.0f, bool quiet = true) {
        int   saved_fd = -1;
        FILE * spool   = nullptr;
        if (quiet) {
            fflush(stderr);
            saved_fd = dup(STDERR_FILENO);
            spool    = tmpfile();
            if (saved_fd >= 0 && spool) dup2(fileno(spool), STDERR_FILENO);
        }
        const bool ok = init_impl(model_path, threads, noise_gate_dbfs);
        if (saved_fd >= 0) {
            fflush(stderr);
            dup2(saved_fd, STDERR_FILENO);
            close(saved_fd);
        }
        if (spool) {
            if (!ok) {                       // load failed — show what it said
                fflush(spool);
                rewind(spool);
                char line[512];
                while (fgets(line, sizeof(line), spool)) fputs(line, stderr);
            }
            fclose(spool);
        }
        return ok;
    }

private:
    bool init_impl(const std::string & model_path, int threads,
                   float noise_gate_dbfs) {
        if (threads > 0) {
            localvqe_options_t opts = localvqe_options_new();
            if (opts) {
                localvqe_options_set_model_path(opts, model_path.c_str());
                localvqe_options_set_threads(opts, threads);
                ctx_ = localvqe_new_with_options(opts);
                localvqe_options_free(opts);
            }
        } else {
            ctx_ = localvqe_new(model_path.c_str());
        }
        if (!ctx_) return false;
        hop_ = localvqe_hop_length(ctx_);
        localvqe_set_noise_gate(ctx_, /*enabled=*/1, noise_gate_dbfs);
        mic_pending_.reserve((size_t) hop_ * 4);
        far_pending_.reserve((size_t) hop_ * 4);
        hop_out_.assign((size_t) hop_, 0.0f);
        return true;
    }

public:
    int  hop()         const { return hop_; }
    int  sample_rate() const { return ctx_ ? localvqe_sample_rate(ctx_) : SAMPLE_RATE; }

    void reset() {
        mic_pending_.clear();
        far_pending_.clear();
        if (ctx_) localvqe_reset(ctx_);
    }

    // Push equal-length paired mic + far samples; cleaned samples are appended
    // to `out` 1:1. Chunks of any size are fine — anything short of a hop is
    // held back and drained on the next call.
    size_t process(const float * mic, const float * far, size_t n,
                   std::vector<float> & out) {
        if (!ctx_ || !mic || !far || n == 0) return 0;
        mic_pending_.insert(mic_pending_.end(), mic, mic + n);
        far_pending_.insert(far_pending_.end(), far, far + n);

        const size_t hop    = (size_t) hop_;
        const size_t before = out.size();
        // Strictly hop-sized calls into the STREAMING entry point. It keeps
        // the adaptive filter and neural state across calls, which is the
        // whole point — the clip API (localvqe_process_f32) resets both per
        // call, so every chunk would restart from a cold filter and leak
        // seconds of uncancelled echo.
        while (mic_pending_.size() >= hop && far_pending_.size() >= hop) {
            const int rc = localvqe_process_frame_f32(ctx_,
                                                      mic_pending_.data(),
                                                      far_pending_.data(),
                                                      (int) hop,
                                                      hop_out_.data());
            if (rc != 0) {
                const char * err = localvqe_last_error(ctx_);
                fprintf(stderr, "vqe: process_frame rc=%d %s\n", rc, err ? err : "");
            } else {
                out.insert(out.end(), hop_out_.begin(), hop_out_.end());
            }
            mic_pending_.erase(mic_pending_.begin(), mic_pending_.begin() + (ptrdiff_t) hop);
            far_pending_.erase(far_pending_.begin(), far_pending_.begin() + (ptrdiff_t) hop);
        }
        return out.size() - before;
    }

private:
    localvqe_ctx_t     ctx_ = 0;
    int                hop_ = 0;
    std::vector<float> mic_pending_;
    std::vector<float> far_pending_;
    std::vector<float> hop_out_;
};
