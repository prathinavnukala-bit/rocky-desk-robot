#pragma once

// On-device "Hey Rocky" wake-word detection using a microWakeWord-trained
// TFLite-Micro model (see hey_rocky_model.h) plus the same Google
// "microfrontend" audio feature extractor microWakeWord's training pipeline
// uses (bundled inside Chirale_TensorFlowLite), so the on-device features
// match what the model was trained on exactly.

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

#include "hey_rocky_model.h"

#define WW_SAMPLE_RATE 16000
#define WW_CHUNK_SAMPLES (WW_SAMPLE_RATE / 100)  // 160 samples = 10ms, matches
                                                  // the frontend's step size
#define WW_NUM_FEATURES 40
#define WW_FRAMES_PER_INFERENCE 3  // model input is [1,3,40] -- the first
                                   // conv layer strides 3 feature-frames at
                                   // a time (see training's --stride 3)
#define WW_TENSOR_ARENA_SIZE (40 * 1024)
#define WW_SMOOTH_WINDOW 5  // number of consecutive inferences averaged
                            // before comparing to the detection threshold --
                            // a single noisy frame should not be able to
                            // trigger on its own (same lesson learned from
                            // the VAD RMS threshold tuning earlier)
// Lowered from 0.85: live testing (with wake-detection timing properly
// synced to when the user actually spoke) showed a genuine "hey rocky"
// attempt peak at 0.714 -- a real, cleanly-shaped detection curve, just
// under the old bar. The training ROC data showed a 0.84 cutoff still only
// had a ~4% miss rate with near-zero false accepts, so there's real margin
// to trade a little false-accept risk for catching quieter/late-night
// attempts like this one.
#define WW_DETECT_THRESHOLD 0.70f
#define WW_NUM_RESOURCE_VARIABLES 6  // the streaming "state_internal" model
                                     // stores its recurrent state in this
                                     // many TFLite resource variables
                                     // (VAR_HANDLE ops) -- confirmed by
                                     // inspecting the trained .tflite file

namespace wakeword {

inline const tflite::Model* g_model = nullptr;
inline tflite::MicroInterpreter* g_interpreter = nullptr;
inline TfLiteTensor* g_input = nullptr;
inline TfLiteTensor* g_output = nullptr;
inline uint8_t* g_tensor_arena = nullptr;

inline FrontendState g_frontend_state;

inline int8_t g_frame_buffer[WW_FRAMES_PER_INFERENCE][WW_NUM_FEATURES];
inline int g_frames_collected = 0;

inline float g_smooth_scores[WW_SMOOTH_WINDOW];
inline int g_smooth_idx = 0;
inline bool g_smooth_filled = false;

inline bool init() {
  g_model = tflite::GetModel(g_hey_rocky_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("WakeWord: model schema version mismatch.");
    return false;
  }

  g_tensor_arena = (uint8_t*) ps_malloc(WW_TENSOR_ARENA_SIZE);
  if (g_tensor_arena == nullptr) {
    Serial.println("WakeWord: failed to allocate tensor arena in PSRAM.");
    return false;
  }

  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(g_tensor_arena, WW_TENSOR_ARENA_SIZE);
  if (allocator == nullptr) {
    Serial.println("WakeWord: MicroAllocator::Create() failed.");
    return false;
  }

  tflite::MicroResourceVariables* resource_variables =
      tflite::MicroResourceVariables::Create(allocator, WW_NUM_RESOURCE_VARIABLES);
  if (resource_variables == nullptr) {
    Serial.println("WakeWord: MicroResourceVariables::Create() failed.");
    return false;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      g_model, resolver, allocator, resource_variables);
  g_interpreter = &static_interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("WakeWord: AllocateTensors() failed -- try a larger WW_TENSOR_ARENA_SIZE.");
    return false;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  FrontendConfig config;
  FrontendFillConfigWithDefaults(&config);
  // These match exactly what microWakeWord's training pipeline used
  // (pymicro_features' MicroFrontend defaults) -- see
  // puddly/pymicro-features python.cpp for the source of truth. A mismatch
  // here would silently produce features the model was never trained on.
  config.window.size_ms = 30;
  config.window.step_size_ms = 10;
  config.filterbank.num_channels = WW_NUM_FEATURES;
  config.filterbank.lower_band_limit = 125.0f;
  config.filterbank.upper_band_limit = 7500.0f;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025f;
  config.noise_reduction.odd_smoothing = 0.06f;
  config.noise_reduction.min_signal_remaining = 0.05f;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95f;
  config.pcan_gain_control.offset = 80.0f;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;

  if (!FrontendPopulateState(&config, &g_frontend_state, WW_SAMPLE_RATE)) {
    Serial.println("WakeWord: FrontendPopulateState() failed.");
    return false;
  }

  g_frames_collected = 0;
  g_smooth_idx = 0;
  g_smooth_filled = false;

  Serial.println("WakeWord: ready.");
  return true;
}

inline void quantizeFrame(const uint16_t* raw, int slot) {
  for (int i = 0; i < WW_NUM_FEATURES; i++) {
    int32_t q = (int32_t) lroundf((float) raw[i] / g_input->params.scale) + g_input->params.zero_point;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    g_frame_buffer[slot][i] = (int8_t) q;
  }
}

inline float runInference() {
  memcpy(g_input->data.int8, g_frame_buffer, sizeof(g_frame_buffer));
  if (g_interpreter->Invoke() != kTfLiteOk) {
    return 0.0f;
  }
  uint8_t raw = g_output->data.uint8[0];
  return ((float) raw - g_output->params.zero_point) * g_output->params.scale;
}

// Clears all per-session state: the smoothing window, the in-progress
// feature-frame buffer, and the model's own internal streaming state
// (resource variables). Call this after a detection, before resuming idle
// ticking -- without it, the smoothing window's `filled` flag never goes
// back false once set, so the last several (high, just-detected) scores
// stay averaged into the very next ticks after a cooldown, making an
// immediate spurious re-detection right after every real one almost
// guaranteed. Confirmed live: this caused continuous re-triggering
// immediately after every response.
inline void reset() {
  g_frames_collected = 0;
  g_smooth_idx = 0;
  g_smooth_filled = false;
  for (int i = 0; i < WW_SMOOTH_WINDOW; i++) g_smooth_scores[i] = 0.0f;
  if (g_interpreter != nullptr) {
    g_interpreter->Reset();
  }
}

// reset() above deliberately leaves the audio front-end's history alone (it
// runs after every detection, where that history is still valid). This is
// the full version -- front-end (noise/gain estimates, window history) AND
// model state -- used by the self-test and as its first-line self-heal.
inline void resetAll() {
  FrontendReset(&g_frontend_state);
  reset();
}

inline float pushSmoothedScore(float score) {
  g_smooth_scores[g_smooth_idx] = score;
  g_smooth_idx = (g_smooth_idx + 1) % WW_SMOOTH_WINDOW;
  if (g_smooth_idx == 0) g_smooth_filled = true;
  int count = g_smooth_filled ? WW_SMOOTH_WINDOW : max(g_smooth_idx, 1);
  float sum = 0;
  for (int i = 0; i < count; i++) sum += g_smooth_scores[i];
  return sum / count;
}

// Feeds `num_samples` of 16kHz int16 PCM through the frontend + model.
// Returns the latest smoothed detection score (0 if no inference ran this
// call), and sets *detected true if that score cleared the threshold.
inline float processAudio(const int16_t* samples, size_t num_samples, bool* detected) {
  *detected = false;
  float last_smoothed = -1.0f;
  size_t idx = 0;

  while (idx + WW_CHUNK_SAMPLES <= num_samples) {
    size_t samples_read = 0;
    struct FrontendOutput out = FrontendProcessSamples(
        &g_frontend_state, samples + idx, WW_CHUNK_SAMPLES, &samples_read);
    idx += samples_read;
    if (samples_read == 0) break;  // avoid a stuck infinite loop

    if (out.size > 0) {
      quantizeFrame(out.values, g_frames_collected);
      g_frames_collected++;

      if (g_frames_collected == WW_FRAMES_PER_INFERENCE) {
        g_frames_collected = 0;
        float score = runInference();
        last_smoothed = pushSmoothedScore(score);
        if (last_smoothed >= WW_DETECT_THRESHOLD) {
          *detected = true;
        }
      }
    }
  }

  return last_smoothed;
}

// Feeds a known-good "Hey Rocky" clip straight through front-end + model
// and returns the peak smoothed score. A healthy detector scores this high
// (>0.9 in testing); a detector that has silently died (observed live:
// exactly 0.000 on every frame, even with the phrase played at it, until a
// reboot) scores ~0. Cause-independent by design: it tests whether the
// detector WORKS, not why it might have stopped.
inline float selfTest(const int16_t* clip, size_t length) {
  resetAll();
  float peak = 0.0f;
  bool detected = false;
  for (size_t i = 0; i + WW_CHUNK_SAMPLES <= length; i += WW_CHUNK_SAMPLES) {
    float score = processAudio(clip + i, WW_CHUNK_SAMPLES, &detected);
    if (score > peak) peak = score;
  }
  resetAll();
  return peak;
}

}  // namespace wakeword
