#pragma once

#include <Arduino.h>
#include <math.h>

struct MotorReading {
  float current;    // amperes
  float vibration;  // signed acceleration in g
  float temp;       // degrees Celsius
};

inline float randomUnit() {
  return static_cast<float>(random(1, 10000)) / 10000.0f;
}

// Gaussian noise using the Box-Muller transform.
inline float gaussNoise(float sigma) {
  const float u1 = randomUnit();
  const float u2 = randomUnit();
  return sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
}

inline float randomPhase() {
  return 2.0f * PI * randomUnit();
}

/*
 * Generates one synthetic feature snapshot.
 *
 * The sketch reports data every 100 ms, which is too slow to reconstruct
 * 25-120 Hz vibration waveforms. Random phase snapshots are therefore used
 * to approximate the distribution of instantaneous readings without claiming
 * that the ESP32 is performing high-rate waveform acquisition.
 */
inline MotorReading generateReading(float modeTimeSeconds, int mode) {
  MotorReading reading{};

  switch (mode) {
    case 0: {  // Normal
      reading.current = 5.0f + gaussNoise(0.08f);
      reading.vibration = 0.2f + gaussNoise(0.015f);
      reading.temp = 45.0f + gaussNoise(0.5f);
      break;
    }

    case 1: {  // Bearing fault
      const float phase = randomPhase();
      reading.vibration =
        0.2f + 0.55f * fabsf(sinf(phase)) + gaussNoise(0.02f);
      reading.current =
        5.0f + 0.15f * sinf(phase) + gaussNoise(0.05f);
      reading.temp = 52.0f + gaussNoise(0.8f);
      break;
    }

    case 2: {  // Overload
      reading.current =
        9.5f
        + 0.3f * sinf(2.0f * PI * modeTimeSeconds)
        + gaussNoise(0.15f);
      reading.vibration = 0.3f + gaussNoise(0.03f);
      reading.temp =
        45.0f + 0.4f * modeTimeSeconds + gaussNoise(1.0f);

      if (reading.temp > 85.0f) {
        reading.temp = 85.0f + gaussNoise(0.5f);
      }
      break;
    }

    case 3: {  // Mechanical imbalance
      const float phase = randomPhase();
      reading.vibration =
        0.2f
        + 0.75f * sinf(phase)
        + 0.15f * sinf(2.0f * phase)
        + gaussNoise(0.03f);
      reading.current =
        5.0f + 0.4f * sinf(phase) + gaussNoise(0.08f);
      reading.temp = 48.0f + gaussNoise(0.6f);
      break;
    }

    default: {
      reading.current = 0.0f;
      reading.vibration = 0.0f;
      reading.temp = 0.0f;
      break;
    }
  }

  return reading;
}

inline const char* faultLabel(int mode) {
  switch (mode) {
    case 0: return "NORMAL";
    case 1: return "BEARING_FAULT";
    case 2: return "OVERLOAD";
    case 3: return "IMBALANCE";
    default: return "UNKNOWN";
  }
}
