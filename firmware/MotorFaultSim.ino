#include <Arduino.h>
#include "fault_equations.h"
#include "wifi_send.h"

// 0 = Normal, 1 = Bearing fault, 2 = Overload, 3 = Imbalance
int faultMode = 0;

unsigned long lastSwitch = 0;
unsigned long modeStart = 0;

void setup() {
  Serial.begin(115200);
  randomSeed(micros());

  connectWiFi();

  modeStart = millis();
  lastSwitch = millis();

  Serial.println("ESP32 Motor Fault Simulator Ready");
  Serial.println("mode,current_A,vibration_g,temp_C,label");
}

void loop() {
  const unsigned long now = millis();

  // Cycle through fault modes every 10 seconds.
  if (now - lastSwitch >= 10000UL) {
    faultMode = (faultMode + 1) % 4;
    lastSwitch = now;
    modeStart = now;
  }

  // Elapsed time within the current simulated condition.
  const float modeTimeSeconds = (now - modeStart) / 1000.0f;

  MotorReading reading = generateReading(modeTimeSeconds, faultMode);

  Serial.printf(
    "%d,%.3f,%.4f,%.2f,%s\n",
    faultMode,
    reading.current,
    reading.vibration,
    reading.temp,
    faultLabel(faultMode)
  );

  sendToServer(reading, faultMode);

  // Report one synthetic sensor snapshot every 100 ms.
  delay(100);
}
