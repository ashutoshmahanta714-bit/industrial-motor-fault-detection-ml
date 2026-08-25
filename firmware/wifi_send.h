#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>

#include "fault_equations.h"
#include "secrets.h"

inline void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  const unsigned long start = millis();
  const unsigned long timeoutMs = 20000UL;

  while (WiFi.status() != WL_CONNECTED && millis() - start < timeoutMs) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected. ESP32 IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWi-Fi connection timed out. The sketch will keep running.");
  }
}

inline void sendToServer(const MotorReading& reading, int faultMode) {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
    return;
  }

  HTTPClient http;
  http.begin(MOTOR_SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<256> document;
  document["current"] = reading.current;
  document["vibration"] = reading.vibration;
  document["temp"] = reading.temp;
  document["fault"] = faultMode;
  document["label"] = faultLabel(faultMode);
  document["timestamp"] = millis();

  String payload;
  serializeJson(document, payload);

  const int statusCode = http.POST(payload);
  if (statusCode < 200 || statusCode >= 300) {
    Serial.printf("HTTP error: %d\n", statusCode);
  }

  http.end();
}
