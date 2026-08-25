from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import threading
import time
import math
import random
import csv
import os
from collections import deque
from datetime import datetime

# ── Suppress TF logs ──────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ── Flask app ─────────────────────────────────────────
app = Flask(__name__)

# ── Global data store (last 100 readings) ─────────────
data_store = deque(maxlen=100)
data_lock  = threading.Lock()

# ── Fault labels ──────────────────────────────────────
FAULT_NAMES = {0: "Normal", 1: "Bearing Fault", 2: "Overload", 3: "Imbalance"}
FAULT_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444", 3: "#3b82f6"}
MAINTENANCE_MSG = {
    0: ("Normal pattern", "Synthetic classifier output — demonstration only", "green"),
    1: ("Bearing-fault pattern", "Synthetic classifier output — inspect the demo condition", "orange"),
    2: ("Overload pattern", "Synthetic classifier output — inspect the demo condition", "red"),
    3: ("Imbalance pattern", "Synthetic classifier output — inspect the demo condition", "blue"),
}

# ── Sensor log for RUL ────────────────────────────────
RUL_LOG     = "sensor_log.csv"
THRESHOLDS  = {"vibration": 1.5, "current": 8.0, "temp": 80.0}

def log_sensor(curr, vib, temp):
    with open(RUL_LOG, "a", newline="") as f:
        csv.writer(f).writerow([time.time(), curr, vib, temp])

# ─────────────────────────────────────────────────────
# SECTION 1 — Random Forest Fault Classifier (existing)
# ─────────────────────────────────────────────────────
def generate_training_data(n=2000):
    X, y = [], []
    for _ in range(n):
        t    = random.uniform(0, 10)
        mode = random.randint(0, 3)
        if mode == 0:
            curr = 5.0  + random.gauss(0, 0.08)
            vib  = 0.2  + random.gauss(0, 0.015)
            temp = 45.0 + random.gauss(0, 0.5)
        elif mode == 1:
            curr = 5.0  + 0.15*math.sin(2*math.pi*120*t) + random.gauss(0, 0.05)
            vib  = 0.2  + 0.55*abs(math.sin(2*math.pi*120*t)) + random.gauss(0, 0.02)
            temp = 52.0 + random.gauss(0, 0.8)
        elif mode == 2:
            curr = 9.5  + 0.3*math.sin(2*math.pi*1.0*t) + random.gauss(0, 0.15)
            vib  = 0.3  + random.gauss(0, 0.03)
            temp = min(45.0 + 0.4*t + random.gauss(0, 1.0), 85.0)
        else:
            curr = 5.0  + 0.4*math.sin(2*math.pi*25*t) + random.gauss(0, 0.08)
            vib  = 0.2  + 0.75*math.sin(2*math.pi*25*t) + 0.15*math.sin(2*math.pi*50*t) + random.gauss(0, 0.03)
            temp = 48.0 + random.gauss(0, 0.6)
        X.append([curr, vib, temp])
        y.append(mode)
    return np.array(X), np.array(y)

print("=" * 55)
print("  [1/2] Training Random Forest Fault Classifier...")
print("=" * 55)
X, y = generate_training_data(4000)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=SEED,
    class_weight="balanced",
)
rf_model.fit(X_train_scaled, y_train)

synthetic_test_accuracy = rf_model.score(X_test_scaled, y_test)
print(f"  ✅ RF held-out synthetic test accuracy: {synthetic_test_accuracy*100:.1f}%")

# ─────────────────────────────────────────────────────
# SECTION 2 — LSTM RUL Predictor (NEW deep learning)
# ─────────────────────────────────────────────────────
#
# How it works:
#   - Input  : last WINDOW_SIZE sensor readings [current, vib, temp]
#   - Output : predicted Remaining Useful Life in days
#
# Training data: synthetic motor degradation sequences.
#   Each sequence simulates a motor running from healthy → failure.
#   At every timestep the true RUL is known, so the LSTM learns
#   to predict it from the sensor pattern.

WINDOW_SIZE = 20      # readings to look back at
NUM_FEATURES = 3      # current, vibration, temp

rul_scaler_X = MinMaxScaler()   # scales inputs  to [0,1]
rul_scaler_y = MinMaxScaler()   # scales RUL     to [0,1]

def generate_degradation_sequences(n_motors=200, steps_per_motor=300):
    """
    Simulate n_motors each running for a random lifetime.
    Returns:
      X_seq : (samples, WINDOW_SIZE, 3)  — sensor windows
      y_rul : (samples,)                 — true RUL in days at each window
    """
    all_X, all_y = [], []

    for _ in range(n_motors):
        # Random motor lifetime 30 – 120 days (representative)
        lifetime_days = random.uniform(30, 120)
        curr_seq, vib_seq, temp_seq = [], [], []

        for step in range(steps_per_motor):
            age    = step / steps_per_motor             # 0 → 1 as motor ages
            rul_d  = lifetime_days * (1 - age)           # true RUL in days

            # Degradation: sensors drift as motor ages
            curr_base = 5.0  + 3.5 * age + random.gauss(0, 0.1)
            vib_base  = 0.2  + 1.3 * age**1.5 + random.gauss(0, 0.02)
            temp_base = 45.0 + 35  * age       + random.gauss(0, 1.0)

            curr_seq.append(curr_base)
            vib_seq.append(vib_base)
            temp_seq.append(temp_base)

            # Only build windows once we have enough history
            if step >= WINDOW_SIZE:
                window = np.column_stack([
                    curr_seq[-WINDOW_SIZE:],
                    vib_seq[-WINDOW_SIZE:],
                    temp_seq[-WINDOW_SIZE:]
                ])
                all_X.append(window)
                all_y.append(rul_d)

    return np.array(all_X), np.array(all_y)

def build_lstm_model():
    model = Sequential([
        LSTM(64, input_shape=(WINDOW_SIZE, NUM_FEATURES),
             return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)          # RUL output (regression)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

print("\n  [2/2] Training LSTM RUL Predictor (Deep Learning)...")
# Generate different synthetic motors for training and validation so that
# overlapping windows from one motor do not appear in both sets.
X_rul_train, y_rul_train = generate_degradation_sequences(
    n_motors=120,
    steps_per_motor=250,
)
X_rul_val, y_rul_val = generate_degradation_sequences(
    n_motors=30,
    steps_per_motor=250,
)

# Fit scalers on training data only.
train_samples = X_rul_train.shape[0]
val_samples = X_rul_val.shape[0]

X_train_flat = X_rul_train.reshape(-1, NUM_FEATURES)
X_val_flat = X_rul_val.reshape(-1, NUM_FEATURES)

X_train_scaled = rul_scaler_X.fit_transform(X_train_flat).reshape(
    train_samples,
    WINDOW_SIZE,
    NUM_FEATURES,
)
X_val_scaled = rul_scaler_X.transform(X_val_flat).reshape(
    val_samples,
    WINDOW_SIZE,
    NUM_FEATURES,
)

y_train_scaled = rul_scaler_y.fit_transform(
    y_rul_train.reshape(-1, 1)
).flatten()
y_val_scaled = rul_scaler_y.transform(
    y_rul_val.reshape(-1, 1)
).flatten()

lstm_model = build_lstm_model()
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = lstm_model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=40,
    batch_size=64,
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[es],
    verbose=0,
)

val_predictions_scaled = lstm_model.predict(X_val_scaled, verbose=0)
val_predictions_days = rul_scaler_y.inverse_transform(
    val_predictions_scaled
).flatten()
mae_days = np.mean(np.abs(val_predictions_days - y_rul_val))

print(f"  ✅ LSTM synthetic validation MAE: ±{mae_days:.1f} days")
print("=" * 55)
print("  Both models ready!")
print("=" * 55)

# ── LSTM RUL Prediction function ──────────────────────
def estimate_rul_lstm():
    """
    Read the last WINDOW_SIZE rows from sensor_log.csv,
    pass through LSTM, return predicted RUL in days.
    Returns None if not enough data yet.
    """
    if not os.path.exists(RUL_LOG):
        return None
    try:
        data = np.loadtxt(RUL_LOG, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if len(data) < WINDOW_SIZE:
            return None           # not enough history yet

        # Last WINDOW_SIZE readings: columns [curr, vib, temp]
        window = data[-WINDOW_SIZE:, 1:4]           # shape (20, 3)
        window_sc = rul_scaler_X.transform(window)  # scale
        window_sc = window_sc.reshape(1, WINDOW_SIZE, NUM_FEATURES)

        pred_sc = lstm_model.predict(window_sc, verbose=0)[0][0]
        pred_days = rul_scaler_y.inverse_transform([[pred_sc]])[0][0]
        return max(0, round(float(pred_days), 1))
    except Exception as e:
        print(f"  [LSTM RUL error] {e}")
        return None

# ── Flask endpoint: receive data from ESP32 ───────────
@app.route('/data', methods=['POST'])
def receive_data():
    try:
        d    = request.get_json()
        curr = float(d['current'])
        vib  = float(d['vibration'])
        temp = float(d['temp'])

        log_sensor(curr, vib, temp)

        # Random Forest — fault classification
        features   = scaler.transform([[curr, vib, temp]])
        pred       = int(rf_model.predict(features)[0])
        proba      = rf_model.predict_proba(features)[0]
        confidence = float(proba[pred]) * 100

        entry = {
            'time':       datetime.now().strftime('%H:%M:%S'),
            'current':    round(curr, 3),
            'vibration':  round(vib, 4),
            'temp':       round(temp, 2),
            'fault':      pred,
            'fault_name': FAULT_NAMES[pred],
            'confidence': round(confidence, 1),
            'esp_label':  d.get('label', ''),
            'timestamp':  time.time()
        }
        with data_lock:
            data_store.append(entry)
        return jsonify({'status': 'ok', 'prediction': FAULT_NAMES[pred]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# ── Flask endpoint: dashboard fetches latest data ─────
@app.route('/api/latest')
def get_latest():
    with data_lock:
        data = list(data_store)
    if not data:
        return jsonify({'readings': [], 'summary': None})

    last  = data[-1]
    fault = last['fault']
    msg   = MAINTENANCE_MSG[fault]

    dist = {FAULT_NAMES[i]: 0 for i in range(4)}
    for d in data:
        dist[d['fault_name']] += 1

    vibs  = [d['vibration'] for d in data[-10:]]
    trend = "rising" if len(vibs) > 3 and vibs[-1] > vibs[0] else "stable"

    # ── LSTM RUL prediction ──
    rul_lstm = estimate_rul_lstm()

    return jsonify({
        'readings': data[-50:],
        'latest':   last,
        'rul':      rul_lstm,           # ← LSTM-based RUL (replaces linear regression)
        'rul_source': 'LSTM' if rul_lstm is not None else 'collecting',
        'maintenance': {
            'title':       msg[0],
            'message':     msg[1],
            'color':       msg[2],
            'fault_color': FAULT_COLORS[fault]
        },
        'distribution': dist,
        'trend':   trend,
        'total':   len(data)
    })

# ── Flask endpoint: manual RUL query ─────────────────
@app.route('/api/rul')
def rul_endpoint():
    rul = estimate_rul_lstm()
    if rul is None:
        return jsonify({'rul': None, 'message': f'Need at least {WINDOW_SIZE} readings'})
    return jsonify({
        'rul_days': rul,
        'model': 'LSTM',
        'window': WINDOW_SIZE,
        'message': f'Synthetic RUL estimate: {rul} days (demonstration only)'
    })

# ── Flask endpoint: serve the dashboard HTML ──────────
@app.route('/')
def dashboard():
    return DASHBOARD_HTML

# ── Dashboard HTML ────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Motor Fault Prediction Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

  :root {
    --bg: #0a0e1a; --panel: #0f1629; --border: #1e2d4a;
    --accent: #00d4ff; --green: #00ff88; --orange: #ffaa00;
    --red: #ff4466; --blue: #4488ff; --text: #c8d8f0; --dim: #4a6080;
    --font-mono: 'Share Tech Mono', monospace;
    --font-main: 'Exo 2', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font-main); min-height: 100vh; overflow-x: hidden; }
  body::before {
    content: ''; position: fixed; inset: 0;
    background-image: linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px; pointer-events: none; z-index: 0;
  }
  .container { position: relative; z-index: 1; padding: 20px; max-width: 1400px; margin: 0 auto; }

  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 24px; border: 1px solid var(--border);
    margin-bottom: 20px; background: rgba(15,22,41,0.9); border-radius: 8px;
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .logo-ring { width: 40px; height: 40px; border: 2px solid var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; position: relative; }
  .logo-ring::before { content: ''; position: absolute; width: 28px; height: 28px; border: 1px solid rgba(0,212,255,0.4); border-radius: 50%; animation: spin 4s linear infinite; }
  .logo-dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .header-title { font-size: 18px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
  .header-sub { font-size: 11px; color: var(--dim); letter-spacing: 3px; text-transform: uppercase; font-family: var(--font-mono); }
  .live-badge { display: flex; align-items: center; gap: 8px; font-family: var(--font-mono); font-size: 12px; color: var(--green); border: 1px solid rgba(0,255,136,0.3); padding: 6px 14px; border-radius: 20px; background: rgba(0,255,136,0.05); }
  .live-dot { width: 7px; height: 7px; background: var(--green); border-radius: 50%; animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.8)} }

  /* AI badge */
  .ai-tag { font-family: var(--font-mono); font-size: 10px; color: var(--accent); border: 1px solid rgba(0,212,255,0.3); padding: 2px 8px; border-radius: 10px; background: rgba(0,212,255,0.05); margin-left: 8px; letter-spacing: 1px; }

  #status-banner { border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; border: 1px solid; display: flex; align-items: center; justify-content: space-between; transition: all 0.5s; }
  .banner-left { display: flex; align-items: center; gap: 14px; }
  .banner-icon { width: 44px; height: 44px; border-radius: 50%; border: 2px solid; display: flex; align-items: center; justify-content: center; font-size: 20px; }
  .banner-title { font-size: 17px; font-weight: 700; }
  .banner-msg { font-size: 13px; margin-top: 3px; }
  .banner-confidence { font-family: var(--font-mono); font-size: 22px; font-weight: 700; }
  .banner-conf-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--dim); text-align: right; }

  /* RUL panel — new prominent display */
  #rul-panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .rul-label { font-size: 10px; letter-spacing: 3px; color: var(--dim); text-transform: uppercase; font-family: var(--font-mono); margin-bottom: 6px; }
  .rul-value { font-size: 36px; font-weight: 700; font-family: var(--font-mono); }
  .rul-unit  { font-size: 14px; color: var(--dim); margin-left: 4px; }
  .rul-model-tag { font-size: 11px; color: var(--accent); font-family: var(--font-mono); margin-top: 4px; }
  .rul-bar-wrap { flex: 1; margin-left: 30px; }
  .rul-bar-bg { height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; margin-top: 8px; }
  .rul-bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s, background 0.5s; }

  .metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }
  .metric-card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; position: relative; overflow: hidden; }
  .metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
  .metric-card.current::before { background: var(--orange); }
  .metric-card.vibration::before { background: var(--accent); }
  .metric-card.temp::before { background: var(--red); }
  .metric-label { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim); margin-bottom: 8px; font-family: var(--font-mono); }
  .metric-value { font-size: 32px; font-weight: 700; font-family: var(--font-mono); line-height: 1; }
  .metric-unit { font-size: 13px; color: var(--dim); margin-left: 4px; }
  .metric-bar { height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px; margin-top: 12px; }
  .metric-fill { height: 100%; border-radius: 2px; transition: width 0.5s; }

  .charts-row { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; margin-bottom: 20px; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; }
  .panel-title { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim); margin-bottom: 14px; font-family: var(--font-mono); display: flex; align-items: center; gap: 8px; }
  .panel-title::before { content: ''; width: 3px; height: 10px; background: var(--accent); display: inline-block; }
  canvas { display: block; width: 100% !important; }

  .dist-item { margin-bottom: 12px; }
  .dist-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; font-family: var(--font-mono); }
  .dist-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; }
  .dist-fill { height: 100%; border-radius: 3px; transition: width 0.6s; }

  .log-panel { margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 12px; }
  th { color: var(--dim); text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); letter-spacing: 1px; font-weight: 400; }
  td { padding: 7px 12px; border-bottom: 1px solid rgba(30,45,74,0.5); }
  tr:hover td { background: rgba(0,212,255,0.03); }
  .fault-pill { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }

  #waiting { text-align: center; padding: 60px 20px; color: var(--dim); font-family: var(--font-mono); }
  .waiting-title { font-size: 16px; margin-bottom: 8px; color: var(--accent); }
  .waiting-sub { font-size: 13px; }

  @media (max-width: 768px) {
    .metrics-row { grid-template-columns: 1fr; }
    .charts-row  { grid-template-columns: 1fr; }
    #rul-panel   { flex-direction: column; }
    .rul-bar-wrap { margin-left: 0; margin-top: 16px; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="header-left">
      <div class="logo-ring"><div class="logo-dot"></div></div>
      <div>
        <div class="header-title">Motor Fault Monitor <span class="ai-tag">LSTM RUL</span></div>
        <div class="header-sub">Predictive Maintenance Prototype — Synthetic ESP32 + ML</div>
      </div>
    </div>
    <div class="live-badge">
      <div class="live-dot"></div>
      LIVE &nbsp;|&nbsp; <span id="total-count">0</span> readings
    </div>
  </header>

  <div id="waiting">
    <div class="waiting-title">Waiting for ESP32 data...</div>
    <div class="waiting-sub">Make sure ESP32 is powered and connected to WiFi.<br>LSTM RUL predictor needs at least 20 readings to activate.</div>
  </div>

  <div id="main-content" style="display:none">

    <div id="status-banner">
      <div class="banner-left">
        <div class="banner-icon" id="banner-icon">&#9881;</div>
        <div>
          <div class="banner-title" id="banner-title">Normal</div>
          <div class="banner-msg"   id="banner-msg">No maintenance needed</div>
        </div>
      </div>
      <div style="text-align:right">
        <div class="banner-confidence" id="banner-conf">--</div>
        <div class="banner-conf-label">RF Confidence</div>
      </div>
    </div>

    <!-- LSTM RUL Panel (new) -->
    <div id="rul-panel">
      <div>
        <div class="rul-label">Synthetic RUL Estimate &nbsp;<span class="ai-tag">LSTM DEMO</span></div>
        <div>
          <span class="rul-value" id="rul-value">--</span>
          <span class="rul-unit" id="rul-unit">days</span>
        </div>
        <div class="rul-model-tag" id="rul-status">Collecting data (need 20 readings)...</div>
      </div>
      <div class="rul-bar-wrap">
        <div style="display:flex;justify-content:space-between;font-size:11px;font-family:var(--font-mono);color:var(--dim)">
          <span>Critical</span><span>Warning</span><span>Safe</span>
        </div>
        <div class="rul-bar-bg">
          <div class="rul-bar-fill" id="rul-bar" style="width:0%;background:var(--green)"></div>
        </div>
        <div style="font-size:10px;color:var(--dim);margin-top:4px;font-family:var(--font-mono)">
          0 ─────────────────── 30 ─────────────────── 120 days
        </div>
      </div>
    </div>

    <div class="metrics-row">
      <div class="metric-card current">
        <div class="metric-label">Current</div>
        <div><span class="metric-value" id="val-current" style="color:var(--orange)">--</span><span class="metric-unit">A</span></div>
        <div class="metric-bar"><div class="metric-fill" id="bar-current" style="background:var(--orange);width:0%"></div></div>
      </div>
      <div class="metric-card vibration">
        <div class="metric-label">Vibration</div>
        <div><span class="metric-value" id="val-vib" style="color:var(--accent)">--</span><span class="metric-unit">g</span></div>
        <div class="metric-bar"><div class="metric-fill" id="bar-vib" style="background:var(--accent);width:0%"></div></div>
      </div>
      <div class="metric-card temp">
        <div class="metric-label">Temperature</div>
        <div><span class="metric-value" id="val-temp" style="color:var(--red)">--</span><span class="metric-unit">°C</span></div>
        <div class="metric-bar"><div class="metric-fill" id="bar-temp" style="background:var(--red);width:0%"></div></div>
      </div>
    </div>

    <div class="charts-row">
      <div class="panel">
        <div class="panel-title">Sensor readings — live waveform</div>
        <canvas id="waveChart" height="180"></canvas>
      </div>
      <div class="panel">
        <div class="panel-title">Fault distribution</div>
        <div id="dist-container" style="margin-top:8px"></div>
      </div>
    </div>

    <div class="panel log-panel">
      <div class="panel-title">Data log — recent readings</div>
      <table>
        <thead>
          <tr>
            <th>Time</th><th>Current (A)</th><th>Vibration (g)</th>
            <th>Temp (°C)</th><th>Prediction</th><th>Confidence</th>
          </tr>
        </thead>
        <tbody id="log-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
const FAULT_COLORS = {
  'Normal': '#00ff88', 'Bearing Fault': '#ffaa00',
  'Overload': '#ff4466', 'Imbalance': '#4488ff'
};
const BANNER_STYLES = {
  green:  { bg:'rgba(0,255,136,0.06)',  border:'rgba(0,255,136,0.3)',  icon:'#00ff88', text:'#00ff88' },
  orange: { bg:'rgba(255,170,0,0.06)',  border:'rgba(255,170,0,0.3)',  icon:'#ffaa00', text:'#ffaa00' },
  red:    { bg:'rgba(255,68,102,0.06)', border:'rgba(255,68,102,0.3)', icon:'#ff4466', text:'#ff4466' },
  blue:   { bg:'rgba(68,136,255,0.06)', border:'rgba(68,136,255,0.3)', icon:'#4488ff', text:'#4488ff' }
};
const ICONS = { green:'&#10003;', orange:'&#9888;', red:'&#9888;', blue:'&#8635;' };

const ctx = document.getElementById('waveChart').getContext('2d');
const waveChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label:'Current (A)', data:[], borderColor:'#ffaa00', borderWidth:1.5, pointRadius:0, tension:0.4, yAxisID:'y' },
      { label:'Vibration (g)', data:[], borderColor:'#00d4ff', borderWidth:1.5, pointRadius:0, tension:0.4, yAxisID:'y1' },
      { label:'Temp (°C)', data:[], borderColor:'#ff4466', borderWidth:1.5, pointRadius:0, tension:0.4, yAxisID:'y2' }
    ]
  },
  options: {
    responsive:true, animation:false,
    interaction:{ mode:'index', intersect:false },
    plugins:{ legend:{ labels:{ color:'#4a6080', font:{ family:'Share Tech Mono', size:11 }, boxWidth:12 } } },
    scales:{
      x:  { ticks:{ color:'#4a6080', font:{ family:'Share Tech Mono', size:10 }, maxTicksLimit:8 }, grid:{ color:'rgba(30,45,74,0.5)' } },
      y:  { position:'left',  ticks:{ color:'#ffaa00', font:{ size:10 } }, grid:{ color:'rgba(30,45,74,0.3)' }, title:{ display:true, text:'A',   color:'#ffaa00', font:{ size:10 } } },
      y1: { position:'right', ticks:{ color:'#00d4ff', font:{ size:10 } }, grid:{ display:false }, title:{ display:true, text:'g',   color:'#00d4ff', font:{ size:10 } } },
      y2: { position:'right', ticks:{ color:'#ff4466', font:{ size:10 } }, grid:{ display:false }, title:{ display:true, text:'°C',  color:'#ff4466', font:{ size:10 } }, offset:true }
    }
  }
});

function updateRUL(rul) {
  const valEl    = document.getElementById('rul-value');
  const unitEl   = document.getElementById('rul-unit');
  const statEl   = document.getElementById('rul-status');
  const barEl    = document.getElementById('rul-bar');
  const MAX_DAYS = 120;

  if (rul === null || rul === undefined) {
    valEl.textContent  = '--';
    statEl.textContent = 'Collecting data (need 20 readings)...';
    statEl.style.color = 'var(--dim)';
    barEl.style.width  = '0%';
    return;
  }

  valEl.textContent = rul;
  const pct = Math.min(rul / MAX_DAYS * 100, 100);
  barEl.style.width = pct + '%';

  if (rul < 5) {
    barEl.style.background = 'var(--red)';
    statEl.textContent     = 'Synthetic estimate: critical range — demonstration only.';
    statEl.style.color     = 'var(--red)';
    valEl.style.color      = 'var(--red)';
  } else if (rul < 30) {
    barEl.style.background = 'var(--orange)';
    statEl.textContent     = 'Synthetic estimate: warning range — demonstration only.';
    statEl.style.color     = 'var(--orange)';
    valEl.style.color      = 'var(--orange)';
  } else {
    barEl.style.background = 'var(--green)';
    statEl.textContent     = 'Synthetic LSTM estimate — demonstration only.';
    statEl.style.color     = 'var(--green)';
    valEl.style.color      = 'var(--green)';
  }
}

function updateDashboard(data) {
  if (!data.latest) return;
  document.getElementById('waiting').style.display      = 'none';
  document.getElementById('main-content').style.display = 'block';

  const l = data.latest;
  const m = data.maintenance;

  document.getElementById('total-count').textContent = data.total;

  const banner = document.getElementById('status-banner');
  const style  = BANNER_STYLES[m.color] || BANNER_STYLES.green;
  banner.style.background  = style.bg;
  banner.style.borderColor = style.border;
  document.getElementById('banner-icon').style.borderColor = style.icon;
  document.getElementById('banner-icon').style.color       = style.icon;
  document.getElementById('banner-icon').innerHTML         = ICONS[m.color];
  document.getElementById('banner-title').textContent      = m.title;
  document.getElementById('banner-title').style.color      = style.text;
  document.getElementById('banner-msg').textContent        = m.message;
  document.getElementById('banner-conf').textContent       = l.confidence.toFixed(1) + '%';
  document.getElementById('banner-conf').style.color       = style.text;

  updateRUL(data.rul);

  document.getElementById('val-current').textContent = l.current.toFixed(2);
  document.getElementById('val-vib').textContent     = l.vibration.toFixed(4);
  document.getElementById('val-temp').textContent    = l.temp.toFixed(1);
  document.getElementById('bar-current').style.width = Math.min(l.current  / 12   * 100, 100) + '%';
  document.getElementById('bar-vib').style.width     = Math.min(Math.abs(l.vibration) / 1.5 * 100, 100) + '%';
  document.getElementById('bar-temp').style.width    = Math.min((l.temp - 20) / 70 * 100, 100) + '%';

  const readings = data.readings;
  waveChart.data.labels              = readings.map(r => r.time);
  waveChart.data.datasets[0].data   = readings.map(r => r.current);
  waveChart.data.datasets[1].data   = readings.map(r => r.vibration);
  waveChart.data.datasets[2].data   = readings.map(r => r.temp);
  waveChart.update('none');

  const dist  = data.distribution;
  const total = Object.values(dist).reduce((a,b) => a+b, 1);
  document.getElementById('dist-container').innerHTML = Object.entries(dist).map(([name, count]) => {
    const pct = Math.round(count / total * 100);
    const col = FAULT_COLORS[name] || '#888';
    return `<div class="dist-item">
      <div class="dist-label"><span>${name}</span><span style="color:${col}">${count}</span></div>
      <div class="dist-track"><div class="dist-fill" style="width:${pct}%;background:${col}"></div></div>
    </div>`;
  }).join('');

  document.getElementById('log-body').innerHTML = [...readings].reverse().slice(0,15).map(r => {
    const col = FAULT_COLORS[r.fault_name] || '#888';
    return `<tr>
      <td>${r.time}</td><td>${r.current.toFixed(3)}</td>
      <td>${r.vibration.toFixed(4)}</td><td>${r.temp.toFixed(1)}</td>
      <td><span class="fault-pill" style="background:${col}22;color:${col};border:1px solid ${col}44">${r.fault_name}</span></td>
      <td style="color:${col}">${r.confidence.toFixed(1)}%</td>
    </tr>`;
  }).join('');
}

async function poll() {
  try {
    const res  = await fetch('/api/latest');
    const data = await res.json();
    updateDashboard(data);
  } catch(e) {}
}

setInterval(poll, 1000);
poll();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Motor Fault Prediction Prototype (Synthetic RF + LSTM Demo)")
    print("=" * 55)
    print("  Dashboard   : http://localhost:5000")
    print("  ESP32 POST  : http://YOUR_LAPTOP_IP:5000/data")
    print(f"  LSTM window : {WINDOW_SIZE} readings needed for RUL")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

