# Industrial Motor Fault Detection using Machine Learning

A simulation-based predictive-maintenance prototype that connects an ESP32 motor-sensor simulator to a Python/Flask machine-learning dashboard.

> **Scope:** This repository uses synthetic sensor data for learning and demonstration. Its model results do not represent performance on a real industrial motor or a validated Remaining Useful Life system.

## Project Objective

The project demonstrates an end-to-end workflow for:

- simulating motor current, vibration, and temperature readings
- representing normal operation and three fault conditions
- sending sensor readings from ESP32 to a Flask API over Wi-Fi
- classifying the current motor condition with Random Forest
- demonstrating an experimental LSTM-based RUL estimate
- displaying readings, predictions, confidence, and maintenance messages on a live dashboard

## System Architecture

```text
ESP32 synthetic sensor simulator
            │
            │ HTTP/JSON over local Wi-Fi
            ▼
      Flask /data endpoint
            │
            ├── Random Forest fault classification
            ├── Synthetic LSTM RUL demonstration
            └── CSV sensor logging
            ▼
       Live browser dashboard
```

## Fault Classes

| Code | Condition | Simulated behaviour |
|---:|---|---|
| 0 | Normal | Values remain close to rated operating conditions |
| 1 | Bearing fault | Increased vibration, current ripple, and moderate temperature rise |
| 2 | Overload | High current and a progressive temperature increase |
| 3 | Imbalance | Increased vibration and modulated current |

## Repository Structure

```text
.
├── server.py
├── requirements.txt
├── .gitignore
└── firmware/
    ├── MotorFaultSim.ino
    ├── fault_equations.h
    ├── wifi_send.h
    └── secrets.example.h
```

## Machine-Learning Approach

### Random Forest classifier

The server generates labelled synthetic samples for the four operating conditions. It uses a stratified train/test split, fits preprocessing only on the training set, and reports accuracy on held-out synthetic test data.

Inputs:

- current in amperes
- vibration in g
- temperature in °C

Output:

- predicted motor condition: Normal, Bearing Fault, Overload, or Imbalance

### Experimental LSTM RUL estimator

The LSTM is trained on separate synthetic motor-degradation sequences. Training and validation motors are generated separately to reduce sequence leakage. Validation MAE is calculated after converting predictions back to days.

The RUL panel is an educational demonstration. A defensible real-world RUL model would require run-to-failure data, operating context, maintenance history, and testing on unseen physical motors.

## Software Requirements

- Python 3
- Arduino IDE
- ESP32 board package
- ArduinoJson library
- Python packages listed in `requirements.txt`

## Setup

### 1. Configure Python

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure the ESP32 safely

Copy the example configuration:

```bash
# Windows
copy firmware\secrets.example.h firmware\secrets.h

# macOS/Linux
cp firmware/secrets.example.h firmware/secrets.h
```

Edit `firmware/secrets.h` and enter:

- your Wi-Fi name
- your Wi-Fi password
- your computer's local Flask-server address

The real `secrets.h` file is ignored by Git and must never be committed.

### 3. Start the Flask server

```bash
python server.py
```

Open the dashboard at:

```text
http://localhost:5000
```

### 4. Upload the ESP32 sketch

Open `firmware/MotorFaultSim.ino` in Arduino IDE, select your ESP32 board, and upload it. The sketch sends a synthetic sensor snapshot every 100 ms and changes fault mode every 10 seconds.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/data` | Receive ESP32 sensor readings |
| `GET` | `/api/latest` | Return recent readings, prediction summary, and RUL estimate |
| `GET` | `/api/rul` | Return the current synthetic LSTM RUL estimate |
| `GET` | `/` | Display the live dashboard |

Example JSON payload:

```json
{
  "current": 5.12,
  "vibration": 0.24,
  "temp": 46.3,
  "fault": 0,
  "label": "NORMAL",
  "timestamp": 10000
}
```

## Important Limitations

- All classifier and RUL training data is synthetic.
- High test accuracy can occur because the class equations are deliberately different.
- The system has not been validated on real motors, sensors, or industrial environments.
- Fault labels are known during simulation; real deployments would not have these labels automatically.
- The Flask development server is intended for local demonstration.
- Maintenance recommendations and RUL values are illustrative, not safety decisions.

## Future Improvements

- collect real current, vibration, and temperature sensor data
- extract time- and frequency-domain vibration features
- compare Random Forest with SVM, XGBoost, and neural-network baselines
- add a confusion matrix, classification report, and saved evaluation artifacts
- save trained models and load them at server startup
- add automated tests and input validation
- evaluate RUL with real run-to-failure datasets

## Author

**Ashutosh Mahanta**  
Data Science and Machine Learning aspirant interested in predictive maintenance, Python, Scikit-learn, and OpenCV.
