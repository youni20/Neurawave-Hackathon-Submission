# NeuroWave Hackathon: Light & Heart Rate Monitor

A Raspberry Pi Pico project that tracks **ambient light** and **heart rate** to explore environmental effects on migraine triggers.

---

## Features

- **Light Monitoring**
  - Analog LDR sensor
  - Smooth rolling average
  - Red/green LEDs indicate brightness
- **Heart Rate Monitoring**
  - Optical pulse sensor
  - Real-time beat detection
  - BPM calculated over 5 seconds
- **Live Terminal Output**
  - Updated every 0.5s
  - Shows raw light, light %, and heart BPM

---

## Hardware

| Component            | Pico Pin |
|----------------------|----------|
| Light sensor (S)     | GP26     |
| Heart rate sensor (S)| GP27     |
| Red LED (+)          | GP4      |
| Green LED (+)        | GP5      |
| LEDs (-)             | GND      |
| Sensors VCC          | 3.3V     |
| Sensors GND          | GND      |

**Additional:** Resistors for LEDs (330–1kΩ), jumper wires, breadboard.

---

## Setup

1. Flash MicroPython onto your Pico.
2. Connect the components as shown above.
3. Open Thonny or your MicroPython IDE.
4. Copy `main.py` to the Pico.
5. Run the script.

---

## Calibration

- `HR_THRESHOLD`: adjust to your pulse sensor signal.
- `RAW_DARK` / `RAW_BRIGHT`: calibrate LDR for your room lighting.
- `THRESHOLD`: light % for red LED trigger.

---

## Usage

- Place your finger on the pulse sensor.
- Observe LED feedback for ambient light.
- Monitor live readings in the terminal:

Light Raw: <raw_value> Light %: <percent> Heart BPM: <bpm>


- Heart BPM is smoothed over 5 seconds; output updates every 0.5 seconds.

---

## Notes

- Script samples heart rate at ~500 Hz for accuracy.
- LEDs respond immediately to light changes.
- Optional: BPM divided by 2 for calibration.

---

## Hackathon Focus

This setup demonstrates how **light intensity and heart rate can be monitored simultaneously**, providing a platform to explore **migraine triggers** in response to environmental conditions.
