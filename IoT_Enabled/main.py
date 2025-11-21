import machine
import time

# --- Light sensor ---
ldr = machine.ADC(26)  # GP26 / ADC0

# --- LEDs ---
red = machine.Pin(4, machine.Pin.OUT)
green = machine.Pin(5, machine.Pin.OUT)

# --- Heart rate sensor ---
hr_sensor = machine.ADC(27)  # GP27 / ADC1
HR_THRESHOLD = 30000          # tune based on your sensor signal
REFRACTORY_MS = 300           # minimum ms between beats

# --- Calibration values for light ---
RAW_DARK = 1150
RAW_BRIGHT = 320
THRESHOLD = 60  # light percent threshold for red LED
window = []
ROLLING_WINDOW = 10

# --- Helper functions ---
def raw_to_percent(raw):
    percent = (RAW_DARK - raw) / (RAW_DARK - RAW_BRIGHT) * 100
    return max(0, min(100, percent))

def rolling_avg(values):
    return sum(values)/len(values) if values else 0

# --- Heartbeat tracking ---
last_beat = 0
beats = 0
bpm_start_time = time.ticks_ms()
BPM_INTERVAL_MS = 5000      # calculate BPM over 5 seconds

PRINT_INTERVAL_MS = 500     # print terminal output every 0.5 seconds
last_print = time.ticks_ms()
current_bpm = 0

print("Starting light + heart rate monitor...")

while True:
    now = time.ticks_ms()

    # --- Heart rate sensor ---
    hr_raw = hr_sensor.read_u16()
    if hr_raw > HR_THRESHOLD and time.ticks_diff(now, last_beat) > REFRACTORY_MS:
        beats += 1
        last_beat = now

    # --- Light sensor ---
    raw_light = ldr.read_u16()
    window.append(raw_light)
    if len(window) > ROLLING_WINDOW:
        window.pop(0)
    avg_light = rolling_avg(window)
    light_percent = raw_to_percent(avg_light)

    # --- LED logic ---
    if light_percent > THRESHOLD:
        red.value(1)
        green.value(0)
    else:
        red.value(0)
        green.value(1)

    # --- BPM calculation every 5 seconds ---
    if time.ticks_diff(now, bpm_start_time) >= BPM_INTERVAL_MS:
        current_bpm = (beats * 60 * 1000) / BPM_INTERVAL_MS  # calculate BPM
        current_bpm /= 2  # optional divide by 2
        beats = 0
        bpm_start_time = now

    # --- Print every line with current BPM ---
    if time.ticks_diff(now, last_print) >= PRINT_INTERVAL_MS:
        print("Light Raw: {}  Light %: {:.1f}%  Heart BPM: {:.1f}".format(raw_light, light_percent, current_bpm))
        last_print = now

    time.sleep(0.002)  # ~500 Hz loop