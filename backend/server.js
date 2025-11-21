import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
// --- SERIAL PORT IMPORTS ---
import { SerialPort } from 'serialport';
import { ReadlineParser } from '@serialport/parser-readline';

const app = express();
const PORT = 3001;

// --- CONFIGURATION ---
const SENSOR_PORT = 'COM3'; // Check device manager!
const BAUD_RATE = 9600;

// --- GLOBAL SENSOR STATE ---
let sensorData = {
    raw: 0,
    percent: 0,
    heartBPM: 0,
    status: "Disconnected"
};

// --- DIRECTORIES ---
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, 'userdata'); // For App State (Login/Dashboard)
const ML_DIR = path.join(__dirname, 'ml_data');     // For ML Model (Flat JSON)

app.use(cors());
app.use(bodyParser.json());

// 1. Create folders automatically if they don't exist
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR);
    console.log(`[System] Created folder: ${DATA_DIR}`);
}
if (!fs.existsSync(ML_DIR)) {
    fs.mkdirSync(ML_DIR);
    console.log(`[System] Created folder: ${ML_DIR}`);
}

// --- SERIAL PORT LOGIC (With Auto-Reconnect) ---
function connectToSensor() {
    console.log(`[Serial] Attempting to connect to ${SENSOR_PORT}...`);
    
    const port = new SerialPort({ path: SENSOR_PORT, baudRate: BAUD_RATE, autoOpen: false });
    const parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));

    port.open((err) => {
        if (err) {
            console.log(`[Serial] Failed to open ${SENSOR_PORT}. Retrying in 5s...`);
            sensorData.status = "Disconnected";
            setTimeout(connectToSensor, 5000); // Retry loop
        }
    });

    port.on('open', () => {
        console.log(`[Serial] Connected to Sensor Hub on ${SENSOR_PORT}`);
        sensorData.status = "Connected";
    });

    port.on('close', () => {
        console.log('[Serial] Port closed/unplugged. Reconnecting...');
        sensorData.status = "Disconnected";
        setTimeout(connectToSensor, 3000);
    });

    port.on('error', (err) => {
        console.error('[Serial] Error: ', err.message);
        sensorData.status = "Disconnected";
    });

    // Read the data stream
    parser.on('data', (line) => {
        // Format example: "Light Raw: 416  Light %: 87.7%  Heart BPM: 96.0"
        
        // 1. Parse Light %
        const lightMatch = line.match(/Light %:\s*([\d.]+)/);
        if (lightMatch) {
            sensorData.percent = parseFloat(lightMatch[1]);
        }

        // 2. Parse Heart BPM
        const heartMatch = line.match(/Heart BPM:\s*([\d.]+)/);
        if (heartMatch) {
            sensorData.heartBPM = parseFloat(heartMatch[1]);
        }
    });
}

// Start the sensor listener
connectToSensor();

// --- ROUTES ---

app.get('/sensor/live', (req, res) => {
    res.json(sensorData);
});

// Compatibility route
app.get('/sensor/light', (req, res) => {
    res.json(sensorData);
});

// --- DUAL SAVE ENDPOINT ---
app.post('/save', (req, res) => {
    const { name, surname, id, fullData } = req.body;

    if (!name || !surname || !id) {
        return res.status(400).json({ error: "Missing identity fields" });
    }

    // --- 1. PREPARE PATHS ---
    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    
    // Filename for App State (e.g., John_Doe_123.json)
    const appFilename = `${safeName}_${safeSurname}_${id}.json`;
    const appFilePath = path.join(DATA_DIR, appFilename);

    // Filename for ML Data (e.g., John_Doe_123_ML.json)
    const mlFilename = `${safeName}_${safeSurname}_${id}_ML.json`;
    const mlFilePath = path.join(ML_DIR, mlFilename);

    // --- 2. SAVE APP STATE (So React works) ---
    fs.writeFile(appFilePath, JSON.stringify(fullData, null, 2), (err) => {
        if (err) {
            console.error("Error saving App Data:", err);
            return res.status(500).json({ error: "Failed to save app data" });
        }
        console.log(`[App Save] ${appFilename}`);

        // --- 3. GENERATE & SAVE ML FORMAT (So Model works) ---
        
        // Extract Data Helpers
        const user = fullData.user || {};
        const sliders = user.sliders || {};
        const logs = fullData.logs || {};
        const logKeys = Object.keys(logs).sort();
        const latestLog = logKeys.length > 0 ? logs[logKeys[logKeys.length - 1]] : {};
        
        // Safe Parser Helper (defaults to 0 if missing/NaN)
        const safeFloat = (val) => {
            const parsed = parseFloat(val);
            return isNaN(parsed) ? 0.0 : parsed;
        };
        const safeInt = (val) => {
            const parsed = parseInt(val);
            return isNaN(parsed) ? 0 : parsed;
        };

        // Build the Strict ML Object
        const mlData = {
            gender: (user.sex || 'female').toLowerCase(),
            
            // Hardcoded/Logic placeholders
            migraine_days_per_month: 5, 
            stress_intensity: safeInt(sliders.stress), 

            // Weather Context
            temp_mean: 15.5,
            wind_mean: 10.2,
            pressure_mean: 1012.5,
            sun_irr_mean: 200.0,
            sun_time_mean: 8.5,
            precip_total: 0.0,
            cloud_mean: 30.0,

            // User Daily Logs (Normalized)
            step_count_normalized: safeFloat((latestLog.steps || 0) / 10), 
            mood_score: safeFloat(latestLog.mood || 0),

            // Live Sensor Data (Brightness Normalized 0-1)
            screen_brightness_normalized: safeFloat((sensorData.percent / 100).toFixed(2)),

            // Sliders (Normalized 0-1)
            stress: safeFloat((sliders.stress || 0) / 10),
            hormonal: safeFloat((sliders.hormonal || 0) / 10),
            sleep: safeFloat((sliders.sleep || 0) / 10),
            weather: safeFloat((sliders.weather || 0) / 10),
            food: safeFloat((sliders.food || 0) / 10),
            sensory: safeFloat((sliders.sensory || 0) / 10),
            physical: safeFloat((sliders.physical || 0) / 10),

            // History logic
            consecutive_migraine_days: 0,
            days_since_last_migraine: 5
        };

        // Write the ML File
        fs.writeFile(mlFilePath, JSON.stringify(mlData, null, 2), (mlErr) => {
            if (mlErr) {
                console.error("Error saving ML Data:", mlErr);
            } else {
                console.log(`[ML Save] ${mlFilename}`);
            }
            
            // Respond Success to Frontend
            res.json({ success: true, filename: appFilename });
        });
    });
});

app.post('/save-triggers', (req, res) => {
    const { name, surname, id, triggerLogs } = req.body;
    const callerId = req.header('X-User-Id');

    if (!name || !surname || !id) return res.status(400).json({ error: "Missing identity fields" });
    if (!callerId || callerId !== id) return res.status(403).json({ error: "Forbidden" });

    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}_triggers.json`;
    const filePath = path.join(DATA_DIR, filename); // Saves to userdata folder

    const incoming = Array.isArray(triggerLogs) ? triggerLogs : [triggerLogs];
    const annotated = incoming.map((entry) => ({ ...entry, serverAddedAt: new Date().toISOString() }));

    fs.readFile(filePath, 'utf8', (readErr, data) => {
        let storage = { meta: { name, surname, id, createdAt: new Date().toISOString() }, logs: [] };
        if (!readErr) {
            try {
                const parsed = JSON.parse(data);
                if (Array.isArray(parsed)) storage.logs = parsed;
                else if (parsed && parsed.logs) storage = parsed;
            } catch (e) {}
        }
        storage.logs = storage.logs.concat(annotated);
        fs.writeFile(filePath, JSON.stringify(storage, null, 2), (writeErr) => {
            if (writeErr) return res.status(500).json({ error: "Failed to save trigger logs" });
            res.json({ success: true, filename, recordCount: storage.logs.length, storage });
        });
    });
});

app.get('/get-triggers/:name/:surname/:id', (req, res) => {
    const { name, surname, id } = req.params;
    const callerId = req.header('X-User-Id');
    if (!callerId || callerId !== id) return res.status(403).json({ error: "Forbidden" });

    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}_triggers.json`;
    const filePath = path.join(DATA_DIR, filename);

    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) return res.json({ triggerLogs: [] });
        try {
            const parsed = JSON.parse(data);
            if (Array.isArray(parsed)) return res.json({ triggerLogs: parsed });
            if (parsed && parsed.logs) return res.json({ triggerLogs: parsed.logs, meta: parsed.meta });
            return res.json({ triggerLogs: [] });
        } catch (parseErr) { res.status(500).json({ error: "Invalid format" }); }
    });
});

app.post('/login', (req, res) => {
    const { username } = req.body;
    if (!username) return res.status(400).json({ error: 'Missing username' });
    const search = username.replace(/[^a-z0-9 ]/gi, '').toLowerCase();

    fs.readdir(DATA_DIR, (err, files) => {
        if (err) return res.status(500).json({ error: 'Failed to read data directory' });
        const candidates = files.filter(f => !f.endsWith('_triggers.json')); // Ignore trigger files
        const reads = candidates.map(f => new Promise((resolve) => {
            const p = path.join(DATA_DIR, f);
            fs.readFile(p, 'utf8', (rErr, data) => {
                if (rErr) return resolve(null);
                try {
                    const parsed = JSON.parse(data);
                    if (parsed && parsed.user && (parsed.user.name || parsed.user.surname)) {
                        const name = (parsed.user.name || '').toString().toLowerCase();
                        const surname = (parsed.user.surname || '').toString().toLowerCase();
                        const combined = `${name} ${surname}`.trim();
                        if (name === search || surname === search || combined === search || combined.includes(search)) {
                            return resolve(parsed);
                        }
                    }
                } catch (e) {}
                resolve(null);
            });
        }));
        Promise.all(reads).then(results => {
            res.json({ matches: results.filter(Boolean) });
        }).catch((e) => res.status(500).json({ error: 'Search failed' }));
    });
});

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`);
    console.log(`  > App Data: ${DATA_DIR}`);
    console.log(`  > ML Data:  ${ML_DIR}`);
    console.log(`  > Sensors:  ${SENSOR_PORT}`);
});