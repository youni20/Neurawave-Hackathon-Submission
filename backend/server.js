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
    heartBPM: 0, // NEW FIELD
    status: "Disconnected"
};

// specific folder path
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, 'userdata');

app.use(cors());
app.use(bodyParser.json());

// 1. Create 'userdata' folder automatically if it doesn't exist
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR);
    console.log(`[System] Created folder: ${DATA_DIR}`);
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
        // Format: "Light Raw: 416  Light %: 87.7%  Heart BPM: 96.0"
        
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

        // Optional logging (uncomment to debug)
        // console.log(`[Sensors] Light: ${sensorData.percent}%, Heart: ${sensorData.heartBPM} BPM`);
    });
}

// Start the sensor listener
connectToSensor();

// --- ROUTES ---

// NEW: Endpoint for the frontend to get ALL sensor data
app.get('/sensor/live', (req, res) => {
    res.json(sensorData);
});

// Keep this for backward compatibility if needed
app.get('/sensor/light', (req, res) => {
    res.json(sensorData);
});

app.post('/save', (req, res) => {
    const { name, surname, id, fullData } = req.body;

    if (!name || !surname || !id) {
        return res.status(400).json({ error: "Missing identity fields" });
    }

    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}.json`;
    const filePath = path.join(DATA_DIR, filename);

    fs.writeFile(filePath, JSON.stringify(fullData, null, 2), (err) => {
        if (err) {
            console.error("Error saving file:", err);
            return res.status(500).json({ error: "Failed to save" });
        }
        console.log(`[Saved] Updated file: ${filename}`);
        res.json({ success: true, filename });
    });
});

app.post('/save-triggers', (req, res) => {
    const { name, surname, id, triggerLogs } = req.body;
    const callerId = req.header('X-User-Id');

    if (!name || !surname || !id) {
        return res.status(400).json({ error: "Missing identity fields" });
    }

    if (!callerId || callerId !== id) {
        return res.status(403).json({ error: "Forbidden" });
    }

    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}_triggers.json`;
    const filePath = path.join(DATA_DIR, filename);

    const incoming = Array.isArray(triggerLogs) ? triggerLogs : [triggerLogs];
    const annotated = incoming.map((entry) => ({
        ...entry,
        serverAddedAt: new Date().toISOString(),
    }));

    fs.readFile(filePath, 'utf8', (readErr, data) => {
        let storage = { meta: { name, surname, id, createdAt: new Date().toISOString() }, logs: [] };

        if (!readErr) {
            try {
                const parsed = JSON.parse(data);
                if (Array.isArray(parsed)) {
                    storage.logs = parsed;
                } else if (parsed && parsed.logs) {
                    storage = parsed;
                }
            } catch (e) {
                console.error("Error parsing existing trigger file", e);
            }
        }

        storage.logs = storage.logs.concat(annotated);

        fs.writeFile(filePath, JSON.stringify(storage, null, 2), (writeErr) => {
            if (writeErr) {
                console.error("Error saving trigger logs:", writeErr);
                return res.status(500).json({ error: "Failed to save trigger logs" });
            }
            console.log(`[Saved] Updated trigger file: ${filename}`);
            res.json({ success: true, filename, recordCount: storage.logs.length, storage });
        });
    });
});

app.get('/get-triggers/:name/:surname/:id', (req, res) => {
    const { name, surname, id } = req.params;
    const callerId = req.header('X-User-Id');
    
    if (!callerId || callerId !== id) {
        return res.status(403).json({ error: "Forbidden" });
    }

    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}_triggers.json`;
    const filePath = path.join(DATA_DIR, filename);

    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            if (err.code === 'ENOENT') return res.json({ triggerLogs: [] });
            return res.status(500).json({ error: "Failed to read trigger logs" });
        }
        try {
            const parsed = JSON.parse(data);
            if (Array.isArray(parsed)) return res.json({ triggerLogs: parsed });
            if (parsed && parsed.logs) return res.json({ triggerLogs: parsed.logs, meta: parsed.meta });
            return res.json({ triggerLogs: [] });
        } catch (parseErr) {
            res.status(500).json({ error: "Invalid trigger logs format" });
        }
    });
});

app.post('/login', (req, res) => {
    const { username } = req.body;
    if (!username) return res.status(400).json({ error: 'Missing username' });

    const search = username.replace(/[^a-z0-9 ]/gi, '').toLowerCase();

    fs.readdir(DATA_DIR, (err, files) => {
        if (err) return res.status(500).json({ error: 'Failed to read data directory' });

        const candidates = files.filter(f => !f.endsWith('_triggers.json'));
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
                            const safeName = (parsed.user.name || '').toString().replace(/[^a-z0-9]/gi, '');
                            const safeSurname = (parsed.user.surname || '').toString().replace(/[^a-z0-9]/gi, '');
                            const triggerPath = path.join(DATA_DIR, `${safeName}_${safeSurname}_${parsed.user.id}_triggers.json`);
                            fs.readFile(triggerPath, 'utf8', (tErr, tData) => {
                                try { 
                                    const tParsed = JSON.parse(tData);
                                    parsed.triggerLogs = tParsed.logs || (Array.isArray(tParsed) ? tParsed : []);
                                } catch { parsed.triggerLogs = []; }
                                return resolve(parsed);
                            });
                            return;
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
    console.log(`Saving user files to: ${DATA_DIR}`);
    console.log(`Monitoring Sensors on: ${SENSOR_PORT}`);
});