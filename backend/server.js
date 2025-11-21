import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const app = express();
const PORT = 3001;

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

app.post('/save', (req, res) => {
    // We expect: { name, surname, id, fullData }
    const { name, surname, id, fullData } = req.body;

    if (!name || !surname || !id) {
        return res.status(400).json({ error: "Missing identity fields" });
    }

    // 2. Create the specific filename: Name_Surname_ID.json
    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}.json`;
    const filePath = path.join(DATA_DIR, filename);

    // 3. Write the file
    fs.writeFile(filePath, JSON.stringify(fullData, null, 2), (err) => {
        if (err) {
            console.error("Error saving file:", err);
            return res.status(500).json({ error: "Failed to save" });
        }
        console.log(`[Saved] Updated file: ${filename}`);
        res.json({ success: true, filename });
    });
});

// NEW: Endpoint for saving trigger logs (AI Insights feature)
app.post('/save-triggers', (req, res) => {
    // We expect: { name, surname, id, triggerLogs }
    const { name, surname, id, triggerLogs } = req.body;

    // Simple access control: require caller to present matching user id in header
    // so they can only access their own logs. Header required: X-User-Id
    const callerId = req.header('X-User-Id');

    if (!name || !surname || !id) {
        return res.status(400).json({ error: "Missing identity fields" });
    }

    if (!callerId || callerId !== id) {
        return res.status(403).json({ error: "Forbidden: caller id does not match target user id" });
    }

    // Create a specific file for trigger logs: Name_Surname_ID_triggers.json
    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}_triggers.json`;
    const filePath = path.join(DATA_DIR, filename);

    // Normalize incoming logs to an array of entries
    const incoming = Array.isArray(triggerLogs) ? triggerLogs : [triggerLogs];

    // Add server-side context/timestamps to each incoming entry
    const annotated = incoming.map((entry) => ({
        ...entry,
        serverAddedAt: new Date().toISOString(),
    }));

    // Read existing file (if any), append new entries, and write back a structured JSON
    fs.readFile(filePath, 'utf8', (readErr, data) => {
        let storage = { meta: { name, surname, id, createdAt: new Date().toISOString() }, logs: [] };

        if (!readErr) {
            try {
                const parsed = JSON.parse(data);
                // If previous file uses plain array format, support that too
                if (Array.isArray(parsed)) {
                    storage.logs = parsed;
                } else if (parsed && parsed.logs) {
                    storage = parsed;
                }
            } catch (e) {
                // If parse fails, start fresh but keep a backup
                console.error("Error parsing existing trigger file, creating fresh storage:", e);
            }
        }

        // Append annotated entries
        storage.logs = storage.logs.concat(annotated);

        // Ensure meta.createdAt remains the original if already present
        if (!storage.meta || !storage.meta.createdAt) {
            storage.meta = { name, surname, id, createdAt: new Date().toISOString() };
        }

        fs.writeFile(filePath, JSON.stringify(storage, null, 2), (writeErr) => {
            if (writeErr) {
                console.error("Error saving trigger logs:", writeErr);
                return res.status(500).json({ error: "Failed to save trigger logs" });
            }
            console.log(`[Saved] Updated trigger file: ${filename} (records: ${storage.logs.length})`);
            res.json({ success: true, filename, recordCount: storage.logs.length, storage });
        });
    });
});

// NEW: Endpoint for retrieving trigger logs
app.get('/get-triggers/:name/:surname/:id', (req, res) => {
    const { name, surname, id } = req.params;

    // Simple access control: require caller to present matching user id in header
    const callerId = req.header('X-User-Id');
    if (!callerId || callerId !== id) {
        return res.status(403).json({ error: "Forbidden: caller id does not match target user id" });
    }

    // Sanitize parameters
    const safeName = name.replace(/[^a-z0-9]/gi, '');
    const safeSurname = surname.replace(/[^a-z0-9]/gi, '');
    const filename = `${safeName}_${safeSurname}_${id}_triggers.json`;
    const filePath = path.join(DATA_DIR, filename);

    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            if (err.code === 'ENOENT') {
                return res.json({ triggerLogs: [] });
            }
            console.error("Error reading trigger logs:", err);
            return res.status(500).json({ error: "Failed to read trigger logs" });
        }

        try {
            const parsed = JSON.parse(data);
            // Support both old array format and new structured format
            if (Array.isArray(parsed)) {
                return res.json({ triggerLogs: parsed });
            }
            if (parsed && parsed.logs) {
                return res.json({ triggerLogs: parsed.logs, meta: parsed.meta });
            }

            // Unknown format fallback
            return res.json({ triggerLogs: [] });
        } catch (parseErr) {
            console.error("Error parsing trigger logs:", parseErr);
            res.status(500).json({ error: "Invalid trigger logs format" });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`);
    console.log(`Saving user files to: ${DATA_DIR}`);
});

// Simple login endpoint: find saved fullData files matching a provided username
app.post('/login', (req, res) => {
    const { username } = req.body;
    if (!username) return res.status(400).json({ error: 'Missing username' });

    const search = username.replace(/[^a-z0-9 ]/gi, '').toLowerCase();

    fs.readdir(DATA_DIR, (err, files) => {
        if (err) {
            console.error('Error reading data dir for login:', err);
            return res.status(500).json({ error: 'Failed to read data directory' });
        }

        const candidates = files.filter(f => !f.endsWith('_triggers.json'));
        const reads = candidates.map(f => new Promise((resolve) => {
            const p = path.join(DATA_DIR, f);
            fs.readFile(p, 'utf8', (rErr, data) => {
                if (rErr) return resolve(null);
                try {
                    const parsed = JSON.parse(data);
                    // Accept file if parsed has user information
                    if (parsed && parsed.user && (parsed.user.name || parsed.user.surname)) {
                        const name = (parsed.user.name || '').toString().toLowerCase();
                        const surname = (parsed.user.surname || '').toString().toLowerCase();
                        const combined = `${name} ${surname}`.trim();
                        if (name === search || surname === search || combined === search || combined.includes(search)) {
                            // also attempt to load triggers file for this user
                            const safeName = (parsed.user.name || '').toString().replace(/[^a-z0-9]/gi, '');
                            const safeSurname = (parsed.user.surname || '').toString().replace(/[^a-z0-9]/gi, '');
                            const triggerFilename = `${safeName}_${safeSurname}_${parsed.user.id}_triggers.json`;
                            const triggerPath = path.join(DATA_DIR, triggerFilename);
                            fs.readFile(triggerPath, 'utf8', (tErr, tData) => {
                                if (!tErr) {
                                    try {
                                        const tParsed = JSON.parse(tData);
                                        // if structured format
                                        parsed.triggerLogs = tParsed && tParsed.logs ? tParsed.logs : (Array.isArray(tParsed) ? tParsed : []);
                                    } catch (e) {
                                        parsed.triggerLogs = [];
                                    }
                                } else {
                                    parsed.triggerLogs = [];
                                }
                                return resolve(parsed);
                            });
                            return;
                        }
                    }
                } catch (e) {
                    // ignore parse errors
                }
                resolve(null);
            });
        }));

        Promise.all(reads).then(results => {
            const matches = results.filter(Boolean);
            return res.json({ matches });
        }).catch((e) => {
            console.error('Login search error:', e);
            res.status(500).json({ error: 'Search failed' });
        });
    });
});