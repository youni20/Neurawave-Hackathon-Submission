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

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`);
    console.log(`Saving user files to: ${DATA_DIR}`);
});