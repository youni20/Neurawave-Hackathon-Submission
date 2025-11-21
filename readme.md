# Neurawave Hackathon Submission

Health monitoring and migraine management platform with AI-powered insights and wellness tools.

## 🏗️ Project Structure

```
NeurawaveHackathon/
├── frontend/                 # React + Vite web application
│   ├── src/
│   │   ├── App.jsx          # Main app with page routing
│   │   ├── Navigation.jsx   # Navigation component
│   │   ├── DoctorPage.jsx   # AI voice assistant
│   │   ├── MusicPage.jsx    # Sonic therapy
│   │   ├── WeatherPage.jsx  # Symptom triggers
│   │   └── *.css            # Styling
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
├── backend/                  # Express.js server
│   ├── server.js            # API server (port 3001)
│   ├── package.json
│   └── userdata/            # User data storage
│
├── data/                     # ML & Data Processing
│   ├── migraine_model/      # XGBoost classifier
│   ├── synthetic_data_100_000/
│   ├── cleaned_data/
│   ├── requirements.txt
│   └── *.py                 # Python scripts
│
└── package.json             # Root config
```

## 🚀 Getting Started

### Prerequisites
- **Node.js** v18+ with npm
- **Python** 3.8+ with pip

### Installation (5 minutes)

```bash
# 1. Navigate to project
cd NeurawaveHackathon

# 2. Install root dependencies
npm install

# 3. Install frontend dependencies
cd frontend
npm install
cd ..

# 4. Install backend dependencies
cd backend
npm install
cd ..

# 5. Install Python dependencies (optional - for ML features)
pip install -r data/requirements.txt
```

## ▶️ Running the App

### Option 1: Run Everything Together (Recommended)

From the project root:
```bash
npm run dev
```

This starts both frontend and backend concurrently:
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:3001

### Option 2: Run Separately

**Terminal 1 - Frontend:**
```bash
cd frontend
npm run dev
```
Opens at http://localhost:5173

**Terminal 2 - Backend:**
```bash
cd backend
npm run dev
```
Runs at http://localhost:3001

**Terminal 3 - Python ML (Optional):**
```bash
cd data
python train_migraine_model.py
```

## 📱 Features & Navigation

The app has 4 main sections accessible from the navigation menu:

### 1. **Dashboard** 🏠
- Health profile setup
- Daily symptom tracking
- Risk assessment
- Daily check-in form

### 2. **Symptom Triggers** ☁️
- Weather analysis (temperature, pressure, humidity, wind)
- Migraine trigger indicators
- Personalized recommendations
- Real-time trigger risk assessment

### 3. **Dr. Neura** 🩺
- AI voice assistant powered by Web Speech API
- Text-to-speech responses
- Conversation history
- Health advice and symptom guidance
- Voice input & playback

### 4. **Sonic Therapy** 🎵
- 4 therapeutic audio modes:
  - **Ambient**: 432 Hz healing frequency
  - **Binaural Beats**: 40 Hz brain entrainment
  - **Nature Sounds**: 250 Hz relaxation
  - **Meditation**: 174 Hz deep peace
- Volume and tempo (BPM) controls
- Audio visualizer with animations
- Session duration recommendations

## 📱 Responsive Design

The app works seamlessly on:
- **📱 Mobile** (< 480px) - Hamburger menu, full-screen views
- **📱 Tablet** (480-768px) - Responsive grids, touch-friendly
- **💻 Desktop** (> 768px) - Full navigation bar, multi-column layouts

## 🛠️ NPM Scripts

### From Root Directory

```bash
npm run dev              # Run frontend + backend together ⭐
npm run dev:frontend    # Frontend only (Vite dev server)
npm run dev:backend     # Backend only (Express server)
npm run build           # Build frontend for production
npm run lint            # Lint frontend code with ESLint
npm run preview         # Preview production build
```

### From Frontend Directory

```bash
cd frontend
npm run dev             # Start Vite dev server
npm run build           # Build for production
npm run preview         # Preview production build
npm run lint            # Run ESLint
```

### From Backend Directory

```bash
cd backend
npm run start           # Start Express server
npm run dev            # Same as start
```

## ⚙️ Configuration

### Frontend Config
- **Vite**: `frontend/vite.config.js` - Bundler settings
- **Tailwind**: `frontend/tailwind.config.js` - CSS framework
- **ESLint**: `frontend/eslint.config.js` - Code linting

### Backend Config
- **Server Port**: 3001
- **User Data Dir**: `backend/userdata/` (auto-created)
- **CORS**: Enabled for frontend requests

### API Endpoint
- **Save Data**: `POST http://localhost:3001/save`
- Format: `{ name, surname, id, fullData }`

## 🔗 API Endpoints

### POST `/save`
Save user health data to server.

**Request:**
```json
{
  "name": "John",
  "surname": "Doe",
  "id": "12345",
  "fullData": {
    "user": { "profile": "data" },
    "logs": { "2025-11-21": "daily_data" }
  }
}
```

**Response:**
```json
{
  "success": true,
  "filename": "john_doe_12345.json"
}
```

## 🧠 ML Features

### Migraine Risk Model
XGBoost model predicts migraine risk based on:
- Stress levels (baseline + daily)
- Sleep quality and hours
- Weather sensitivity & barometric pressure
- Hormonal factors
- Sensory sensitivity

**Location**: `data/migraine_model/`

**Training**:
```bash
cd data
python train_migraine_model.py
```

## 🌐 Browser Support

- Chrome/Edge (Chromium) ✅
- Firefox ✅
- Safari ✅
- Mobile browsers ✅

## 📝 Technology Stack

### Frontend
- **React** 19.2.0 - UI framework
- **Vite** 7.2.4 - Bundler & dev server
- **Tailwind CSS** 3.4.17 - Styling
- **Framer Motion** 12.23.24 - Animations
- **Lucide React** - Icon library
- **Web Speech API** - Voice recognition

### Backend
- **Express** 5.1.0 - Web framework
- **CORS** - Cross-origin support
- **Body Parser** - JSON parsing

### ML/Data
- **XGBoost** - Classification model
- **Python** 3.8+ - Data processing
- **Pandas, NumPy, Scikit-learn** - Data science

## 🚀 Building for Production

```bash
# Build frontend
npm run build

# Output location
frontend/dist/

# Deploy dist/ folder to hosting (Vercel, Netlify, etc.)
# Deploy backend to cloud (Heroku, AWS, Azure, etc.)
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Windows: Kill process on port
netstat -ano | findstr :3001
taskkill /PID [pid] /F

# Mac/Linux
lsof -i :3001
kill -9 [pid]
```

### Dependencies Not Installing
```bash
# Clear npm cache and reinstall
npm cache clean --force
rm -r node_modules frontend/node_modules backend/node_modules
npm install
cd frontend && npm install && cd ..
cd backend && npm install && cd ..
```

### Vite Not Starting
```bash
# Clear Vite cache
rm -r frontend/.vite
npm run dev
```

### Voice API Not Working
- Ensure you have microphone permissions granted
- Voice API requires HTTPS in production (localhost works)
- Test in Chrome/Edge first
- Check browser console for errors

### Backend Not Connecting
- Ensure backend is running on port 3001
- Check if `backend/userdata/` folder exists
- Verify CORS is enabled in `backend/server.js`

## 📂 File Locations Reference

| Component | Path |
|-----------|------|
| Main App | `frontend/src/App.jsx` |
| Navigation | `frontend/src/Navigation.jsx` |
| Doctor AI | `frontend/src/DoctorPage.jsx` |
| Music Therapy | `frontend/src/MusicPage.jsx` |
| Weather Triggers | `frontend/src/WeatherPage.jsx` |
| Backend API | `backend/server.js` |
| ML Model | `data/migraine_model/` |
| Styling | `frontend/src/index.css` |

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Make changes and test
3. Commit: `git commit -m "Add feature"`
4. Push: `git push origin feature/name`
5. Open pull request

## 📄 License

See LICENSE file for details.

---

**Quick Start Summary:**
```bash
# Install
npm install && cd frontend && npm install && cd .. && cd backend && npm install && cd ..

# Run
npm run dev

# Visit
http://localhost:5173
```

**Happy coding! 🎉**
