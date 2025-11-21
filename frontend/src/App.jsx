import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Wind, Sun, Brain, Moon, CloudRain, Settings, 
  ChevronRight, ChevronLeft, Check, Zap, Save, 
  Thermometer, Activity, AlertCircle
} from 'lucide-react';
import Navigation from './Navigation';
import DoctorPage from './DoctorPage';
import MusicPage from './MusicPage';
import WeatherPage from './WeatherPage';
import TriggerTracker from './TriggerTracker';
import AIInsights from './AIInsights';

// --- CONFIGURATION & UTILS ---

const CONSTANTS = {
  WEATHER_KALMAR: { temp: 14, condition: 'Overcast', pressure: 1012, humidity: 82 },
  API_URL: 'http://localhost:3001/save'
};

// --- DATA MANAGER (Handles Logic & Saving) ---

class DataManager {
  static STORAGE_KEY = 'neuraflow_full_data';

  // 1. Sync to Server with specific Naming Convention
  static async syncToServer(fullData) {
    const { user } = fullData;
    if (!user || !user.name || !user.surname || !user.id) return;

    try {
      await fetch(CONSTANTS.API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: user.name,
          surname: user.surname,
          id: user.id,
          fullData: fullData
        })
      });
      console.log("Synced to server successfully.");
    } catch (e) {
      console.warn("Server offline. Data saved to LocalStorage only.");
    }
  }

  static loadData() {
    const data = localStorage.getItem(this.STORAGE_KEY);
    return data ? JSON.parse(data) : null;
  }

  static saveData(fullData) {
    // Save Local
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(fullData));
    // Save Server
    this.syncToServer(fullData);
  }

  static generateId() {
    return Math.floor(Math.random() * 1000000).toString();
  }

  // --- THE V3 ALGORITHM LOGIC ---
  static calculateRisk(user, daily) {
    if (!user || !daily) return null;

    // Normalize inputs (0-10 -> 0.0-1.0)
    const baseStress = user.sliders.stress / 10;
    const baseHormonal = user.sliders.hormonal / 10;
    const baseWeatherSens = user.sliders.weather / 10;
    const baseSleep = user.sliders.sleep / 10; // Base sleep quality
    
    const dailyStress = daily.stress / 10;
    const dailySleepHours = daily.sleep; // 0-9
    const dailyFocus = daily.focus / 10;

    // 1. Migraine Risk Calculation
    // Factors: Stress (Base+Daily), Sleep Debt (Inverse of hours), Weather Pressure + Sensitivity
    const sleepDebt = (9 - dailySleepHours) / 9; // High if sleep is low
    const weatherImpact = (baseWeatherSens * 0.8); // Simulated pressure impact
    
    let migraineRaw = (dailyStress * 0.35) + (baseStress * 0.15) + (sleepDebt * 0.25) + (weatherImpact * 0.15) + (baseHormonal * 0.1);
    const migrainePct = Math.min(Math.round(migraineRaw * 100), 99);

    // 2. Sleep Quality (V3 Requirement)
    let sleepQual = 'Medium';
    let sleepScoreColor = 'text-yellow-500';
    if (dailySleepHours <= 3) { sleepQual = 'Bad'; sleepScoreColor = 'text-red-500'; }
    if (dailySleepHours >= 7) { sleepQual = 'Good'; sleepScoreColor = 'text-green-500'; }

    // 3. ADHD Risk (V3 Requirement)
    // High Risk if: Low Focus + High Sensory Sensitivity
    const sensorySens = user.sliders.sensory / 10;
    const adhdScore = ((1 - dailyFocus) * 0.6) + (sensorySens * 0.4);
    let adhdLabel = adhdScore > 0.65 ? 'High' : adhdScore > 0.35 ? 'Medium' : 'Low';

    // 4. Anxiety Risk (V3 Requirement)
    // High Risk if: High Stress + High Weather Sensitivity
    const anxietyScore = (dailyStress * 0.7) + (weatherImpact * 0.3);
    let anxietyLabel = anxietyScore > 0.7 ? 'High' : anxietyScore > 0.4 ? 'Medium' : 'Low';

    return {
      migraine: migrainePct,
      sleep: { label: sleepQual, val: dailySleepHours, color: sleepScoreColor },
      adhd: adhdLabel,
      anxiety: anxietyLabel,
      raw: { migraineRaw, adhdScore, anxietyScore }
    };
  }

  static getAIRecommendation(metrics) {
    const { migraine, sleep, anxiety, adhd } = metrics;
    
    if (migraine > 75) return "URGENT: Your biometrics indicate a very high risk of migraine onset. Triggers: High stress & Sleep debt. Action: Take acute medication if prescribed, dim lights, and use the Neurawave device for 20 minutes immediately.";
    if (anxiety === 'High') return "High Anxiety detected. The current barometric pressure in Kalmar is likely amplifying your stress response. Action: Perform 4-7-8 breathing exercises for 5 minutes to stimulate the Vagus nerve.";
    if (adhd === 'High') return "Focus levels are critical. Your sensory processing sensitivity is peaking. Action: Reduce environmental noise, put on headphones, and attempt a 'Focus Session' with the app.";
    if (sleep.label === 'Bad') return "Severe sleep deprivation detected. This lowers your migraine threshold significantly. Action: Prioritize hydration and do not consume caffeine after 12:00 PM.";
    
    return "You are in the green zone. Your neural stability is high. Keep up your current routine of hydration and movement.";
  }
}

// --- UI COMPONENTS ---

const Slider = ({ label, value, onChange, max = 10, min = 0, helpText = "" }) => (
  <div className="mb-4">
    <div className="flex justify-between text-sm font-medium text-slate-700 mb-1">
      <span>{label}</span>
      <span className="text-teal-600 font-bold">{value}</span>
    </div>
    <input 
      type="range" min={min} max={max} value={value} 
      onChange={(e) => onChange(parseInt(e.target.value))}
      className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-500"
    />
    {helpText && <p className="text-xs text-slate-400 mt-1">{helpText}</p>}
  </div>
);

const Card = ({ children, className = "" }) => (
  <div className={`bg-white/90 backdrop-blur-sm border border-white/50 shadow-sm rounded-2xl p-5 ${className}`}>
    {children}
  </div>
);

const Button = ({ children, onClick, className = "" }) => (
  <button onClick={onClick} className={`bg-gradient-to-r from-teal-500 to-blue-600 text-white px-6 py-3 rounded-xl font-semibold shadow-md active:scale-95 transition-transform ${className}`}>
    {children}
  </button>
);

// --- 1. ONBOARDING (The Questionnaire V1) ---

const Onboarding = ({ onComplete }) => {
  const [step, setStep] = useState(0);
  const [data, setData] = useState({
    name: '', surname: '', dob: '', sex: '',
    history: [], // Q3
    sliders: { // Q4
      stress: 5, hormonal: 5, sleep: 5, weather: 5, 
      food: 5, sensory: 5, physical: 5
    }
  });

  const toggleHistory = (item) => {
    setData(prev => ({
      ...prev,
      history: prev.history.includes(item) ? prev.history.filter(i => i !== item) : [...prev.history, item]
    }));
  };

  const updateSlider = (key, val) => setData(p => ({...p, sliders: {...p.sliders, [key]: val}}));

  const finish = () => {
    // Create unique ID and save
    const finalUser = { ...data, id: DataManager.generateId() };
    onComplete(finalUser);
  };

  const steps = [
    // Q1: Animation
    <div className="text-center space-y-6 py-10">
      <motion.div animate={{ rotate: 360 }} transition={{ duration: 20, repeat: Infinity }} className="mx-auto w-32 h-32 bg-gradient-to-tr from-teal-400 to-blue-500 rounded-full blur-xl opacity-60 absolute left-0 right-0" />
      <Brain size={80} className="text-slate-800 mx-auto relative z-10" />
      <h1 className="text-3xl font-bold text-slate-800">Welcome to Neurawave</h1>
      <p className="text-slate-500">Your personalized neural weather forecast.</p>
      <Button onClick={() => setStep(1)} className="w-full mt-8">Start Setup</Button>
    </div>,

    // Q2: Static Data
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-slate-800">About You</h2>
      <input placeholder="First Name" className="w-full p-3 rounded-xl border bg-white" onChange={e => setData({...data, name: e.target.value})} />
      <input placeholder="Last Name" className="w-full p-3 rounded-xl border bg-white" onChange={e => setData({...data, surname: e.target.value})} />
      <input type="date" className="w-full p-3 rounded-xl border bg-white" onChange={e => setData({...data, dob: e.target.value})} />
      <select className="w-full p-3 rounded-xl border bg-white" onChange={e => setData({...data, sex: e.target.value})}>
        <option>Biological Sex</option><option>Male</option><option>Female</option><option>Other</option>
      </select>
      <Button onClick={() => setStep(2)} className="w-full">Next</Button>
    </div>,

    // Q3: Medical History (Visual)
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-slate-800">Medical History</h2>
      <p className="text-sm text-slate-500">Select all that apply (Private)</p>
      <div className="grid grid-cols-2 gap-3">
        {['Alcoholism', 'Painkillers', 'Migraine History', 'Epilepsy', 'Trauma', 'Sleep Disorder', 'Depression', 'Cardiovascular'].map(item => (
          <div key={item} onClick={() => toggleHistory(item)} 
            className={`p-3 text-sm rounded-xl border cursor-pointer transition-colors ${data.history.includes(item) ? 'bg-teal-100 border-teal-500 text-teal-800' : 'bg-white border-slate-200'}`}>
            {item}
          </div>
        ))}
      </div>
      <div className="flex gap-3">
        <button onClick={() => setStep(1)} className="px-4 py-3 text-slate-400">Back</button>
        <Button onClick={() => setStep(3)} className="flex-1">Next</Button>
      </div>
    </div>,

    // Q4: Sliders
    <div className="space-y-2">
      <h2 className="text-2xl font-bold text-slate-800">Baseline Sensitivity</h2>
      <div className="h-[60vh] overflow-y-auto pr-2">
        {Object.keys(data.sliders).map(key => (
          <Slider key={key} label={key.charAt(0).toUpperCase() + key.slice(1)} value={data.sliders[key]} onChange={(v) => updateSlider(key, v)} />
        ))}
      </div>
      <div className="flex gap-3 pt-4">
        <button onClick={() => setStep(2)} className="px-4 py-3 text-slate-400">Back</button>
        <Button onClick={() => setStep(4)} className="flex-1">Next</Button>
      </div>
    </div>,

    // Q5: Review
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">Confirm Data</h2>
      <Card className="space-y-2 text-sm">
        <p><strong>Name:</strong> {data.name} {data.surname}</p>
        <p><strong>Dob:</strong> {data.dob}</p>
        <p><strong>History:</strong> {data.history.length > 0 ? data.history.join(', ') : 'None'}</p>
        <p><strong>Baseline Factors:</strong> {Object.keys(data.sliders).length} Set</p>
      </Card>
      <div className="flex gap-3">
        <button onClick={() => setStep(3)} className="px-4 py-3 text-slate-400">Back</button>
        <Button onClick={finish} className="flex-1">Submit</Button>
      </div>
    </div>
  ];

  return (
    <div className="min-h-screen bg-slate-50 p-6 flex flex-col justify-center max-w-md mx-auto">
      <AnimatePresence mode="wait">
        <motion.div key={step} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
          {steps[step]}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

// --- 2. DAILY CHECK-IN (V1 Phase 2 + V3 Additions) ---

const DailyCheckIn = ({ onComplete }) => {
  const [vals, setVals] = useState({
    stress: 5, mood: 5, energy: 5, focus: 5, // V1/V2
    steps: 5, // V2
    sleep: 7 // V3 Addition
  });

  const submit = () => onComplete(vals);

  return (
    <div className="min-h-screen bg-slate-50 p-6 flex flex-col justify-center max-w-md mx-auto">
      <h1 className="text-3xl font-bold text-slate-800 mb-6">Daily Check-in</h1>
      <div className="space-y-1 overflow-y-auto max-h-[70vh] pr-2">
        <Slider label="Daily Stress" value={vals.stress} onChange={v => setVals({...vals, stress: v})} />
        <Slider label="Mood" value={vals.mood} onChange={v => setVals({...vals, mood: v})} />
        <Slider label="Energy Level" value={vals.energy} onChange={v => setVals({...vals, energy: v})} />
        <Slider label="Focus Level" value={vals.focus} onChange={v => setVals({...vals, focus: v})} />
        <Slider label="Sleep Duration (Hours)" value={vals.sleep} max={9} onChange={v => setVals({...vals, sleep: v})} />
        <Slider label="Step Count" value={vals.steps} max={10} min={1} helpText={`${vals.steps * 1000} steps estimated`} onChange={v => setVals({...vals, steps: v})} />
      </div>
      <Button onClick={submit} className="w-full mt-6">Generate Forecast</Button>
    </div>
  );
};

// --- 3. DASHBOARD (V1 Phase 3 + V2/V3 Additions) ---

const Dashboard = ({ fullData, onEditSettings }) => {
  const [view, setView] = useState('home');
  const [metrics, setMetrics] = useState(null);
  const [brightness, setBrightness] = useState('50%');

  // Get today's date key YYYY-MM-DD
  const todayKey = new Date().toISOString().split('T')[0];
  const dailyLog = fullData.logs[todayKey];

  useEffect(() => {
    // Calculate Risk
    const riskData = DataManager.calculateRisk(fullData.user, dailyLog);
    setMetrics(riskData);
    // Mock Brightness (V2)
    setBrightness(Math.floor(Math.random() * (100 - 30) + 30) + '%');
  }, [fullData]);

  if (!metrics) return <div>Loading Algorithm...</div>;

  // RECOMMENDATION PAGE
  if (view === 'recs') {
    return (
      <div className="min-h-screen bg-slate-50 p-6">
        <div className="max-w-md mx-auto space-y-6">
          <button onClick={() => setView('home')} className="flex items-center text-slate-500 mb-4"><ChevronLeft size={20}/> Back to Dashboard</button>
          <h1 className="text-2xl font-bold text-slate-800">AI Recommendation</h1>
          
          <Card className="bg-gradient-to-br from-teal-50 to-blue-50 border-teal-200">
            <div className="flex items-center mb-2 text-teal-700 font-bold"><Brain size={20} className="mr-2"/> Analysis</div>
            <p className="text-slate-700 leading-relaxed">{DataManager.getAIRecommendation(metrics)}</p>
          </Card>

          <h3 className="font-bold text-slate-600 mt-4">Contributing Metrics</h3>
          <div className="space-y-3">
            <Card className="flex justify-between items-center"><span>Migraine Risk</span><span className="font-bold text-slate-800">{metrics.migraine}%</span></Card>
            <Card className="flex justify-between items-center"><span>Sleep Quality</span><span className={`font-bold ${metrics.sleep.color}`}>{metrics.sleep.label}</span></Card>
            <Card className="flex justify-between items-center"><span>Anxiety Index</span><span className="font-bold text-slate-800">{metrics.anxiety}</span></Card>
          </div>
        </div>
      </div>
    );
  }

  // HOME PAGE
  return (
    <div className="min-h-screen bg-slate-50 pb-12">
      {/* Header */}
      <div className="bg-white px-6 py-4 rounded-b-3xl shadow-sm mb-6">
        <div className="max-w-md mx-auto flex justify-between items-center">
          <div>
            <h1 className="font-bold text-lg text-slate-800">{fullData.user.name} {fullData.user.surname}</h1>
            <p className="text-xs text-slate-400">ID: {fullData.user.id}</p>
          </div>
          <button onClick={onEditSettings} className="p-2 bg-slate-100 rounded-full"><Settings size={20} className="text-slate-600"/></button>
        </div>
      </div>

      <div className="max-w-md mx-auto px-6 space-y-6">
        
        {/* PROBABILITY CIRCLE (V3 Logic) */}
        <div className="relative flex justify-center py-4">
          <div className="w-64 h-64 relative flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90">
              <circle cx="128" cy="128" r="110" stroke="#e2e8f0" strokeWidth="18" fill="transparent" />
              <motion.circle 
                cx="128" cy="128" r="110" 
                stroke={metrics.migraine > 60 ? '#ef4444' : metrics.migraine > 30 ? '#f59e0b' : '#14b8a6'} 
                strokeWidth="18" fill="transparent" strokeLinecap="round"
                strokeDasharray={691}
                initial={{ strokeDashoffset: 691 }}
                animate={{ strokeDashoffset: 691 - (691 * metrics.migraine / 100) }}
                transition={{ duration: 1.5 }}
              />
            </svg>
            <div className="absolute text-center">
              <span className="text-5xl font-bold text-slate-800">{metrics.migraine}%</span>
              <p className="text-sm text-slate-500 font-medium uppercase tracking-wide mt-1">Migraine Risk</p>
            </div>
          </div>
        </div>

        {/* SECONDARY GAUGES (V3) */}
        <div className="grid grid-cols-3 gap-3">
          <Card className="flex flex-col items-center p-3 py-4">
            <Moon size={20} className="text-indigo-500 mb-2"/>
            <span className="text-[10px] uppercase text-slate-400 font-bold">Sleep</span>
            <span className={`font-bold ${metrics.sleep.color}`}>{metrics.sleep.label}</span>
          </Card>
          <Card className="flex flex-col items-center p-3 py-4">
            <Zap size={20} className="text-yellow-500 mb-2"/>
            <span className="text-[10px] uppercase text-slate-400 font-bold">ADHD Risk</span>
            <span className="font-bold text-slate-700">{metrics.adhd}</span>
          </Card>
          <Card className="flex flex-col items-center p-3 py-4">
            <Activity size={20} className="text-teal-500 mb-2"/>
            <span className="text-[10px] uppercase text-slate-400 font-bold">Anxiety</span>
            <span className="font-bold text-slate-700">{metrics.anxiety}</span>
          </Card>
        </div>

        {/* WEATHER & ENV (V2) */}
        <Card className="bg-gradient-to-br from-blue-50 to-teal-50 border-none">
          <div className="flex justify-between items-center mb-4 border-b border-slate-200/50 pb-2">
            <div className="flex items-center font-bold text-slate-700"><CloudRain size={18} className="mr-2"/> Kalmar, SE</div>
            <span className="text-xs bg-white px-2 py-1 rounded-md text-slate-500 shadow-sm">Live</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div><p className="text-xs text-slate-400">Temp</p><p className="font-bold text-slate-700">{CONSTANTS.WEATHER_KALMAR.temp}°C</p></div>
            <div><p className="text-xs text-slate-400">Pressure</p><p className="font-bold text-slate-700">{CONSTANTS.WEATHER_KALMAR.pressure} hPa</p></div>
            <div><p className="text-xs text-slate-400">Light</p><p className="font-bold text-slate-700">{brightness}</p></div>
          </div>
        </Card>

        {/* HISTORY CALENDAR (V1 Phase 3) */}
        <div>
          <h3 className="font-bold text-slate-700 mb-2 ml-1">7-Day History</h3>
          <div className="grid grid-cols-7 gap-2 h-24">
            {[...Array(7)].map((_, i) => {
              // Mock data for history visualization
              const h = Math.floor(Math.random() * 80) + 10;
              const isHigh = h > 60;
              return (
                <div key={i} className="relative group h-full bg-slate-200 rounded-lg overflow-hidden flex items-end">
                  <div style={{height: `${h}%`}} className={`w-full transition-all ${isHigh ? 'bg-red-400' : 'bg-teal-400'}`}></div>
                  {/* Tooltip on Hover */}
                  <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-black/80 text-white text-xs py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
                    Risk: {h}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <Button onClick={() => setView('recs')} className="w-full py-4 flex items-center justify-center">
          Get AI Recommendation <ChevronRight size={18} className="ml-2"/>
        </Button>

      </div>
    </div>
  );
};

// --- ROOT APP ---

export default function App() {
  const [fullData, setFullData] = useState(null);
  const [mode, setMode] = useState('loading'); // onboarding | daily | dashboard
  const [currentPage, setCurrentPage] = useState('dashboard'); // dashboard | weather | doctor | music | triggers | insights

  useEffect(() => {
    const loaded = DataManager.loadData();
    const today = new Date().toISOString().split('T')[0];

    if (loaded && loaded.user) {
      setFullData(loaded);
      // Check if daily log exists for today
      if (loaded.logs && loaded.logs[today]) {
        setMode('dashboard');
      } else {
        setMode('daily');
      }
    } else {
      setMode('onboarding');
    }
  }, []);

  const handleOnboardingFinish = (userProfile) => {
    const newData = { user: userProfile, logs: {} };
    setFullData(newData);
    DataManager.saveData(newData);
    setMode('daily');
  };

  const handleDailyFinish = (dailyStats) => {
    const today = new Date().toISOString().split('T')[0];
    const newData = { ...fullData, logs: { ...fullData.logs, [today]: dailyStats } };
    setFullData(newData);
    DataManager.saveData(newData);
    setMode('dashboard');
  };

  if (mode === 'loading') return <div className="min-h-screen bg-slate-50"/>;

  return (
    <div className="font-sans text-slate-900 bg-gradient-to-br from-slate-900 to-slate-800 min-h-screen">
      {mode === 'onboarding' && <Onboarding onComplete={handleOnboardingFinish} />}
      {mode === 'daily' && <DailyCheckIn onComplete={handleDailyFinish} />}
      {mode === 'dashboard' && (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 py-6 px-4">
          <Navigation currentPage={currentPage} setCurrentPage={setCurrentPage} />
          
          <AnimatePresence mode="wait">
            {currentPage === 'dashboard' && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Dashboard fullData={fullData} onEditSettings={() => setMode('onboarding')} />
              </motion.div>
            )}
            {currentPage === 'weather' && (
              <motion.div
                key="weather"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <WeatherPage />
              </motion.div>
            )}
            {currentPage === 'doctor' && (
              <motion.div
                key="doctor"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <DoctorPage />
              </motion.div>
            )}
            {currentPage === 'music' && (
              <motion.div
                key="music"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <MusicPage />
              </motion.div>
            )}
            {currentPage === 'triggers' && (
              <motion.div
                key="triggers"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <TriggerTracker onDataChange={(data) => setFullData({...fullData, triggerLogs: data.triggerLogs})} />
              </motion.div>
            )}
            {currentPage === 'insights' && (
              <motion.div
                key="insights"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <AIInsights />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}