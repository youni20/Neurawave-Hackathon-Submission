import React, { useState, useEffect } from 'react';
import { Plus, Trash2, TrendingUp, AlertCircle, Calendar, Brain, X, Loader, Sun } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// --- CONSTANTS ---
// 1. Python AI Server (Dr. Neura)
const AI_API_URL = "http://100.89.109.97:5000"; 
// 2. Node.js Data/Sensor Server (Localhost)
const DATA_API_URL = "http://localhost:3001";

const SYMPTOM_OPTIONS = [
  { id: 'migraine', label: 'Migraine', icon: '🧠' },
  { id: 'nausea', label: 'Nausea', icon: '🤢' },
  { id: 'light_sensitivity', label: 'Light Sensitivity', icon: '☀️' },
  { id: 'sound_sensitivity', label: 'Sound Sensitivity', icon: '🔊' },
  { id: 'fatigue', label: 'Fatigue', icon: '😴' },
  { id: 'dizziness', label: 'Dizziness', icon: '🌀' },
  { id: 'brain_fog', label: 'Brain Fog', icon: '🌫️' },
  { id: 'neck_pain', label: 'Neck Pain', icon: '🪨' },
];

const TRIGGER_OPTIONS = [
  { id: 'stress', label: 'Stress', icon: '😰' },
  { id: 'sleep', label: 'Poor Sleep', icon: '😪' },
  { id: 'caffeine', label: 'Caffeine', icon: '☕' },
  { id: 'weather', label: 'Weather Change', icon: '🌦️' },
  { id: 'food', label: 'Food/Drink', icon: '🍽️' },
  { id: 'dehydration', label: 'Dehydration', icon: '💧' },
  { id: 'exercise', label: 'Exercise', icon: '🏃' },
  { id: 'hormones', label: 'Hormonal Changes', icon: '⚗️' },
  { id: 'medication', label: 'Medication Change', icon: '💊' },
  { id: 'screen_time', label: 'Screen Time', icon: '💻' },
];

// --- ADVICE MODAL COMPONENT ---
const AdviceModal = ({ isOpen, onClose, loading, adviceData }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <motion.div 
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="bg-slate-900 border border-slate-700 w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="bg-slate-800 p-4 flex justify-between items-center border-b border-slate-700">
          <div className="flex items-center gap-2">
            <div className="bg-teal-500/20 p-2 rounded-full">
              <Brain size={20} className="text-teal-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Dr. Neura Analysis</h3>
              <p className="text-xs text-slate-400">AI-Powered Log Review</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 min-h-[200px] flex flex-col justify-center">
          {loading ? (
            <div className="flex flex-col items-center text-slate-400 space-y-4">
              <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}>
                <Loader size={40} className="text-teal-500" />
              </motion.div>
              <p>Analyzing trigger patterns...</p>
            </div>
          ) : adviceData ? (
            <div className="space-y-4">
              {/* Emotion Badge */}
              <div className="flex justify-center">
                 <span className={`px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase
                    ${adviceData.emotion === 'HAPPY' ? 'bg-green-500/20 text-green-400' : 
                      adviceData.emotion === 'SAD' ? 'bg-red-500/20 text-red-400' : 
                      'bg-blue-500/20 text-blue-400'}`}>
                    Analysis: {adviceData.emotion}
                 </span>
              </div>

              {/* The Advice Text */}
              <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
                <p className="text-slate-200 text-lg leading-relaxed font-medium">
                  "{adviceData.advice}"
                </p>
              </div>
              
              <p className="text-center text-xs text-slate-500 mt-4">
                *AI advice is for informational purposes only. Consult a real doctor for medical decisions.
              </p>
            </div>
          ) : (
            <p className="text-center text-red-400">Failed to generate advice. Please try again.</p>
          )}
        </div>

        {/* Footer */}
        {!loading && (
          <div className="p-4 border-t border-slate-700 flex justify-end">
            <button onClick={onClose} className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg transition">
              Close
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
};

// --- MAIN COMPONENT ---

export default function TriggerTracker({ onDataChange, user }) {
  const [triggerLogs, setTriggerLogs] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [selectedTriggers, setSelectedTriggers] = useState([]);
  const [severity, setSeverity] = useState(5);
  const [notes, setNotes] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [triggerStats, setTriggerStats] = useState({});
  
  // AI Modal State
  const [showAdviceModal, setShowAdviceModal] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState(null);

  // Sensor State
  const [lightLevel, setLightLevel] = useState(0);
  const [sensorStatus, setSensorStatus] = useState('Disconnected');

  // 1. Poll for Light Sensor Data (From Node Backend)
  useEffect(() => {
    const fetchSensor = async () => {
      try {
        const res = await fetch(`${DATA_API_URL}/sensor/light`);
        const data = await res.json();
        setLightLevel(data.percent);
        setSensorStatus(data.status);
      } catch (e) {
        // console.warn("Sensor fetch failed", e);
        setSensorStatus('Disconnected');
      }
    };

    fetchSensor();
    const interval = setInterval(fetchSensor, 1000); // Run every 1 second
    return () => clearInterval(interval);
  }, []);

  // 2. Load existing logs (From Node Backend)
  useEffect(() => {
    const load = async () => {
      if (user && user.name && user.surname && user.id) {
        try {
          const resp = await fetch(`${DATA_API_URL}/get-triggers/${encodeURIComponent(user.name)}/${encodeURIComponent(user.surname)}/${encodeURIComponent(user.id)}`, {
            headers: { 'X-User-Id': user.id }
          });
          if (resp.ok) {
            const json = await resp.json();
            const logs = json.triggerLogs || [];
            setTriggerLogs(logs.reverse ? logs.reverse() : logs);
            calculateStats(logs);
            localStorage.setItem('triggerLogs', JSON.stringify(logs));
            return;
          }
        } catch (e) {
          console.warn('Failed to load triggers from server, falling back to localStorage', e);
        }
      }
      const savedLogs = JSON.parse(localStorage.getItem('triggerLogs')) || [];
      setTriggerLogs(savedLogs);
      calculateStats(savedLogs);
    };
    load();
  }, [user]);

  const calculateStats = (logs) => {
    const stats = {};
    logs.forEach(log => {
      log.triggers.forEach(trigger => {
        stats[trigger] = (stats[trigger] || 0) + 1;
      });
    });
    setTriggerStats(stats);
  };

  // --- AI HANDLER (To Python Backend) ---
  const handleAnalyzeLog = async (logEntry) => {
    setShowAdviceModal(true);
    setIsAnalyzing(true);
    setAiResult(null);

    try {
      // Note: Using AI_API_URL here
      const response = await fetch(`${AI_API_URL}/analyze_log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logEntry)
      });

      if (!response.ok) throw new Error("Failed to fetch advice");

      const data = await response.json();
      setAiResult(data);
    } catch (error) {
      console.error("AI Error:", error);
      setAiResult(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleSymptom = (symptomId) => {
    setSelectedSymptoms(prev => prev.includes(symptomId) ? prev.filter(id => id !== symptomId) : [...prev, symptomId]);
  };

  const toggleTrigger = (triggerId) => {
    setSelectedTriggers(prev => prev.includes(triggerId) ? prev.filter(id => id !== triggerId) : [...prev, triggerId]);
  };

  const handleSubmit = () => {
    if (selectedSymptoms.length === 0) {
      alert('Please select at least one symptom');
      return;
    }

    // If light is high during logging, suggest adding "Light Sensitivity" or "Weather"
    const autoTriggers = [...selectedTriggers];
    // Example: If light sensor is > 80%, automatically assume light is a factor if not selected
    // if (lightLevel > 80 && !autoTriggers.includes('weather')) autoTriggers.push('weather'); 

    const newLog = {
      id: Date.now(),
      date: new Date().toISOString(),
      symptoms: selectedSymptoms,
      triggers: autoTriggers, // Use the modified triggers
      severity,
      notes: `${notes} (Env Light: ${lightLevel}%)`, // Auto-append light data to notes
    };

    const updatedLogs = [newLog, ...triggerLogs];
    setTriggerLogs(updatedLogs);
    calculateStats(updatedLogs);
    onDataChange && onDataChange({ triggerLogs: updatedLogs });

    // Saving Logic
    if (user && user.name) {
        fetch(`${DATA_API_URL}/save-triggers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-User-Id': user.id },
            body: JSON.stringify({ name: user.name, surname: user.surname, id: user.id, triggerLogs: newLog })
        }).catch(e => console.warn("Save failed", e));
    }
    localStorage.setItem('triggerLogs', JSON.stringify(updatedLogs));

    // Reset form
    setSelectedSymptoms([]);
    setSelectedTriggers([]);
    setSeverity(5);
    setNotes('');
    setShowForm(false);
  };

  const handleDelete = (id) => {
    const updatedLogs = triggerLogs.filter(log => log.id !== id);
    setTriggerLogs(updatedLogs);
    calculateStats(updatedLogs);
    onDataChange && onDataChange({ triggerLogs: updatedLogs });
    localStorage.setItem('triggerLogs', JSON.stringify(updatedLogs));
  };

  const getMostCommonTriggers = () => {
    return Object.entries(triggerStats).sort((a, b) => b[1] - a[1]).slice(0, 3);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4 md:p-6 relative">
      
      <AdviceModal 
        isOpen={showAdviceModal} 
        onClose={() => setShowAdviceModal(false)} 
        loading={isAnalyzing} 
        adviceData={aiResult} 
      />

      <h1 className="text-3xl md:text-4xl font-bold text-center mb-2 text-slate-100">Trigger Tracker</h1>
      <p className="text-center text-slate-400 mb-8">Log your symptoms and discover what triggers them</p>

      {/* --- LIGHT SENSOR CARD --- */}
      <div className={`rounded-xl p-6 mb-8 text-white transition-colors duration-500 border border-white/10 shadow-lg
        ${sensorStatus === 'Disconnected' ? 'bg-slate-700' : 
          lightLevel > 60 ? 'bg-gradient-to-r from-red-600 to-orange-600' : 'bg-gradient-to-r from-emerald-600 to-green-500'}
      `}>
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="bg-white/20 p-3 rounded-full">
              <Sun size={24} className="text-white" />
            </div>
            <div>
              <p className="text-sm opacity-90 uppercase font-bold tracking-wider">Live Light Sensor</p>
              {sensorStatus === 'Disconnected' ? (
                <p className="text-xs text-slate-400">Sensor Offline (Check COM3)</p>
              ) : (
                <p className="text-3xl font-bold">{lightLevel}%</p>
              )}
            </div>
          </div>

          {sensorStatus !== 'Disconnected' && (
            <div className="text-right">
              <p className="text-sm font-bold opacity-80">
                {lightLevel > 60 ? "HIGH INTENSITY" : "SAFE LEVEL"}
              </p>
              <p className="text-xs opacity-60">
                {lightLevel > 60 ? "Consider dimming lights" : "Environment is optimal"}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Stats Overview */}
      {triggerLogs.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-gradient-to-br from-purple-600 to-purple-500 rounded-xl p-6 text-white">
            <TrendingUp size={24} className="mb-2" />
            <p className="text-sm opacity-80">Total Logs</p>
            <p className="text-3xl font-bold">{triggerLogs.length}</p>
          </div>
          <div className="bg-gradient-to-br from-blue-600 to-blue-500 rounded-xl p-6 text-white">
            <AlertCircle size={24} className="mb-2" />
            <p className="text-sm opacity-80">Unique Symptoms</p>
            <p className="text-3xl font-bold">{new Set(triggerLogs.flatMap(l => l.symptoms)).size}</p>
          </div>
          <div className="bg-gradient-to-br from-pink-600 to-pink-500 rounded-xl p-6 text-white">
            <Calendar size={24} className="mb-2" />
            <p className="text-sm opacity-80">Top Trigger</p>
            <p className="text-2xl font-bold">
              {getMostCommonTriggers()[0] 
                ? TRIGGER_OPTIONS.find(t => t.id === getMostCommonTriggers()[0][0])?.label 
                : 'None'}
            </p>
          </div>
        </div>
      )}

      {/* Add New Log Button */}
      <button
        onClick={() => setShowForm(!showForm)}
        className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold py-3 rounded-xl mb-8 flex items-center justify-center gap-2 hover:shadow-lg transition"
      >
        <Plus size={20} />
        {showForm ? 'Cancel' : 'Log New Entry'}
      </button>

      {/* Entry Form */}
      <AnimatePresence>
      {showForm && (
        <motion.div 
            initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
            className="bg-slate-800 border border-slate-700 rounded-2xl p-6 mb-8"
        >
          {/* Symptoms Selection */}
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-100 mb-4">What symptoms are you experiencing?</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {SYMPTOM_OPTIONS.map(symptom => (
                <button
                  key={symptom.id}
                  onClick={() => toggleSymptom(symptom.id)}
                  className={`p-3 rounded-lg font-semibold transition ${
                    selectedSymptoms.includes(symptom.id) ? 'bg-cyan-500 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  <span className="text-lg mr-2">{symptom.icon}</span>
                  {symptom.label}
                </button>
              ))}
            </div>
          </div>

          {/* Triggers Selection */}
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-100 mb-4">What might be triggering this?</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {TRIGGER_OPTIONS.map(trigger => (
                <button
                  key={trigger.id}
                  onClick={() => toggleTrigger(trigger.id)}
                  className={`p-3 rounded-lg font-semibold transition text-sm ${
                    selectedTriggers.includes(trigger.id) ? 'bg-pink-500 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  <span className="text-lg mr-1">{trigger.icon}</span>
                  {trigger.label}
                </button>
              ))}
            </div>
          </div>

          {/* Severity & Notes */}
          <div className="mb-6">
            <label className="text-slate-100 font-semibold block mb-3">
              Symptom Severity: <span className="text-cyan-400">{severity}/10</span>
            </label>
            <input type="range" min="1" max="10" value={severity} onChange={(e) => setSeverity(Number(e.target.value))} className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer" />
          </div>
          <div className="mb-6">
            <label className="text-slate-100 font-semibold block mb-3">Additional Notes</label>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Details..." className="w-full bg-slate-700 text-slate-100 rounded-lg p-4 border border-slate-600 focus:border-cyan-500 outline-none" rows="3" />
          </div>

          <button onClick={handleSubmit} className="w-full bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold py-3 rounded-xl hover:shadow-lg transition">
            Save Entry
          </button>
        </motion.div>
      )}
      </AnimatePresence>

      {/* Previous Logs */}
      <div className="space-y-4">
        <h3 className="text-2xl font-bold text-slate-100 mb-4">Recent Entries</h3>
        {triggerLogs.length === 0 ? (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-8 text-center">
            <p className="text-slate-400">No entries yet.</p>
          </div>
        ) : (
          triggerLogs.slice(0, 10).map(log => (
            <div key={log.id} className="bg-slate-800 border border-slate-700 rounded-xl p-4 hover:border-slate-500 transition">
              <div className="flex justify-between items-start mb-3">
                <div>
                  <p className="text-slate-400 text-sm flex items-center mb-2">{formatDate(log.date)}</p>
                  <div className="flex gap-2 flex-wrap">
                    {log.symptoms.map(s => {
                      const sym = SYMPTOM_OPTIONS.find(opt => opt.id === s);
                      return <span key={s} className="bg-cyan-500/20 text-cyan-300 px-3 py-1 rounded-full text-sm">{sym?.icon} {sym?.label}</span>
                    })}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <button onClick={() => handleDelete(log.id)} className="text-red-400 hover:text-red-300 p-2"><Trash2 size={20} /></button>
                  
                  {/* --- THE BRAIN BUTTON --- */}
                  <button 
                    onClick={() => handleAnalyzeLog(log)} 
                    className="text-teal-400 hover:text-teal-300 p-2 bg-slate-700/50 rounded-full hover:bg-slate-700 transition"
                    title="Get AI Advice"
                  >
                    <Brain size={20} />
                  </button>

                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mt-4">
                 <div>
                    <p className="text-slate-500 text-xs uppercase font-bold">Severity</p>
                    <p className="text-yellow-400 font-bold">{log.severity}/10</p>
                 </div>
                 <div>
                    <p className="text-slate-500 text-xs uppercase font-bold">Triggers</p>
                    <div className="flex gap-1 flex-wrap">
                        {log.triggers.length > 0 ? log.triggers.map(t => {
                            const trg = TRIGGER_OPTIONS.find(opt => opt.id === t);
                            return <span key={t} className="text-xs text-pink-300 bg-pink-500/10 px-1 rounded">{trg?.label}</span>
                        }) : <span className="text-xs text-slate-500">None</span>}
                    </div>
                 </div>
              </div>
              {log.notes && (
                <div className="mt-3 pt-3 border-t border-slate-700/50">
                   <p className="text-slate-400 text-xs italic">{log.notes}</p>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}