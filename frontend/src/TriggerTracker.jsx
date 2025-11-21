import React, { useState, useEffect } from 'react';
import { Plus, Trash2, TrendingUp, AlertCircle, Calendar, Brain } from 'lucide-react';

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

export default function TriggerTracker({ onDataChange, user }) {
  const [triggerLogs, setTriggerLogs] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [selectedTriggers, setSelectedTriggers] = useState([]);
  const [severity, setSeverity] = useState(5);
  const [notes, setNotes] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [triggerStats, setTriggerStats] = useState({});
  const [aiStatusMap, setAiStatusMap] = useState(() => JSON.parse(localStorage.getItem('aiStatusMap') || '{}'));

  // Load existing logs from backend (if user provided) or from localStorage as fallback
  useEffect(() => {
    const load = async () => {
      if (user && user.name && user.surname && user.id) {
        try {
          const resp = await fetch(`http://localhost:3001/get-triggers/${encodeURIComponent(user.name)}/${encodeURIComponent(user.surname)}/${encodeURIComponent(user.id)}`, {
            headers: { 'X-User-Id': user.id }
          });
          if (resp.ok) {
            const json = await resp.json();
            const logs = json.triggerLogs || [];
            setTriggerLogs(logs.reverse ? logs.reverse() : logs);
            calculateStats(logs);
            // keep localStorage in sync for offline UX
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
      // initialize ai status map for existing logs
      const savedAi = JSON.parse(localStorage.getItem('aiStatusMap') || '{}');
      setAiStatusMap(savedAi);
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

  const toggleSymptom = (symptomId) => {
    setSelectedSymptoms(prev =>
      prev.includes(symptomId)
        ? prev.filter(id => id !== symptomId)
        : [...prev, symptomId]
    );
  };

  const toggleTrigger = (triggerId) => {
    setSelectedTriggers(prev =>
      prev.includes(triggerId)
        ? prev.filter(id => id !== triggerId)
        : [...prev, triggerId]
    );
  };

  const handleSubmit = () => {
    if (selectedSymptoms.length === 0) {
      alert('Please select at least one symptom');
      return;
    }

    const newLog = {
      id: Date.now(),
      date: new Date().toISOString(),
      symptoms: selectedSymptoms,
      triggers: selectedTriggers,
      severity,
      notes,
    };

    // Optimistically update UI
    const updatedLogs = [newLog, ...triggerLogs];
    setTriggerLogs(updatedLogs);
    calculateStats(updatedLogs);
    onDataChange && onDataChange({ triggerLogs: updatedLogs });

    // Persist to backend if user present, otherwise save to localStorage
    if (user && user.name && user.surname && user.id) {
      (async () => {
        try {
          // mark as pending for AI send
          setAiStatusMap(prev => {
            const next = { ...prev, [newLog.id]: 'pending' };
            localStorage.setItem('aiStatusMap', JSON.stringify(next));
            return next;
          });
          const resp = await fetch('http://localhost:3001/save-triggers', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-User-Id': user.id
            },
            body: JSON.stringify({ name: user.name, surname: user.surname, id: user.id, triggerLogs: newLog })
          });
          if (resp.ok) {
            const json = await resp.json();
            // If server returns the full storage, use it to update UI
            if (json && json.storage && Array.isArray(json.storage.logs)) {
              setTriggerLogs(json.storage.logs.slice().reverse());
              calculateStats(json.storage.logs);
              localStorage.setItem('triggerLogs', JSON.stringify(json.storage.logs));
            } else {
              // fallback: append locally
              const fallback = [newLog, ...triggerLogs];
              localStorage.setItem('triggerLogs', JSON.stringify(fallback));
            }
          } else {
            console.warn('Server responded with non-OK status saving trigger');
            localStorage.setItem('triggerLogs', JSON.stringify(updatedLogs));
          }
        } catch (e) {
          console.warn('Failed to save trigger to server, saved locally', e);
          localStorage.setItem('triggerLogs', JSON.stringify(updatedLogs));
        }

        // Attempt to send the log to the user's LLM endpoint (non-blocking)
        (async () => {
          try {
            const llmResp = await fetch('http://100.89.109.97:5000/', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(newLog),
            });
            if (llmResp.ok) {
              setAiStatusMap(prev => {
                const next = { ...prev, [newLog.id]: 'sent' };
                localStorage.setItem('aiStatusMap', JSON.stringify(next));
                return next;
              });
            } else {
              setAiStatusMap(prev => {
                const next = { ...prev, [newLog.id]: 'failed' };
                localStorage.setItem('aiStatusMap', JSON.stringify(next));
                return next;
              });
            }
          } catch (err) {
            setAiStatusMap(prev => {
              const next = { ...prev, [newLog.id]: 'failed' };
              localStorage.setItem('aiStatusMap', JSON.stringify(next));
              return next;
            });
            console.warn('Failed to send log to LLM endpoint', err);
          }
        })();
      })();
    } else {
      localStorage.setItem('triggerLogs', JSON.stringify(updatedLogs));
    }

    // Reset form
    setSelectedSymptoms([]);
    setSelectedTriggers([]);
    setSeverity(5);
    setNotes('');
    setShowForm(false);
    calculateStats(updatedLogs);
  };

  const handleDelete = (id) => {
    const updatedLogs = triggerLogs.filter(log => log.id !== id);
    setTriggerLogs(updatedLogs);
    calculateStats(updatedLogs);
    onDataChange && onDataChange({ triggerLogs: updatedLogs });

    // Persist deletion to localStorage for now. Server-side deletion not implemented.
    localStorage.setItem('triggerLogs', JSON.stringify(updatedLogs));
    // remove any saved ai status
    setAiStatusMap(prev => {
      const next = { ...prev };
      delete next[id];
      localStorage.setItem('aiStatusMap', JSON.stringify(next));
      return next;
    });
  };

  const getMostCommonTriggers = () => {
    return Object.entries(triggerStats)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4 md:p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-center mb-2 text-slate-100">Trigger Tracker</h1>
      <p className="text-center text-slate-400 mb-8">Log your symptoms and discover what triggers them</p>

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
      {showForm && (
        <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 mb-8">
          {/* Symptoms Selection */}
          <div className="mb-6">
            <h3 className="text-xl font-bold text-slate-100 mb-4">What symptoms are you experiencing?</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {SYMPTOM_OPTIONS.map(symptom => (
                <button
                  key={symptom.id}
                  onClick={() => toggleSymptom(symptom.id)}
                  className={`p-3 rounded-lg font-semibold transition ${
                    selectedSymptoms.includes(symptom.id)
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
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
                    selectedTriggers.includes(trigger.id)
                      ? 'bg-pink-500 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  <span className="text-lg mr-1">{trigger.icon}</span>
                  {trigger.label}
                </button>
              ))}
            </div>
          </div>

          {/* Severity Slider */}
          <div className="mb-6">
            <label className="text-slate-100 font-semibold block mb-3">
              Symptom Severity: <span className="text-cyan-400">{severity}/10</span>
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={severity}
              onChange={(e) => setSeverity(Number(e.target.value))}
              className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Notes */}
          <div className="mb-6">
            <label className="text-slate-100 font-semibold block mb-3">Additional Notes</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Any additional details about your symptoms..."
              className="w-full bg-slate-700 text-slate-100 rounded-lg p-4 border border-slate-600 focus:border-cyan-500 outline-none"
              rows="3"
            />
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            className="w-full bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold py-3 rounded-xl hover:shadow-lg transition"
          >
            Save Entry
          </button>
        </div>
      )}

      {/* Previous Logs */}
      <div className="space-y-4">
        <h3 className="text-2xl font-bold text-slate-100 mb-4">Recent Entries</h3>
        {triggerLogs.length === 0 ? (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-8 text-center">
            <p className="text-slate-400">No entries yet. Start tracking your symptoms!</p>
          </div>
        ) : (
          triggerLogs.slice(0, 10).map(log => (
            <div key={log.id} className="bg-slate-800 border border-slate-700 rounded-xl p-4">
              <div className="flex justify-between items-start mb-3">
                <div>
                      <p className="text-slate-400 text-sm flex items-center">
                        {formatDate(log.date)}
                        <span className="ml-2">
                          <Brain size={14} className={
                            aiStatusMap[log.id] === 'pending' ? 'text-yellow-400' :
                            aiStatusMap[log.id] === 'sent' ? 'text-green-400' :
                            aiStatusMap[log.id] === 'failed' ? 'text-red-400' : 'text-gray-400'
                          } />
                        </span>
                      </p>
                  <div className="flex gap-2 flex-wrap mt-2">
                    {log.symptoms.map(symptom => {
                      const sym = SYMPTOM_OPTIONS.find(s => s.id === symptom);
                      return (
                        <span key={symptom} className="bg-cyan-500 bg-opacity-20 text-cyan-300 px-3 py-1 rounded-full text-sm">
                          {sym?.icon} {sym?.label}
                        </span>
                      );
                    })}
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(log.id)}
                  className="text-red-400 hover:text-red-300 transition"
                >
                  <Trash2 size={20} />
                </button>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-slate-400 text-sm">Severity</p>
                  <p className="text-lg font-bold text-yellow-400">{log.severity}/10</p>
                </div>
                <div>
                  <p className="text-slate-400 text-sm">Identified Triggers</p>
                  <div className="flex gap-2 flex-wrap">
                    {log.triggers.length > 0 ? (
                      log.triggers.map(trigger => {
                        const trg = TRIGGER_OPTIONS.find(t => t.id === trigger);
                        return (
                          <span key={trigger} className="bg-pink-500 bg-opacity-20 text-pink-300 px-2 py-1 rounded text-xs">
                            {trg?.icon} {trg?.label}
                          </span>
                        );
                      })
                    ) : (
                      <span className="text-slate-400 text-sm">None identified</span>
                    )}
                  </div>
                </div>
              </div>
              {log.notes && (
                <div className="mt-3 pt-3 border-t border-slate-700">
                  <p className="text-slate-300 text-sm">{log.notes}</p>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
