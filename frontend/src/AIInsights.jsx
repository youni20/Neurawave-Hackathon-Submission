import React, { useState, useEffect } from 'react';
import { Brain, TrendingDown, AlertTriangle, Zap, BarChart3, Clock, CheckCircle } from 'lucide-react';

const TRIGGER_OPTIONS = [
  { id: 'stress', label: 'Stress' },
  { id: 'sleep', label: 'Poor Sleep' },
  { id: 'caffeine', label: 'Caffeine' },
  { id: 'weather', label: 'Weather Change' },
  { id: 'food', label: 'Food/Drink' },
  { id: 'dehydration', label: 'Dehydration' },
  { id: 'exercise', label: 'Exercise' },
  { id: 'hormones', label: 'Hormonal Changes' },
  { id: 'medication', label: 'Medication Change' },
  { id: 'screen_time', label: 'Screen Time' },
];

export default function AIInsights() {
  const [triggerLogs, setTriggerLogs] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [triggerCorrelations, setTriggerCorrelations] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadAndAnalyzeData = async () => {
      setLoading(true);
      // Load trigger logs
      const savedLogs = JSON.parse(localStorage.getItem('triggerLogs')) || [];
      setTriggerLogs(savedLogs);

      if (savedLogs.length > 0) {
        // Analyze triggers and correlations
        analyzeData(savedLogs);
      }
      setLoading(false);
    };

    loadAndAnalyzeData();
  }, []);

  const analyzeData = (logs) => {
    // Calculate trigger correlations with symptoms
    const correlations = {};
    const symptomFrequency = {};
    let totalSeverity = 0;

    logs.forEach(log => {
      totalSeverity += log.severity;
      log.symptoms.forEach(symptom => {
        symptomFrequency[symptom] = (symptomFrequency[symptom] || 0) + 1;
      });

      log.triggers.forEach(trigger => {
        if (!correlations[trigger]) {
          correlations[trigger] = { count: 0, avgSeverity: 0, totalSeverity: 0 };
        }
        correlations[trigger].count++;
        correlations[trigger].totalSeverity += log.severity;
        correlations[trigger].avgSeverity = correlations[trigger].totalSeverity / correlations[trigger].count;
      });
    });

    setTriggerCorrelations(correlations);

    // Generate prediction
    const prediction = generatePrediction(logs, correlations);
    setPrediction(prediction);

    // Generate recommendations
    const recs = generateRecommendations(correlations, logs);
    setRecommendations(recs);
  };

  const generatePrediction = (logs, correlations) => {
    if (logs.length === 0) return null;

    const recentLogs = logs.slice(0, 5);
    const avgSeverity = recentLogs.reduce((sum, log) => sum + log.severity, 0) / recentLogs.length;
    
    // Calculate migraine risk based on multiple factors
    let riskScore = 0;
    
    // Factor 1: Recent severity trend (weight: 30%)
    riskScore += avgSeverity / 10 * 0.3;

    // Factor 2: High-correlation triggers recently identified (weight: 40%)
    const recentTriggers = {};
    recentLogs.forEach(log => {
      log.triggers.forEach(trigger => {
        recentTriggers[trigger] = (recentTriggers[trigger] || 0) + 1;
      });
    });

    const highRiskTriggers = Object.entries(correlations)
      .filter(([_, data]) => data.avgSeverity > 6)
      .map(([trigger, _]) => trigger);

    const matchingHighRiskTriggers = Object.keys(recentTriggers).filter(t => highRiskTriggers.includes(t));
    riskScore += (matchingHighRiskTriggers.length / Math.max(highRiskTriggers.length, 1)) * 0.4;

    // Factor 3: Frequency of symptoms (weight: 30%)
    riskScore += Math.min(logs.length / 20, 1) * 0.3;

    // Convert to percentage
    const riskPercentage = Math.round(riskScore * 100);
    
    let riskLevel = 'Low';
    let riskColor = 'text-green-400';
    
    if (riskPercentage > 70) {
      riskLevel = 'Critical';
      riskColor = 'text-red-400';
    } else if (riskPercentage > 50) {
      riskLevel = 'High';
      riskColor = 'text-orange-400';
    } else if (riskPercentage > 30) {
      riskLevel = 'Moderate';
      riskColor = 'text-yellow-400';
    }

    return {
      riskPercentage,
      riskLevel,
      riskColor,
      topTriggers: Object.entries(correlations)
        .sort((a, b) => b[1].avgSeverity - a[1].avgSeverity)
        .slice(0, 3)
        .map(([id, data]) => ({
          id,
          label: TRIGGER_OPTIONS.find(t => t.id === id)?.label || id,
          avgSeverity: data.avgSeverity.toFixed(1),
          frequency: data.count,
        })),
    };
  };

  const generateRecommendations = (correlations, logs) => {
    const recs = [];

    // Find top triggers
    const topTriggers = Object.entries(correlations)
      .sort((a, b) => b[1].avgSeverity - a[1].avgSeverity)
      .slice(0, 3)
      .map(([id, _]) => id);

    // Generate specific recommendations based on triggers
    topTriggers.forEach(triggerId => {
      const trigger = TRIGGER_OPTIONS.find(t => t.id === triggerId);
      
      let recommendation = {};
      switch (triggerId) {
        case 'stress':
          recommendation = {
            icon: '🧘',
            title: 'Stress Management',
            actions: ['Practice daily meditation (10-15 min)', 'Use the Music page for relaxation', 'Take regular breaks'],
            priority: 'High',
          };
          break;
        case 'sleep':
          recommendation = {
            icon: '😴',
            title: 'Improve Sleep Quality',
            actions: ['Maintain consistent sleep schedule', 'Avoid screens 1 hour before bed', 'Keep bedroom cool and dark'],
            priority: 'High',
          };
          break;
        case 'caffeine':
          recommendation = {
            icon: '☕',
            title: 'Caffeine Management',
            actions: ['Limit caffeine intake to morning only', 'Track daily caffeine consumption', 'Gradually reduce if heavily dependent'],
            priority: 'Medium',
          };
          break;
        case 'weather':
          recommendation = {
            icon: '🌦️',
            title: 'Weather Preparedness',
            actions: ['Check weather forecasts daily', 'Monitor barometric pressure changes', 'Prepare preventive measures in advance'],
            priority: 'Medium',
          };
          break;
        case 'dehydration':
          recommendation = {
            icon: '💧',
            title: 'Hydration Protocol',
            actions: ['Drink 8-10 glasses of water daily', 'Stay hydrated during work', 'Set hydration reminders'],
            priority: 'High',
          };
          break;
        case 'screen_time':
          recommendation = {
            icon: '💻',
            title: 'Screen Time Reduction',
            actions: ['Follow 20-20-20 rule (every 20 min, look away for 20 sec)', 'Adjust screen brightness', 'Use blue light filter'],
            priority: 'Medium',
          };
          break;
        case 'exercise':
          recommendation = {
            icon: '🏃',
            title: 'Exercise Planning',
            actions: ['Warm up before exercise', 'Stay hydrated during workouts', 'Avoid overexertion'],
            priority: 'Low',
          };
          break;
        case 'food':
          recommendation = {
            icon: '🍽️',
            title: 'Dietary Adjustment',
            actions: ['Identify trigger foods and eliminate', 'Eat regular, balanced meals', 'Keep food diary'],
            priority: 'Medium',
          };
          break;
        case 'hormones':
          recommendation = {
            icon: '⚗️',
            title: 'Hormonal Tracking',
            actions: ['Track your cycle patterns', 'Plan prevention around peak risk days', 'Consult healthcare provider'],
            priority: 'Medium',
          };
          break;
        default:
          recommendation = {
            icon: '✓',
            title: `Manage ${trigger?.label}`,
            actions: ['Monitor this trigger closely', 'Note patterns when it occurs'],
            priority: 'Low',
          };
      }

      recs.push({
        ...recommendation,
        triggerId,
      });
    });

    return recs;
  };

  if (loading) {
    return (
      <div className="w-full max-w-4xl mx-auto p-4 md:p-6 flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Brain size={48} className="mx-auto text-cyan-400 mb-4 animate-pulse" />
          <p className="text-slate-300">Analyzing your data...</p>
        </div>
      </div>
    );
  }

  if (triggerLogs.length === 0) {
    return (
      <div className="w-full max-w-4xl mx-auto p-4 md:p-6">
        <h1 className="text-3xl md:text-4xl font-bold text-center mb-8 text-slate-100">AI Insights</h1>
        <div className="bg-slate-800 border border-slate-700 rounded-2xl p-12 text-center">
          <Brain size={48} className="mx-auto text-slate-500 mb-4" />
          <p className="text-slate-400 mb-4">No data to analyze yet.</p>
          <p className="text-slate-500 text-sm">Start logging your symptoms in the Trigger Tracker to get personalized AI insights!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto p-4 md:p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-center mb-2 text-slate-100">AI Insights</h1>
      <p className="text-center text-slate-400 mb-8">Personalized migraine prediction and recommendations powered by machine learning</p>

      {/* Risk Prediction Card */}
      {prediction && (
        <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-slate-100">Migraine Risk Assessment</h2>
            <Brain size={32} className={prediction.riskColor} />
          </div>

          {/* Risk Gauge */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="flex flex-col items-center justify-center">
              <div className="relative w-40 h-40">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  {/* Background circle */}
                  <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" strokeWidth="8" />
                  {/* Progress circle */}
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke={prediction.riskColor === 'text-green-400' ? '#22c55e' :
                            prediction.riskColor === 'text-yellow-400' ? '#eab308' :
                            prediction.riskColor === 'text-orange-400' ? '#f97316' : '#ef4444'}
                    strokeWidth="8"
                    strokeDasharray={`${prediction.riskPercentage * 2.83} 283`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <p className={`text-4xl font-bold ${prediction.riskColor}`}>{prediction.riskPercentage}%</p>
                  <p className="text-slate-400 text-sm">Risk Level</p>
                </div>
              </div>
              <p className={`text-2xl font-bold mt-4 ${prediction.riskColor}`}>{prediction.riskLevel}</p>
            </div>

            {/* Risk Details */}
            <div className="flex flex-col justify-center space-y-4">
              <div>
                <p className="text-slate-400 text-sm mb-1">Assessment Based On:</p>
                <ul className="space-y-2">
                  <li className="flex items-center gap-2 text-slate-200">
                    <CheckCircle size={16} className="text-cyan-400" />
                    <span>{triggerLogs.length} logged entries analyzed</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-200">
                    <BarChart3 size={16} className="text-cyan-400" />
                    <span>{prediction.topTriggers.length} key triggers identified</span>
                  </li>
                  <li className="flex items-center gap-2 text-slate-200">
                    <Zap size={16} className="text-cyan-400" />
                    <span>Severity trends calculated</span>
                  </li>
                </ul>
              </div>

              {/* Interpretation */}
              <div className="bg-slate-700 bg-opacity-50 rounded-lg p-3 border-l-4 border-cyan-400">
                <p className="text-slate-100 text-sm">
                  {prediction.riskPercentage > 70
                    ? '⚠️ Critical risk detected. Consider preventive measures immediately.'
                    : prediction.riskPercentage > 50
                    ? '⚡ Elevated risk. Be proactive with your health management.'
                    : prediction.riskPercentage > 30
                    ? '⚙️ Moderate risk. Continue monitoring and prevention.'
                    : '✅ Low risk. Maintain your current healthy habits.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Top Triggers Analysis */}
      {prediction && prediction.topTriggers.length > 0 && (
        <div className="mb-8">
          <h3 className="text-2xl font-bold text-slate-100 mb-4">Your Top Migraine Triggers</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {prediction.topTriggers.map((trigger, index) => (
              <div
                key={trigger.id}
                className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-xl p-5"
              >
                <div className="flex items-start justify-between mb-3">
                  <h4 className="text-lg font-bold text-slate-100">{trigger.label}</h4>
                  <span className="bg-cyan-500 bg-opacity-20 text-cyan-300 px-2 py-1 rounded text-xs font-bold">
                    #{index + 1}
                  </span>
                </div>
                <div className="space-y-2">
                  <div>
                    <p className="text-slate-400 text-xs mb-1">Avg Severity</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-slate-700 rounded-full h-2">
                        <div
                          className="bg-red-500 h-2 rounded-full"
                          style={{ width: `${(trigger.avgSeverity / 10) * 100}%` }}
                        />
                      </div>
                      <span className="text-red-400 font-bold text-sm">{trigger.avgSeverity}/10</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-slate-400 text-xs">Occurrences</p>
                    <p className="text-cyan-300 font-bold">{trigger.frequency} times</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Personalized Recommendations */}
      {recommendations.length > 0 && (
        <div>
          <h3 className="text-2xl font-bold text-slate-100 mb-4">Personalized Recommendations</h3>
          <div className="space-y-4">
            {recommendations.map((rec, index) => (
              <div
                key={index}
                className="bg-gradient-to-r from-slate-800 to-slate-900 border border-slate-700 rounded-xl p-6 hover:border-cyan-500 transition"
              >
                <div className="flex items-start gap-4">
                  <div className="text-3xl">{rec.icon}</div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-lg font-bold text-slate-100">{rec.title}</h4>
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-bold ${
                          rec.priority === 'High'
                            ? 'bg-red-500 bg-opacity-20 text-red-300'
                            : rec.priority === 'Medium'
                            ? 'bg-yellow-500 bg-opacity-20 text-yellow-300'
                            : 'bg-green-500 bg-opacity-20 text-green-300'
                        }`}
                      >
                        {rec.priority} Priority
                      </span>
                    </div>
                    <ul className="space-y-2">
                      {rec.actions.map((action, i) => (
                        <li key={i} className="flex items-start gap-2 text-slate-300 text-sm">
                          <CheckCircle size={16} className="text-cyan-400 mt-0.5 flex-shrink-0" />
                          <span>{action}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Summary */}
      <div className="mt-8 bg-slate-800 bg-opacity-50 border border-slate-700 rounded-xl p-4">
        <p className="text-slate-400 text-sm text-center">
          💡 Tip: The more entries you log, the more accurate these AI insights become. Keep tracking your symptoms for better predictions!
        </p>
      </div>
    </div>
  );
}
