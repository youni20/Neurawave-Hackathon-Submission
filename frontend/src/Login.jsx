import React, { useState } from 'react';

export default function Login({ onLogin, onCancel }) {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [matches, setMatches] = useState([]);

  const submit = async () => {
    setError(null);
    setLoading(true);
    try {
      const resp = await fetch('http://localhost:3001/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
      });
      const json = await resp.json();
      setLoading(false);
      if (!resp.ok) {
        setError(json.error || 'Login failed');
        return;
      }
      if (Array.isArray(json.matches) && json.matches.length > 0) {
        setMatches(json.matches);
        // if exactly one, pick it
        if (json.matches.length === 1) {
          const match = json.matches[0];
          // if triggerLogs provided by server, store locally for immediate UI
          if (match.triggerLogs) localStorage.setItem('triggerLogs', JSON.stringify(match.triggerLogs));
          onLogin(match);
        }
      } else {
        setError('No matching profiles found');
      }
    } catch (e) {
      setLoading(false);
      setError('Network error');
    }
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-white/5 rounded-xl">
      <h2 className="text-lg font-bold mb-3 text-slate-100">Log in</h2>
      <p className="text-sm text-slate-400 mb-4">Enter your first name, last name, or full name to find your profile.</p>
      <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="e.g. Arturs" className="w-full p-3 rounded mb-3 bg-slate-700 text-white" />
      <div className="flex gap-3">
        <button onClick={submit} disabled={loading} className="flex-1 bg-cyan-500 py-2 rounded font-semibold">{loading ? 'Searching...' : 'Find'}</button>
        <button onClick={onCancel} className="bg-slate-600 py-2 rounded">Cancel</button>
      </div>

      {error && <p className="text-red-400 mt-3">{error}</p>}

      {matches.length > 1 && (
        <div className="mt-4">
          <p className="text-sm text-slate-300 mb-2">Multiple matches — pick your profile:</p>
          <div className="space-y-2">
            {matches.map((m, i) => (
              <button key={i} onClick={() => {
                if (m.triggerLogs) localStorage.setItem('triggerLogs', JSON.stringify(m.triggerLogs));
                onLogin(m);
              }} className="w-full text-left p-3 bg-slate-700 rounded">
                {m.user?.name} {m.user?.surname} (ID: {m.user?.id})
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
