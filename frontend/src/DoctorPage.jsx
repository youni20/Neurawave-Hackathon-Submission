import React, { useState, useRef } from 'react';
import { Mic, Send, Volume2 } from 'lucide-react';

export default function DoctorPage() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [history, setHistory] = useState([]);
  const recognitionRef = useRef(null);

  // Initialize Speech Recognition
  React.useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;

      recognitionRef.current.onstart = () => setIsListening(true);
      recognitionRef.current.onend = () => setIsListening(false);
      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((result) => result[0].transcript)
          .join('');
        setInput(transcript);
      };
    }
  }, []);

  const toggleMic = () => {
    if (recognitionRef.current) {
      if (isListening) {
        recognitionRef.current.stop();
      } else {
        recognitionRef.current.start();
      }
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add to history
    setHistory([...history, { type: 'user', text: input }]);
    
    // Simulate doctor response
    const responses = [
      'I understand. Let me help you with that.',
      'That sounds important. Have you considered...',
      'Thank you for sharing. How has this affected you?',
      'I see. What symptoms have you noticed?',
    ];
    
    const randomResponse = responses[Math.floor(Math.random() * responses.length)];
    setResponse(randomResponse);
    setHistory((prev) => [...prev, { type: 'assistant', text: randomResponse }]);
    setInput('');
  };

  const speak = (text) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-4 md:p-6">
      {/* Character Display */}
      <div className="flex justify-center mb-6">
        <div className="w-32 h-40 md:w-48 md:h-64 bg-gradient-to-br from-slate-400 to-slate-600 rounded-2xl flex items-center justify-center">
          <svg viewBox="0 0 100 140" className="w-full h-full p-4" style={{ filter: 'drop-shadow(0 4px 12px rgba(0,0,0,0.2))' }}>
            {/* Character SVG */}
            <path d="M 35 70 L 30 110 L 40 135 L 60 135 L 70 110 L 65 70 Z" fill="white" stroke="#e2e8f0" strokeWidth="1.5" />
            <circle cx="45" cy="50" r="18" fill="white" stroke="#e2e8f0" strokeWidth="1.5" />
            <circle cx="40" cy="48" r="4" fill="#000" />
            <circle cx="50" cy="48" r="4" fill="#000" />
            <path d="M 38 58 Q 45 62 52 58" stroke="#000" strokeWidth="1.5" fill="none" strokeLinecap="round" />
          </svg>
        </div>
      </div>

      {/* Title */}
      <h1 className="text-3xl md:text-4xl font-bold text-center mb-2 text-slate-100">Dr. Neura</h1>
      <p className="text-center text-slate-400 mb-6">Voice-enabled health advisor</p>

      {/* Response Box */}
      {response && (
        <div className="bg-blue-500 bg-opacity-10 border border-blue-500 rounded-xl p-4 mb-6 animate-slideIn">
          <p className="text-slate-200 mb-3">{response}</p>
          <button
            onClick={() => speak(response)}
            className="flex items-center gap-2 px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold transition text-sm"
          >
            <Volume2 size={16} />
            Play
          </button>
        </div>
      )}

      {/* Input Section */}
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask Dr. Neura..."
          className="flex-1 px-4 py-3 bg-slate-700 text-white rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none transition"
        />
        <button
          onClick={toggleMic}
          className={`px-4 py-3 rounded-lg font-semibold transition flex items-center gap-2 ${
            isListening
              ? 'bg-red-500 text-white animate-pulse'
              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
          }`}
        >
          <Mic size={20} />
        </button>
        <button
          onClick={handleSend}
          className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold transition flex items-center gap-2"
        >
          <Send size={20} />
        </button>
      </div>

      {/* History */}
      {history.length > 0 && (
        <div className="bg-slate-700 bg-opacity-50 rounded-xl p-4 max-h-64 overflow-y-auto">
          <p className="text-slate-400 text-sm font-semibold mb-3">Conversation History</p>
          <div className="space-y-2">
            {history.map((msg, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-lg text-sm ${
                  msg.type === 'user'
                    ? 'bg-blue-600 text-white ml-4'
                    : 'bg-slate-600 text-slate-100 mr-4'
                }`}
              >
                {msg.text}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
