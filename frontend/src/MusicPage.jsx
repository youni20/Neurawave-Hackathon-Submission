import React, { useState, useRef, useEffect } from 'react';
import { Music, Play, Pause, Volume2, Zap } from 'lucide-react';

export default function MusicPage() {
  const [selectedGenre, setSelectedGenre] = useState('ambient');
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [tempo, setTempo] = useState(60);
  const audioRef = useRef(new (window.AudioContext || window.webkitAudioContext)());
  const oscillatorRef = useRef(null);
  const gainRef = useRef(null);

  const genres = {
    ambient: {
      name: 'Ambient',
      description: 'Calming, atmospheric soundscapes',
      frequency: 432,
      color: 'from-purple-600 to-blue-600',
    },
    binaural: {
      name: 'Binaural Beats',
      description: 'Brain wave entrainment',
      frequency: 40,
      color: 'from-blue-600 to-cyan-600',
    },
    nature: {
      name: 'Nature Sounds',
      description: 'Rain, forest, ocean',
      frequency: 250,
      color: 'from-green-600 to-emerald-600',
    },
    meditation: {
      name: 'Meditation',
      description: 'Deep relaxation tones',
      frequency: 174,
      color: 'from-orange-600 to-red-600',
    },
  };

  const currentGenre = genres[selectedGenre];

  useEffect(() => {
    return () => {
      if (oscillatorRef.current) {
        oscillatorRef.current.stop();
      }
    };
  }, []);

  const togglePlayback = () => {
    if (isPlaying) {
      stopAudio();
    } else {
      playAudio();
    }
  };

  const playAudio = () => {
    const audioContext = audioRef.current;

    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }

    // Create oscillator
    const oscillator = audioContext.createOscillator();
    oscillator.type = 'sine';
    oscillator.frequency.value = currentGenre.frequency;

    // Create gain node
    const gainNode = audioContext.createGain();
    gainNode.gain.value = volume * 0.1;

    // Connect
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.start();

    oscillatorRef.current = oscillator;
    gainRef.current = gainNode;
    setIsPlaying(true);
  };

  const stopAudio = () => {
    if (oscillatorRef.current) {
      oscillatorRef.current.stop();
      oscillatorRef.current = null;
    }
    setIsPlaying(false);
  };

  const handleVolumeChange = (value) => {
    setVolume(value);
    if (gainRef.current) {
      gainRef.current.gain.value = value * 0.1;
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4 md:p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-center mb-2 text-slate-100">Sonic Therapy</h1>
      <p className="text-center text-slate-400 mb-8">Audio-based pain management and relaxation</p>

      {/* Genre Selection Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {Object.entries(genres).map(([key, genre]) => (
          <button
            key={key}
            onClick={() => {
              if (isPlaying) stopAudio();
              setSelectedGenre(key);
            }}
            className={`p-4 rounded-xl font-semibold transition-all transform hover:scale-105 ${
              selectedGenre === key
                ? `bg-gradient-to-br ${genre.color} text-white shadow-lg`
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            <Music size={24} className="mx-auto mb-2" />
            <p className="font-bold">{genre.name}</p>
            <p className="text-xs opacity-75">{genre.description}</p>
          </button>
        ))}
      </div>

      {/* Main Player Card */}
      <div className={`bg-gradient-to-br ${currentGenre.color} rounded-2xl p-6 md:p-10 mb-8 shadow-xl`}>
        <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">{currentGenre.name}</h2>
        <p className="text-white opacity-90 mb-8">{currentGenre.description}</p>

        {/* Visualizer */}
        <div className="bg-black bg-opacity-30 rounded-lg h-32 md:h-48 mb-8 flex items-center justify-center overflow-hidden">
          <svg viewBox="0 0 200 100" className="w-full h-full" style={{ filter: 'drop-shadow(0 0 10px rgba(255,255,255,0.3))' }}>
            {[...Array(20)].map((_, i) => (
              <rect
                key={i}
                x={i * 10}
                y={50 - Math.random() * 30}
                width="8"
                height={Math.random() * 50}
                fill="white"
                opacity={isPlaying ? 0.8 : 0.3}
                style={{
                  animation: isPlaying ? `wave 0.5s ease-in-out infinite` : 'none',
                  animationDelay: `${i * 0.05}s`,
                }}
              />
            ))}
          </svg>
        </div>

        {/* Controls */}
        <div className="space-y-6">
          {/* Play/Pause Button */}
          <button
            onClick={togglePlayback}
            className={`w-full py-4 rounded-xl font-bold text-lg transition-all transform hover:scale-105 flex items-center justify-center gap-3 ${
              isPlaying
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-white text-blue-600 hover:bg-gray-100'
            }`}
          >
            {isPlaying ? (
              <>
                <Pause size={28} />
                Stop Therapy
              </>
            ) : (
              <>
                <Play size={28} />
                Start Therapy
              </>
            )}
          </button>

          {/* Volume Control */}
          <div className="bg-white bg-opacity-10 rounded-xl p-4">
            <label className="flex items-center gap-3 text-white font-semibold mb-3">
              <Volume2 size={20} />
              Volume
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={volume * 100}
              onChange={(e) => handleVolumeChange(e.target.value / 100)}
              className="w-full accent-white cursor-pointer"
            />
            <p className="text-white opacity-75 text-sm mt-2">{Math.round(volume * 100)}%</p>
          </div>

          {/* Tempo Control */}
          <div className="bg-white bg-opacity-10 rounded-xl p-4">
            <label className="flex items-center gap-3 text-white font-semibold mb-3">
              <Zap size={20} />
              Tempo (BPM)
            </label>
            <input
              type="range"
              min="30"
              max="180"
              value={tempo}
              onChange={(e) => setTempo(e.target.value)}
              className="w-full accent-white cursor-pointer"
            />
            <p className="text-white opacity-75 text-sm mt-2">{tempo} BPM</p>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { title: '10 Minutes', desc: 'Quick relaxation' },
          { title: '30 Minutes', desc: 'Deep meditation' },
          { title: '60 Minutes', desc: 'Full therapy session' },
        ].map((rec, idx) => (
          <div key={idx} className="bg-slate-700 rounded-xl p-6 text-center">
            <p className="text-xl font-bold text-white mb-2">{rec.title}</p>
            <p className="text-slate-400">{rec.desc}</p>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes wave {
          0%, 100% { height: 20px; }
          50% { height: 50px; }
        }
      `}</style>
    </div>
  );
}
