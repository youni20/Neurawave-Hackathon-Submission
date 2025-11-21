import React, { useState, useEffect } from 'react';
import { Cloud, CloudRain, Wind, Droplets, Eye, Gauge } from 'lucide-react';

export default function WeatherPage() {
  const [weather, setWeather] = useState({
    temp: 14,
    condition: 'Overcast',
    pressure: 1012,
    humidity: 82,
    windSpeed: 12,
    visibility: 10,
    location: 'Kalmar, Sweden',
  });

  const [triggerRisks, setTriggerRisks] = useState({
    pressure: 'High',
    humidity: 'Moderate',
    wind: 'Low',
    temperature: 'Moderate',
  });

  useEffect(() => {
    calculateTriggerRisks();
  }, [weather]);

  const calculateTriggerRisks = () => {
    // Simplified trigger risk calculation based on weather
    const pressureLow = weather.pressure < 1000;
    const humidityHigh = weather.humidity > 70;
    const windHigh = weather.windSpeed > 20;
    const tempExtreme = weather.temp < 5 || weather.temp > 30;

    setTriggerRisks({
      pressure: pressureLow ? 'High' : weather.pressure < 1010 ? 'Moderate' : 'Low',
      humidity: humidityHigh ? 'High' : weather.humidity > 50 ? 'Moderate' : 'Low',
      wind: windHigh ? 'High' : weather.windSpeed > 15 ? 'Moderate' : 'Low',
      temperature: tempExtreme ? 'High' : 'Moderate',
    });
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'High':
        return 'text-red-400 bg-red-500 bg-opacity-10 border-red-500';
      case 'Moderate':
        return 'text-yellow-400 bg-yellow-500 bg-opacity-10 border-yellow-500';
      default:
        return 'text-green-400 bg-green-500 bg-opacity-10 border-green-500';
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-4 md:p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-center mb-2 text-slate-100">Symptom Triggers</h1>
      <p className="text-center text-slate-400 mb-8">How weather affects your health</p>

      {/* Weather Card */}
      <div className="bg-gradient-to-br from-blue-600 to-blue-500 rounded-2xl p-8 mb-8 shadow-2xl">
        <p className="text-blue-100 text-center mb-4">{weather.location}</p>
        <h2 className="text-7xl font-bold text-white text-center mb-4">{weather.temp}°C</h2>
        <p className="text-2xl text-blue-50 text-center mb-6">{weather.condition}</p>
        
        {/* Weather Details Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center">
            <Wind size={24} className="mx-auto mb-2 text-blue-100" />
            <p className="text-sm text-blue-100 opacity-75">Wind Speed</p>
            <p className="text-xl font-bold text-white">{weather.windSpeed} km/h</p>
          </div>
          
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center">
            <Droplets size={24} className="mx-auto mb-2 text-blue-100" />
            <p className="text-sm text-blue-100 opacity-75">Humidity</p>
            <p className="text-xl font-bold text-white">{weather.humidity}%</p>
          </div>
          
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center">
            <Gauge size={24} className="mx-auto mb-2 text-blue-100" />
            <p className="text-sm text-blue-100 opacity-75">Pressure</p>
            <p className="text-xl font-bold text-white">{weather.pressure} mb</p>
          </div>
          
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center">
            <Eye size={24} className="mx-auto mb-2 text-blue-100" />
            <p className="text-sm text-blue-100 opacity-75">Visibility</p>
            <p className="text-xl font-bold text-white">{weather.visibility} km</p>
          </div>
          
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center col-span-2 md:col-span-1">
            <Cloud size={24} className="mx-auto mb-2 text-blue-100" />
            <p className="text-sm text-blue-100 opacity-75">Condition</p>
            <p className="text-lg font-bold text-white">{weather.condition}</p>
          </div>
        </div>
      </div>

      {/* Trigger Risk Analysis */}
      <div className="space-y-4 mb-8">
        <h3 className="text-2xl font-bold text-slate-100 mb-4">Your Trigger Risks</h3>
        
        {Object.entries(triggerRisks).map(([key, risk]) => {
          const icon = {
            pressure: <Gauge size={24} />,
            humidity: <Droplets size={24} />,
            wind: <Wind size={24} />,
            temperature: <CloudRain size={24} />,
          }[key];

          const label = {
            pressure: 'Barometric Pressure',
            humidity: 'Humidity Level',
            wind: 'Wind Speed',
            temperature: 'Temperature',
          }[key];

          return (
            <div
              key={key}
              className={`flex items-center justify-between p-4 rounded-xl border ${getRiskColor(risk)}`}
            >
              <div className="flex items-center gap-4">
                {icon}
                <span className="font-semibold text-slate-100">{label}</span>
              </div>
              <span className="px-4 py-2 bg-white bg-opacity-10 rounded-lg font-bold text-slate-100">
                {risk}
              </span>
            </div>
          );
        })}
      </div>

      {/* Recommendations */}
      <div className="bg-slate-700 bg-opacity-50 rounded-2xl p-6 border border-slate-600">
        <h3 className="text-xl font-bold text-slate-100 mb-4">Recommendations</h3>
        <ul className="space-y-3 text-slate-300">
          <li className="flex gap-3">
            <span className="text-blue-400">✓</span>
            <span>Stay hydrated - humidity is high today</span>
          </li>
          <li className="flex gap-3">
            <span className="text-blue-400">✓</span>
            <span>Watch for pressure headaches with current barometric conditions</span>
          </li>
          <li className="flex gap-3">
            <span className="text-blue-400">✓</span>
            <span>Consider indoor activities given wind conditions</span>
          </li>
          <li className="flex gap-3">
            <span className="text-blue-400">✓</span>
            <span>Use Dr. Neura if symptoms develop</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
