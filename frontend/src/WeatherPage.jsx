import React, { useState, useEffect } from 'react';
import { Cloud, CloudRain, Wind, Droplets, Eye, Gauge, Loader, Sun, Moon, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function WeatherPage() {
  // Initial states for API operation
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [triggerRisks, setTriggerRisks] = useState({});

  // Kalmar, Sweden Coordinates and required variables
  const API_URL = "https://api.open-meteo.com/v1/forecast?latitude=56.66&longitude=16.36&current=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,weather_code,is_day&timezone=Europe%2FBerlin";
  const LOCATION = 'Kalmar, Sweden';

  // --- 1. API Fetching Logic (Reinstated) ---
  useEffect(() => {
    fetchWeather();
  }, []);

  // Recalculate risks whenever weather data is successfully updated
  useEffect(() => {
    if (weather) {
      calculateTriggerRisks(weather);
    }
  }, [weather]);


  const fetchWeather = async () => {
    setLoading(true);
    setError(null);
    try {
        const response = await fetch(API_URL);
        if (!response.ok) throw new Error("API returned non-OK status.");
        const data = await response.json();
        
        const current = data.current;
        
        if (!current || current.temperature_2m === undefined) {
          throw new Error("Incomplete weather data structure received.");
        }

        // Processed and Mapped Data Structure
        const weatherData = {
            temp: Math.round(current.temperature_2m),
            condition: mapWmoCode(current.weather_code),
            pressure: Math.round(current.pressure_msl),
            // Removed humidity and visibility fields
            windSpeed: Math.round(current.wind_speed_10m),
            location: LOCATION,
            isDay: current.is_day === 1
        };

        setWeather(weatherData);
        setLoading(false);
    } catch (err) {
        console.error("Weather Fetch Error:", err);
        setError(`Failed to load live weather. Error: ${err.message}`);
        setLoading(false);
    }
  };

  // --- 2. WMO Code Mapping ---
  const mapWmoCode = (code) => {
      if (code === 0) return "Clear Sky";
      if (code >= 1 && code <= 3) return "Overcast"; // Mainly clear to Overcast grouped
      if (code === 45 || code === 48) return "Foggy";
      if (code >= 51 && code <= 55) return "Drizzle";
      if (code >= 61 && code <= 65) return "Rain";
      if (code >= 80 && code <= 82) return "Rain Showers";
      if (code >= 71 && code <= 77) return "Snow";
      if (code >= 95) return "Thunderstorm";
      return "Cloudy";
  };

  // --- 3. Risk Calculation (Uses live data, removed humidity) ---
  const calculateTriggerRisks = (data) => {
    // Risk thresholds
    const pressureLow = data.pressure < 1000;
    // const humidityHigh = data.humidity > 70; // REMOVED
    const windHigh = data.windSpeed > 20;
    const tempExtreme = data.temp < 5 || data.temp > 30;

    setTriggerRisks({
      pressure: pressureLow ? 'High' : data.pressure < 1010 ? 'Moderate' : 'Low',
      // humidity: humidityHigh ? 'High' : data.humidity > 50 ? 'Moderate' : 'Low', // REMOVED
      wind: windHigh ? 'High' : data.windSpeed > 15 ? 'Moderate' : 'Low',
      temperature: tempExtreme ? 'High' : 'Moderate',
    });
  };

  // --- 4. UI Helpers ---
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

  const getWeatherIcon = (condition, isDay) => {
    const lowerCaseCondition = condition.toLowerCase();
    if (lowerCaseCondition.includes('clear')) return isDay ? <Sun size={24} /> : <Moon size={24} />;
    if (lowerCaseCondition.includes('rain') || lowerCaseCondition.includes('shower') || lowerCaseCondition.includes('drizzle')) return <CloudRain size={24} />;
    return <Cloud size={24} />;
  };


  // --- Loading/Error Render (Robust) ---
  if (loading) return <div className="flex flex-col items-center justify-center h-[50vh] text-slate-400">
    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }} className="mb-4">
      <Loader size={40} className="text-blue-500"/>
    </motion.div>
    <p className="text-lg">Fetching live environmental data...</p>
  </div>;
  
  if (error) return <div className="text-center text-red-400 mt-10 p-6 bg-slate-700 rounded-xl max-w-md mx-auto">
    <AlertCircle className="inline-block mb-3"/>
    <p className="font-bold">Data Load Error</p>
    <p className="text-sm">{error}</p>
    <button onClick={fetchWeather} className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg">Retry</button>
  </div>;


  // --- Main Content Render (Using live data from 'weather') ---
  return (
    <div className="w-full max-w-2xl mx-auto p-4 md:p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-center mb-2 text-slate-100">Symptom Triggers</h1>
      <p className="text-center text-slate-400 mb-8">How weather affects your health</p>

      {/* Weather Card */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-blue-600 to-blue-500 rounded-2xl p-8 mb-8 shadow-2xl"
      >
        <p className="text-blue-100 text-center mb-4">{weather.location}</p>
        <h2 className="text-7xl font-bold text-white text-center mb-4">{weather.temp}°C</h2>
        <p className="text-2xl text-blue-50 text-center mb-6 flex items-center justify-center gap-2">
          {getWeatherIcon(weather.condition, weather.isDay)}
          {weather.condition}
        </p>
        
        {/* Weather Details Grid (Now only showing 3 core values) */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center">
            <Wind size={24} className="mx-auto mb-2 text-blue-100" />
            <p className="text-sm text-blue-100 opacity-75">Wind Speed</p>
            <p className="text-xl font-bold text-white">{weather.windSpeed} km/h</p>
          </div>
          
          {/* HUMIDITY REMOVED */}
          
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center">
            <Gauge size={24} className="mx-auto mb-2 text-blue-100" />
            <p className="text-sm text-blue-100 opacity-75">Pressure</p>
            <p className="text-xl font-bold text-white">{weather.pressure} mb</p>
          </div>
          
          {/* VISIBILITY REMOVED */}
          
          <div className="bg-white bg-opacity-10 rounded-xl p-4 text-center">
            {getWeatherIcon(weather.condition, weather.isDay)}
            <p className="text-sm text-blue-100 opacity-75">Condition</p>
            <p className="text-lg font-bold text-white">{weather.condition}</p>
          </div>
        </div>
      </motion.div>

      {/* Trigger Risk Analysis (Removed Humidity) */}
      <div className="space-y-4 mb-8">
        <h3 className="text-2xl font-bold text-slate-100 mb-4">Your Trigger Risks</h3>
        
        {Object.entries(triggerRisks).map(([key, risk]) => {
          // Skip if key is humidity
          if (key === 'humidity') return null;

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

      {/* Recommendations (Removed Humidity) */}
      <div className="bg-slate-700 bg-opacity-50 rounded-2xl p-6 border border-slate-600">
        <h3 className="text-xl font-bold text-slate-100 mb-4">Recommendations</h3>
        <ul className="space-y-3 text-slate-300">
          
          {triggerRisks.pressure === 'High' && (
             <li className="flex gap-3">
               <span className="text-blue-400">✓</span>
               <span>Watch for pressure headaches with current barometric conditions</span>
             </li>
          )}
          {triggerRisks.wind === 'High' && (
             <li className="flex gap-3">
               <span className="text-blue-400">✓</span>
               <span>Consider indoor activities given wind conditions</span>
             </li>
          )}
          <li className="flex gap-3">
            <span className="text-blue-400">✓</span>
            <span>Use Dr. Neura if symptoms develop</span>
          </li>
        </ul>
      </div>
    </div>
  );
}