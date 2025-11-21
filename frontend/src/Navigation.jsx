import React from 'react';
import { Music, Stethoscope, Cloud, Menu, X, Zap, Brain } from 'lucide-react';
import { useState } from 'react';

export default function Navigation({ currentPage, setCurrentPage, setMode, onAIClick }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '🏠' },
    { id: 'weather', label: 'Symptom Triggers', icon: Cloud, component: Cloud },
    { id: 'triggers', label: 'Track Triggers', icon: Zap, component: Zap },
    { id: 'doctor', label: 'Dr. Neura', icon: Stethoscope, component: Stethoscope },
    { id: 'music', label: 'Sonic Therapy', icon: Music, component: Music },
    // { id: 'login', label: 'Log In', icon: '🔑' }, // REMOVED: Login is only available pre-dashboard
  ];

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        className="md:hidden fixed top-4 right-4 z-50 p-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition"
      >
        {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Desktop Navigation */}
      <nav className="hidden md:flex gap-4 mb-6 flex-wrap justify-center">
        {navItems.map((item) => {
          const IconComponent = item.component;
          
          return (
            <button
              key={item.id}
              onClick={() => {
                // For static standalone pages, navigate directly to HTML files
                if (item.id === 'doctor') {
                  window.location.href = '/doctor.html';
                  return;
                }
                if (item.id === 'music') {
                  window.location.href = '/music.html';
                  return;
                }

                // Default: internal SPA routing
                setCurrentPage(item.id);
                setMobileMenuOpen(false);
              }}
              className={`nav-button flex items-center gap-2 rounded-lg font-semibold transition-all ${
                currentPage === item.id
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-200 hover:bg-slate-600'
              }`}
            >
              {IconComponent ? (
                <IconComponent className="nav-icon" />
              ) : (
                <span className="nav-icon inline-flex items-center justify-center">{item.icon}</span>
              )}
              <span className="hidden sm:inline">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Mobile Navigation Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed top-16 right-4 z-40 bg-slate-800 rounded-lg shadow-xl border border-slate-700 p-2 w-48">
          {navItems.map((item) => {
            const IconComponent = item.component;
            return (
              <button
                key={item.id}
                onClick={() => {
                  // Directly open standalone pages for doctor/music
                  if (item.id === 'doctor') {
                    window.location.href = '/doctor.html';
                    return;
                  }
                  if (item.id === 'music') {
                    window.location.href = '/music.html';
                    return;
                  }

                  // Default: internal SPA routing
                  setCurrentPage(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full nav-button flex items-center gap-3 rounded-lg font-semibold transition-all text-left ${
                  currentPage === item.id
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-200 hover:bg-slate-700'
                }`}
              >
                {IconComponent ? <IconComponent className="nav-icon" /> : <span className="nav-icon inline-flex items-center justify-center">{item.icon}</span>}
                {item.label}
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}