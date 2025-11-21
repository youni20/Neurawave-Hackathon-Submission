import React from 'react';
import { Music, Stethoscope, Cloud, Menu, X } from 'lucide-react';
import { useState } from 'react';

export default function Navigation({ currentPage, setCurrentPage }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '🏠' },
    { id: 'weather', label: 'Symptom Triggers', icon: Cloud, component: Cloud },
    { id: 'doctor', label: 'Dr. Neura', icon: Stethoscope, component: Stethoscope },
    { id: 'music', label: 'Sonic Therapy', icon: Music, component: Music },
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
                setCurrentPage(item.id);
                setMobileMenuOpen(false);
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition-all ${
                currentPage === item.id
                  ? 'bg-blue-500 text-white shadow-lg'
                  : 'bg-slate-700 text-gray-200 hover:bg-slate-600'
              }`}
            >
              {IconComponent ? <IconComponent size={20} /> : item.icon}
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
                  setCurrentPage(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-semibold transition-all text-left ${
                  currentPage === item.id
                    ? 'bg-blue-500 text-white'
                    : 'text-gray-200 hover:bg-slate-700'
                }`}
              >
                {IconComponent ? <IconComponent size={20} /> : item.icon}
                {item.label}
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}
