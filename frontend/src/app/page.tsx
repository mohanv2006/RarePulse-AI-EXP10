import React from 'react';
import Link from 'next/link';
import { Activity, ShieldCheck, ArrowRight, Zap, Target, GitCommit } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between">
      <header className="border-b border-slate-800 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-teal-500/20 border border-teal-500/30 text-teal-400 flex items-center justify-center">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-white flex items-center gap-2">
              RarePulse <span className="text-xs px-2 py-0.5 rounded bg-teal-500/20 text-teal-300 font-mono">AI</span>
            </h1>
            <p className="text-xs text-slate-400">Longitudinal Rare Disease Detection System</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="px-4 py-2 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-semibold transition shadow flex items-center gap-1.5">
            Launch Clinical Dashboard <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-16 text-center space-y-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-teal-500/10 text-teal-300 border border-teal-500/20">
          <Zap className="w-3.5 h-3.5" /> Breakthrough Clinical AI for Extreme Class Imbalance (&lt;1.5% Prevalence)
        </div>

        <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white leading-tight">
          Detecting Rare Diseases <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400">
            2.7+ Years Earlier
          </span> from Longitudinal EHR Trajectories
        </h2>

        <p className="text-slate-400 text-sm md:text-base max-w-2xl mx-auto leading-relaxed">
          RarePulse AI combines temporal GRU networks, masked additive attention pooling, and dynamically modulated binary focal loss to identify subtle multi-year prodromes in massive longitudinal electronic health records.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-6 text-left">
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
            <Target className="w-6 h-6 text-teal-400 mb-2" />
            <h3 className="font-semibold text-white text-sm">Focal Loss Optimization</h3>
            <p className="text-xs text-slate-400 mt-1">Solves 1:100 extreme imbalance by suppressing easy-negative gradients by up to 100x.</p>
          </div>

          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
            <GitCommit className="w-6 h-6 text-cyan-400 mb-2" />
            <h3 className="font-semibold text-white text-sm">Visit-Level Attention XAI</h3>
            <p className="text-xs text-slate-400 mt-1">Extracts exact historical encounters triggering the alert with clinical rationale.</p>
          </div>

          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800">
            <Activity className="w-6 h-6 text-purple-400 mb-2" />
            <h3 className="font-semibold text-white text-sm">Diagnostic Odyssey Shortener</h3>
            <p className="text-xs text-slate-400 mt-1">Shortens traditional 5-7 year diagnostic delays down to initial prodromal encounters.</p>
          </div>
        </div>

        <div className="pt-8">
          <Link href="/dashboard" className="px-6 py-3 rounded-xl bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-bold text-sm hover:opacity-95 transition shadow-lg inline-flex items-center gap-2">
            Open Clinical CDSS Console <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </main>

      <footer className="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
        RarePulse AI Clinical Prototype &copy; 2026. HIPAA-Deidentified Compliant CDSS Architecture.
      </footer>
    </div>
  );
}
