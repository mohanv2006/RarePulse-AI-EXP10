/**
 * RarePulse AI - Clinical Frontend Application Logic
 */

let currentPatientId = null;
let currentSearchTerm = "";
let prChartInstance = null;

// Preset Data for Ingestion Simulator
const SIMULATION_PRESETS = {
  amyloidosis: {
    name: "Julian Montgomery",
    age: 62,
    gender: "Male",
    encounters: [
      {
        date: "2021-03-15",
        days_from_baseline: 74,
        icd_code: "G62.9",
        diagnosis: "Peripheral neuropathy / Bilateral carpal tunnel syndrome",
        abnormal_lab_flag: 0,
        lab_notes: "Normal complete metabolic panel",
        medications: "Gabapentin 300mg"
      },
      {
        date: "2022-11-04",
        days_from_baseline: 673,
        icd_code: "M54.5",
        diagnosis: "Low back pain",
        abnormal_lab_flag: 0,
        lab_notes: "Normal lab panel",
        medications: "PRN Ibuprofen"
      },
      {
        date: "2023-09-18",
        days_from_baseline: 991,
        icd_code: "R00.0",
        diagnosis: "Unexplained tachycardia & palpitations",
        abnormal_lab_flag: 1,
        lab_notes: "Elevated NT-proBNP (1,450 pg/mL)",
        medications: "Metoprolol Succinate 25mg"
      },
      {
        date: "2025-01-20",
        days_from_baseline: 1481,
        icd_code: "E85.82",
        diagnosis: "Wild-type transthyretin amyloidosis (Cardiomyopathy)",
        abnormal_lab_flag: 1,
        lab_notes: "Markedly elevated NT-proBNP (4,200 pg/mL), PYP Scan Positive",
        medications: "Tafamidis 61mg"
      }
    ]
  },
  fabry: {
    name: "Caleb Sterling",
    age: 44,
    gender: "Male",
    encounters: [
      {
        date: "2021-06-10",
        days_from_baseline: 161,
        icd_code: "R20.2",
        diagnosis: "Acroparesthesia / Episodic neuropathic burning pain",
        abnormal_lab_flag: 0,
        lab_notes: "Normal CRP and ESR",
        medications: "Carbamazepine 200mg"
      },
      {
        date: "2023-04-12",
        days_from_baseline: 832,
        icd_code: "N04.9",
        diagnosis: "Nephrotic syndrome & persistent microalbuminuria",
        abnormal_lab_flag: 1,
        lab_notes: "Elevated Urine Protein/Creatinine (0.85 g/g)",
        medications: "Lisinopril 10mg"
      },
      {
        date: "2024-10-05",
        days_from_baseline: 1373,
        icd_code: "I42.0",
        diagnosis: "Hypertrophic cardiomyopathy & Angiokeratomas",
        abnormal_lab_flag: 1,
        lab_notes: "Low alpha-galactosidase A enzyme activity",
        medications: "Agalsidase beta infusions"
      }
    ]
  },
  scleroderma: {
    name: "Elena Rostova",
    age: 51,
    gender: "Female",
    encounters: [
      {
        date: "2021-04-20",
        days_from_baseline: 110,
        icd_code: "I73.0",
        diagnosis: "Raynaud's phenomenon / Digital blanching and pain",
        abnormal_lab_flag: 0,
        lab_notes: "Borderline ANA 1:160 speckled",
        medications: "Nifedipine 30mg ER"
      },
      {
        date: "2023-01-15",
        days_from_baseline: 745,
        icd_code: "K21.0",
        diagnosis: "Severe esophageal dysmotility & GERD",
        abnormal_lab_flag: 1,
        lab_notes: "Positive Anti-Scl-70 (Topoisomerase I) Antibodies",
        medications: "Omeprazole 40mg BID"
      },
      {
        date: "2024-11-28",
        days_from_baseline: 1427,
        icd_code: "M34.0",
        diagnosis: "Diffuse cutaneous systemic sclerosis with ILD",
        abnormal_lab_flag: 1,
        lab_notes: "High ESR/CRP, Decreased DLCO (52%)",
        medications: "Mycophenolate Mofetil 1000mg BID"
      }
    ]
  },
  control: {
    name: "Marcus Vance",
    age: 55,
    gender: "Male",
    encounters: [
      {
        date: "2021-08-12",
        days_from_baseline: 224,
        icd_code: "I10",
        diagnosis: "Essential (primary) hypertension",
        abnormal_lab_flag: 0,
        lab_notes: "Electrolytes & Creatinine normal",
        medications: "Amlodipine 5mg"
      },
      {
        date: "2023-05-19",
        days_from_baseline: 869,
        icd_code: "E11.9",
        diagnosis: "Type 2 diabetes mellitus without complications",
        abnormal_lab_flag: 0,
        lab_notes: "HbA1c 6.4% (Controlled)",
        medications: "Metformin 500mg"
      },
      {
        date: "2025-02-10",
        days_from_baseline: 1501,
        icd_code: "Z00.00",
        diagnosis: "Encounter for general adult medical examination",
        abnormal_lab_flag: 0,
        lab_notes: "Comprehensive metabolic panel normal",
        medications: "Routine vitamins"
      }
    ]
  }
};

// -----------------------------------------------------------------------------
// SPA Navigation Router
// -----------------------------------------------------------------------------
function navigate(viewName) {
  const views = ['dashboard', 'patient', 'metrics', 'simulate'];
  const titles = {
    dashboard: 'Clinical Surveillance Dashboard',
    patient: 'Longitudinal Patient Trajectory & Explainability (XAI)',
    metrics: 'Model Diagnostics & Imbalance Loss Analysis',
    simulate: 'Longitudinal EHR Simulator & Real-Time Inference'
  };

  views.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) {
      if (v === viewName || (viewName === 'patient-view' && v === 'patient')) {
        el.classList.remove('hidden');
      } else {
        el.classList.add('hidden');
      }
    }
  });

  // Update Sidebar Active Styles
  ['dashboard', 'patient-view', 'metrics', 'simulate'].forEach(navId => {
    const btn = document.getElementById(`nav-${navId}`);
    if (btn) {
      if (navId === viewName || (navId === 'patient-view' && viewName === 'patient')) {
        btn.className = "w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg font-medium transition-all text-brand-400 bg-brand-500/10 border border-brand-500/20";
      } else {
        btn.className = "w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg font-medium transition-all text-slate-400 hover:text-white hover:bg-clinical-hover";
      }
    }
  });

  const pageKey = viewName === 'patient-view' ? 'patient' : viewName;
  const pageTitleEl = document.getElementById('page-title');
  if (pageTitleEl && titles[pageKey]) {
    pageTitleEl.innerText = titles[pageKey];
  }

  // View-specific initializers
  if (viewName === 'dashboard') {
    loadDashboardStats();
    loadPatientsTable();
  } else if (viewName === 'metrics') {
    loadMetricsView();
  } else if (viewName === 'simulate') {
    if (!document.getElementById('sim-encounters-list').children.length) {
      loadSimulationPreset('amyloidosis');
    }
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
  setTimeout(() => lucide.createIcons(), 50);
}

// -----------------------------------------------------------------------------
// View 1: Clinical Dashboard Functions
// -----------------------------------------------------------------------------
async function loadDashboardStats() {
  try {
    const res = await fetch('/api/dashboard/stats');
    const data = await res.json();
    
    document.getElementById('kpi-total-screened').innerText = data.kpis.total_screened.toLocaleString();
    document.getElementById('kpi-flagged-cases').innerText = data.kpis.flagged_high_risk;
    document.getElementById('kpi-lead-time').innerText = `${data.kpis.avg_lead_time_years} Years`;
    document.getElementById('kpi-prauc').innerText = data.kpis.pr_auc.toFixed(4);
    document.getElementById('kpi-bal-acc').innerText = `${(data.kpis.balanced_accuracy * 100).toFixed(1)}%`;
  } catch (err) {
    console.error('Error fetching dashboard stats:', err);
  }
}

async function loadPatientsTable() {
  const tableBody = document.getElementById('patients-table-body');
  const riskFilter = document.getElementById('filter-risk').value;
  const conditionFilter = document.getElementById('filter-condition').value;

  try {
    const params = new URLSearchParams({
      search: currentSearchTerm,
      risk_tier: riskFilter,
      condition: conditionFilter,
      limit: 100
    });

    const res = await fetch(`/api/patients?${params.toString()}`);
    const data = await res.json();

    document.getElementById('patient-count-badge').innerText = data.total;

    if (data.patients.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="7" class="py-8 text-center text-slate-400">
            No patients found matching the selected filters.
          </td>
        </tr>
      `;
      return;
    }

    tableBody.innerHTML = data.patients.map(p => {
      const isHighRisk = p.risk_tier === "High Risk";
      const riskClass = isHighRisk 
        ? "bg-rose-500/20 text-rose-400 border border-rose-500/30" 
        : (p.risk_tier === "Moderate Risk" ? "bg-amber-500/20 text-amber-400 border border-amber-500/30" : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20");

      const triageClass = p.triage_status === "Review Needed"
        ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
        : (p.triage_status === "Genetic Testing Ordered" ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30" : "bg-slate-800 text-slate-300 border-slate-700");

      const leadTimeBadge = p.lead_time_years > 0
        ? `<span class="text-brand-400 font-mono font-semibold">+${p.lead_time_years} yrs earlier</span>`
        : `<span class="text-slate-500 font-mono">--</span>`;

      return `
        <tr class="hover:bg-clinical-hover/60 transition cursor-pointer" onclick="openPatientDetail('${p.patient_id}')">
          <td class="py-3.5 px-4">
            <div class="font-mono font-bold text-white">${p.patient_id}</div>
            <div class="text-[11px] text-slate-400 font-sans">${p.name}</div>
          </td>
          <td class="py-3.5 px-4 text-slate-300">
            ${p.age}y / ${p.gender}
          </td>
          <td class="py-3.5 px-4">
            <span class="font-medium text-slate-200">${p.target_condition}</span>
            <div class="text-[10px] text-slate-500">${p.total_encounters} encounters (5y)</div>
          </td>
          <td class="py-3.5 px-4">
            <span class="px-2 py-0.5 rounded-full text-xs font-mono font-semibold ${riskClass}">
              ${p.risk_score.toFixed(1)}%
            </span>
          </td>
          <td class="py-3.5 px-4">
            ${leadTimeBadge}
          </td>
          <td class="py-3.5 px-4">
            <span class="px-2 py-0.5 rounded-full text-[11px] font-medium border ${triageClass}">
              ${p.triage_status}
            </span>
          </td>
          <td class="py-3.5 px-4 text-right" onclick="event.stopPropagation()">
            <button onclick="openPatientDetail('${p.patient_id}')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-brand-600 hover:text-white text-slate-300 text-xs transition">
              Inspect
            </button>
          </td>
        </tr>
      `;
    }).join('');

    lucide.createIcons();
  } catch (err) {
    console.error('Error loading patients roster:', err);
  }
}

function handleSearch(term) {
  currentSearchTerm = term;
  loadPatientsTable();
}

// -----------------------------------------------------------------------------
// View 2: Patient Longitudinal Detail & XAI
// -----------------------------------------------------------------------------
async function openPatientDetail(patientId) {
  currentPatientId = patientId;
  navigate('patient-view');

  try {
    const res = await fetch(`/api/patients/${patientId}`);
    const data = await res.json();
    const p = data.patient;
    const odyssey = data.diagnostic_odyssey;

    // Set Patient Headers
    document.getElementById('detail-patient-id').innerText = p.patient_id;
    document.getElementById('detail-patient-name').innerText = p.name;
    document.getElementById('detail-patient-meta').innerText = `${p.age} y/o ${p.gender}`;
    document.getElementById('detail-physician').innerText = p.primary_physician;

    const riskBadge = document.getElementById('detail-risk-badge');
    if (p.risk_score >= 70) {
      riskBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30";
      riskBadge.innerText = `${p.risk_score.toFixed(1)}% Risk (Critical Alert)`;
    } else if (p.risk_score >= 35) {
      riskBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30";
      riskBadge.innerText = `${p.risk_score.toFixed(1)}% Risk (Moderate)`;
    } else {
      riskBadge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
      riskBadge.innerText = `${p.risk_score.toFixed(1)}% Risk (Low Risk)`;
    }

    document.getElementById('detail-triage-badge').innerText = p.triage_status;
    document.getElementById('update-triage-select').value = p.triage_status;

    // Set Diagnostic Odyssey Comparison
    if (p.rare_label === 1) {
      document.getElementById('odyssey-lead-time-tag').innerText = `Early Detection Window: ${p.lead_time_years} Years Ahead`;
      document.getElementById('standard-odyssey-val').innerText = `~${p.standard_odyssey_years} Years`;
      document.getElementById('rarepulse-detection-val').innerText = `~${p.rarepulse_detection_years} Years (${p.encounters[1]?.year || 'Year 2-3'})`;
      
      const pctWidth = Math.min(90, Math.round((p.rarepulse_detection_years / p.standard_odyssey_years) * 100));
      document.getElementById('rarepulse-progress-bar').style.width = `${pctWidth}%`;
    } else {
      document.getElementById('odyssey-lead-time-tag').innerText = `Control Patient - No Odyssey Identified`;
      document.getElementById('standard-odyssey-val').innerText = `N/A`;
      document.getElementById('rarepulse-detection-val').innerText = `Low Risk Profile`;
      document.getElementById('rarepulse-progress-bar').style.width = `15%`;
    }

    // Render XAI Attention Breakdown
    renderAttentionBreakdown(p.visit_attributions, p.top_triggers);

    // Render 5-Year Longitudinal Care Timeline
    renderCareTimeline(p.encounters, p.visit_attributions, p.top_triggers);

    lucide.createIcons();
  } catch (err) {
    console.error('Error fetching patient detail:', err);
  }
}

function renderAttentionBreakdown(attributions, topTriggers) {
  const container = document.getElementById('attention-breakdown-container');
  if (!attributions || attributions.length === 0) {
    container.innerHTML = `<p class="text-xs text-slate-400">No attention breakdown available.</p>`;
    return;
  }

  const triggerIds = (topTriggers || []).map(t => t.encounter_id);

  container.innerHTML = attributions.map((attr, idx) => {
    const isTopTrigger = triggerIds.includes(attr.encounter_id);
    const scorePct = parseFloat(attr.attention_percentage);
    const borderClass = isTopTrigger ? 'border-amber-500/50 bg-amber-950/20' : 'border-slate-800 bg-slate-950/60';
    const tagBadge = isTopTrigger 
      ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono">PRIMARY TRIGGER</span>`
      : ``;

    return `
      <div class="p-3.5 rounded-lg border ${borderClass} transition flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div class="space-y-1 flex-1">
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs text-slate-400 font-semibold">${attr.encounter_id}</span>
            <span class="text-xs text-slate-300">(${attr.date})</span>
            <span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-cyan-300">${attr.icd_code}</span>
            ${tagBadge}
          </div>
          <div class="text-xs text-slate-200 font-medium">${attr.diagnosis}</div>
          <div class="text-[11px] text-slate-400">Labs: <span class="${attr.abnormal_lab_flag ? 'text-amber-400 font-semibold' : 'text-slate-300'}">${attr.lab_notes}</span></div>
        </div>

        <div class="w-full md:w-48 shrink-0">
          <div class="flex justify-between text-[11px] font-mono mb-1">
            <span class="text-slate-400">Attention Attribution</span>
            <span class="${isTopTrigger ? 'text-amber-300 font-bold' : 'text-slate-300'}">${attr.attention_percentage}</span>
          </div>
          <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
            <div class="${isTopTrigger ? 'bg-amber-400' : 'bg-slate-500'} h-full rounded-full" style="width: ${Math.max(4, scorePct)}%;"></div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderCareTimeline(encounters, attributions, topTriggers) {
  const container = document.getElementById('timeline-encounters-container');
  const triggerIds = (topTriggers || []).map(t => t.encounter_id);
  const attnMap = {};
  (attributions || []).forEach(a => { attnMap[a.encounter_id] = a; });

  container.innerHTML = encounters.map((enc, idx) => {
    const isTopTrigger = triggerIds.includes(enc.encounter_id);
    const attnData = attnMap[enc.encounter_id];
    const dotClass = isTopTrigger 
      ? 'bg-amber-400 border-amber-300 ring-4 ring-amber-500/20' 
      : (enc.abnormal_lab_flag ? 'bg-rose-400 border-rose-300 ring-2 ring-rose-500/20' : 'bg-brand-500 border-brand-400');

    return `
      <div class="relative group">
        <!-- Timeline Marker Dot -->
        <div class="absolute -left-[31px] top-1.5 w-3.5 h-3.5 rounded-full border-2 ${dotClass}"></div>
        
        <!-- Encounter Card -->
        <div class="bg-slate-950/80 border ${isTopTrigger ? 'border-amber-500/40 bg-amber-950/10' : 'border-slate-800'} rounded-xl p-4 transition hover:border-slate-700">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
            <div class="flex items-center gap-2">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-slate-800 text-brand-300">${enc.year}</span>
              <span class="text-xs font-semibold text-white">${enc.date}</span>
              <span class="text-xs text-slate-400 font-mono">(${enc.encounter_id})</span>
            </div>
            ${attnData ? `<span class="text-[11px] font-mono text-slate-400">Attention: <span class="${isTopTrigger ? 'text-amber-300 font-bold' : 'text-slate-300'}">${attnData.attention_percentage}</span></span>` : ''}
          </div>

          <div class="space-y-1.5 text-xs">
            <div class="flex items-center gap-2">
              <span class="px-1.5 py-0.5 rounded text-[11px] font-mono bg-teal-500/10 text-brand-400 font-bold border border-teal-500/20">${enc.icd_code}</span>
              <span class="text-slate-100 font-semibold">${enc.diagnosis}</span>
            </div>

            <div class="text-slate-400">
              <span class="text-slate-500 font-medium">Lab Panel:</span> 
              <span class="${enc.abnormal_lab_flag ? 'text-rose-400 font-semibold' : 'text-slate-300'}">${enc.lab_notes}</span>
            </div>

            <div class="text-slate-400">
              <span class="text-slate-500 font-medium">Medications:</span> 
              <span class="text-slate-300">${enc.medications}</span>
            </div>

            ${enc.clinical_notes ? `<div class="text-slate-400 italic text-[11px] pt-1 border-t border-slate-900">"${enc.clinical_notes}"</div>` : ''}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

async function saveTriageStatus() {
  if (!currentPatientId) return;
  const newStatus = document.getElementById('update-triage-select').value;

  try {
    const res = await fetch(`/api/triage/${currentPatientId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus })
    });
    
    if (res.ok) {
      document.getElementById('detail-triage-badge').innerText = newStatus;
      alert(`Triage status for ${currentPatientId} updated to: ${newStatus}`);
    }
  } catch (err) {
    console.error('Error updating triage status:', err);
  }
}

// -----------------------------------------------------------------------------
// View 3: Model Diagnostics & Metrics
// -----------------------------------------------------------------------------
async function loadMetricsView() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    const m = data.metrics;

    document.getElementById('metrics-prauc').innerText = m.pr_auc.toFixed(4);
    document.getElementById('metrics-rocauc').innerText = m.roc_auc.toFixed(4);
    document.getElementById('metrics-balacc').innerText = `${(m.balanced_accuracy * 100).toFixed(2)}%`;

    const cm = data.confusion_matrix;
    document.getElementById('cm-tn').innerText = cm[0][0];
    document.getElementById('cm-fp').innerText = cm[0][1];
    document.getElementById('cm-fn').innerText = cm[1][0];
    document.getElementById('cm-tp').innerText = cm[1][1];

    // Render PR Curve Chart
    renderPRChart(data.pr_curve_data);
  } catch (err) {
    console.error('Error loading metrics:', err);
  }
}

function renderPRChart(curveData) {
  const ctx = document.getElementById('pr-chart');
  if (!ctx) return;

  if (prChartInstance) {
    prChartInstance.destroy();
  }

  const labels = curveData.map(d => d.recall.toFixed(1));
  const precisions = curveData.map(d => d.precision);

  prChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'RarePulse AI (Focal Loss)',
          data: precisions,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          borderWidth: 2.5,
          fill: true,
          tension: 0.2
        },
        {
          label: 'Standard Cross-Entropy Baseline',
          data: [0.45, 0.42, 0.38, 0.30, 0.22, 0.15, 0.08],
          borderColor: '#f43f5e',
          borderDash: [5, 5],
          borderWidth: 1.5,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Recall', color: '#94a3b8' },
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          title: { display: true, text: 'Precision', color: '#94a3b8' },
          min: 0,
          max: 1.05,
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#f1f5f9' }
        }
      }
    }
  });
}

// -----------------------------------------------------------------------------
// View 4: EHR Ingestion Simulator & Inference
// -----------------------------------------------------------------------------
function loadSimulationPreset(presetKey) {
  const preset = SIMULATION_PRESETS[presetKey];
  if (!preset) return;

  document.getElementById('sim-name').value = preset.name;
  document.getElementById('sim-age').value = preset.age;
  document.getElementById('sim-gender').value = preset.gender;

  const container = document.getElementById('sim-encounters-list');
  container.innerHTML = '';

  preset.encounters.forEach(enc => {
    addSimulatedEncounterRow(enc);
  });

  document.getElementById('sim-results-card').classList.add('hidden');
  lucide.createIcons();
}

function addSimulatedEncounterRow(data = null) {
  const container = document.getElementById('sim-encounters-list');
  const rowId = `sim-row-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`;

  const defaultData = data || {
    date: new Date().toISOString().split('T')[0],
    days_from_baseline: 100,
    icd_code: "Z00.00",
    diagnosis: "Encounter for general adult examination",
    abnormal_lab_flag: 0,
    lab_notes: "Routine normal panel",
    medications: "None"
  };

  const rowHtml = `
    <div id="${rowId}" class="p-3.5 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
      <div class="flex items-center justify-between">
        <span class="text-xs font-mono font-bold text-teal-400">Encounter Record</span>
        <button onclick="document.getElementById('${rowId}').remove()" class="text-xs text-rose-400 hover:text-rose-300">
          Remove
        </button>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs">
        <div>
          <label class="text-[10px] text-slate-400 block">Date</label>
          <input type="date" value="${defaultData.date}" class="sim-enc-date w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-white">
        </div>
        <div>
          <label class="text-[10px] text-slate-400 block">ICD-10 Code</label>
          <input type="text" value="${defaultData.icd_code}" class="sim-enc-icd w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-white font-mono">
        </div>
        <div class="sm:col-span-2">
          <label class="text-[10px] text-slate-400 block">Diagnosis Description</label>
          <input type="text" value="${defaultData.diagnosis}" class="sim-enc-diag w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-white">
        </div>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
        <div>
          <label class="text-[10px] text-slate-400 block">Abnormal Lab Flag</label>
          <select class="sim-enc-labflag w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-white">
            <option value="0" ${defaultData.abnormal_lab_flag === 0 ? 'selected' : ''}>0 - Normal / Reference Range</option>
            <option value="1" ${defaultData.abnormal_lab_flag === 1 ? 'selected' : ''}>1 - Elevated Biomarkers / Abnormal</option>
          </select>
        </div>
        <div>
          <label class="text-[10px] text-slate-400 block">Lab Notes / Biomarkers</label>
          <input type="text" value="${defaultData.lab_notes}" class="sim-enc-labnotes w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-white">
        </div>
        <div>
          <label class="text-[10px] text-slate-400 block">Medications Prescribed</label>
          <input type="text" value="${defaultData.medications}" class="sim-enc-meds w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-white">
        </div>
      </div>
    </div>
  `;

  container.insertAdjacentHTML('beforeend', rowHtml);
}

async function executeSimulationInference() {
  const patientName = document.getElementById('sim-name').value;
  const age = parseInt(document.getElementById('sim-age').value) || 50;
  const gender = document.getElementById('sim-gender').value;

  const container = document.getElementById('sim-encounters-list');
  const rows = container.querySelectorAll('[id^="sim-row-"]');

  if (rows.length === 0) {
    alert('Please add at least 1 encounter to run inference.');
    return;
  }

  const encounters = [];
  rows.forEach((row, idx) => {
    const date = row.querySelector('.sim-enc-date').value;
    const icd_code = row.querySelector('.sim-enc-icd').value;
    const diagnosis = row.querySelector('.sim-enc-diag').value;
    const abnormal_lab_flag = parseInt(row.querySelector('.sim-enc-labflag').value);
    const lab_notes = row.querySelector('.sim-enc-labnotes').value;
    const medications = row.querySelector('.sim-enc-meds').value;

    encounters.push({
      date,
      days_from_baseline: idx * 250 + 50,
      icd_code,
      diagnosis,
      abnormal_lab_flag,
      lab_notes,
      medications
    });
  });

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_name: patientName,
        age,
        gender,
        encounters
      })
    });

    const data = await res.json();
    const pred = data.prediction;

    const resultsCard = document.getElementById('sim-results-card');
    resultsCard.classList.remove('hidden');

    document.getElementById('sim-res-prob').innerText = `${pred.risk_score_pct.toFixed(1)}%`;
    
    const tierEl = document.getElementById('sim-res-tier');
    tierEl.innerText = pred.risk_tier;
    if (pred.risk_tier === "High Risk") {
      tierEl.className = "text-2xl font-bold font-mono text-rose-400 mt-1";
    } else if (pred.risk_tier === "Moderate Risk") {
      tierEl.className = "text-2xl font-bold font-mono text-amber-400 mt-1";
    } else {
      tierEl.className = "text-2xl font-bold font-mono text-emerald-400 mt-1";
    }

    document.getElementById('sim-res-condition').innerText = data.likely_condition;

    // Render Simulation Attributions
    const attrContainer = document.getElementById('sim-res-attributions');
    const triggerIds = (pred.top_triggers || []).map(t => t.encounter_id);

    attrContainer.innerHTML = pred.visit_attributions.map((attr, idx) => {
      const isTop = triggerIds.includes(attr.encounter_id);
      return `
        <div class="p-3 bg-slate-950 border ${isTop ? 'border-amber-500/50 bg-amber-950/20' : 'border-slate-800'} rounded-lg flex items-center justify-between text-xs">
          <div>
            <span class="font-bold text-white font-mono">${attr.icd_code}</span> - ${attr.diagnosis}
            <div class="text-[11px] text-slate-400">Date: ${attr.date} | Labs: ${attr.lab_notes}</div>
          </div>
          <div class="text-right">
            <span class="font-mono ${isTop ? 'text-amber-300 font-bold' : 'text-slate-300'}">${attr.attention_percentage}</span>
            ${isTop ? '<span class="block text-[10px] text-amber-400 font-semibold">PRIMARY TRIGGER</span>' : ''}
          </div>
        </div>
      `;
    }).join('');

    resultsCard.scrollIntoView({ behavior: 'smooth' });
    lucide.createIcons();
  } catch (err) {
    console.error('Error simulating inference:', err);
  }
}

function triggerSimulateAmyloidosis() {
  navigate('simulate');
  loadSimulationPreset('amyloidosis');
  setTimeout(() => executeSimulationInference(), 150);
}

// -----------------------------------------------------------------------------
// Initialization on Page Load
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  navigate('dashboard');
});
