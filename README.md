# RarePulse AI: Longitudinal Rare Disease Detection Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-00B4D8.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade Clinical Decision Support System (CDSS) for the **early detection of rare diseases from longitudinal Electronic Health Record (EHR) sequences** under extreme class imbalance (<1.5% positive prevalence).

---

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install streamlit torch pandas numpy scikit-learn
```

### Step 2: Launch Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🩺 System Architecture & Features

### 1. Clean Native UI & Theme Switcher
- Clean, stable native Streamlit interface with standard input containers and markdown headers.
- **Theme Palette Options**:
  - **Deep Slate Navy** (`#0B0F17` / `#161F30`, Teal accent `#0D9488`)
  - **Gemini Pitch Dark** (`#131314` / `#1E1F20`, Blue accent `#4E8DF5`)
  - **Clinical Dark Teal** (`#061A1E` / `#0D2D35`, Emerald accent `#14B8A6`)
  - **Light Medical Theme** (`#F8FAFC` / `#FFFFFF`, Blue accent `#0284C7`)

### 2. Dual-Authentication Flow
- 🩺 **Clinician / Doctor Portal**: Physician ID (`DOC-2026-X`) + **⚡ Demo Doctor Login**.
- 👤 **Patient Portal**: Patient ID (`PAT_0962`) + **⚡ Demo Patient Login**.

### 3. Longitudinal Patient Records & Quick Actions
- Interactive cohort selector for:
  - **`PAT_0962`**: Transthyretin Amyloidosis (*Progressive Neuropathy & Cardiac Biomarker Spike*)
  - **`PAT_0114`**: Fabry Disease (*Unexplained Microalbuminuria & Burning Extremity Pain*)
  - **`PAT_0480`**: Healthy Control (*Routine Primary Care Checkup*)
- Custom EHR file uploader (`.csv`, `.json`, `.txt`).
- Quick analysis buttons:
  - `[⚡ Predict Rare Condition Risk]`
  - `[📋 View 5-Year Symptom Timeline]`
  - `[🔍 Show Clinical Rationale & Attribution]`

### 4. Role-Tailored Clinical Intelligence
- **Doctor Mode**: Neural trajectory reports with exact ICD-10 sequences (`G62.9` $\to$ `R00.0` $\to$ `E85.82`), visit-level attention attribution percentages ($46.0\%$, $38.0\%$), lead-time savings ($\sim 3.4$ years earlier), and confirmatory testing workup ($^{99m}\text{Tc-PYP}$ scintigraphy, $\alpha$-Gal A assays, *TTR*/*GLA* genetic panels).
- **Patient Mode**: Reassuring, empathetic language explaining connected bodily signals, questions for their physician, and recommended specialist types.

---

## 💻 Standalone PyTorch CLI Prototype
To run the pure PyTorch training, evaluation (PR-AUC, Balanced Accuracy), and terminal alert generator:
```bash
python rarepulse_ai.py
```
