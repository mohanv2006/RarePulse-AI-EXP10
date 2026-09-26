"""
RarePulse AI - Data Generator Module
====================================
Generates realistic multi-year longitudinal EHR synthetic data for rare disease detection
directly integrated with `Rare_Diseases_Dataset.xlsx` (10 rare disease phenotypes).
Simulates ~1.5% prevalence of rare diseases along with realistic common outpatient/inpatient control records.
"""

import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

EXCEL_FILE_PATH = "Rare_Diseases_Dataset.xlsx"

def load_rare_diseases_knowledge_base(file_path: str = EXCEL_FILE_PATH) -> pd.DataFrame:
    """Reads phenotypic definitions directly from Rare_Diseases_Dataset.xlsx."""
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    parent_path = os.path.join("..", file_path)
    if os.path.exists(parent_path):
        return pd.read_excel(parent_path)
    return pd.DataFrame()

COMMON_ICD_CODES = [
    ("E11.9", "Type 2 diabetes mellitus without complications", "Endocrine"),
    ("I10", "Essential (primary) hypertension", "Cardiovascular"),
    ("J45.909", "Unspecified asthma, uncomplicated", "Respiratory"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis", "Gastrointestinal"),
    ("M54.5", "Low back pain", "Musculoskeletal"),
    ("R53.83", "Other fatigue", "General"),
    ("J06.9", "Acute upper respiratory infection, unspecified", "Respiratory"),
    ("K58.9", "Irritable bowel syndrome without diarrhea", "Gastrointestinal"),
    ("F41.1", "Generalized anxiety disorder", "Psychiatric"),
    ("E78.5", "Hyperlipidemia, unspecified", "Metabolic"),
    ("N39.0", "Urinary tract infection, site not specified", "Renal"),
    ("G43.909", "Migraine, unspecified, not intractable", "Neurological"),
    ("M25.50", "Pain in unspecified joint", "Musculoskeletal"),
    ("R05", "Cough", "Respiratory"),
    ("R10.9", "Unspecified abdominal pain", "Gastrointestinal"),
    ("K29.70", "Gastritis, unspecified, without bleeding", "Gastrointestinal"),
    ("H10.9", "Unspecified conjunctivitis", "Ophthalmology"),
    ("L20.9", "Atopic dermatitis, unspecified", "Dermatology"),
    ("R51.9", "Headache, unspecified", "Neurological"),
    ("Z00.00", "Encounter for general adult medical examination", "Routine"),
]

PRODROMAL_CODE_MAPPING = {
    "Fabry Disease": {
        "early_codes": [("R20.2", "Acroparesthesia / Extremity burning pain"), ("K58.9", "Irritable bowel syndrome misdiagnosis")],
        "mid_codes": [("N04.9", "Microalbuminuria / Nephrotic syndrome"), ("R00.0", "Tachycardia / Palpitations")],
        "lab_profile": "Elevated urine protein/creatinine ratio & low alpha-Gal A enzyme"
    },
    "Gaucher Disease (Type 1)": {
        "early_codes": [("R53.83", "Chronic fatigue"), ("R23.3", "Spontaneous ecchymoses / easy bruising")],
        "mid_codes": [("M89.9", "Bone pain / osteopenia"), ("R16.2", "Hepatosplenomegaly")],
        "lab_profile": "Thrombocytopenia (low platelets) & anemia on CBC"
    },
    "Pompe Disease (Late-Onset)": {
        "early_codes": [("M62.81", "Proximal muscle weakness"), ("R06.02", "Shortness of breath on exertion")],
        "mid_codes": [("R29.6", "Frequent falls / gait instability"), ("J96.10", "Chronic respiratory insufficiency")],
        "lab_profile": "Elevated serum creatine kinase (CK > 800 U/L)"
    },
    "Wilson Disease": {
        "early_codes": [("R25.1", "Intention tremor"), ("F41.1", "Anxiety / subtle behavioral changes")],
        "mid_codes": [("R47.81", "Dysarthria / slurred speech"), ("K76.9", "Unexplained liver disorder")],
        "lab_profile": "Persistently elevated transaminases (ALT/AST) & low ceruloplasmin"
    },
    "Amyloidosis, Hereditary Transthyretin (hATTR)": {
        "early_codes": [("G62.9", "Peripheral sensorimotor neuropathy"), ("G56.0", "Bilateral carpal tunnel syndrome")],
        "mid_codes": [("R00.0", "Cardiac palpitations"), ("I50.9", "Heart failure with preserved EF (HFpEF)")],
        "lab_profile": "Markedly elevated NT-proBNP (>1,400 pg/mL) & Troponin-T"
    },
    "Huntington's Disease": {
        "early_codes": [("F32.9", "Depressive episode / mood changes"), ("R27.8", "Mild coordination difficulty")],
        "mid_codes": [("R25.8", "Choreiform involuntary movements"), ("F03.90", "Unspecified dementia / cognitive decline")],
        "lab_profile": "Non-specific metabolic panel; caudate volume reduction flag"
    },
    "Amyotrophic Lateral Sclerosis (ALS)": {
        "early_codes": [("R25.2", "Muscle cramps and fasciculations"), ("M62.81", "Asymmetrical focal limb weakness / foot drop")],
        "mid_codes": [("R47.02", "Dysphasia / Bulbar speech difficulty"), ("R13.10", "Dysphagia / swallowing difficulty")],
        "lab_profile": "Normal routine labs; positive EMG active denervation"
    },
    "Vascular Ehlers-Danlos Syndrome (vEDS)": {
        "early_codes": [("R23.3", "Spontaneous extensive hematomas / easy bruising"), ("M35.7", "Hypermobility spectrum disorder")],
        "mid_codes": [("I72.9", "Aneurysm / arterial dissection"), ("J98.2", "Spontaneous pneumothorax")],
        "lab_profile": "Normal standard coagulation panel (PT/INR/aPTT)"
    },
    "Duchenne Muscular Dystrophy": {
        "early_codes": [("R62.0", "Delayed gross motor milestones"), ("R26.89", "Abnormal waddling gait / toe-walking")],
        "mid_codes": [("M62.81", "Proximal muscle pseudohypertrophy"), ("M41.9", "Progressive scoliosis")],
        "lab_profile": "Massively elevated serum creatine kinase (CK > 10,000 U/L)"
    },
    "Prader-Willi Syndrome": {
        "early_codes": [("P92.9", "Neonatal feeding difficulty / hypotonia"), ("R62.51", "Failure to thrive (infancy)")],
        "mid_codes": [("E66.01", "Morbid obesity due to insatiable hyperphagia"), ("E23.0", "Hypopituitarism / growth hormone deficiency")],
        "lab_profile": "Abnormal DNA methylation analysis (chromosome 15q11-q13)"
    }
}

FIRST_NAMES = ["Eleanor", "Marcus", "Sophia", "Liam", "Ava", "Noah", "Isabella", "Lucas", "Mia", "Ethan", "Charlotte", "Oliver", "Amelia", "James", "Harper", "Benjamin", "Evelyn", "Alexander", "Abigail", "Henry", "Emily", "Daniel", "Elizabeth", "Matthew", "Chloe", "Samuel"]
LAST_NAMES = ["Vance", "Chen", "Sterling", "Patel", "Kowalski", "Rodriguez", "O'Connor", "Dubois", "Al-Mansoor", "Takahashi", "Jensen", "Morales", "Goldstein", "Navarro", "Sinclair", "Bergstrom", "Mercer", "Adeyemi", "Petrov", "Washington"]
PHYSICIANS = ["Dr. Sarah Lin, MD (Cardiology)", "Dr. Robert Vance, MD (Rheumatology)", "Dr. Maya Patel, MD (Neurology)", "Dr. David Kim, MD (Internal Medicine)", "Dr. Elena Rostova, MD (Genetics)"]

@dataclass
class ClinicalEncounter:
    encounter_id: str
    patient_id: str
    visit_date: str
    year_offset: int
    icd_code: str
    diagnosis: str
    category: str
    abnormal_lab_flag: int
    lab_notes: str
    medication: str
    physician: str
    is_prodromal_signal: bool

@dataclass
class LongitudinalPatient:
    patient_id: str
    name: str
    age: int
    gender: str
    is_rare_case: bool
    condition_name: str
    icd_code_final: str
    prevalence_rate: str
    genetic_cause: str
    avg_diagnostic_delay: str
    encounters: List[ClinicalEncounter]
    total_visits: int

def generate_cohort(
    num_patients: int = 1000,
    prevalence: float = 0.015,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)
    
    df_rare = load_rare_diseases_knowledge_base()
    disease_rows = df_rare.to_dict(orient="records") if not df_rare.empty else []
    
    num_pos = int(round(num_patients * prevalence))
    num_neg = num_patients - num_pos
    
    labels = [1] * num_pos + [0] * num_neg
    random.shuffle(labels)

    patients_list = []
    all_visits = []
    common_codes_only = [c[0] for c in COMMON_ICD_CODES]

    start_date = datetime(2021, 1, 1)

    for idx, is_pos in enumerate(labels):
        pid = f"PAT-{idx+1:04d}"
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"
        gender = random.choice(["Female", "Male"])
        
        num_visits = random.randint(4, 9)
        visit_days = sorted(random.sample(range(1, 1800), num_visits))
        encounters = []

        if is_pos and disease_rows:
            d_info = random.choice(disease_rows)
            d_name = d_info.get("Disease Name", "Rare Condition")
            icd_final = d_info.get("ICD-10 Code", "R69")
            gen_cause = d_info.get("Inheritance / Genetic Cause", "Unknown")
            diag_delay = d_info.get("Avg. Diagnostic Delay", "4-7 years")
            prev_rate = d_info.get("Estimated Prevalence", "1 in 50,000")
            
            p_map = PRODROMAL_CODE_MAPPING.get(d_name, {
                "early_codes": [("R53.83", "Non-specific fatigue")],
                "mid_codes": [("R10.9", "Unspecified symptoms")],
                "lab_profile": "Elevated inflammatory markers"
            })
            
            early_code, early_desc = random.choice(p_map["early_codes"])
            mid_code, mid_desc = random.choice(p_map["mid_codes"])

            for v_idx, day_offset in enumerate(visit_days):
                v_date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                enc_id = f"ENC-{pid}-{v_idx+1}"
                doc = random.choice(PHYSICIANS)

                if v_idx == 0:
                    code, desc, is_prod, lab_flag = early_code, early_desc, True, 0
                    lab_notes = "Routine panel unremarkable"
                    med = "Symptomatic relief"
                elif v_idx == len(visit_days) // 2:
                    code, desc, is_prod, lab_flag = mid_code, mid_desc, True, 1
                    lab_notes = p_map["lab_profile"]
                    med = "Targeted therapy / Specialist referral"
                elif v_idx == len(visit_days) - 1:
                    code, desc, is_prod, lab_flag = icd_final, f"{d_name} (Confirmed Diagnosis)", True, 1
                    lab_notes = "Confirmatory diagnostic biomarker positive"
                    med = str(d_info.get("Treatment / Management", "Specialist treatment"))[:50]
                else:
                    c_tuple = random.choice(COMMON_ICD_CODES)
                    code, desc, is_prod, lab_flag = c_tuple[0], c_tuple[1], False, 0
                    lab_notes = "Within normal limits"
                    med = "Standard maintenance"

                enc = ClinicalEncounter(
                    encounter_id=enc_id,
                    patient_id=pid,
                    visit_date=v_date,
                    year_offset=day_offset // 365 + 1,
                    icd_code=code,
                    diagnosis=desc,
                    category="Rare Prodrome" if is_prod else "General",
                    abnormal_lab_flag=lab_flag,
                    lab_notes=lab_notes,
                    medication=med,
                    physician=doc,
                    is_prodromal_signal=is_prod
                )
                encounters.append(enc)
                all_visits.append(asdict(enc))

            patient = LongitudinalPatient(
                patient_id=pid,
                name=name,
                age=random.randint(18, 75),
                gender=gender,
                is_rare_case=True,
                condition_name=d_name,
                icd_code_final=icd_final,
                prevalence_rate=str(prev_rate),
                genetic_cause=str(gen_cause),
                avg_diagnostic_delay=str(diag_delay),
                encounters=encounters,
                total_visits=len(encounters)
            )
        else:
            for v_idx, day_offset in enumerate(visit_days):
                v_date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                enc_id = f"ENC-{pid}-{v_idx+1}"
                doc = random.choice(PHYSICIANS)
                c_tuple = random.choice(COMMON_ICD_CODES)

                enc = ClinicalEncounter(
                    encounter_id=enc_id,
                    patient_id=pid,
                    visit_date=v_date,
                    year_offset=day_offset // 365 + 1,
                    icd_code=c_tuple[0],
                    diagnosis=c_tuple[1],
                    category=c_tuple[2],
                    abnormal_lab_flag=0,
                    lab_notes="Routine bloodwork normal",
                    medication="Standard care",
                    physician=doc,
                    is_prodromal_signal=False
                )
                encounters.append(enc)
                all_visits.append(asdict(enc))

            patient = LongitudinalPatient(
                patient_id=pid,
                name=name,
                age=random.randint(22, 80),
                gender=gender,
                is_rare_case=False,
                condition_name="None / Routine Control",
                icd_code_final="Z00.00",
                prevalence_rate="Baseline Population",
                genetic_cause="None",
                avg_diagnostic_delay="0 years",
                encounters=encounters,
                total_visits=len(encounters)
            )

        patients_list.append(asdict(patient))

    visits_df = pd.DataFrame(all_visits)
    return patients_list, visits_df
