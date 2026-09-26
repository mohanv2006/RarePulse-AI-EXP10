"""
RarePulse AI: Longitudinal Rare Disease Detection Under Extreme Class Imbalance
==============================================================================
A self-contained clinical machine learning prototype featuring:
1. Dynamic Dataset Ingestion: Reads phenotypic definitions, ICD-10 codes, and trajectory
   patterns directly from `Rare_Diseases_Dataset.xlsx` using pandas.read_excel().
2. Realistic Longitudinal Synthetic EHR Sequence Generator (1.5% rare disease prevalence)
   modeling multi-year trajectories for 10 rare diseases.
3. PyTorch Temporal Bidirectional GRU with Masked Additive Attention (Bahdanau-style).
4. Numerically Stable Binary Focal Loss (alpha=0.85, gamma=2.0) for extreme class imbalance.
5. Clinical Explainability Engine extracting top-k influential visits and longitudinal attributions.
6. Stratified Validation (PR-AUC, Balanced Accuracy, ROC-AUC, F1-Score).
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------------------------------------------------------
# 0. Reproducibility Configuration
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------------------------------------------------------------
# 1. Dynamic Ingestion of Rare_Diseases_Dataset.xlsx
# -----------------------------------------------------------------------------
EXCEL_FILE_PATH = "Rare_Diseases_Dataset.xlsx"

def load_rare_diseases_knowledge_base(file_path: str = EXCEL_FILE_PATH) -> pd.DataFrame:
    """
    Reads the phenotypic definitions, ICD-10 codes, and trajectory patterns
    directly from Rare_Diseases_Dataset.xlsx using pandas.read_excel.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset '{file_path}' not found in current directory.")
    
    df_rare = pd.read_excel(file_path)
    print(f"[*] Successfully loaded '{file_path}' ({len(df_rare)} rare disease phenotypes).")
    return df_rare

# Baseline Common Outpatient ICD-10 Codes
COMMON_ICD_CODES = [
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I10", "Essential (primary) hypertension"),
    ("J45.909", "Unspecified asthma, uncomplicated"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
    ("M54.5", "Low back pain"),
    ("R53.83", "Other fatigue"),
    ("J06.9", "Acute upper respiratory infection, unspecified"),
    ("K58.9", "Irritable bowel syndrome without diarrhea"),
    ("F41.1", "Generalized anxiety disorder"),
    ("E78.5", "Hyperlipidemia, unspecified"),
    ("N39.0", "Urinary tract infection, site not specified"),
    ("G43.909", "Migraine, unspecified, not intractable"),
    ("M25.50", "Pain in unspecified joint"),
    ("R05", "Cough"),
    ("R10.9", "Unspecified abdominal pain"),
    ("K29.70", "Gastritis, unspecified, without bleeding"),
    ("H10.9", "Unspecified conjunctivitis"),
    ("L20.9", "Atopic dermatitis, unspecified"),
    ("R51.9", "Headache, unspecified"),
    ("Z00.00", "Encounter for general adult medical examination"),
]

# Prodromal code mapping derived from Key Early Symptoms & Common Misdiagnoses
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
        "early_codes": [("R29.6", "Asymmetric limb weakness & falls"), ("M62.50", "Muscle fasciculations & wasting")],
        "mid_codes": [("R47.81", "Bulbar slurred speech"), ("R13.10", "Dysphagia / swallowing difficulty")],
        "lab_profile": "Normal CSF/inflammatory panel; denervation on EMG notes"
    },
    "Ehlers-Danlos Syndrome (Vascular Type)": {
        "early_codes": [("M35.7", "Joint hypermobility syndrome"), ("R23.3", "Unexplained subcutaneous bruising")],
        "mid_codes": [("M24.40", "Recurrent joint dislocation"), ("I72.9", "Aneurysm / vascular fragility event")],
        "lab_profile": "Normal coagulation panel despite recurrent hematomas"
    },
    "Duchenne Muscular Dystrophy": {
        "early_codes": [("R62.0", "Delayed gross motor milestones"), ("R26.89", "Abnormal waddling gait / toe-walking")],
        "mid_codes": [("M62.81", "Proximal muscle pseudohypertrophy"), ("M41.9", "Progressive scoliosis")],
        "lab_profile": "Massively elevated serum creatine kinase (CK > 10,000 U/L)"
    },
    "Prader-Willi Syndrome": {
        "early_codes": [("P94.2", "Infantile hypotonia ('floppy infant')"), ("P92.9", "Feeding difficulty / failure to thrive")],
        "mid_codes": [("E66.01", "Rapid childhood hyperphagia & morbid obesity"), ("R62.50", "Developmental delay")],
        "lab_profile": "Hypothalamic-pituitary endocrine abnormalities & insulin resistance"
    }
}

@dataclass
class SyntheticEHRConfig:
    num_patients: int = 1000
    prevalence: float = 0.015  # Exactly 1.5% positive (15 positive, 985 negative)
    min_visits: int = 3
    max_visits: int = 8
    timeline_days: int = 1825  # 5 years timeline span

# -----------------------------------------------------------------------------
# 2. Synthetic Longitudinal Data Generator from Excel Phenotypes
# -----------------------------------------------------------------------------
def generate_synthetic_ehr_from_excel(
    df_rare: pd.DataFrame,
    config: SyntheticEHRConfig = SyntheticEHRConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Generates realistic longitudinal EHR visits across a 5-year timeline
    grounded in the exact phenotypes from Rare_Diseases_Dataset.xlsx.
    """
    num_pos = int(round(config.num_patients * config.prevalence))
    num_neg = config.num_patients - num_pos

    patient_labels = [1] * num_pos + [0] * num_neg
    random.shuffle(patient_labels)

    visits_records = []
    patient_records = []
    icd_code_descriptions = dict(COMMON_ICD_CODES)

    # Populate descriptions from Excel
    for _, row in df_rare.iterrows():
        d_name = row["Disease Name"]
        d_code = str(row["ICD-10 Code"]).strip()
        icd_code_descriptions[d_code] = f"{d_name} (Definitive Rare Diagnosis)"

    # Also include prodromal descriptions
    for d_name, p_data in PRODROMAL_CODE_MAPPING.items():
        for code, desc in p_data["early_codes"] + p_data["mid_codes"]:
            icd_code_descriptions[code] = desc

    common_codes_list = [c[0] for c in COMMON_ICD_CODES]
    rare_disease_list = df_rare.to_dict("records")

    for p_idx, is_positive in enumerate(patient_labels):
        patient_id = f"PAT_{p_idx+1:04d}"
        num_visits = random.randint(config.min_visits, config.max_visits)
        visit_days = sorted(random.sample(range(10, config.timeline_days), num_visits))

        if is_positive:
            # Assign a specific rare disease from the Excel dataset
            rare_profile = random.choice(rare_disease_list)
            disease_name = rare_profile["Disease Name"]
            final_icd_code = str(rare_profile["ICD-10 Code"]).strip()
            prodrome_info = PRODROMAL_CODE_MAPPING.get(disease_name, {
                "early_codes": [("G62.9", "Early prodromal symptom")],
                "mid_codes": [("R00.0", "Intermediate complication")],
                "lab_profile": "Biomarker spike"
            })

            patient_records.append({
                "patient_id": patient_id,
                "rare_condition_label": 1,
                "disease_name": disease_name,
                "icd_code_final": final_icd_code,
                "category": rare_profile["Category"],
                "avg_diagnostic_delay": rare_profile["Avg. Diagnostic Delay"],
                "key_symptoms": rare_profile["Key Early Symptoms"],
                "diagnostic_tests": rare_profile["Diagnostic Tests"],
                "treatment": rare_profile["Treatment / Management"],
                "num_visits": num_visits
            })

            early_idx = 0
            mid_idx = max(1, num_visits // 2)
            late_idx = num_visits - 1

            for v_idx, day in enumerate(visit_days):
                visit_id = f"VIS_{patient_id}_{v_idx+1}"
                timestamp = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)

                if v_idx == early_idx:
                    # Early Phase (Year 1-2): Prodromal symptom code
                    code_tuple = random.choice(prodrome_info["early_codes"])
                    code = code_tuple[0]
                    lab_flag = 1 if random.random() < 0.3 else 0
                elif v_idx == mid_idx:
                    # Intermediate Phase (Year 2-3): Biomarker spike & progressive code
                    code_tuple = random.choice(prodrome_info["mid_codes"])
                    code = code_tuple[0]
                    lab_flag = 1  # Key biomarker signal
                elif v_idx == late_idx:
                    # Late Phase (Year 4-5): Definitive condition code
                    code = final_icd_code
                    lab_flag = 1 if random.random() < 0.85 else 0
                else:
                    # Background routine visit
                    code = random.choice(common_codes_list)
                    lab_flag = 1 if random.random() < 0.15 else 0

                visits_records.append({
                    "visit_id": visit_id,
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "days_since_start": day,
                    "icd_code": code,
                    "abnormal_lab_flag": lab_flag,
                    "rare_condition_label": 1,
                    "disease_name": disease_name
                })
        else:
            # Control / Negative Cohort
            patient_records.append({
                "patient_id": patient_id,
                "rare_condition_label": 0,
                "disease_name": "None / Routine Outpatient Control",
                "icd_code_final": "Z00.00",
                "category": "General Population",
                "avg_diagnostic_delay": "N/A",
                "key_symptoms": "Standard primary care concerns",
                "diagnostic_tests": "Routine screening",
                "treatment": "Standard care",
                "num_visits": num_visits
            })

            for v_idx, day in enumerate(visit_days):
                visit_id = f"VIS_{patient_id}_{v_idx+1}"
                timestamp = pd.Timestamp("2021-01-01") + pd.Timedelta(days=day)
                code = random.choice(common_codes_list)
                lab_flag = 1 if random.random() < 0.08 else 0

                visits_records.append({
                    "visit_id": visit_id,
                    "patient_id": patient_id,
                    "timestamp": timestamp,
                    "days_since_start": day,
                    "icd_code": code,
                    "abnormal_lab_flag": lab_flag,
                    "rare_condition_label": 0,
                    "disease_name": "Control"
                })

    df_visits = pd.DataFrame(visits_records)
    df_patients = pd.DataFrame(patient_records)

    return df_visits, df_patients, icd_code_descriptions

# -----------------------------------------------------------------------------
# 3. Vocabulary & Dataset Pipeline
# -----------------------------------------------------------------------------
class EHRVocabulary:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.code2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2code = {0: self.pad_token, 1: self.unk_token}

    def build_vocab(self, codes: List[str]):
        for code in sorted(set(codes)):
            if code not in self.code2idx:
                idx = len(self.code2idx)
                self.code2idx[code] = idx
                self.idx2code[idx] = code

    def encode(self, code: str) -> int:
        return self.code2idx.get(code, self.code2idx[self.unk_token])

    def decode(self, idx: int) -> str:
        return self.idx2code.get(idx, self.unk_token)

    def __len__(self):
        return len(self.code2idx)


class LongitudinalEHRDataset(Dataset):
    def __init__(
        self,
        df_visits: pd.DataFrame,
        vocab: EHRVocabulary,
        patient_ids: List[str],
        max_seq_len: int = 8
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.samples = []
        
        df_sub = df_visits[df_visits["patient_id"].isin(patient_ids)].copy()
        grouped = df_sub.groupby("patient_id")

        for pid in patient_ids:
            group = grouped.get_group(pid).sort_values("days_since_start")
            seq_len = min(len(group), max_seq_len)

            codes = [self.vocab.encode(c) for c in group["icd_code"].tolist()[:seq_len]]
            labs = group["abnormal_lab_flag"].tolist()[:seq_len]
            days = group["days_since_start"].tolist()[:seq_len]
            label = int(group["rare_condition_label"].iloc[0])
            disease_name = group["disease_name"].iloc[0]
            raw_codes = group["icd_code"].tolist()[:seq_len]
            timestamps = group["timestamp"].tolist()[:seq_len]
            visit_ids = group["visit_id"].tolist()[:seq_len]

            # Compute inter-visit time deltas in years (normalized)
            deltas = [0.0]
            for i in range(1, seq_len):
                d_diff = (days[i] - days[i-1]) / 365.25
                deltas.append(d_diff)

            # Pad sequences
            pad_len = max_seq_len - seq_len
            padded_codes = codes + [0] * pad_len
            padded_labs = labs + [0] * pad_len
            padded_deltas = deltas + [0.0] * pad_len
            mask = [1.0] * seq_len + [0.0] * pad_len

            self.samples.append({
                "patient_id": pid,
                "codes": torch.tensor(padded_codes, dtype=torch.long),
                "labs": torch.tensor(padded_labs, dtype=torch.float32).unsqueeze(-1),
                "deltas": torch.tensor(padded_deltas, dtype=torch.float32).unsqueeze(-1),
                "mask": torch.tensor(mask, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.float32).unsqueeze(-1),
                "seq_len": seq_len,
                "disease_name": disease_name,
                "raw_codes": raw_codes,
                "raw_labs": labs,
                "raw_timestamps": timestamps,
                "raw_visit_ids": visit_ids,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        "codes": torch.stack([b["codes"] for b in batch]),
        "labs": torch.stack([b["labs"] for b in batch]),
        "deltas": torch.stack([b["deltas"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "patient_ids": [b["patient_id"] for b in batch],
        "disease_names": [b["disease_name"] for b in batch],
    }

# -----------------------------------------------------------------------------
# 4. PyTorch Model Architecture (Bidirectional GRU + Additive Attention)
# -----------------------------------------------------------------------------
class MaskedAdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.proj(hidden_states))
        scores = self.context_vector(energy).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attention_weights


class RarePulseRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 48,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.code_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        visit_feature_dim = embedding_dim + 1 + 1  # Code + Lab + Time Delta
        self.input_projection = nn.Sequential(
            nn.Linear(visit_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        gru_out_dim = hidden_dim * 2
        self.attention = MaskedAdditiveAttention(hidden_dim=gru_out_dim, attention_dim=32)
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        codes: torch.Tensor,
        labs: torch.Tensor,
        deltas: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.code_embedding(codes)
        features = torch.cat([embedded, labs, deltas], dim=-1)
        proj = self.input_projection(features)
        gru_out, _ = self.gru(proj)
        context, attn_weights = self.attention(gru_out, mask)
        logits = self.classifier(context)
        return logits, attn_weights

# -----------------------------------------------------------------------------
# 5. Binary Focal Loss for Extreme Class Imbalance (<1.5% Positive)
# -----------------------------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        loss = focal_weight * bce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

# -----------------------------------------------------------------------------
# 6. Training & Evaluation Loops
# -----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 0.003
) -> Dict[str, Any]:
    criterion = BinaryFocalLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_pr_auc = 0.0
    best_weights = None

    print("\n--- Training RarePulse Sequential GRU with Focal Loss ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            codes = batch["codes"].to(device)
            labs = batch["labs"].to(device)
            deltas = batch["deltas"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(codes, labs, deltas, mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * codes.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # Validation evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                codes = batch["codes"].to(device)
                labs = batch["labs"].to(device)
                deltas = batch["deltas"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(codes, labs, deltas, mask)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy().flatten())

        val_pr_auc = average_precision_score(val_targets, val_preds)
        val_roc_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.0

        if val_pr_auc >= best_pr_auc:
            best_pr_auc = val_pr_auc
            best_weights = model.state_dict().copy()

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Focal Loss: {train_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {"best_pr_auc": best_pr_auc}


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, Any]:
    model.eval()
    preds, targets, pids, dnames = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            codes = batch["codes"].to(device)
            labs = batch["labs"].to(device)
            deltas = batch["deltas"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(codes, labs, deltas, mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)
            targets.extend(labels.cpu().numpy().flatten())
            pids.extend(batch["patient_ids"])
            dnames.extend(batch["disease_names"])

    preds = np.array(preds)
    targets = np.array(targets)
    bin_preds = (preds >= threshold).astype(int)

    pr_auc = average_precision_score(targets, preds)
    roc_auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.0
    bal_acc = balanced_accuracy_score(targets, bin_preds)
    cm = confusion_matrix(targets, bin_preds)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "predictions": preds,
        "targets": targets,
        "patient_ids": pids,
        "disease_names": dnames
    }

# -----------------------------------------------------------------------------
# 7. Main Execution Flow
# -----------------------------------------------------------------------------
def run_pipeline():
    print("=" * 80)
    print("  RarePulse AI: Longitudinal Rare Disease ML Pipeline from Excel Knowledge Base")
    print("=" * 80)

    # 1. Load Excel definitions
    df_rare = load_rare_diseases_knowledge_base(EXCEL_FILE_PATH)

    # 2. Synthesize longitudinal cohort
    print("\n[*] Generating synthetic multi-year EHR trajectories for 1,000 patients...")
    df_visits, df_patients, icd_descriptions = generate_synthetic_ehr_from_excel(df_rare)
    
    pos_count = (df_patients["rare_condition_label"] == 1).sum()
    neg_count = (df_patients["rare_condition_label"] == 0).sum()
    print(f"[+] Total Patients: {len(df_patients)} | Positive (Rare): {pos_count} ({pos_count/len(df_patients)*100:.1f}%) | Negative (Control): {neg_count}")
    print(f"[+] Total Clinical Encounters: {len(df_visits)} visits across 5-year timeline.")

    # 3. Build vocabulary
    vocab = EHRVocabulary()
    vocab.build_vocab(df_visits["icd_code"].tolist())
    print(f"[+] Encoded Vocabulary Size: {len(vocab)} unique medical entities.")

    # 4. Stratified Train / Test Split
    patient_ids = df_patients["patient_id"].tolist()
    labels = df_patients["rare_condition_label"].tolist()

    train_ids, test_ids = train_test_split(patient_ids, test_size=0.25, stratify=labels, random_state=42)
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.20,
        stratify=[df_patients.set_index("patient_id").loc[pid]["rare_condition_label"] for pid in train_ids],
        random_state=42
    )

    train_dataset = LongitudinalEHRDataset(df_visits, vocab, train_ids)
    val_dataset = LongitudinalEHRDataset(df_visits, vocab, val_ids)
    test_dataset = LongitudinalEHRDataset(df_visits, vocab, test_ids)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 5. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RarePulseRNN(vocab_size=len(vocab), embedding_dim=32, hidden_dim=48, dropout=0.2).to(device)

    # 6. Train Model
    train_model(model, train_loader, val_loader, device, epochs=15, lr=0.003)

    # 7. Evaluate on Test Cohort
    test_results = evaluate_dataset(model, test_loader, device, threshold=0.5)

    print("\n" + "=" * 80)
    print("  FINAL MODEL EVALUATION (Test Cohort)")
    print("=" * 80)
    print(f"Precision-Recall AUC (PR-AUC) : {test_results['pr_auc']:.4f} (Baseline ~ 0.015)")
    print(f"ROC-AUC                       : {test_results['roc_auc']:.4f}")
    print(f"Balanced Accuracy             : {test_results['balanced_accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(test_results["confusion_matrix"])

    # 8. Clinical Explainability Case Studies
    print("\n" + "=" * 80)
    print("  CLINICAL XAI EXPLANATIONS (Sample Trajectories from Excel Dataset)")
    print("=" * 80)

    pos_samples = [s for s in test_dataset.samples if s["label"].item() == 1.0]
    for s in pos_samples[:3]:
        pid = s["patient_id"]
        dname = s["disease_name"]
        
        with torch.no_grad():
            c = s["codes"].unsqueeze(0).to(device)
            l = s["labs"].unsqueeze(0).to(device)
            d = s["deltas"].unsqueeze(0).to(device)
            m = s["mask"].unsqueeze(0).to(device)
            logits, attn = model(c, l, d, m)
            prob = torch.sigmoid(logits).item()
            attn_weights = attn.squeeze(0).cpu().numpy()[:s["seq_len"]]
            norm_attn = attn_weights / attn_weights.sum() if attn_weights.sum() > 0 else attn_weights

        print(f"\n[ALERT] Patient ID: {pid} | Target Phenotype: {dname}")
        print(f"        Predicted Rare Condition Risk: {prob * 100:.1f}%")
        print("        Longitudinal Sequence & Attention Attribution:")
        for i in range(s["seq_len"]):
            code = s["raw_codes"][i]
            desc = icd_descriptions.get(code, "Medical Diagnostic Encounter")
            date_str = s["raw_timestamps"][i].strftime("%Y-%m-%d")
            lab_str = "YES (Abnormal Biomarker)" if s["raw_labs"][i] == 1 else "Normal"
            pct = norm_attn[i] * 100
            mark = "[ALERT]" if pct >= 30.0 else "       "
            print(f"        {mark} Visit {i+1} ({date_str}) | ICD: {code:<7} ({desc}) | Labs: {lab_str} | Attn: {pct:.1f}%")

    print("\n[SUCCESS] Pipeline execution complete.")

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    run_pipeline()
