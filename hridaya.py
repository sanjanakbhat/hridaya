# -*- coding: utf-8 -*-
"""
HRIDAYA (हृदय) - Automated ECG Arrhythmia & Anomaly Diagnostic Hub   [v2 - clinical-safety edition]
Run:  streamlit run hridaya.py --server.maxUploadSize 1000

Folder layout (keep these next to hridaya.py):
    hridaya.py
    logo.png                 <- HRIDAYA emblem (sidebar, home page, favicon, reports)
    profile.png.webp         <- reviewer photo (root folder; WebP format)
    assets/dhairya.png       <- optional alternate reviewer photo
    requirements.txt         <- streamlit numpy pandas scipy scikit-learn matplotlib seaborn wfdb pillow
                                (optional, for the CNN / MC-Dropout / contrastive option: torch)

What is new in v2 (engineering roadmap implemented)
 Phase 1  Patient-wise evaluation: GroupKFold cross-validation by record (no beat of a test patient is ever
          seen in training). Headline metric = patient-wise macro-F1, with per-fold spread.
 Phase 2  Class imbalance: balanced Random-Forest weights, class-weighted CE / Focal Loss (PyTorch model),
          physiology-aware augmentation of minority classes (baseline wander, HF noise, time-stretch,
          amplitude jitter) applied to TRAINING folds only.
 Phase 3  Optional 1D-CNN fused with tabular context (RR features + optional age/sex from a metadata CSV),
          optional SimCLR-style contrastive pre-training (on the uploaded, unlabeled beats of the training fold).
 Phase 4  Safety layer: reject option ("Uncertain / Refer to Cardiologist") via softmax threshold and
          Monte-Carlo Dropout spread; Signal-Quality Index gatekeeper; Grad-CAM (CNN) / occlusion (RF)
          explanations; hybrid rule-engine cross-check of AI beat labels; external-cohort validation page;
          Model Card / data-lineage / audit-log page with an ISO 13485 / IEC 62304 evidence checklist.

Design rules (unchanged)
 * Every number shown is computed from the uploaded data. Nothing is hard-coded or random-generated.
 * Beat classifier is trained and tested ONLY on labels that come with the upload.
 * The 13 conditions are screened with signal-processing rules; where the leads cannot support a
   condition the app says "Not assessable" instead of guessing.
 * Research prototype only - NOT a certified medical device. The regulatory page lists evidence artefacts
   a future SaMD submission would need; it does not claim compliance.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import html
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from urllib.parse import quote_plus

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

try:
    import wfdb

    WFDB_OK = True
except ImportError:  # pragma: no cover
    WFDB_OK = False

try:  # optional deep-learning stack
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except Exception:  # pragma: no cover
    TORCH_OK = False

# ======================================================================
# 0. CONSTANTS
# ======================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

OWNER = {
    "name": "Sanjana K Bhat",
    "phone": "9535309338",
    "email": "sanjanakbhatunitech@gmail.com",
}

REVIEWER = {
    "name": "Dr. Dhairya Thakker",
    "role": "Consultant Cardiologist",
    "base": "Mulund & Northeast Mumbai (visiting consultant)",
    "photo_candidates": (
        "profile.png.webp",
        "assets/profile.png.webp",
        "assets/dhairya.png",
        "assets/dhairya.jpg",
        "dhairya.png",
        "dhairya.jpg",
    ),
    "profile": "https://mymedisage.com/profile/dr-dhairya-thakker",
}


def _first_existing(paths):
    for p in paths:
        fp = os.path.join(APP_DIR, p)
        if os.path.isfile(fp):
            return fp
    return None


def find_logo():
    return _first_existing(("logo.png", os.path.join("assets", "logo.png"), "Elegant_Devanagari_Heart_Emblem.png"))


LOGO = find_logo()


def _b64_of(path):
    if not path:
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def logo_b64():
    return _b64_of(LOGO)


def reviewer_photo_path():
    return _first_existing(REVIEWER["photo_candidates"])


def reviewer_photo_b64():
    return _b64_of(reviewer_photo_path())


CONDITIONS = [
    dict(id=1, name="Atrial Fibrillation (AFib)", tier="Critical",
         desc="Irregularly irregular rhythm, absent P waves; risk of blood clots/stroke.",
         needs="RR intervals (any lead)"),
    dict(id=2, name="Sinus Bradycardia", tier="Moderate",
         desc="Heart rate below 60 bpm.", needs="RR intervals (+P waves)"),
    dict(id=3, name="Sinus Tachycardia", tier="Moderate",
         desc="Heart rate above 100 bpm.", needs="RR intervals (+P waves)"),
    dict(id=4, name="First-Degree AV Block", tier="Moderate",
         desc="PR interval longer than 0.20 s.", needs="Visible P wave"),
    dict(id=5, name="Second-Degree AV Block (Mobitz I / Wenckebach)", tier="Moderate",
         desc="PR progressively lengthens until a beat is dropped.", needs="PR + pauses"),
    dict(id=6, name="Second-Degree AV Block (Mobitz II)", tier="Critical",
         desc="Sudden dropped beats without progressive PR lengthening.", needs="PR + pauses"),
    dict(id=7, name="Third-Degree (Complete) Heart Block", tier="Critical",
         desc="Atria and ventricles beat independently (AV dissociation).", needs="Slow regular RR + variable PR"),
    dict(id=8, name="Right Bundle Branch Block (RBBB)", tier="Low",
         desc="Wide QRS with an 'M' (rSR') pattern, best seen in V1.", needs="QRS width (+V1 ideal)"),
    dict(id=9, name="Left Bundle Branch Block (LBBB)", tier="Critical",
         desc="Wide, broad monophasic QRS; masks ischemic changes.", needs="QRS width (+V1/V5 ideal)"),
    dict(id=10, name="ST-Elevation MI (STEMI) pattern", tier="Critical",
         desc="ST elevation of at least 0.1 mV (acute full-thickness infarct).", needs="Calibrated mV, narrow QRS, >=2 leads to confirm"),
    dict(id=11, name="ST Depression / Ischemia", tier="Critical",
         desc="Downward ST displacement of at least 0.1 mV.", needs="Calibrated mV, narrow QRS"),
    dict(id=12, name="Left Ventricular Hypertrophy (LVH)", tier="Moderate",
         desc="Tall R / deep S waves (Sokolow-Lyon: S in V1 + R in V5/V6 >= 3.5 mV).", needs="V1 AND V5/V6 leads, mV units"),
    dict(id=13, name="QT Interval Prolongation", tier="Critical",
         desc="Corrected QT (Bazett) above 450 ms; risk of Torsades de Pointes.", needs="Visible T wave"),
]
COND_BY_ID = {c["id"]: c for c in CONDITIONS}
TIER_ICON = {"Critical": "🔴 Critical / Urgent Review", "Moderate": "🟡 Moderate / Monitor", "Low": "🟢 Low Risk / Benign Variant"}
TIER_RANK = {"Low": 1, "Moderate": 2, "Critical": 3}
FLAGGED = ("Detected", "Suspected")

AAMI_CLASSES = {
    "N": ["N", "L", "R", "j", "e"],
    "S": ["a", "J", "A", "S"],
    "V": ["V", "E"],
    "F": ["F"],
    "Q": ["/", "f", "Q"],
}
ANN_TO_AAMI = {s: c for c, syms in AAMI_CLASSES.items() for s in syms}
BEAT_SYMBOLS = set(ANN_TO_AAMI)               # annotation symbols treated as beats
NORMAL_CONDUCTED = {"N", "L", "R", "e", "j"}  # used for PR/QRS/ST/QT measurement
AAMI_ORDER = ["N", "S", "V", "F", "Q"]
CLASS_ORDER = AAMI_ORDER + ["Normal", "Abnormal"]
CRITICAL_CLASSES = ("S", "V", "F", "Abnormal")  # classes where a miss is clinically costly

SIG_TITLE = "Annotated ECG records (waveform + RR features)"
SCHEME_TITLE = {
    "mitbih": "Pre-segmented beat files (labels 0-4 = N, S, V, F, Q)",
    "ptbdb": "Pre-segmented beat files (PTB-DB: 0 = Normal, 1 = Abnormal)",
}
AAMI_LABEL_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
STD_FS = (100, 125, 128, 200, 250, 256, 360, 500, 512, 1000, 2000)
IGNORE_PREFIX = ("readme", "license", "records", "sha256", "annotators", "index", "changelog")

SEG_LEN = 188          # every signal beat is resampled to this many points
SEG_DUR_S = 0.52       # 0.25 s before + 0.27 s after the R peak

# Engineering milestones from the roadmap. These are TARGETS to compare against, not claims of achievement.
TARGETS = dict(macro_f1=0.88, crit_sens=0.93)

MODEL_KINDS = {"rf": "Random Forest (fast, CPU)", "cnn": "1D-CNN + tabular fusion + MC-Dropout (PyTorch)"}

DOCTORS = [
    dict(city="Bengaluru", name="Dr. Nagamani A. C.", exp="23 Years", spec="MBBS, DM - Cardiology",
         loc="Apollo Clinic, Koramangala, Bengaluru"),
    dict(city="Bengaluru", name="Dr. Anusha A. Rao", exp="8 Years", spec="MBBS, MD, DNB (Cardiology)",
         loc="Manipal Hospitals Old Airport Road / Apollo Clinic, Indiranagar, Bengaluru"),
    dict(city="Bengaluru", name="Dr. Shilpa Jayaprakash", exp="12 Years", spec="MBBS, MD General Medicine, DM Cardiology",
         loc="Apollo Clinic, JP Nagar, Bengaluru"),
    dict(city="Bengaluru", name="Dr. K. Nikhitha Reddy (Dr. K. Nikhitha)", exp="12 Years", spec="MBBS, MD General Medicine, DM Cardiology",
         loc="Apollo Clinic, HSR Layout, Bengaluru"),
    dict(city="Bengaluru", name="Dr. Disha R. Shetty", exp="8 Years", spec="MBBS, MD, DM Cardiology",
         loc="Apollo One, Electronic City, Bengaluru"),
]


def _ver(pkg):
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return "not installed"


def audit(event, detail=""):
    """Append to the in-session audit trail (shown on the Model Card page)."""
    try:
        log = st.session_state.setdefault("audit_log", [])
        log.append({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "event": event, "detail": str(detail)[:300]})
    except Exception:
        pass


# ======================================================================
# 1. SIGNAL LOADING (WFDB, CSV/TXT signal, annotation text, beat matrices)
# ======================================================================
def _fill_nans(x):
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _strip_q(s):
    return str(s).strip().strip("'\"").strip()


def _time_to_s(v):
    """'0:00.003' / '1:02:03.5' / '12.5' -> seconds (NaN when not a time)."""
    v = _strip_q(v)
    if _is_float(v):
        return float(v)
    try:
        secs = 0.0
        for p in v.split(":"):
            secs = secs * 60 + float(p)
        return secs
    except ValueError:
        return np.nan


def _snap_fs(fs):
    if fs is None or not np.isfinite(fs) or fs <= 0:
        return None
    best = min(STD_FS, key=lambda s: abs(s - fs))
    return float(best) if abs(best - fs) / best < 0.03 else float(round(fs, 2))


def estimate_fs_from_time(series):
    n = len(series)
    if n < 10:
        return None
    idx = np.unique(np.linspace(0, n - 1, min(n, 2000)).astype(int))
    t = np.array([_time_to_s(v) for v in series.iloc[idx]], dtype=float)
    ok = np.isfinite(t)
    if ok.sum() < 3:
        return None
    t, ii = t[ok], idx[ok]
    if t[-1] <= t[0]:
        return None
    return _snap_fs((ii[-1] - ii[0]) / (t[-1] - t[0]))


def build_rhythm(samples, symbols, notes, n):
    marks = [(int(s), str(a).replace("\x00", "").strip().lstrip("(")) for s, sym, a in zip(samples, symbols, notes) if sym == "+"]
    out = []
    for k, (s, lab) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else n
        out.append((s, end, lab))
    return out


def load_wfdb_record(base):
    """Load a WFDB record (.hea + .dat/.mat) and its optional .atr annotation."""
    rec = wfdb.rdrecord(base)
    sig = rec.p_signal
    calibrated = True
    if sig is None:
        sig = rec.d_signal.astype(float)
        calibrated = False
    sig = np.column_stack([_fill_nans(sig[:, i]) for i in range(sig.shape[1])])
    units = [(u or "").strip().lower() for u in (rec.units or [])] or [""] * sig.shape[1]
    cal = []
    for i, u in enumerate(units):
        if u == "mv" and calibrated:
            cal.append(True)
        elif u in ("uv", "µv", "μv") and calibrated:
            sig[:, i] = sig[:, i] / 1000.0
            cal.append(True)
        else:
            cal.append(False)
    out = dict(kind="signal", signal=sig, fs=float(rec.fs), fs_source="WFDB header", sig_names=list(rec.sig_name),
               calibrated=cal, ann_samples=None, ann_symbols=None, rhythm=[])
    try:
        ann = wfdb.rdann(base, "atr")
        out["ann_samples"] = np.asarray(ann.sample, dtype=int)
        out["ann_symbols"] = np.asarray(ann.symbol)
        notes = list(ann.aux_note) if ann.aux_note is not None else [""] * len(ann.symbol)
        out["rhythm"] = build_rhythm(ann.sample, ann.symbol, notes, sig.shape[0])
    except Exception:
        pass
    return out


# ---- text / CSV helpers ------------------------------------------------
ANN_RE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(\S+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*(.*)$")


def _first_lines(path, n=6):
    try:
        with open(path, "r", errors="ignore") as f:
            return [f.readline() for _ in range(n)]
    except OSError:
        return []


def looks_like_annotation(path):
    """PhysioNet annotation text ('Time  Sample #  Type  Sub  Chan  Num  Aux') or a CSV with sample/type columns."""
    for ln in _first_lines(path):
        low = ln.lower()
        if "sample" in low and ("type" in low or "symbol" in low):
            return True
        m = ANN_RE.match(ln)
        if m and "," not in ln and not _is_float(m.group(3)):
            return True
    return False


def parse_annotation_file(path):
    with open(path, "r", errors="ignore") as f:
        lines = f.read().splitlines()
    samples, symbols, notes, times = [], [], [], []
    head = next((ln for ln in lines if ln.strip()), "")
    if "," in head and not ANN_RE.match(head):  # CSV-style annotation table
        df = pd.read_csv(path)
        cols = {_strip_q(c).lower(): c for c in df.columns}
        sc = next((cols[k] for k in cols if k.startswith("sample")), None)
        tc = next((cols[k] for k in cols if k in ("type", "symbol", "label", "beat", "annotation")), None)
        ac = next((cols[k] for k in cols if k in ("aux", "note", "rhythm")), None)
        if sc is None or tc is None:
            raise ValueError("annotation CSV needs 'sample' and 'type' columns")
        samples = pd.to_numeric(df[sc], errors="coerce").fillna(-1).astype(int).tolist()
        symbols = df[tc].astype(str).str.strip().tolist()
        notes = df[ac].fillna("").astype(str).tolist() if ac else [""] * len(samples)
        times = None
    else:
        for ln in lines:
            m = ANN_RE.match(ln)
            if m and not _is_float(m.group(3)):
                times.append(_time_to_s(m.group(1)))
                samples.append(int(m.group(2)))
                symbols.append(m.group(3))
                notes.append(m.group(7).strip())
    if not samples:
        raise ValueError("no annotation rows could be parsed")
    order = np.argsort(samples, kind="stable")
    samples = np.asarray(samples, dtype=int)[order]
    symbols = np.asarray(symbols)[order]
    notes = [notes[i] for i in order]
    t = np.asarray(times, dtype=float)[order] if times is not None else None
    return dict(samples=samples, symbols=symbols, notes=notes, times=t)


def fs_from_annotation(ann):
    t = ann.get("times")
    if t is None:
        return None
    ok = np.isfinite(t) & (t > 1.0)
    if ok.sum() < 3:
        return None
    return _snap_fs(float(np.median(ann["samples"][ok] / t[ok])))


def _read_table(path):
    line = next((ln for ln in _first_lines(path, 5) if ln.strip()), "")
    sep = next((s for s in (",", ";", "\t") if s in line), r"\s+")
    kw = dict(sep=sep, engine="python") if sep == r"\s+" else dict(sep=sep)
    first = pd.read_csv(path, nrows=1, header=None, **kw)
    headerless = all(_is_float(_strip_q(v)) for v in first.iloc[0].tolist())
    df = pd.read_csv(path, header=None if headerless else 0, **kw)
    df.columns = [_strip_q(c) for c in df.columns]
    return df


def make_beat_matrix(num, path):
    """Pre-segmented beats: one beat per row, last column = class label."""
    lab = num.iloc[:, -1].values.astype(float)
    M = num.iloc[:, :-1].values.astype(float)
    ok = np.isfinite(lab) & (lab == np.round(lab))
    if ok.mean() < 0.99 or np.nanmax(lab) > 4 or np.nanmin(lab) < 0:
        raise ValueError("looks like a pre-segmented beat file, but its last column is not a 0-4 class label")
    lab, M = lab[ok].astype(int), np.nan_to_num(M[ok])
    low = os.path.basename(path).lower()
    if "ptbdb" in low:
        if lab.max() > 1:
            raise ValueError("PTB-DB beat file should have labels 0/1 only")
        scheme, names = "ptbdb", {0: "Normal", 1: "Abnormal"}
    else:
        scheme, names = "mitbih", AAMI_LABEL_NAMES
    role = "test" if "test" in low else ("train" if "train" in low else "all")
    return dict(kind="beats", X=M, y=np.array([names[i] for i in lab]), scheme=scheme, role=role)


def load_csv_record(path, ann_path, opts):
    df = _read_table(path)
    names = list(df.columns)
    low = [n.lower() for n in names]
    time_col = next((n for n, l in zip(names, low) if "time" in l or "elapsed" in l or "interval" in l), None)
    idx_cols = [n for n, l in zip(names, low)
                if n != time_col and (l.startswith("sample") or l in ("index", "idx", "#") or l.startswith("unnamed"))]
    data_cols = [n for n in names if n != time_col and n not in idx_cols]
    if not data_cols:
        raise ValueError("no numeric ECG columns")

    # optional units row, e.g. 'hh:mm:ss.mmm','mV','mV'
    unit_of = {}
    if len(df) > 1 and all(not _is_float(_strip_q(v)) for v in df.iloc[0][data_cols]):
        unit_of = {c: _strip_q(v).lower().replace("µ", "u").replace("μ", "u") for c, v in df.iloc[0][data_cols].items()}
        df = df.iloc[1:].reset_index(drop=True)

    num = df[data_cols].apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all")
    if num.shape[1] == 0:
        raise ValueError("no numeric ECG columns")
    if num.shape[1] >= 100 and len(num) >= 20:
        return make_beat_matrix(num, path)

    num = num.iloc[:, :12]
    sig = np.column_stack([_fill_nans(num.iloc[:, i].values) for i in range(num.shape[1])])
    cal = []
    for i, c in enumerate(num.columns):
        u = unit_of.get(c)
        if u is None:
            u = {"mV": "mv", "µV": "uv", "arbitrary": ""}[opts["csv_units"]]
        if u == "mv":
            cal.append(True)
        elif u == "uv":
            sig[:, i] = sig[:, i] / 1000.0
            cal.append(True)
        elif u == "v":
            sig[:, i] = sig[:, i] * 1000.0
            cal.append(True)
        else:
            cal.append(False)

    ann = parse_annotation_file(ann_path) if ann_path else None
    fs, src = None, ""
    if not opts.get("force_fs"):
        if time_col is not None:
            fs = estimate_fs_from_time(df[time_col])
            src = f"time column '{time_col}'"
        if fs is None and ann is not None:
            fs = fs_from_annotation(ann)
            src = "annotation Sample#/Time ratio"
    if fs is None:
        fs, src = float(opts["csv_fs"]), "sidebar setting"

    out = dict(kind="signal", signal=sig, fs=float(fs), fs_source=src, sig_names=[str(c) for c in num.columns],
               calibrated=cal, ann_samples=None, ann_symbols=None, rhythm=[])
    if ann is not None:
        ok = (ann["samples"] >= 0) & (ann["samples"] < sig.shape[0])
        out["ann_samples"] = ann["samples"][ok]
        out["ann_symbols"] = ann["symbols"][ok]
        notes = [n for n, k in zip(ann["notes"], ok) if k]
        out["rhythm"] = build_rhythm(out["ann_samples"], out["ann_symbols"], notes, sig.shape[0])
    return out


def pick_leads(names):
    up = [str(n).upper().strip() for n in names]
    main = 0
    for pref in ("MLII", "II", "LEAD II", "I"):
        if pref in up:
            main = up.index(pref)
            break
    v1 = up.index("V1") if "V1" in up else None
    v5 = None
    for pref in ("V5", "V6", "V4"):
        if pref in up:
            v5 = up.index(pref)
            break
    return main, v1, v5


# ---- optional patient metadata (age / sex) -------------------------------
def _norm_stem(stem, ann=False):
    s = stem.lower()
    if ann:
        for tok in ("annotations", "annotation", "annot", "atr", "ann", "labels", "label"):
            s = s.replace(tok, "")
    for ch in ("_", "-", " ", "."):
        s = s.replace(ch, "")
    return s


def parse_demographics(file_obj):
    """CSV with columns: record|id|patient, age, sex|gender  ->  {normalised record: (age, sex_code)}; sex M=1, F=0."""
    df = pd.read_csv(file_obj)
    cols = {str(c).strip().lower(): c for c in df.columns}
    rc = next((cols[k] for k in ("record", "record_id", "id", "patient", "patient_id", "file", "name") if k in cols), None)
    ac = cols.get("age")
    sc = next((cols[k] for k in ("sex", "gender") if k in cols), None)
    if rc is None or (ac is None and sc is None):
        raise ValueError("metadata CSV needs a record/id column plus an 'age' and/or 'sex' column")
    out = {}
    for _, r in df.iterrows():
        key = _norm_stem(os.path.splitext(str(r[rc]).strip())[0])
        age = pd.to_numeric(r[ac], errors="coerce") if ac else np.nan
        sx = str(r[sc]).strip().upper()[:1] if sc else ""
        out[key] = (float(age) if np.isfinite(age) else np.nan, 1.0 if sx == "M" else (0.0 if sx == "F" else np.nan))
    return out


def demo_features(age, sex, age_fill):
    """Normalised demographic feature columns (missing -> cohort median age / 0.5 sex)."""
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    a = np.where(np.isfinite(age), age, age_fill) / 100.0
    s = np.where(np.isfinite(sex), sex, 0.5)
    return np.column_stack([a, s]).astype(np.float32)


# ======================================================================
# 2. SIGNAL PROCESSING
# ======================================================================
def bandpass(x, fs, lo, hi, order=2):
    nyq = fs / 2.0
    hi = min(hi, 0.9 * nyq)
    if len(x) < 40 or lo >= hi:
        return np.asarray(x, dtype=float)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def lowpass(x, fs, fc, order=2):
    nyq = fs / 2.0
    if len(x) < 40 or fc >= 0.9 * nyq:
        return np.asarray(x, dtype=float)
    b, a = butter(order, fc / nyq, btype="low")
    return filtfilt(b, a, x)


def detect_r_peaks(x, fs):
    """Pan-Tompkins-style detector (band-pass, derivative, squaring, integration)."""
    f = bandpass(x, fs, 5, 15)
    d = np.gradient(f)
    w = max(1, int(0.12 * fs))
    integ = np.convolve(d * d, np.ones(w) / w, mode="same")
    cand, _ = find_peaks(integ, distance=max(1, int(0.25 * fs)))
    if len(cand) == 0:
        return np.array([], dtype=int)
    seg = int(10 * fs)
    keep = []
    for s in range(0, len(integ), seg):
        c = cand[(cand >= s) & (cand < s + seg)]
        if len(c) == 0:
            continue
        thr = 0.3 * np.percentile(integ[c], 90)
        keep.extend(c[integ[c] >= thr])
    f2 = bandpass(x, fs, 3, 25)
    half = max(1, int(0.06 * fs))
    peaks = []
    for p in keep:
        a, b = max(0, p - half), min(len(x), p + half + 1)
        peaks.append(a + int(np.argmax(np.abs(f2[a:b]))))
    peaks = np.unique(peaks)
    out = []
    for p in peaks:  # enforce refractory period, keep the larger deflection
        if out and p - out[-1] < 0.25 * fs:
            if abs(f2[p]) > abs(f2[out[-1]]):
                out[-1] = p
        else:
            out.append(p)
    return np.array(out, dtype=int)


def get_beats(rec, x, fs):
    """Beat positions: reference annotations when present, otherwise detected R peaks."""
    if rec["ann_samples"] is not None:
        mask = np.array([s in BEAT_SYMBOLS for s in rec["ann_symbols"]])
        if mask.sum() >= 20:
            samples = rec["ann_samples"][mask]
            symbols = rec["ann_symbols"][mask]
            ok = (samples >= 0) & (samples < len(x))
            return samples[ok].astype(int), symbols[ok]
    return detect_r_peaks(x, fs), None


# ---- Signal Quality Index (SQI) gatekeeper -------------------------------
def signal_quality(x, fs, peaks, win_s=5.0, thr=0.5):
    """Per-window quality score in [0,1] from: flat-line, clipping, high-frequency noise share,
    baseline-wander share, implausible beat count, and beat-to-template correlation."""
    xs = np.asarray(x, dtype=float)
    n = len(xs)
    w = max(int(win_s * fs), 1)
    nw = max(1, int(np.ceil(n / w)))
    bw = lowpass(xs, fs, 0.7)
    hf = bandpass(xs, fs, 40, min(100.0, 0.45 * fs)) if fs >= 120 else np.zeros_like(xs)
    gstd = xs.std() + 1e-12
    half = max(2, int(0.2 * fs))
    peaks = np.asarray(peaks, dtype=int)
    scores = np.full(nw, np.nan)
    reasons = [""] * nw
    for k in range(nw):
        a, b = k * w, min(n, (k + 1) * w)
        seg = xs[a:b]
        if len(seg) < w // 2:
            continue
        if seg.std() < 0.02 * gstd or np.ptp(seg) == 0:
            scores[k], reasons[k] = 0.0, "flat-line"
            continue
        sc, why = 1.0, []
        clip = float(np.mean(seg == seg.max()) + np.mean(seg == seg.min()))
        if clip > 0.05:
            sc -= 0.4
            why.append("clipping")
        v = np.var(seg) + 1e-12
        if np.var(hf[a:b]) / v > 0.20:
            sc -= 0.3
            why.append("HF noise")
        if np.var(bw[a:b]) / v > 0.7:
            sc -= 0.2
            why.append("baseline wander")
        pk = peaks[(peaks >= a) & (peaks < b)]
        hr_w = len(pk) / ((b - a) / fs) * 60.0
        if len(pk) < 2 or hr_w < 20 or hr_w > 260:
            sc -= 0.4
            why.append("implausible beat count")
        elif len(pk) >= 4:
            pk2 = pk[(pk - half >= 0) & (pk + half < n)]
            if len(pk2) >= 4:
                M = xs[pk2[:, None] + np.arange(-half, half)[None, :]]
                M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
                T = np.median(M, 0)
                T = (T - T.mean()) / (T.std() + 1e-9)
                mc = float(np.mean(M @ T / len(T)))
                if mc < 0.85:
                    sc -= 0.2
                    why.append("beats noisy vs template")
                if mc < 0.6:
                    sc -= 0.2
                    why.append("beats unlike template")
        scores[k], reasons[k] = max(0.0, sc), ", ".join(why)
    valid = np.isfinite(scores)
    return dict(win=w, scores=scores, reasons=reasons,
                overall=float(np.nanmean(scores)) if valid.any() else float("nan"),
                clean_frac=float(np.mean(scores[valid] >= thr)) if valid.any() else float("nan"))


def beat_clean(samples, sqi, thr):
    idx = np.minimum(np.asarray(samples) // sqi["win"], len(sqi["scores"]) - 1)
    return np.nan_to_num(sqi["scores"][idx], nan=1.0) >= thr


def _walk(arr, start, step, thr, limit, run=3):
    n, cnt = len(arr), 0
    last = start
    for k in range(limit):
        i = start + step * k
        if i < 0 or i >= n:
            break
        last = i
        if arr[i] < thr:
            cnt += 1
            if cnt >= run:
                return i - step * (run - 1)
        else:
            cnt = 0
    return last


def measure_beat(xf, fs, r, prev_r=None, next_r=None):
    """Approximate delineation of one beat. Returns None if the beat can't be measured."""
    n = len(xf)
    if prev_r is None:
        prev_r = r - int(0.8 * fs)
    if next_r is None:
        next_r = r + int(0.8 * fs)
    if r - int(0.40 * fs) < 0 or r + int(0.60 * fs) >= n:
        return None
    k = max(2, int(0.04 * fs))
    d = np.abs(np.gradient(xf))
    if d[r - k:r + k + 1].max() <= 0:
        return None
    s_on = r - k + int(np.argmax(d[r - k:r + 1]))
    s_off = r + int(np.argmax(d[r:r + k + 1]))
    thr = 0.12 * d[r - k:r + k + 1].max()
    lim = int(0.12 * fs)
    on = _walk(d, s_on, -1, thr, lim)
    off = _walk(d, s_off, +1, thr, lim)
    if off <= on:
        return None
    iso_a, iso_b = on - int(0.035 * fs), on - int(0.005 * fs)
    if iso_a < 0 or iso_b <= iso_a:
        return None
    iso = float(np.median(xf[iso_a:iso_b]))
    qrs_seg = xf[on:off + 1]
    out = dict(qrs_ms=(off - on) / fs * 1000.0,
               r_amp=float(qrs_seg.max() - iso), s_amp=float(iso - qrs_seg.min()),
               n_pos_peaks=0, neg_dominant=False,
               st_mv=np.nan, pr_s=None, qt_s=None, qtc_ms=None)
    rng = float(qrs_seg.max() - qrs_seg.min())
    if rng > 0:
        pk, _ = find_peaks(qrs_seg, prominence=0.2 * rng, height=iso)
        out["n_pos_peaks"] = len(pk)
        out["neg_dominant"] = (iso - qrs_seg.min()) > 2.0 * max(qrs_seg.max() - iso, 1e-9)
    # ST level: J point + 60 ms relative to PR-segment baseline
    st_i = off + int(0.06 * fs)
    if st_i < n:
        out["st_mv"] = float(xf[st_i] - iso)
    ref_amp = max(abs(xf[r] - iso), 1e-9)
    # P wave / PR interval
    a = max(on - int(0.28 * fs), prev_r + int(0.15 * fs))
    b = on - int(0.03 * fs)
    if b - a >= int(0.06 * fs) and a >= 0:
        seg = xf[a:b + 1]
        kk = int(np.argmax(seg))
        p_amp = float(seg[kk] - iso)
        if 0 < kk < len(seg) - 1 and p_amp >= 0.05 * ref_amp and seg[kk] >= seg[kk - 1] and seg[kk] >= seg[kk + 1]:
            p_idx = a + kk
            j = p_idx
            while j > a and xf[j] > iso + 0.2 * p_amp and p_idx - j < int(0.12 * fs):
                j -= 1
            p_on = j
            pr = (on - p_on) / fs
            if 0.06 <= pr <= 0.40:
                out["pr_s"] = pr
    # T wave / QT interval
    ta = off + int(0.08 * fs)
    tb = min(off + int(0.45 * fs), r + int(0.65 * (next_r - r)))
    if tb - ta >= int(0.06 * fs) and tb < n:
        seg = xf[ta:tb + 1] - iso
        kk = int(np.argmax(np.abs(seg)))
        t_amp = float(abs(seg[kk]))
        if 0 < kk < len(seg) - 1 and t_amp >= 0.05 * ref_amp:
            e = np.abs(xf - iso)
            t_end = _walk(e, ta + kk, +1, 0.15 * t_amp, int(0.25 * fs))
            qt = (t_end - on) / fs
            rr = (r - prev_r) / fs
            if 0.25 <= qt <= 0.65 and rr > 0.3:
                out["qt_s"] = qt
                out["qtc_ms"] = qt / np.sqrt(rr) * 1000.0
    return out


def measure_lead(col, fs, samples, sel):
    xf = bandpass(col, fs, 0.5, 40)
    res = []
    for i in sel:
        m = measure_beat(xf, fs, samples[i],
                         samples[i - 1] if i > 0 else None,
                         samples[i + 1] if i + 1 < len(samples) else None)
        if m is not None:
            res.append(m)
    return res, xf


def _med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


# ======================================================================
# 3. 13-CONDITION RULE ENGINE
# ======================================================================
def C(status, evidence):
    return {"status": status, "evidence": evidence}


def analyze_record(rec, name, max_beats=300, min_sqi=0.5):
    fs = rec["fs"]
    sig = rec["signal"]
    names = rec["sig_names"]
    cal = rec["calibrated"]
    main, v1, v5 = pick_leads(names)
    x = sig[:, main]
    samples, symbols = get_beats(rec, x, fs)
    annotated = symbols is not None

    res = dict(name=name, fs=fs, fs_source=rec.get("fs_source", ""), duration_s=sig.shape[0] / fs,
               leads=", ".join(map(str, names)),
               main_lead=str(names[main]), n_beats=int(len(samples)), has_ann=annotated,
               class_counts={}, hr_med=np.nan, hr_min=np.nan, hr_max=np.nan, rr_cv=np.nan,
               qrs_ms=np.nan, pr_ms=np.nan, p_ratio=np.nan, qtc_ms=np.nan, st_mv={}, sokolow=np.nan,
               ectopic_burden=np.nan, conditions={}, truth=None, risk="Unknown",
               sqi=np.nan, sqi_clean=np.nan)
    conds = {c["id"]: C("Inconclusive", "Too few detectable beats") for c in CONDITIONS}
    res["conditions"] = conds
    if len(samples) < 20:
        return res

    if annotated:
        cc = {k: 0 for k in AAMI_ORDER}
        for s in symbols:
            cc[ANN_TO_AAMI[s]] += 1
        res["class_counts"] = cc
        res["ectopic_burden"] = 1.0 - cc["N"] / max(1, sum(cc.values()))

    # ---- Signal-quality gatekeeper: refuse to screen garbage
    sqi = signal_quality(x, fs, samples, thr=min_sqi)
    res["sqi"], res["sqi_clean"] = sqi["overall"], sqi["clean_frac"]
    if np.isfinite(sqi["clean_frac"]) and sqi["clean_frac"] < 0.5:
        msg = f"Signal quality too low to analyse (SQI {sqi['overall']:.2f}; only {sqi['clean_frac']:.0%} of 5-s windows usable) - please re-record"
        for c in CONDITIONS:
            conds[c["id"]] = C("Inconclusive", msg)
        res["risk"] = "Unreliable signal"
        return res
    clean = beat_clean(samples, sqi, min_sqi)

    rr = np.diff(samples) / fs
    valid = (rr > 0.25) & (rr < 3.0)
    if valid.sum() < 5:
        return res
    hr = 60.0 / rr[valid]
    hr_med = float(np.median(hr))
    res.update(hr_med=hr_med, hr_min=float(np.percentile(hr, 2)), hr_max=float(np.percentile(hr, 98)),
               rr_cv=float(np.std(rr[valid]) / np.mean(rr[valid])))
    pct_b, pct_t = float(np.mean(hr < 60)) * 100, float(np.mean(hr > 100)) * 100

    ne = np.array([s in NORMAL_CONDUCTED for s in symbols]) if annotated else np.ones(len(samples), bool)

    # ---- eligible beats for morphology measurements (non-ectopic, plausible RR, clean signal)
    elig = [i for i in range(1, len(samples) - 1)
            if ne[i] and ne[i - 1] and clean[i] and 0.3 < (samples[i] - samples[i - 1]) / fs < 2.0]
    if len(elig) > max_beats:
        elig = list(np.array(elig)[np.linspace(0, len(elig) - 1, max_beats).astype(int)])
    lead_ids = list(range(sig.shape[1])) if sig.shape[1] <= 4 else sorted({main, v1, v5} - {None})
    meas, xfs = {}, {}
    for ch in lead_ids:
        meas[ch], xfs[ch] = measure_lead(sig[:, ch], fs, samples, elig)
    mm = meas.get(main, [])

    qrs_ms = _med([m["qrs_ms"] for m in mm])
    pr_list = [m["pr_s"] for m in mm if m["pr_s"] is not None]
    p_ratio = len(pr_list) / len(mm) if mm else float("nan")
    pr_ms = _med(pr_list) * 1000 if pr_list else float("nan")
    pr_std = float(np.std(pr_list)) if len(pr_list) > 5 else float("nan")
    qtc_vals = [m["qtc_ms"] for m in mm if m["qtc_ms"] is not None]
    t_ratio = len(qtc_vals) / len(mm) if mm else float("nan")
    qtc_ms = _med(qtc_vals)
    res.update(qrs_ms=qrs_ms, pr_ms=pr_ms, p_ratio=p_ratio, qtc_ms=qtc_ms)
    p_txt = "n/a" if not np.isfinite(p_ratio) else f"{p_ratio:.0%}"

    # ---- 1. AFib: RR irregularity in 30-beat windows (ectopic/noisy windows skipped)
    flags = []
    win = 30
    for s in range(0, len(rr) - win + 1, win // 2):
        w = rr[s:s + win]
        if ne[s:s + win + 1].mean() < 0.9 or clean[s:s + win + 1].mean() < 0.9 or w.min() < 0.25 or w.max() > 2.5:
            continue
        m_ = w.mean()
        cv = w.std() / m_
        rmssd = np.sqrt(np.mean(np.diff(w) ** 2)) / m_
        flags.append(cv > 0.15 and rmssd > 0.15)
    afib_frac = float(np.mean(flags)) if flags else float("nan")
    if not flags:
        conds[1] = C("Inconclusive", "No clean 30-beat windows available (too many ectopic/noisy beats)")
    elif afib_frac >= 0.5:
        st_ = "Detected" if (not np.isfinite(p_ratio) or p_ratio < 0.6) else "Suspected"
        conds[1] = C(st_, f"{afib_frac:.0%} of {len(flags)} windows irregularly irregular (CV>0.15 & RMSSD/RR>0.15); P detected in {p_txt} of beats")
    elif afib_frac >= 0.2:
        conds[1] = C("Suspected", f"Paroxysmal-type pattern: {afib_frac:.0%} of {len(flags)} windows irregular")
    else:
        conds[1] = C("Not detected", f"Only {afib_frac:.0%} of {len(flags)} windows irregular")
    afib_on = conds[1]["status"] == "Detected"

    # ---- 2/3. Rate
    if afib_on:
        for cid in (2, 3):
            conds[cid] = C("Inconclusive", f"AFib present - rate is ventricular, not sinus (median {hr_med:.0f} bpm)")
    else:
        sinus_ok = np.isfinite(p_ratio) and p_ratio >= 0.6
        for cid, hit, label in ((2, hr_med < 60, f"<60 bpm ({pct_b:.0f}% of beats)"),
                                (3, hr_med > 100, f">100 bpm ({pct_t:.0f}% of beats)")):
            if hit:
                conds[cid] = C("Detected" if sinus_ok else "Suspected",
                               f"Median HR {hr_med:.0f} bpm, {label}; P before QRS in {p_txt}" + ("" if sinus_ok else " - sinus origin not confirmed"))
            else:
                conds[cid] = C("Not detected", f"Median HR {hr_med:.0f} bpm")

    # ---- 4. First-degree AV block
    if not np.isfinite(pr_ms) or (np.isfinite(p_ratio) and p_ratio < 0.4):
        conds[4] = C("Inconclusive", f"P waves not reliably located in lead {names[main]} (found in {p_txt})")
    elif pr_ms > 200:
        conds[4] = C("Detected", f"Median PR ~{pr_ms:.0f} ms (>200 ms; approximate threshold-based delineation)")
    else:
        conds[4] = C("Not detected", f"Median PR ~{pr_ms:.0f} ms")

    # ---- 5/6. Mobitz via pauses
    prog = fixed = 0
    mb_cache = {}

    def pr_at(i):
        if i not in mb_cache:
            m = measure_beat(xfs[main], fs, samples[i], samples[i - 1] if i > 0 else None,
                             samples[i + 1] if i + 1 < len(samples) else None)
            mb_cache[i] = m["pr_s"] if m else None
        return mb_cache[i]

    n_pause = 0
    if not afib_on:
        for j in range(5, len(rr)):
            med_ = np.median(rr[j - 5:j])
            if med_ > 0.3 and 1.6 * med_ < rr[j] < 2.6 * med_:
                n_pause += 1
                if n_pause > 40:
                    break
                prs = [pr_at(k) for k in range(j - 3, j + 1)]
                prs = [p for p in prs if p is not None]
                if len(prs) >= 3:
                    steps = np.diff(prs)
                    if np.all(steps >= -0.005) and prs[-1] - prs[0] >= 0.02:
                        prog += 1
                    elif max(prs) - min(prs) <= 0.02:
                        fixed += 1
    if afib_on:
        conds[5] = conds[6] = C("Inconclusive", "AFib present - AV-block pattern analysis skipped")
    elif n_pause == 0:
        conds[5] = C("Not detected", "No dropped-beat pauses (RR > 1.6x local median)")
        conds[6] = C("Not detected", "No dropped-beat pauses (RR > 1.6x local median)")
    else:
        ev = f"{n_pause} pauses; {prog} with progressive PR lengthening, {fixed} with constant PR"
        conds[5] = C("Suspected" if prog >= 2 and prog > fixed else "Not detected", ev)
        conds[6] = C("Suspected" if fixed >= 2 and fixed >= prog else "Not detected", ev)

    # ---- 7. Complete heart block (low confidence: no independent P-P tracking)
    if hr_med < 50 and res["rr_cv"] < 0.12:
        if (np.isfinite(pr_std) and pr_std > 0.05) or (np.isfinite(p_ratio) and p_ratio < 0.5):
            conds[7] = C("Suspected", f"Slow regular ventricular rate ({hr_med:.0f} bpm) with variable/absent PR relation (low confidence)")
        else:
            conds[7] = C("Not detected", f"Slow regular rhythm but stable PR relation (SD {pr_std*1000 if np.isfinite(pr_std) else float('nan'):.0f} ms)")
    else:
        conds[7] = C("Not detected", f"Ventricular rate {hr_med:.0f} bpm, RR CV {res['rr_cv']:.2f}")

    # ---- 8/9. Bundle branch block (morphology lead: V1 if present, else main)
    bb_lead = v1 if (v1 is not None and meas.get(v1)) else main
    bm = meas.get(bb_lead, [])
    qrs_bb = _med([m["qrs_ms"] for m in bm]) if bm else qrs_ms
    if not np.isfinite(qrs_bb):
        conds[8] = conds[9] = C("Inconclusive", "QRS width could not be measured")
    elif qrs_bb < 120:
        ev = f"Measured QRS ~{qrs_bb:.0f} ms (<120 ms)"
        conds[8] = conds[9] = C("Not detected", ev)
    else:
        m_frac = float(np.mean([m["n_pos_peaks"] >= 2 for m in bm])) if bm else 0
        n_frac = float(np.mean([m["neg_dominant"] for m in bm])) if bm else 0
        conf = "" if bb_lead == v1 else " (non-V1 lead: low confidence on side)"
        ev = f"QRS ~{qrs_bb:.0f} ms in {names[bb_lead]}; M-shaped in {m_frac:.0%}, broad negative in {n_frac:.0%} of beats{conf}"
        if m_frac >= 0.5:
            conds[8] = C("Suspected", ev)
            conds[9] = C("Not detected", ev)
        elif v1 is not None and bb_lead == v1 and n_frac >= 0.5:
            conds[8] = C("Not detected", ev)
            conds[9] = C("Suspected", ev)
        elif bb_lead != v1:
            conds[8] = C("Not detected", ev)
            conds[9] = C("Suspected", ev)
        else:
            conds[8] = conds[9] = C("Inconclusive", "Wide QRS but side not determinable. " + ev)

    # ---- 10/11. ST deviation
    st_mv = {}
    for ch, ml in meas.items():
        if ml and cal[ch]:
            st_mv[str(names[ch])] = _med([m["st_mv"] for m in ml])
    res["st_mv"] = st_mv
    if not st_mv:
        conds[10] = conds[11] = C("Not assessable", "Signal not calibrated in mV")
    elif np.isfinite(qrs_ms) and qrs_ms >= 120:
        ev = f"QRS ~{qrs_ms:.0f} ms (bundle-branch pattern) makes ST interpretation unreliable"
        conds[10] = conds[11] = C("Inconclusive", ev)
    else:
        vals = {k: v for k, v in st_mv.items() if np.isfinite(v)}
        up = [k for k, v in vals.items() if v >= 0.10]
        dn = [k for k, v in vals.items() if v <= -0.10]
        txt = ", ".join(f"{k}: {v:+.2f} mV" for k, v in vals.items())
        multi = "multi-lead" if len(up) >= 2 else "single-lead; >=2 contiguous leads needed to confirm"
        conds[10] = C("Suspected", f"ST >= +0.10 mV ({multi}). {txt}") if up else C("Not detected", f"ST at J+60ms: {txt}")
        conds[11] = C("Suspected", f"ST <= -0.10 mV in {', '.join(dn)}. {txt}") if dn else C("Not detected", f"ST at J+60ms: {txt}")

    # ---- 12. LVH (Sokolow-Lyon) needs V1 and V5/V6
    if v1 is None or v5 is None or v1 == v5 or not (cal[v1] and cal[v5]) or not meas.get(v1) or not meas.get(v5):
        conds[12] = C("Not assessable", f"Needs calibrated V1 and V5/V6 leads; file has: {res['leads']}")
    else:
        s_v1 = _med([m["s_amp"] for m in meas[v1]])
        r_v5 = _med([m["r_amp"] for m in meas[v5]])
        sok = s_v1 + r_v5
        res["sokolow"] = sok
        ev = f"Sokolow-Lyon: S(V1) {s_v1:.2f} + R({names[v5]}) {r_v5:.2f} = {sok:.2f} mV (criterion >=3.5)"
        conds[12] = C("Suspected" if sok >= 3.5 else "Not detected", ev)

    # ---- 13. QT prolongation (Bazett)
    if not np.isfinite(qtc_ms) or (np.isfinite(t_ratio) and t_ratio < 0.4):
        conds[13] = C("Inconclusive", f"T-wave end not reliably located (found in {0 if not np.isfinite(t_ratio) else t_ratio:.0%} of beats)")
    elif qtc_ms > 450:
        extra = "; Bazett over-corrects at HR>100" if hr_med > 100 else ""
        conds[13] = C("Suspected", f"Median QTc ~{qtc_ms:.0f} ms (>450 ms){extra}")
    else:
        conds[13] = C("Not detected", f"Median QTc ~{qtc_ms:.0f} ms")

    # ---- annotation-derived truth (VALIDATION ONLY - never used by the rules above)
    if annotated:
        total = float(sig.shape[0])
        dur = {}
        for a_, b_, lab in rec["rhythm"]:
            dur[lab] = dur.get(lab, 0) + (b_ - a_)
        res["truth"] = dict(
            afib=dur.get("AFIB", 0) / total >= 0.10,
            mobitz=dur.get("BII", 0) > 0,
            lbbb=float(np.mean(symbols == "L")) > 0.5,
            rbbb=float(np.mean(symbols == "R")) > 0.5,
        )

    flagged_tiers = [COND_BY_ID[i]["tier"] for i, c in conds.items() if c["status"] in FLAGGED]
    top = max((TIER_RANK[t] for t in flagged_tiers), default=0)
    if top == 3:
        res["risk"] = "High"
    elif top == 2 or (np.isfinite(res["ectopic_burden"]) and res["ectopic_burden"] > 0.10):
        res["risk"] = "Moderate"
    else:
        res["risk"] = "Low"
    return res


# ======================================================================
# 4. FEATURES, AUGMENTATION, MODELS
# ======================================================================
def _stats_block(seg, z):
    g = np.gradient(z, axis=1)
    h = z.shape[1] // 2
    return np.column_stack([seg.mean(1), seg.std(1), seg.max(1), seg.min(1), np.ptp(seg, axis=1),
                            g.mean(1), g.std(1), g.max(1), g.min(1),
                            z[:, :h].std(1), z[:, h:].std(1), np.abs(z[:, h:]).mean(1)])


def zscore(seg):
    seg = np.asarray(seg, dtype=np.float64)
    m, s = seg.mean(1, keepdims=True), seg.std(1, keepdims=True)
    return (seg - m) / np.where(s > 0, s, 1.0)


def extract_segments(x, fs, samples):
    """Cut beats around the given R positions; resample to SEG_LEN; add local RR context features."""
    samples = np.asarray(samples, dtype=int)
    pre, post = int(round(0.25 * fs)), int(round(0.27 * fs))
    L = pre + post
    ok = (samples - pre >= 0) & (samples + post <= len(x))
    s_ok = samples[ok]
    idx = s_ok[:, None] + np.arange(-pre, post)[None, :]
    seg = x[idx]
    pos = np.linspace(0, L - 1, SEG_LEN)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, L - 1)
    fr = pos - lo
    seg = seg[:, lo] * (1 - fr) + seg[:, hi] * fr
    rr = np.diff(samples) / fs
    loc = pd.Series(rr).rolling(11, center=True, min_periods=1).median().values
    loc_b = np.r_[loc, loc[-1]]
    prev = np.r_[loc_b[0], rr]
    nxt = np.r_[rr, loc_b[-1]]
    rrf = np.column_stack([prev, nxt, prev / loc_b, nxt / loc_b, loc_b])[ok]
    return seg.astype(np.float32), rrf.astype(np.float32), ok


def build_inputs(seg, rrf=None, demo=None):
    """z-scored waveform (for CNN / RF) + tabular context (waveform stats, RR features, optional age/sex)."""
    seg = np.nan_to_num(np.asarray(seg, dtype=np.float64))
    z = zscore(seg)
    blocks = [_stats_block(seg, z)]
    if rrf is not None:
        blocks.append(rrf)
    if demo is not None:
        blocks.append(demo)
    tab = np.nan_to_num(np.hstack(blocks)).astype(np.float32)
    return z.astype(np.float32), tab


def augment_segments(seg, rng, dur_s=SEG_DUR_S):
    """Physiology-aware augmentation: time-stretch (rate change), amplitude jitter, baseline wander, HF noise."""
    seg = np.asarray(seg, dtype=np.float64)
    n, L = seg.shape
    sd = seg.std(1, keepdims=True) + 1e-9
    f = rng.uniform(0.85, 1.15, n)
    c = (L - 1) / 2.0
    pos = np.clip(c + (np.arange(L)[None, :] - c) / f[:, None], 0, L - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, L - 1)
    fr = pos - lo
    rows = np.arange(n)[:, None]
    out = seg[rows, lo] * (1 - fr) + seg[rows, hi] * fr
    out = out * rng.uniform(0.9, 1.1, (n, 1))
    t = np.linspace(0, dur_s, L)[None, :]
    out = out + rng.uniform(0.05, 0.4, (n, 1)) * sd * np.sin(2 * np.pi * rng.uniform(0.15, 0.6, (n, 1)) * t + rng.uniform(0, 2 * np.pi, (n, 1)))
    out = out + rng.normal(0, 1, (n, L)) * rng.uniform(0.01, 0.12, (n, 1)) * sd
    return out.astype(np.float32), f


def augment_minority(seg, rrf, demo, y, ratio, rng, max_mult=6):
    """Bring each minority class up to `ratio` x the majority count (capped at max_mult x its own size).
    New beats are synthesised from TRAINING beats only and are appended after the originals."""
    if ratio <= 0 or len(y) == 0:
        return seg, rrf, demo, y
    cnt = Counter(y)
    target = int(ratio * max(cnt.values()))
    S, R, D, Y = [], [], [], []
    for c, k in cnt.items():
        if k >= target or k < 5:
            continue
        n_new = min(target - k, max_mult * k)
        pick = rng.choice(np.where(y == c)[0], n_new, replace=True)
        s_new, f = augment_segments(seg[pick], rng)
        S.append(s_new)
        Y.extend([c] * n_new)
        if rrf is not None:
            r = rrf[pick].copy()
            r[:, [0, 1, 4]] *= f[:, None]  # slower waveform -> longer RR
            R.append(r)
        if demo is not None:
            D.append(demo[pick])
    if not S:
        return seg, rrf, demo, y
    seg2 = np.vstack([seg] + S)
    rrf2 = np.vstack([rrf] + R) if rrf is not None else None
    demo2 = np.vstack([demo] + D) if demo is not None else None
    return seg2, rrf2, demo2, np.concatenate([y, np.array(Y)])


if TORCH_OK:

    class FocalLoss(nn.Module):
        """Focal loss (Lin et al.): down-weights easy 'Normal' beats so gradients focus on rare arrhythmias."""

        def __init__(self, gamma=2.0, weight=None):
            super().__init__()
            self.gamma, self.weight = gamma, weight

        def forward(self, logits, target):
            logp = F.log_softmax(logits, 1)
            lp = logp.gather(1, target[:, None]).squeeze(1)
            pt = lp.exp()
            w = self.weight[target] if self.weight is not None else 1.0
            return (-w * (1 - pt) ** self.gamma * lp).mean()

    class ECGNet(nn.Module):
        """1D-CNN over the beat waveform fused with a dense branch for tabular context (RR, age, sex...)."""

        def __init__(self, tab_dim, n_cls, p=0.3):
            super().__init__()
            self.c1, self.b1 = nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16)
            self.c2, self.b2 = nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32)
            self.c3, self.b3 = nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64)
            self.drop = nn.Dropout(p)
            self.tab = nn.Sequential(nn.Linear(tab_dim, 32), nn.ReLU(), nn.Dropout(p))
            self.head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, n_cls))
            self.last_feat = None

        def encode(self, z):  # z: (B,1,L)
            h = self.drop(F.max_pool1d(F.relu(self.b1(self.c1(z))), 2))
            h = self.drop(F.max_pool1d(F.relu(self.b2(self.c2(h))), 2))
            return F.relu(self.b3(self.c3(h)))

        def forward(self, z, t):  # z: (B,L)
            f = self.encode(z.unsqueeze(1))
            if f.requires_grad:
                f.retain_grad()
            self.last_feat = f
            return self.head(torch.cat([f.mean(2), self.tab(t)], 1))

    def nt_xent(a, b, tau=0.5):
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        n = a.shape[0]
        reps = torch.cat([a, b], 0)
        sim = reps @ reps.T / tau - 1e9 * torch.eye(2 * n)
        tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
        return F.cross_entropy(sim, tgt)

    def pretrain_encoder(net, seg, epochs, rng, cap=20000, bs=256):
        """SimCLR-style contrastive pre-training of the conv encoder on UNLABELED training beats."""
        proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
        params = [p for m in (net.c1, net.b1, net.c2, net.b2, net.c3, net.b3, proj) for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=1e-3)
        idx_all = rng.choice(len(seg), min(cap, len(seg)), replace=False)
        net.train()
        proj.train()
        for _ in range(epochs):
            rng.shuffle(idx_all)
            for i in range(0, len(idx_all) - bs + 1, bs):
                b = seg[idx_all[i:i + bs]]
                v1 = torch.from_numpy(zscore(augment_segments(b, rng)[0]).astype(np.float32))
                v2 = torch.from_numpy(zscore(augment_segments(b, rng)[0]).astype(np.float32))
                g1 = proj(net.encode(v1.unsqueeze(1)).mean(2))
                g2 = proj(net.encode(v2.unsqueeze(1)).mean(2))
                loss = nt_xent(g1, g2)
                opt.zero_grad()
                loss.backward()
                opt.step()


class BeatModel:
    """Unified wrapper: Random Forest or PyTorch CNN. predict() -> (probabilities aligned to `labels`, spread)."""

    def __init__(self, kind, labels, opts):
        self.kind = kind if (kind != "cnn" or TORCH_OK) else "rf"
        self.labels = list(labels)
        self.opts = opts
        self.m = None
        self.params = {}

    @property
    def use_spread(self):
        return self.kind == "cnn"

    def fit(self, z, tab, y, seg_for_pretrain=None):
        o = self.opts
        if self.kind == "rf":
            self.m = RandomForestClassifier(n_estimators=o["n_trees"], max_depth=25, min_samples_split=4,
                                            class_weight="balanced_subsample", random_state=42, n_jobs=-1)
            self.m.fit(np.hstack([z, tab]), y)
            self.params = dict(model="RandomForest", n_estimators=o["n_trees"], max_depth=25, min_samples_split=4,
                               class_weight="balanced_subsample", random_state=42)
        else:
            self._fit_torch(z, tab, y, seg_for_pretrain)
        return self

    def _fit_torch(self, z, tab, y, pretrain_seg):
        o = self.opts
        torch.manual_seed(42)
        rng = np.random.RandomState(42)
        self.t_mu, self.t_sd = tab.mean(0), tab.std(0) + 1e-6
        lab_idx = {c: i for i, c in enumerate(self.labels)}
        yi = np.array([lab_idx[v] for v in y])
        C_ = len(self.labels)
        net = ECGNet(tab.shape[1], C_)
        if o.get("pretrain") and pretrain_seg is not None and len(pretrain_seg) > 512:
            pretrain_encoder(net, pretrain_seg, o["pretrain_epochs"], rng)
        cnt = np.bincount(yi, minlength=C_).astype(float)
        w = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0)
        w = w / w[cnt > 0].mean()
        wt = torch.tensor(w, dtype=torch.float32)
        loss_fn = FocalLoss(2.0, wt) if o["loss"] == "focal" else nn.CrossEntropyLoss(weight=wt)
        Xz = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
        Xt = torch.from_numpy(((tab - self.t_mu) / self.t_sd).astype(np.float32))
        Y = torch.from_numpy(yi.astype(np.int64))
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Y)
        for _ in range(o["epochs"]):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                b = perm[i:i + 256]
                if len(b) < 2:
                    continue
                loss = loss_fn(net(Xz[b], Xt[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        self.m = net
        self.params = dict(model="ECGNet(1D-CNN + tabular fusion)", loss=o["loss"], gamma=2.0 if o["loss"] == "focal" else None,
                           epochs=o["epochs"], mc_passes=o["mc_passes"], contrastive_pretrain=bool(o.get("pretrain")),
                           pretrain_epochs=o["pretrain_epochs"] if o.get("pretrain") else 0, optimizer="Adam lr=2e-3 wd=1e-4",
                           dropout=0.3, seed=42)

    def predict(self, z, tab):
        if self.kind == "rf":
            X = np.hstack([z, tab]).astype(np.float32)
            Pm = self.m.predict_proba(X)
            P = np.zeros((len(X), len(self.labels)))
            P[:, [self.labels.index(c) for c in self.m.classes_]] = Pm
            ent = -(P * np.log(P + 1e-12)).sum(1) / np.log(max(2, len(self.labels)))
            return P, ent  # for RF the "spread" is normalised predictive entropy (informational only)
        net = self.m
        net.eval()
        for mod in net.modules():  # Monte-Carlo Dropout: dropout stays ON, BatchNorm stays in eval mode
            if isinstance(mod, nn.Dropout):
                mod.train()
        Xt = ((tab - self.t_mu) / self.t_sd).astype(np.float32)
        z = np.ascontiguousarray(z, dtype=np.float32)
        T = self.opts["mc_passes"]
        outs = []
        with torch.no_grad():
            for i in range(0, len(z), 2048):
                zb, tb = torch.from_numpy(z[i:i + 2048]), torch.from_numpy(Xt[i:i + 2048])
                outs.append(torch.stack([F.softmax(net(zb, tb), 1) for _ in range(T)]).numpy())
        net.eval()
        P = np.concatenate(outs, 1)  # (T, n, C)
        mean = P.mean(0)
        pred = mean.argmax(1)
        spread = P[:, np.arange(len(pred)), pred].std(0)
        return mean, spread

    def explain(self, z1, tab1, cls_idx, win=12, stride=4):
        """Saliency over the beat: Grad-CAM (CNN) or occlusion sensitivity (RF). Returned in [0,1]."""
        L = len(z1)
        if self.kind == "cnn":
            net = self.m
            net.eval()
            zt = torch.from_numpy(np.asarray(z1, dtype=np.float32)[None, :])
            tt = torch.from_numpy(((tab1 - self.t_mu) / self.t_sd)[None, :].astype(np.float32))
            net.zero_grad()
            out = net(zt, tt)
            feat = net.last_feat
            out[0, cls_idx].backward()
            w = feat.grad[0].mean(1)
            cam = torch.relu((w[:, None] * feat[0]).sum(0)).detach().numpy()
            cam = np.interp(np.linspace(0, len(cam) - 1, L), np.arange(len(cam)), cam)
            return cam / (cam.max() + 1e-9)
        starts = list(range(0, L - win + 1, stride))
        Z = np.tile(np.asarray(z1, dtype=np.float32), (len(starts) + 1, 1))
        for k, s in enumerate(starts):
            Z[k + 1, s:s + win] = 0.0
        P, _ = self.predict(Z, np.tile(tab1, (len(Z), 1)))
        sal, cnt = np.zeros(L), np.zeros(L)
        for k, s in enumerate(starts):
            sal[s:s + win] += max(0.0, P[0, cls_idx] - P[k + 1, cls_idx])
            cnt[s:s + win] += 1
        sal /= np.maximum(cnt, 1)
        return sal / (sal.max() + 1e-9)


# ======================================================================
# 5. SAFETY LAYER: reject option, rule-engine cross-check, decision metrics
# ======================================================================
def width_proxy_ms(seg, dur_s=SEG_DUR_S):
    """Rough QRS-width proxy from a resampled beat: samples within +-120 ms of the peak above 25% of the peak deviation."""
    seg = np.asarray(seg, dtype=np.float64)
    L = seg.shape[1]
    dev = np.abs(seg - np.median(seg, 1, keepdims=True))
    c = dev.argmax(1)
    half = max(2, int(0.12 / dur_s * L))
    idx = np.clip(c[:, None] + np.arange(-half, half + 1)[None, :], 0, L - 1)
    win = np.take_along_axis(dev, idx, 1)
    return (win > 0.25 * dev.max(1, keepdims=True)).sum(1) * dur_s / L * 1000.0


def rule_conflicts(pred, rrf, seg):
    """Hybrid rule override: the AI says S/V/F (ectopic) but the beat is neither premature nor wide.
    Ectopic beats are normally early (RR ratio < ~0.88) or wide; a perfectly on-time narrow 'ectopic' is suspicious."""
    if rrf is None:
        return np.zeros(len(pred), bool)
    r1, r2 = rrf[:, 2], rrf[:, 3]
    regular = (r1 > 0.88) & (r1 < 1.12) & (r2 > 0.88) & (r2 < 1.12)
    return np.isin(pred, ["S", "V", "F"]) & regular & (width_proxy_ms(seg) < 105)


def apply_decisions(proba, spread, labels, conf_thr, max_spread, use_spread, clean=None, conflict=None):
    pred = np.array(labels)[proba.argmax(1)]
    conf = proba.max(1)
    status = np.full(len(pred), "accepted", dtype=object)
    unc = conf < conf_thr
    if use_spread:
        unc = unc | (spread > max_spread)
    status[unc] = "uncertain"
    if conflict is not None:
        status[(status == "accepted") & conflict] = "rule_conflict"
    if clean is not None:
        status[~clean] = "low_sqi"
    return pred, conf, status


def decision_metrics(y, pred, status, labels):
    acc_m = status == "accepted"
    cov = float(acc_m.mean()) if len(y) else float("nan")
    out = dict(coverage=cov, n=int(len(y)), n_accepted=int(acc_m.sum()),
               n_uncertain=int((status == "uncertain").sum()), n_conflict=int((status == "rule_conflict").sum()),
               n_low_sqi=int((status == "low_sqi").sum()))
    if acc_m.any():
        out["acc_accepted"] = float(accuracy_score(y[acc_m], pred[acc_m]))
        out["f1_accepted"] = float(f1_score(y[acc_m], pred[acc_m], average="macro", zero_division=0))
    else:
        out["acc_accepted"] = out["f1_accepted"] = float("nan")
    rows = []
    for c in labels:
        t = y == c
        if t.sum() == 0:
            continue
        rows.append({"True class": c, "Beats": int(t.sum()),
                     "Confident & correct": f"{np.mean(acc_m[t] & (pred[t] == c)):.1%}",
                     "Referred to cardiologist": f"{np.mean(~acc_m[t]):.1%}",
                     "Confident & WRONG": f"{np.mean(acc_m[t] & (pred[t] != c)):.1%}"})
    out["per_class"] = pd.DataFrame(rows)
    return out


def sweep_table(oof, thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95), max_spread=0.25):
    rows = []
    for thr in thresholds:
        pred, _, status = apply_decisions(oof["proba"], oof["spread"], oof["labels"], thr, max_spread,
                                          oof["use_spread"], oof["clean"], oof["conflict"])
        d = decision_metrics(oof["y"], pred, status, oof["labels"])
        rows.append({"Confidence threshold": thr, "Coverage (auto-decided)": d["coverage"],
                     "Accuracy on decided": d["acc_accepted"], "Macro-F1 on decided": d["f1_accepted"]})
    return pd.DataFrame(rows)


# ======================================================================
# 6. EVALUATION (patient-wise GroupKFold + beat-level secondary)
# ======================================================================
def metrics_from_pred(y_true, y_pred):
    present = set(y_true) | set(y_pred)
    labels = [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    sens, spec = [], []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        if tp + fn > 0:
            sens.append(tp / (tp + fn))
        if tn + fp > 0:
            spec.append(tn / (tn + fp))
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
                macro_sens=float(np.mean(sens)), macro_spec=float(np.mean(spec)),
                cm=cm, labels=labels, report=rep, n_test=int(len(y_true)))


def crit_recall(m):
    """Mean recall over the clinically costly classes present (S, V, F, or Abnormal)."""
    vals = [m["report"][c]["recall"] for c in CRITICAL_CLASSES if c in m["labels"] and m["report"][c]["support"] > 0]
    return float(np.mean(vals)) if vals else float("nan")


def _sub(D, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in D.items()}


def evaluate_signal(D, opts, cb=None):
    """D: beats from annotated signal records. Returns (ml dict, model bundle) or (None, None)."""
    cb = cb or (lambda f, t: None)
    notes = []
    y = D["y"]
    cnt = pd.Series(y).value_counts()
    rare = cnt[cnt < 10].index.tolist()
    if rare:
        D = _sub(D, ~np.isin(y, rare))
        y = D["y"]
        notes.append(f"Classes with fewer than 10 beats were excluded from evaluation: {', '.join(rare)}")
    present = set(y)
    labels = [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))
    if len(labels) < 2:
        return None, None
    n = len(y)
    seg, rrf, demo, clean, groups = D["seg"], D["rrf"], D["demo"], D["clean"], D["groups"]
    z_all, tab_all = build_inputs(seg, rrf, demo)
    rng = np.random.RandomState(42)
    aug_log = {"added": 0}

    def fit_on(idx):
        tr = idx[clean[idx]]
        seg_a, rrf_a, demo_a, y_a = augment_minority(seg[tr], rrf[tr], demo[tr] if demo is not None else None,
                                                      y[tr], opts["aug_ratio"] / 100.0, rng)
        z_tr, tab_tr = z_all[tr], tab_all[tr]
        n_new = len(y_a) - len(tr)
        if n_new > 0:
            z_new, tab_new = build_inputs(seg_a[len(tr):], rrf_a[len(tr):], demo_a[len(tr):] if demo_a is not None else None)
            z_tr, tab_tr = np.vstack([z_tr, z_new]), np.vstack([tab_tr, tab_new])
        aug_log["added"] = n_new
        return BeatModel(opts["model_kind"], labels, opts).fit(z_tr, tab_tr, y_a, seg_for_pretrain=seg[tr])

    n_groups = len(np.unique(groups))
    k = min(opts["cv_folds"], n_groups)
    oof_p = np.zeros((n, len(labels)), np.float32)
    oof_s = np.zeros(n, np.float32)
    have = np.zeros(n, bool)
    patient = None
    if k >= 3:
        f1s, accs, fold_sizes = [], [], []
        for fi, (tr, te) in enumerate(GroupKFold(n_splits=k).split(z_all, y, groups)):
            cb(0.05 + 0.6 * fi / k, f"Patient-wise fold {fi + 1}/{k}: training on {len(np.unique(groups[tr]))} records, testing on {len(np.unique(groups[te]))} unseen records")
            m = fit_on(tr)
            P, S = m.predict(z_all[te], tab_all[te])
            oof_p[te], oof_s[te], have[te] = P, S, True
            pr = np.array(labels)[P.argmax(1)]
            f1s.append(float(f1_score(y[te], pr, average="macro", labels=sorted(set(y[te]) | set(pr)), zero_division=0)))
            accs.append(float(accuracy_score(y[te], pr)))
            fold_sizes.append(int(len(np.unique(groups[te]))))
        pooled = metrics_from_pred(y, np.array(labels)[oof_p.argmax(1)])
        patient = dict(k=k, fold_f1=f1s, fold_acc=accs, fold_records=fold_sizes, mean_f1=float(np.mean(f1s)), std_f1=float(np.std(f1s)),
                       mean_acc=float(np.mean(accs)), pooled=pooled, n_records=int(n_groups))
    else:
        notes.append(f"Patient-wise cross-validation needs at least 3 records (found {n_groups}). Showing beat-level figures only - "
                     "these are OPTIMISTIC because beats of the same recording appear in train and test.")

    beat = None
    final_needed = True
    if opts["beat_level"] or patient is None:
        cb(0.7, "Beat-level stratified 80/20 split (secondary)")
        tr_i, te_i = train_test_split(np.arange(n), test_size=0.20, random_state=42, stratify=y)
        m = fit_on(tr_i)
        P, S = m.predict(z_all[te_i], tab_all[te_i])
        beat = metrics_from_pred(y[te_i], np.array(labels)[P.argmax(1)])
        if patient is None:
            oof_p[te_i], oof_s[te_i], have[te_i] = P, S, True
    cb(0.85, "Training final model on all records (kept for XAI, safety page and external validation)")
    final = fit_on(np.arange(n))
    cb(1.0, "Done")

    pred_all = np.array(labels)[oof_p.argmax(1)]
    h = have
    oof = dict(proba=oof_p[h], spread=oof_s[h], y=y[h], labels=labels, use_spread=final.use_spread,
               clean=clean[h], conflict=rule_conflicts(pred_all[h], rrf[h], seg[h]),
               rec=groups[h], age=D["age"][h] if D["age"] is not None else None,
               sex=D["sex"][h] if D["sex"] is not None else None)
    ml = dict(schema="signal", kind=final.kind, labels=labels, patient=patient, beat=beat, oof=oof, notes=notes,
              class_counts={k_: int(v) for k_, v in pd.Series(y).value_counts().items()},
              n_beats=int(n), n_records=int(n_groups), n_augmented=int(aug_log["added"]),
              params=final.params,
              split_desc=(f"{patient['k']}-fold GroupKFold by record (patient-wise)" if patient else "Beat-level stratified 80/20 split"))
    bundle = dict(model=final, labels=labels, schema="signal", demo_used=demo is not None,
                  age_fill=D.get("age_fill", 60.0), kind=final.kind, params=final.params)
    return ml, bundle


def evaluate_matrix(parts, scheme, opts, cb=None):
    cb = cb or (lambda f, t: None)
    notes = []
    if parts["train"] and parts["test"]:
        Xtr = np.vstack([p[0] for p in parts["train"]])
        ytr = np.concatenate([p[1] for p in parts["train"]])
        Xte = np.vstack([p[0] for p in parts["test"]])
        yte = np.concatenate([p[1] for p in parts["test"]])
        i1, i2 = _cap_idx(len(ytr), opts["max_beats"]), _cap_idx(len(yte), opts["max_beats"])
        Xtr, ytr, Xte, yte = Xtr[i1], ytr[i1], Xte[i2], yte[i2]
        split_desc = "Official train/test files (trained on the train file, scored on the unseen test file)"
    else:
        allp = parts["all"] + parts["train"] + parts["test"]
        Xm = np.vstack([p[0] for p in allp])
        ym = np.concatenate([p[1] for p in allp])
        i1 = _cap_idx(len(ym), opts["max_beats"])
        Xm, ym = Xm[i1], ym[i1]
        cnt = pd.Series(ym).value_counts()
        rare = cnt[cnt < 10].index.tolist()
        if rare:
            keep = ~np.isin(ym, rare)
            Xm, ym = Xm[keep], ym[keep]
            notes.append(f"Classes with fewer than 10 beats were excluded: {', '.join(rare)}")
        if len(set(ym)) < 2:
            return None, None
        Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.20, random_state=42, stratify=ym)
        split_desc = "Beat-level stratified 80/20 split"
    cnt = pd.Series(ytr).value_counts()
    rare = cnt[cnt < 10].index.tolist()
    if rare:
        keep = ~np.isin(ytr, rare)
        Xtr, ytr = Xtr[keep], ytr[keep]
        notes.append(f"Classes with fewer than 10 training beats were excluded: {', '.join(rare)}")
    m_ = np.isin(yte, np.unique(ytr))
    Xte, yte = Xte[m_], yte[m_]
    if len(set(ytr)) < 2 or len(yte) == 0:
        return None, None
    present = set(ytr) | set(yte)
    labels = [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))
    notes.append("These files carry no patient identifiers, so patient-wise validation is impossible. Public beat-matrix files are shuffled "
                 "across patients, so figures here are beat-level and likely OPTIMISTIC versus truly unseen patients.")
    rng = np.random.RandomState(42)
    cb(0.2, "Augmenting minority beats")
    Xa, _, _, ya = augment_minority(Xtr, None, None, ytr, opts["aug_ratio"] / 100.0, rng)
    z_tr, tab_tr = build_inputs(Xa)
    z_te, tab_te = build_inputs(Xte)
    cb(0.4, "Training")
    model = BeatModel(opts["model_kind"], labels, opts).fit(z_tr, tab_tr, ya, seg_for_pretrain=Xtr)
    P, S = model.predict(z_te, tab_te)
    beat = metrics_from_pred(yte, np.array(labels)[P.argmax(1)])
    cb(1.0, "Done")
    oof = dict(proba=P, spread=S, y=yte, labels=labels, use_spread=model.use_spread, clean=None, conflict=None,
               rec=None, age=None, sex=None)
    ml = dict(schema=scheme, kind=model.kind, labels=labels, patient=None, beat=beat, oof=oof, notes=notes,
              class_counts={k_: int(v) for k_, v in pd.Series(np.r_[ytr, yte]).value_counts().items()},
              n_beats=int(len(ytr) + len(yte)), n_records=None, n_augmented=int(len(ya) - len(ytr)), params=model.params,
              split_desc=split_desc)
    bundle = dict(model=model, labels=labels, schema=scheme, demo_used=False, age_fill=60.0, kind=model.kind, params=model.params)
    return ml, bundle


def per_record_table(oof, rec_names, conf_thr, max_spread):
    if oof["rec"] is None:
        return pd.DataFrame()
    pred, _, status = apply_decisions(oof["proba"], oof["spread"], oof["labels"], conf_thr, max_spread,
                                      oof["use_spread"], oof["clean"], oof["conflict"])
    rows = []
    for g in np.unique(oof["rec"]):
        m = oof["rec"] == g
        yt, pr = oof["y"][m], pred[m]
        rows.append({"Record": rec_names.get(int(g), str(g)), "Beats": int(m.sum()),
                     "Accuracy": float(accuracy_score(yt, pr)),
                     "Macro-F1": float(f1_score(yt, pr, average="macro", labels=sorted(set(yt) | set(pr)), zero_division=0)) if len(set(yt)) > 1 else np.nan,
                     "Referred (uncertain/conflict/low-SQI)": float(np.mean(status[m] != "accepted")),
                     "Rule conflicts": int(np.sum(status[m] == "rule_conflict"))})
    return pd.DataFrame(rows).sort_values("Accuracy")


def subgroup_table(oof):
    """Bias check: out-of-fold performance by sex and age band (only if a metadata CSV was supplied)."""
    if oof["rec"] is None or oof["age"] is None:
        return pd.DataFrame()
    pred = np.array(oof["labels"])[oof["proba"].argmax(1)]
    age, sex = oof["age"].astype(float), oof["sex"].astype(float)
    groups = []
    groups += [(f"Sex = {nm}", sex == v) for nm, v in (("Male", 1.0), ("Female", 0.0))]
    for nm, lo, hi in (("Age < 40", 0, 40), ("Age 40-59", 40, 60), ("Age 60-79", 60, 80), ("Age >= 80", 80, 200)):
        groups.append((nm, (age >= lo) & (age < hi)))
    rows = []
    for nm, m in groups:
        if m.sum() < 30:
            continue
        yt, pr = oof["y"][m], pred[m]
        rows.append({"Subgroup": nm, "Beats": int(m.sum()), "Records": int(len(np.unique(oof["rec"][m]))),
                     "Accuracy": float(accuracy_score(yt, pr)),
                     "Macro-F1": float(f1_score(yt, pr, average="macro", labels=sorted(set(yt) | set(pr)), zero_division=0))})
    return pd.DataFrame(rows)


# ======================================================================
# 7. DATASET PIPELINE
# ======================================================================
def safe_extract(uploaded, dest):
    with zipfile.ZipFile(uploaded) as zf:
        root = os.path.realpath(dest)
        for m in zf.infolist():
            if m.filename.startswith("__MACOSX") or m.is_dir():
                continue
            target = os.path.realpath(os.path.join(dest, m.filename))
            if not target.startswith(root + os.sep):
                continue  # zip-slip protection
            zf.extract(m, dest)


def ingest_uploads(files, dest):
    """Accepts .zip archives and/or loose files (csv, txt, hea, dat, atr, mat)."""
    for f in files:
        if f.name.lower().endswith(".zip"):
            safe_extract(f, dest)
        else:
            name = os.path.basename(f.name)
            with open(os.path.join(dest, name), "wb") as out:
                out.write(f.getbuffer())


def discover_records(root):
    """Returns {key: (kind, path, annotation_path_or_None)}; pairs 100.csv with 100annotations.txt automatically."""
    wf, sig_files, ann_files = {}, [], []
    for d, _, files in os.walk(root):
        if "__MACOSX" in d:
            continue
        for f in sorted(files):
            low = f.lower()
            if f.startswith("._") or low.startswith(IGNORE_PREFIX):
                continue
            full = os.path.join(d, f)
            if low.endswith(".hea"):
                wf[(d, f[:-4])] = full[:-4]
            elif low.endswith((".csv", ".txt")):
                (ann_files if looks_like_annotation(full) else sig_files).append((d, f, full))
    recs = {}
    for (d, stem), base in wf.items():
        key = stem if stem not in recs else f"{os.path.basename(d)}/{stem}"
        recs[key] = ("wfdb", base, None)
    for d, f, full in sig_files:
        stem = os.path.splitext(f)[0]
        if (d, stem) in wf:
            continue
        ns = _norm_stem(stem)
        cand = [a for a in ann_files if a[0] == d and _norm_stem(os.path.splitext(a[1])[0], True) == ns]
        cand = cand or [a for a in ann_files if _norm_stem(os.path.splitext(a[1])[0], True) == ns]
        if cand:
            ann = cand[0][2]
        elif len(sig_files) == 1 and len(ann_files) == 1:
            ann = ann_files[0][2]
        else:
            ann = None
        key = stem if stem not in recs else f"{os.path.basename(d)}/{stem}"
        recs[key] = ("csv", full, ann)
    return recs


def load_any(kind, path, ann, opts):
    if kind == "wfdb":
        if not WFDB_OK:
            raise RuntimeError("wfdb is not installed")
        return load_wfdb_record(path)
    return load_csv_record(path, ann, opts)


def _cap_idx(n, cap, seed=42):
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, cap, replace=False))


def _sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unreadable"


def lineage_for(name, kind, path, ann):
    files = []
    if kind == "wfdb":
        files = [path + e for e in (".hea", ".dat", ".mat", ".atr") if os.path.isfile(path + e)]
    else:
        files = [path] + ([ann] if ann else [])
    return [{"Record": name, "File": os.path.basename(f), "Bytes": os.path.getsize(f), "SHA-256": _sha256(f)} for f in files]


def extract_record_beats(rec, name, gid, demo_map, min_sqi):
    """Labelled beats from one annotated signal record -> arrays used for training / external validation."""
    if rec["ann_samples"] is None:
        return None
    main, _, _ = pick_leads(rec["sig_names"])
    x = rec["signal"][:, main]
    mask = np.array([s in ANN_TO_AAMI for s in rec["ann_symbols"]])
    smp, sym = rec["ann_samples"][mask], rec["ann_symbols"][mask]
    if len(smp) <= 20:
        return None
    seg, rrf, ok = extract_segments(x, rec["fs"], smp)
    sqi = signal_quality(x, rec["fs"], smp, thr=min_sqi)
    clean = beat_clean(smp, sqi, min_sqi)[ok]
    key = _norm_stem(os.path.basename(str(name)))
    age, sx = demo_map.get(key, (np.nan, np.nan)) if demo_map else (np.nan, np.nan)
    m = int(ok.sum())
    return dict(seg=seg, rrf=rrf, y=np.array([ANN_TO_AAMI[s] for s in sym[ok]]), clean=clean,
                groups=np.full(m, gid), age=np.full(m, age, dtype=float), sex=np.full(m, sx, dtype=float))


def concat_beats(chunks, demo_map):
    D = {k: np.concatenate([c[k] for c in chunks]) if k in ("y", "clean", "groups", "age", "sex") else np.vstack([c[k] for c in chunks])
         for k in ("seg", "rrf", "y", "clean", "groups", "age", "sex")}
    D["demo"], D["age_fill"] = None, 60.0
    if demo_map:
        D["age_fill"] = float(np.nanmedian(D["age"])) if np.isfinite(D["age"]).any() else 60.0
        D["demo"] = demo_features(D["age"], D["sex"], D["age_fill"])
    else:
        D["age"] = D["sex"] = None
    return D


def run_pipeline(recs, opts, dataset_name, demo_map=None):
    per, skipped, beat_files, lineage, chunks, gnames = {}, {}, {}, [], [], {}
    mats = {}
    bar = st.progress(0.0, text="Analysing files...")
    for k, (name, (kind, path, ann)) in enumerate(recs.items()):
        bar.progress((k + 1) / max(1, len(recs)), text=f"Analysing {name} ({k + 1}/{len(recs)})")
        try:
            lineage.extend(lineage_for(name, kind, path, ann))
            rec = load_any(kind, path, ann, opts)
            if rec.get("kind") == "beats":
                s = mats.setdefault(rec["scheme"], dict(train=[], test=[], all=[]))
                s[rec["role"]].append((rec["X"], rec["y"]))
                beat_files[name] = {"File": name, "Type": SCHEME_TITLE[rec["scheme"]], "Role": rec["role"],
                                    "Beats": len(rec["y"]), "Class counts": str(dict(Counter(rec["y"])))}
                continue
            if rec["signal"].shape[0] < rec["fs"] * 5:
                raise ValueError("record shorter than 5 s")
            per[name] = analyze_record(rec, name, min_sqi=opts["min_sqi"])
            b = extract_record_beats(rec, name, k, demo_map, opts["min_sqi"])
            if b is not None:
                chunks.append(b)
                gnames[k] = name
        except Exception as e:  # keep going - one bad file must not kill the run
            skipped[name] = f"{type(e).__name__}: {e}"
    bar.empty()

    ml_all, bundles = {}, {}
    prog = st.progress(0.0, text="Training and evaluating the beat classifier...")
    cb = lambda f, t: prog.progress(min(1.0, max(0.0, f)), text=t)
    if chunks:
        D = concat_beats(chunks, demo_map)
        sel = _cap_idx(len(D["y"]), opts["max_beats"])
        D = {k_: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(D["y"]) else v) for k_, v in D.items()}
        if len(D["y"]) >= 200:
            ml, bd = evaluate_signal(D, opts, cb)
            if ml:
                ml["rec_names"] = gnames
                ml_all[SIG_TITLE], bundles[SIG_TITLE] = ml, bd
    for scheme, parts in mats.items():
        ml, bd = evaluate_matrix(parts, scheme, opts, cb)
        if ml:
            ml_all[SCHEME_TITLE[scheme]], bundles[SCHEME_TITLE[scheme]] = ml, bd
    prog.empty()

    title0 = next(iter(ml_all)) if ml_all else None
    ml = ml_all[title0] if title0 else None
    ml_msg = None
    if not ml_all:
        if per:
            ml_msg = ("No reference annotations (.atr, annotation .txt) or labelled beat files were found, so classification accuracy "
                      "cannot be computed. The 13-condition rule screening below still ran on the raw signals.")
        else:
            ml_msg = "Not enough labelled beats / classes to train and test a classifier."
    audit("analysis_run", f"dataset={dataset_name}; records={len(per)}; model={opts['model_kind']}; "
                          f"aug={opts['aug_ratio']}%; folds={opts['cv_folds']}; conf_thr={opts['conf_thr']}")
    return dict(per=per, skipped=skipped, ml=ml, ml_all=ml_all, bundles=bundles, bundle=bundles.get(title0), ml_msg=ml_msg,
                beat_files=beat_files, dataset=dataset_name, recs={n: recs[n] for n in per}, lineage=lineage,
                demo_map=demo_map, opts=dict(opts), time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


def agreement_table(per):
    rows = []
    spec = [("AFib", 1, "afib"), ("2nd-degree AV block (Mobitz I or II)", None, "mobitz"),
            ("LBBB", 9, "lbbb"), ("RBBB", 8, "rbbb")]
    for label, cid, key in spec:
        tp = fp = fn = tn = 0
        for r in per.values():
            if r["truth"] is None:
                continue
            if cid is None:
                pred = r["conditions"][5]["status"] in FLAGGED or r["conditions"][6]["status"] in FLAGGED
            else:
                pred = r["conditions"][cid]["status"] in FLAGGED
            truth = r["truth"][key]
            tp += pred and truth
            fp += pred and not truth
            fn += (not pred) and truth
            tn += (not pred) and (not truth)
        n = tp + fp + fn + tn
        if n == 0:
            continue
        rows.append({"Condition": label, "Records checked": n, "Annotated positive": tp + fn,
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "Sensitivity": f"{tp / (tp + fn):.0%}" if tp + fn else "n/a (no positives)",
                     "Specificity": f"{tn / (tn + fp):.0%}" if tn + fp else "n/a"})
    return pd.DataFrame(rows)


def condition_summary(per):
    rows = []
    for c in CONDITIONS:
        cnt = {"Detected": 0, "Suspected": 0, "Not detected": 0, "Inconclusive": 0, "Not assessable": 0}
        for r in per.values():
            cnt[r["conditions"][c["id"]]["status"]] += 1
        tested = len(per) - cnt["Not assessable"]
        rows.append({"#": c["id"], "Condition": c["name"], "Risk tier": TIER_ICON[c["tier"]],
                     "Records tested": tested, "Detected": cnt["Detected"], "Suspected": cnt["Suspected"],
                     "Not detected": cnt["Not detected"], "Inconclusive": cnt["Inconclusive"],
                     "Not assessable": cnt["Not assessable"], "Needs": c["needs"]})
    return pd.DataFrame(rows)


def record_conditions_df(res):
    rows = []
    for c in CONDITIONS:
        r = res["conditions"][c["id"]]
        tested = "No" if r["status"] == "Not assessable" else ("Partly" if r["status"] == "Inconclusive" else "Yes")
        icon = {"Detected": "🔴", "Suspected": "🟠", "Not detected": "🟢", "Inconclusive": "⚪", "Not assessable": "⚫"}[r["status"]]
        rows.append({"#": c["id"], "Condition": c["name"], "Tested?": tested,
                     "Result": f"{icon} {r['status']}", "Risk tier": TIER_ICON[c["tier"]], "Evidence": r["evidence"]})
    return pd.DataFrame(rows)


# ======================================================================
# 8. PLOTS
# ======================================================================
def plot_confusion(cm, labels, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title(f"{title} - counts")
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="RdPu", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title(f"{title} - row-normalised (sensitivity)")
    for a in axes:
        a.set_xlabel("Predicted class")
        a.set_ylabel("True class (reference label)")
    fig.tight_layout()
    return fig


def plot_waveform(rec, start_s, dur_s, samples, symbols):
    fs = rec["fs"]
    a, b = int(start_s * fs), int((start_s + dur_s) * fs)
    n_lead = min(2, rec["signal"].shape[1])
    fig, axes = plt.subplots(n_lead, 1, figsize=(11, 2.8 * n_lead), sharex=True, squeeze=False)
    t = np.arange(a, min(b, rec["signal"].shape[0])) / fs
    in_win = (samples >= a) & (samples < b)
    for i in range(n_lead):
        ax = axes[i, 0]
        y = rec["signal"][a:b, i]
        ax.plot(t[:len(y)], y, color="#d90429", lw=1.1)
        pk = samples[in_win]
        ax.scatter(pk / fs, rec["signal"][pk, i], s=18, color="#1f4e79", zorder=3)
        if symbols is not None:
            for s_, sym in zip(pk, symbols[in_win]):
                ax.text(s_ / fs, rec["signal"][s_, i], sym, fontsize=8, color="#5A2A38", ha="center", va="bottom", weight="bold")
        ax.set_ylabel(f"{rec['sig_names'][i]}")
        ax.grid(True, ls="--", alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].set_title("ECG with beat markers" + (" (reference annotation symbols)" if symbols is not None else " (detected R peaks)"))
    fig.tight_layout()
    return fig


def plot_sqi(sqi, fs, thr):
    sc = sqi["scores"]
    t = (np.arange(len(sc)) + 0.5) * sqi["win"] / fs
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.bar(t, np.nan_to_num(sc), width=sqi["win"] / fs * 0.9, color=["#2a9d8f" if (np.isfinite(s) and s >= thr) else "#d90429" for s in sc])
    ax.axhline(thr, color="#3A1F1F", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SQI per 5-s window")
    ax.set_title(f"Signal Quality Index (green = analysed, red = rejected; threshold {thr:.2f})")
    fig.tight_layout()
    return fig


def plot_saliency(z, sal, title):
    t = np.linspace(-250, 270, len(z))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, z, color="#d90429", lw=1.6)
    ax.fill_between(t, z.min(), z.max(), where=sal > 0.5, color="#D4AF37", alpha=0.35, label="High attribution")
    ax2 = ax.twinx()
    ax2.plot(t, sal, color="#1f4e79", lw=1, alpha=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Attribution (0-1)")
    ax.axvline(0, color="#3A1F1F", ls=":", lw=1)
    ax.set_xlabel("ms relative to R peak")
    ax.set_ylabel("z-scored beat")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ======================================================================
# 9. MODEL CARD / REGULATORY EVIDENCE
# ======================================================================
def build_model_card(results):
    ml, bd, o = results["ml"], results.get("bundle"), results.get("opts", {})
    card = dict(
        generated=results["time"], dataset=results["dataset"], author=OWNER["name"],
        reviewer=f"{REVIEWER['name']} ({REVIEWER['role']}) - review of workflow; not a regulatory sign-off",
        intended_use="Research / education screening aid for ECG beat classification and rule-based rhythm screening. "
                     "NOT intended for diagnosis, triage or treatment decisions.",
        environment=dict(python=sys.version.split()[0], platform=platform.platform(), numpy=_ver("numpy"), scipy=_ver("scipy"),
                         scikit_learn=_ver("scikit-learn"), pandas=_ver("pandas"), wfdb=_ver("wfdb"), torch=_ver("torch"),
                         streamlit=_ver("streamlit")),
        options=o, lineage=results.get("lineage", []),
    )
    if ml:
        card["model"] = dict(kind=ml["kind"], params=ml["params"], labels=ml["labels"], split=ml["split_desc"],
                             class_counts=ml["class_counts"], n_beats=ml["n_beats"], n_records=ml["n_records"],
                             n_synthetic_training_beats_added_last_fit=ml["n_augmented"])
        p, b = ml["patient"], ml["beat"]
        card["performance"] = dict(
            patient_wise=dict(macro_f1_mean=p["mean_f1"], macro_f1_sd=p["std_f1"], fold_f1=p["fold_f1"], accuracy_mean=p["mean_acc"],
                              pooled_macro_f1=p["pooled"]["macro_f1"], critical_class_recall=crit_recall(p["pooled"]),
                              folds=p["k"], records=p["n_records"]) if p else "not computed (needs >= 3 annotated records)",
            beat_level=dict(accuracy=b["accuracy"], macro_f1=b["macro_f1"]) if b else "not computed")
        oof = ml["oof"]
        pred, _, status = apply_decisions(oof["proba"], oof["spread"], oof["labels"], o.get("conf_thr", 0.85), o.get("max_spread", 0.25),
                                          oof["use_spread"], oof["clean"], oof["conflict"])
        d = decision_metrics(oof["y"], pred, status, oof["labels"])
        card["reject_option"] = dict(confidence_threshold=o.get("conf_thr", 0.85), coverage=d["coverage"],
                                     accuracy_on_decided=d["acc_accepted"], macro_f1_on_decided=d["f1_accepted"],
                                     referred_uncertain=d["n_uncertain"], referred_rule_conflict=d["n_conflict"],
                                     referred_low_sqi=d["n_low_sqi"])
        sg = subgroup_table(oof)
        card["subgroup_performance"] = sg.to_dict("records") if len(sg) else "not available (no age/sex metadata supplied)"
    card["limitations"] = [
        "Trained and evaluated only on the uploaded data; no claim is made beyond it.",
        "Public annotation sets (e.g. MIT-BIH) contain few patients and a single lead (MLII) in most records; subgroup coverage is limited.",
        "Beat-level and pooled figures are dominated by the majority 'N' class; use macro-F1 and per-class recall.",
        "Rule-engine delineation (PR/QT/ST) is threshold-based and approximate; BBB side and Mobitz detection are low-confidence.",
        "SQI and rule-conflict thresholds are heuristics that have not been clinically calibrated.",
        "No prospective, multi-centre or device-diversity validation has been performed unless the External Validation page was used.",
    ]
    return card


def card_markdown(card):
    L = [f"# HRIDAYA Model Card & Evidence Pack", f"*Generated {card['generated']} - dataset: {card['dataset']}*", "",
         "## Intended use", card["intended_use"], "", "## Authorship / review", f"- Author: {card['author']}", f"- {card['reviewer']}", ""]
    if "model" in card:
        m = card["model"]
        L += ["## Model", f"- Type: {m['kind']}", f"- Labels: {', '.join(m['labels'])}", f"- Split: {m['split']}",
              f"- Beats: {m['n_beats']:,}; records: {m['n_records']}", f"- Class counts: {m['class_counts']}",
              f"- Synthetic training beats added in final fit: {m['n_synthetic_training_beats_added_last_fit']:,}",
              f"- Hyper-parameters: `{json.dumps(m['params'], default=str)}`", ""]
        L += ["## Performance (measured on this upload)", "```json", json.dumps(card["performance"], indent=2, default=float), "```", ""]
        L += ["## Reject option (Uncertain / Refer to Cardiologist)", "```json", json.dumps(card["reject_option"], indent=2, default=float), "```", ""]
        L += ["## Subgroup / bias check", "```json", json.dumps(card["subgroup_performance"], indent=2, default=float), "```", ""]
    L += ["## Known limitations"] + [f"- {x}" for x in card["limitations"]] + ["", "## Environment", "```json",
          json.dumps(card["environment"], indent=2), "```", "", "## Data lineage (SHA-256)", ""]
    L += ["| Record | File | Bytes | SHA-256 |", "|---|---|---|---|"]
    L += [f"| {r['Record']} | {r['File']} | {r['Bytes']:,} | `{r['SHA-256']}` |" for r in card["lineage"]]
    return "\n".join(L)


def regulatory_checklist():
    rows = [
        ("ISO 13485", "Documented quality management / design controls", "Not provided - requires an organisational QMS (procedures, CAPA, supplier control).", "Gap"),
        ("ISO 13485 / IEC 62304", "Traceability: data lineage, versions, parameters", "Model Card + SHA-256 lineage + environment versions + audit log (this page).", "Artefact provided"),
        ("IEC 62304", "Software development plan, requirements, architecture, unit/integration tests, SOUP list", "Source is in one file; no formal plan/tests. SOUP (numpy, scipy, scikit-learn, torch, wfdb, streamlit) versions are recorded.", "Partial"),
        ("IEC 62304", "Software safety classification", "Not performed. A diagnostic-support function would likely be Class B or C - to be decided in a formal risk process.", "Gap"),
        ("ISO 14971", "Risk management file (hazards, controls, residual risk)", "Technical controls exist: reject option, SQI gate, rule cross-check, XAI, disclaimers. No formal hazard analysis.", "Partial"),
        ("IEC 62366-1", "Usability engineering", "Not performed (no formative/summative usability study).", "Gap"),
        ("FDA/ML-SaMD guidance", "Patient-wise validation, subgroup analysis, uncertainty handling", "GroupKFold patient-wise metrics, subgroup table (needs age/sex CSV), reject option.", "Artefact provided"),
        ("FDA/ML-SaMD guidance", "Independent external clinical validation (different site / device)", "External Validation page evaluates a frozen model on a second dataset. A real multi-centre study is still required.", "Partial"),
        ("Data protection (DPDP Act 2023 / HIPAA / GDPR as applicable)", "Privacy, consent, de-identification", "App warns not to upload identifiable data; no consent/retention framework.", "Gap"),
        ("Regulatory pathway (CDSCO India / FDA / CE-MDR)", "Classification, clinical evaluation, post-market surveillance", "Not started. HRIDAYA remains a research prototype.", "Gap"),
    ]
    return pd.DataFrame(rows, columns=["Standard / guidance", "Requirement", "What this app provides", "Status"])


# ======================================================================
# 10. UI HELPERS
# ======================================================================
CSS = """
<style>
:root { --maroon:#8B1E23; --gold:#D4AF37; --cream:#FAF7F0; --beige:#E8DCC6; --brown:#3A1F1F; }
.stApp { background:var(--cream); color:var(--brown); }
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp p,.stApp li,.stApp label { color:var(--brown); }
.stApp h1,.stApp h2 { color:var(--maroon); }
[data-testid="stHeader"] { background:var(--maroon); }
[data-testid="stSidebar"] { background:var(--beige); }
[data-testid="stSidebar"] * { color:var(--brown); }
[data-testid="stSidebar"] img { border-radius:12px; }
[data-testid="stFileUploader"] section { background:#FFFFFF; border:2px dashed var(--gold); }
.stButton>button,.stDownloadButton>button { background:var(--maroon); color:#FFFFFF; border:0; border-radius:10px; }
.stButton>button:hover,.stDownloadButton>button:hover { background:#6E171B; color:#FFFFFF; }
div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--beige); border-left:4px solid var(--gold); border-radius:10px; padding:12px; }
div[data-testid="stMetric"] * { color:var(--brown) !important; }
.doc-card { background:#FFFFFF; border:1px solid var(--beige); border-left:4px solid var(--maroon); border-radius:10px; padding:14px 18px; margin-bottom:12px; color:var(--brown); }
.reviewer-photo { width:100%; max-width:260px; border-radius:16px; border:3px solid var(--gold); box-shadow:0 4px 14px rgba(58,31,31,.25); }
.reviewer-badge { background:#FFFFFF; border:1px solid var(--gold); border-radius:10px; padding:8px 12px; font-size:0.9rem; color:var(--brown); margin-top:10px; }
.disclaimer { background:#FFFFFF; border:1px solid var(--gold); border-radius:10px; padding:10px 14px; margin-bottom:14px; font-size:0.9rem; color:var(--brown); }
.footer { text-align:center; font-size:0.85rem; color:var(--brown); border-top:1px solid var(--beige); padding-top:10px; margin-top:30px; }
</style>
"""


def footer():
    st.markdown(
        f"<div class='footer'>© {datetime.date.today().year} {OWNER['name']} - All Rights Reserved &nbsp;|&nbsp; "
        f"📞 {OWNER['phone']} &nbsp;|&nbsp; ✉️ {OWNER['email']}<br>"
        f"Clinically reviewed by {html.escape(REVIEWER['name'])}, {html.escape(REVIEWER['role'])} (see 🩺 Clinical Review).<br>"
        "Research prototype - not a medical device; does not diagnose, treat, cure or prevent any condition. Consult a qualified healthcare professional.</div>",
        unsafe_allow_html=True)


def results_or_warn():
    r = st.session_state.get("results")
    if r and not r["per"] and r.get("ml_all"):
        st.warning("⚠️ The uploaded data contains only pre-segmented beats (no continuous signal), so rhythm / PR / QRS / QT screening is not possible. "
                   "Upload raw signal records (WFDB or CSV with time/sample data) to use this page.")
        footer()
        return None
    if not r or not r["per"]:
        st.warning("⚠️ Upload ECG data on **📁 ECG Data Upload** and click **Run full analysis** first.")
        footer()
        return None
    return r


def ml_or_warn():
    r = st.session_state.get("results")
    if not r or not r.get("ml_all"):
        st.warning("⚠️ No trained classifier yet. Upload **labelled** ECG data on **📁 ECG Data Upload** and run the analysis first.")
        footer()
        return None
    return r


def show_ml_one(ml):
    p, b = ml["patient"], ml["beat"]
    st.caption(f"Model: **{MODEL_KINDS.get(ml['kind'], ml['kind'])}** - evaluation: {ml['split_desc']} - "
               f"{ml['n_beats']:,} beats" + (f" from {ml['n_records']} records" if ml["n_records"] else "") +
               (f" - {ml['n_augmented']:,} synthetic minority beats added to the final training set" if ml["n_augmented"] else ""))
    for n in ml["notes"]:
        st.warning(n)
    if p:
        pooled = p["pooled"]
        st.markdown(f"**Primary result - patient-wise {p['k']}-fold GroupKFold (every test record was unseen during training)**")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Macro-F1 (mean of folds)", f"{p['mean_f1'] * 100:.1f}%", f"± {p['std_f1'] * 100:.1f}% SD", delta_color="off")
        q2.metric("Macro-F1 (pooled)", f"{pooled['macro_f1'] * 100:.1f}%")
        q3.metric("Critical-class recall (S/V/F)", "n/a" if not np.isfinite(crit_recall(pooled)) else f"{crit_recall(pooled) * 100:.1f}%")
        q4.metric("Accuracy (mean of folds)", f"{p['mean_acc'] * 100:.1f}%")
        f1 = p["mean_f1"]
        cr = crit_recall(pooled)
        msg = (f"Milestone check (roadmap targets: macro-F1 ≥ {TARGETS['macro_f1']:.0%}, critical-class recall ≥ {TARGETS['crit_sens']:.0%}): "
               f"measured macro-F1 {f1:.1%}" + (f", critical recall {cr:.1%}" if np.isfinite(cr) else "") + ". ")
        if f1 >= TARGETS["macro_f1"] and (not np.isfinite(cr) or cr >= TARGETS["crit_sens"]):
            st.success("✅ " + msg + "This meets the engineering milestone on THIS dataset only; it is not evidence of clinical readiness.")
        else:
            st.error("⚠️ " + msg + "Below milestone - reported as measured. Pooled figures hide per-fold variation; see fold table.")
        st.dataframe(pd.DataFrame({"Fold": range(1, p["k"] + 1), "Unseen records": p["fold_records"],
                                   "Macro-F1": np.round(p["fold_f1"], 3), "Accuracy": np.round(p["fold_acc"], 3)}),
                     width='stretch', hide_index=True)
        cm_m = pooled
    else:
        cm_m = b
    if b:
        st.markdown("**Beat-level hold-out (secondary - optimistic, beats of one patient can sit in both train and test)**" if p else
                    "**Beat-level hold-out result**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{b['accuracy'] * 100:.2f}%")
        c2.metric("Macro F1", f"{b['macro_f1'] * 100:.2f}%")
        c3.metric("Macro sensitivity", f"{b['macro_sens'] * 100:.2f}%")
        c4.metric("Macro specificity", f"{b['macro_spec'] * 100:.2f}%")
        c5.metric("Test beats", f"{b['n_test']:,}")
    st.caption(f"Class mix: {ml['class_counts']}. Accuracy is dominated by the majority class - judge the model by macro-F1 and per-class recall. "
               "99.99% accuracy on unseen patients is not a realistic target (label noise, biology, artefacts, rare classes).")
    st.markdown("**🧮 Confusion matrix** (" + ("pooled out-of-fold, patient-wise" if p else "hold-out test beats") + ")")
    fig = plot_confusion(cm_m["cm"], cm_m["labels"], "Beat classifier")
    st.pyplot(fig)
    plt.close(fig)
    rep = pd.DataFrame(cm_m["report"]).T
    rep = rep.loc[[i for i in rep.index if i in cm_m["labels"] or i in ("macro avg", "weighted avg")]]
    st.dataframe(rep.round(3), width='stretch')
    if "N" in cm_m["labels"]:
        st.caption("N = normal/BBB/escape, S = supraventricular ectopic, V = ventricular ectopic, F = fusion, Q = paced/unclassifiable (AAMI EC57 grouping).")


def show_ml_all(results):
    st.markdown("### 🎯 Classification performance (computed from the uploaded labels)")
    ml_all = results.get("ml_all") or {}
    if not ml_all:
        st.warning(results.get("ml_msg") or "Classifier could not be evaluated.")
        return
    for title, ml in ml_all.items():
        st.markdown(f"#### {title}")
        show_ml_one(ml)
        o = results["opts"]
        pred, _, status = apply_decisions(ml["oof"]["proba"], ml["oof"]["spread"], ml["oof"]["labels"], o["conf_thr"], o["max_spread"],
                                          ml["oof"]["use_spread"], ml["oof"]["clean"], ml["oof"]["conflict"])
        d = decision_metrics(ml["oof"]["y"], pred, status, ml["oof"]["labels"])
        st.info(f"🛡️ **Safety layer at confidence ≥ {o['conf_thr']:.2f}:** the model decides {d['coverage']:.1%} of beats automatically "
                f"(accuracy on those: {d['acc_accepted']:.1%}); the rest are routed to **Uncertain / Refer to Cardiologist** "
                f"({d['n_uncertain']:,} low-confidence, {d['n_conflict']:,} rule-conflict, {d['n_low_sqi']:,} low-quality-signal). "
                "Open **🛡️ AI Safety & Explainability** for details.")


def show_waveform_block(results, key_prefix):
    names = sorted(results["per"].keys())
    sel = st.selectbox("Record", names, key=f"{key_prefix}_rec")
    kind, path, ann = results["recs"][sel]
    try:
        rec = load_any(kind, path, ann, st.session_state["opts"])
    except Exception as e:
        st.error(f"Could not reload record: {e}")
        return sel, None
    res = results["per"][sel]
    main, _, _ = pick_leads(rec["sig_names"])
    samples, symbols = get_beats(rec, rec["signal"][:, main], rec["fs"])
    total = rec["signal"].shape[0] / rec["fs"]
    c1, c2 = st.columns(2)
    dur = c1.slider("Window length (s)", 3, 20, 10, key=f"{key_prefix}_dur")
    start = c2.slider("Start (s)", 0.0, float(max(0.0, total - dur)), 0.0, step=1.0, key=f"{key_prefix}_start")
    fig = plot_waveform(rec, start, dur, samples, symbols)
    st.pyplot(fig)
    plt.close(fig)
    return sel, res


def report_html(results, sel):
    res = results["per"][sel]
    ml = results["ml"]
    rows = "".join(
        f"<tr><td>{c['id']}</td><td>{html.escape(c['name'])}</td><td>{html.escape(res['conditions'][c['id']]['status'])}</td>"
        f"<td>{html.escape(TIER_ICON[c['tier']])}</td><td>{html.escape(res['conditions'][c['id']]['evidence'])}</td></tr>"
        for c in CONDITIONS)
    if ml and ml["patient"]:
        p = ml["patient"]
        acc = (f"Patient-wise ({p['k']}-fold GroupKFold) macro-F1 {p['mean_f1']:.2%} ± {p['std_f1']:.2%}; "
               f"pooled macro-F1 {p['pooled']['macro_f1']:.2%}; {p['n_records']} records")
    elif ml and ml["beat"]:
        acc = f"Beat-level only (optimistic): accuracy {ml['beat']['accuracy']:.2%}, macro-F1 {ml['beat']['macro_f1']:.2%}"
    else:
        acc = "Not computed (no reference labels)"
    f_ = lambda v: "n/a" if not np.isfinite(v) else f"{v:.0f}"
    sqi_txt = "n/a" if not np.isfinite(res["sqi"]) else f"{res['sqi']:.2f}"
    lg = logo_b64()
    logo_tag = f"<img src='data:image/png;base64,{lg}' style='height:110px'><br>" if lg else ""
    return f"""<html><head><meta charset="utf-8"><title>HRIDAYA report</title>
<style>body{{font-family:Arial;margin:30px;color:#3A1F1F}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #D4AF37;padding:6px;font-size:13px}}th{{background:#F3E7C9}}h1{{color:#8B1E23}}</style></head><body>
{logo_tag}<h1>HRIDAYA - ECG Screening Report</h1>
<p><b>Dataset:</b> {html.escape(results['dataset'])} &nbsp; <b>Record:</b> {html.escape(sel)} &nbsp; <b>Generated:</b> {results['time']}</p>
<p><b>Overall screening risk:</b> {res['risk']} &nbsp; <b>Median HR:</b> {f_(res['hr_med'])} bpm &nbsp;
<b>QRS:</b> {f_(res['qrs_ms'])} ms &nbsp; <b>PR:</b> {f_(res['pr_ms'])} ms &nbsp; <b>QTc:</b> {f_(res['qtc_ms'])} ms &nbsp; <b>Signal quality (SQI):</b> {sqi_txt}</p>
<p><b>Beat-classifier performance (dataset level):</b> {html.escape(acc)}</p>
<table><tr><th>#</th><th>Condition</th><th>Result</th><th>Risk tier</th><th>Evidence</th></tr>{rows}</table>
<p style="margin-top:20px"><i>Automated screening output; not a diagnosis. AI beat labels below the confidence threshold are routed to "Uncertain / Refer to Cardiologist". Requires review and sign-off by a qualified clinician.</i></p>
<p>Reviewer: ______________________ &nbsp; Signature: ______________ &nbsp; Date: ____________</p>
<hr><p>Project clinically reviewed by {html.escape(REVIEWER['name'])}, {html.escape(REVIEWER['role'])}.</p>
<p>© {datetime.date.today().year} {OWNER['name']} - All Rights Reserved | {OWNER['phone']} | {OWNER['email']}</p>
</body></html>"""


# ======================================================================
# 11. PAGES
# ======================================================================
def page_home():
    if LOGO:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.image(LOGO, width=360)
    st.title("💓 HRIDAYA (हृदय): ECG Research Companion")
    st.markdown("##### Patient-wise validated beat classification, safety layer & 13-condition clinical screener")
    st.info("Every figure in this app is computed from the data you upload - nothing is pre-filled. "
            "Go to **📁 ECG Data Upload** to begin.")
    st.markdown("### The 13 conditions screened")
    st.dataframe(pd.DataFrame([{"#": c["id"], "Condition": c["name"], "Description": c["desc"],
                                "Risk tier": TIER_ICON[c["tier"]], "Needs": c["needs"]} for c in CONDITIONS]),
                 width='stretch', hide_index=True)
    st.markdown("### What the v2 pipeline does")
    st.markdown(
        "1. **Patient-wise evaluation** - GroupKFold by record, so no test patient is ever seen in training. Headline metric: macro-F1 with per-fold spread.\n"
        "2. **Imbalance handling** - balanced class weights, focal / class-weighted loss (CNN option), and physiology-aware augmentation "
        "(baseline wander, muscle-noise, time-stretch) of minority classes in training folds only.\n"
        "3. **Context** - RR features and optional age / sex (upload a small metadata CSV); optional contrastive pre-training of the CNN encoder.\n"
        "4. **Safety layer** - *Uncertain / Refer to Cardiologist* output (confidence + Monte-Carlo Dropout), a Signal-Quality gatekeeper, "
        "a rule-engine cross-check of AI beat labels, and Grad-CAM / occlusion explanations.\n"
        "5. **External validation** - freeze the trained model and test it on a different dataset.\n"
        "6. **Evidence pack** - model card, SHA-256 data lineage, versions, audit log and an honest ISO 13485 / IEC 62304 gap checklist.")
    st.markdown("### Supported uploads (any ECG database)")
    st.markdown(
        "Upload a **.zip** or select the **individual files together**. Nested folders are searched recursively.\n"
        "- **WFDB records** (`.hea` + `.dat`/`.mat`, optional `.atr`): MIT-BIH, INCART, European ST-T, PTB, PhysioNet challenge sets, etc.\n"
        "- **CSV/TXT signal + annotation text:** `100.csv` next to `100annotations.txt` is paired automatically.\n"
        "- **Pre-segmented beat CSVs** (Kaggle `mitbih_train.csv` / `mitbih_test.csv`): accuracy + confusion matrix only (no patient IDs, so no patient-wise validation).\n"
        "- **CSV/TXT with no annotations:** the 13-condition screening runs, but accuracy can't be computed.\n"
        "- **Optional metadata CSV** (`record,age,sex`): adds demographics as model input and enables the subgroup / bias table.")
    st.markdown("### Honest expectations")
    st.markdown("Patient-wise macro-F1 around 0.88-0.92 on diverse records is a demanding target; 99.99% accuracy is not achievable on unseen patients. "
                "HRIDAYA reports what it measures and refers uncertain beats to a clinician instead of guessing.")
    footer()


def page_upload():
    st.title("📁 ECG Data Upload")
    if not WFDB_OK:
        st.warning("The `wfdb` package is missing, so WFDB (.hea/.dat) records can't be read. CSV/TXT uploads still work. Run `pip install wfdb` to enable WFDB.")
    ups = st.file_uploader(
        "Upload ECG data: a .zip, or the individual files together (e.g. 100.csv + 100annotations.txt, or .hea + .dat + .atr)",
        type=["zip", "csv", "txt", "hea", "dat", "atr", "mat"], accept_multiple_files=True)
    st.caption("Unknown sampling rate? Set it under **Analysis options** in the sidebar. "
               "Beat-matrix files (one beat per row, last column = class 0-4) give accuracy + confusion matrix only.")
    with st.expander("➕ Optional: patient metadata (age / sex) for context features and bias analysis"):
        dm = st.file_uploader("CSV with columns: record (or id), age, sex (M/F)", type=["csv"], key="demo_up")
        if dm is not None:
            try:
                st.session_state["demo_map"] = parse_demographics(dm)
                st.success(f"Metadata for {len(st.session_state['demo_map'])} record(s) loaded. It is used in the NEXT analysis run.")
            except Exception as e:
                st.session_state["demo_map"] = None
                st.error(f"Could not read metadata: {e}")
        else:
            st.session_state["demo_map"] = None
        st.caption("With only a handful of patients, demographic inputs cannot be expected to help and may overfit; the subgroup table needs ≥30 beats per group.")
    opts = st.session_state["opts"]
    if opts["model_kind"] == "cnn" and not TORCH_OK:
        st.warning("PyTorch is not installed - falling back to the Random Forest. Add `torch` to requirements.txt to enable the CNN / MC-Dropout option.")

    if ups:
        sig = "|".join(f"{u.name}:{u.size}" for u in ups)
        if st.session_state.get("upload_sig") != sig:
            old = st.session_state.get("workdir")
            if old and os.path.isdir(old):
                shutil.rmtree(old, ignore_errors=True)
            wd = tempfile.mkdtemp(prefix="hridaya_")
            with st.spinner("Reading uploaded files..."):
                ingest_uploads(ups, wd)
            st.session_state.update(workdir=wd, upload_sig=sig,
                                    upload_name=ups[0].name if len(ups) == 1 else f"{ups[0].name} (+{len(ups) - 1} files)",
                                    results=None, found=discover_records(wd))
            audit("upload", f"{len(ups)} file(s)")
        found = st.session_state["found"]
        if not found:
            st.error("No ECG data found. Need WFDB files (`.hea`/`.dat`) or `.csv`/`.txt` signals / beat files in the upload.")
        else:
            wf = sum(1 for v in found.values() if v[0] == "wfdb")
            paired = sum(1 for v in found.values() if v[0] == "csv" and v[2])
            st.success(f"Found {len(found)} record file(s): {wf} WFDB, {len(found) - wf} CSV/TXT "
                       f"({paired} paired with an annotation text file).")
            if st.button("▶ Run full analysis", type="primary"):
                st.session_state["results"] = run_pipeline(found, opts, st.session_state["upload_name"], st.session_state.get("demo_map"))

    results = st.session_state.get("results")
    if not results:
        st.caption("No analysis yet - results appear here after you upload data and run the analysis.")
        footer()
        return
    per = results["per"]
    st.markdown("---")
    st.header(f"Results - {results['dataset']}")
    if results["skipped"]:
        with st.expander(f"⚠️ {len(results['skipped'])} file(s) skipped"):
            st.json(results["skipped"])

    show_ml_all(results)

    if results["beat_files"]:
        st.markdown("### 🧬 Pre-segmented beat files read")
        st.dataframe(pd.DataFrame(list(results["beat_files"].values())), width='stretch', hide_index=True)

    if not per:
        if results["ml_all"]:
            st.info("These files contain isolated beats only (no continuous signal or timing), so the 13-condition rhythm / PR / QT screening "
                    "was not run. Upload raw signal records to get it.")
        else:
            st.error("No record could be analysed.")
        footer()
        return

    st.markdown("### 🩺 Abnormalities detected - 13-condition evaluation (all records)")
    st.dataframe(condition_summary(per), width='stretch', hide_index=True)
    st.caption("**Records tested** = records where the condition could be evaluated with the available leads. "
               "Detected/Suspected are screening flags, not diagnoses.")

    ag = agreement_table(per)
    if len(ag):
        st.markdown("### ✅ Rule-engine validation against dataset annotations")
        st.dataframe(ag, width='stretch', hide_index=True)
        st.caption("Record-level agreement between the rule engine (signal only) and the reference rhythm/beat annotations.")

    st.markdown("### ⚠️ Risk stratification per record")
    df = pd.DataFrame([{
        "Record": n, "Risk": r["risk"], "Beats": r["n_beats"], "Fs (Hz)": round(r["fs"], 1), "Fs source": r["fs_source"],
        "SQI": round(r["sqi"], 2) if np.isfinite(r["sqi"]) else None,
        "HR (bpm)": round(r["hr_med"], 0) if np.isfinite(r["hr_med"]) else None,
        "QRS (ms)": round(r["qrs_ms"], 0) if np.isfinite(r["qrs_ms"]) else None,
        "PR (ms)": round(r["pr_ms"], 0) if np.isfinite(r["pr_ms"]) else None,
        "QTc (ms)": round(r["qtc_ms"], 0) if np.isfinite(r["qtc_ms"]) else None,
        "Ectopic burden": f"{r['ectopic_burden']:.1%}" if np.isfinite(r["ectopic_burden"]) else "n/a",
        "Flagged": ", ".join(COND_BY_ID[i]["name"].split(" (")[0] for i, c in r["conditions"].items() if c["status"] in FLAGGED) or "-",
    } for n, r in per.items()])
    st.dataframe(df, width='stretch', hide_index=True)
    if any(r["fs_source"] == "sidebar setting" for r in per.values()):
        st.caption("Records marked *sidebar setting* had no time column to read the sampling rate from - check the value under Analysis options.")

    st.markdown("### 📈 Waveform")
    sel, res = show_waveform_block(results, "up")
    if res:
        st.markdown(f"**13-condition findings for record {sel}**")
        st.dataframe(record_conditions_df(res), width='stretch', hide_index=True)
    footer()


def page_screener():
    st.title("🔍 Live Waveform & 13-Condition Screener")
    results = results_or_warn()
    if not results:
        return
    sel, res = show_waveform_block(results, "sc")
    if not res:
        return
    if res["risk"] == "Unreliable signal":
        st.error(f"🚫 Signal quality too low to analyse this record (SQI {res['sqi']:.2f}). Please re-record - no screening result is shown.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sampling rate", f"{res['fs']:.0f} Hz")
    m2.metric("Duration", f"{res['duration_s'] / 60:.1f} min")
    m3.metric("Beats", f"{res['n_beats']:,}")
    m4.metric("Screening risk", res["risk"])
    st.caption(f"Leads in file: {res['leads']} - analysis lead: {res['main_lead']} - sampling rate from: {res['fs_source']}")
    m5, m6, m7, m8 = st.columns(4)
    f = lambda v, u: "n/a" if not np.isfinite(v) else f"{v:.0f} {u}"
    m5.metric("Median HR", f(res["hr_med"], "bpm"))
    m6.metric("QRS (approx.)", f(res["qrs_ms"], "ms"))
    m7.metric("PR (approx.)", f(res["pr_ms"], "ms"))
    m8.metric("QTc Bazett (approx.)", f(res["qtc_ms"], "ms"))
    st.markdown("### All 13 conditions - tested or not, with evidence")
    st.dataframe(record_conditions_df(res), width='stretch', hide_index=True)
    if res["class_counts"]:
        st.markdown("### Beat classes in this record (reference annotations)")
        st.bar_chart(pd.Series(res["class_counts"]))
    footer()


def page_risk():
    st.title("⚠️ Clinical Risk Summary")
    results = results_or_warn()
    if not results:
        return
    per = results["per"]
    n_high = sum(r["risk"] == "High" for r in per.values())
    n_mod = sum(r["risk"] == "Moderate" for r in per.values())
    n_low = sum(r["risk"] == "Low" for r in per.values())
    n_bad = sum(r["risk"] == "Unreliable signal" for r in per.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 High-risk records", n_high)
    c2.metric("🟡 Moderate-risk records", n_mod)
    c3.metric("🟢 Low-risk records", n_low)
    c4.metric("⚫ Unreliable signal", n_bad)
    st.caption("High = any Critical-tier condition Detected/Suspected. Moderate = Moderate-tier flag or ectopic burden >10%. Otherwise Low. "
               "Unreliable = rejected by the signal-quality gatekeeper (not analysed).")
    sel = st.selectbox("Record", sorted(per.keys()))
    r = per[sel]
    st.subheader(f"Record {sel}: {r['risk']}" + ("" if r["risk"] == "Unreliable signal" else " risk"))
    flagged = [(i, c) for i, c in r["conditions"].items() if c["status"] in FLAGGED]
    if r["risk"] == "Unreliable signal":
        st.error(r["conditions"][1]["evidence"])
    elif flagged:
        for i, c in flagged:
            box = st.error if COND_BY_ID[i]["tier"] == "Critical" else st.warning
            box(f"**{COND_BY_ID[i]['name']}** - {c['status']} ({TIER_ICON[COND_BY_ID[i]['tier']]})\n\n{c['evidence']}")
    else:
        st.success("No condition flagged by the screener for this record.")
    if np.isfinite(r["ectopic_burden"]):
        st.metric("Ectopic / non-normal beat burden (reference annotations)", f"{r['ectopic_burden']:.1%}")
    if r["st_mv"]:
        st.markdown("**Median ST deviation (J+60 ms)**")
        st.table(pd.DataFrame({"Lead": list(r["st_mv"].keys()), "ST (mV)": [round(v, 3) for v in r["st_mv"].values()]}))
    if r["risk"] in ("High", "Moderate"):
        st.info("Consider cardiology review - see **🏥 Cardiologists (Bengaluru)**.")
    footer()


def page_safety():
    st.title("🛡️ AI Safety & Explainability")
    st.caption("Reject option, signal-quality gatekeeper, rule-engine cross-check and explanations - the layer that stops the model from guessing.")
    results = ml_or_warn()
    if not results:
        return
    o = results["opts"]
    title = st.selectbox("Classifier", list(results["ml_all"].keys())) if len(results["ml_all"]) > 1 else next(iter(results["ml_all"]))
    ml, bundle = results["ml_all"][title], results["bundles"][title]
    oof = ml["oof"]

    st.markdown("### 1. Reject option - *Uncertain / Refer to Cardiologist*")
    c1, c2 = st.columns(2)
    thr = c1.slider("Minimum confidence to auto-decide", 0.30, 0.99, float(o["conf_thr"]), 0.01)
    spread = c2.slider("Maximum MC-Dropout spread (CNN only)", 0.05, 0.50, float(o["max_spread"]), 0.01,
                       disabled=not oof["use_spread"])
    pred, conf, status = apply_decisions(oof["proba"], oof["spread"], oof["labels"], thr, spread, oof["use_spread"], oof["clean"], oof["conflict"])
    d = decision_metrics(oof["y"], pred, status, oof["labels"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Auto-decided (coverage)", f"{d['coverage']:.1%}")
    k2.metric("Accuracy on decided beats", "n/a" if not np.isfinite(d["acc_accepted"]) else f"{d['acc_accepted']:.1%}")
    k3.metric("Macro-F1 on decided beats", "n/a" if not np.isfinite(d["f1_accepted"]) else f"{d['f1_accepted']:.1%}")
    k4.metric("Referred to human", f"{d['n'] - d['n_accepted']:,}")
    st.caption(f"Referred: {d['n_uncertain']:,} low-confidence, {d['n_conflict']:,} rule-engine conflict, {d['n_low_sqi']:,} low signal quality. "
               + ("Uncertainty = Monte-Carlo Dropout (dropout kept on at inference; spread = SD of the predicted-class probability across passes)."
                  if oof["use_spread"] else
                  "Random Forest: confidence = mean tree vote share (the 'spread' column is just normalised entropy, not used to reject)."))
    st.markdown("**What happens to each true class** - the key safety table (a good reject option converts 'confident & wrong' into 'referred'):")
    st.dataframe(d["per_class"], width='stretch', hide_index=True)
    sw = sweep_table(oof, max_spread=spread)
    st.markdown("**Coverage vs accuracy trade-off**")
    st.line_chart(sw.set_index("Confidence threshold")[["Coverage (auto-decided)", "Accuracy on decided", "Macro-F1 on decided"]])
    st.dataframe(sw.round(3), width='stretch', hide_index=True)

    if oof["rec"] is not None:
        st.markdown("### 2. Per-record reliability & bias check (out-of-fold, patient-wise)")
        pr = per_record_table(oof, ml.get("rec_names", {}), thr, spread)
        st.dataframe(pr.round(3), width='stretch', hide_index=True)
        st.caption("A wide spread between best and worst record is the real generalisation risk - pooled numbers hide it.")
        sg = subgroup_table(oof)
        if len(sg):
            st.markdown("**Subgroup performance (from the metadata CSV)**")
            st.dataframe(sg.round(3), width='stretch', hide_index=True)
        else:
            st.caption("Subgroup table appears when a metadata CSV (record, age, sex) is supplied and each group has ≥30 beats.")

    st.markdown("### 3. Hybrid rule-engine cross-check")
    if oof["conflict"] is not None:
        nconf = int(oof["conflict"].sum())
        truth_ect = np.isin(oof["y"], ["S", "V", "F"])
        st.write(f"The rule check flags AI labels of S/V/F where the beat arrives on time (RR ratio 0.88-1.12 on both sides) **and** looks narrow "
                 f"(width proxy < 105 ms). Flagged: **{nconf:,}** beats, of which **{int((oof['conflict'] & truth_ect).sum()):,}** were truly ectopic "
                 f"(cost: a possible missed ectopic goes to a human instead of being auto-labelled) and **{int((oof['conflict'] & ~truth_ect).sum()):,}** "
                 "were false alarms caught before reaching a clinician.")
    else:
        st.caption("Not available for beat-matrix files (no RR timing).")

    st.markdown("### 4. Signal-quality gatekeeper")
    per = results["per"]
    if per:
        sq = pd.DataFrame([{"Record": n, "SQI (mean)": r["sqi"], "Usable 5-s windows": r["sqi_clean"], "Status": r["risk"]} for n, r in per.items()])
        st.dataframe(sq.round(3), width='stretch', hide_index=True)
        sel = st.selectbox("Inspect SQI for record", sorted(per.keys()), key="sqi_rec")
        kind, path, ann = results["recs"][sel]
        try:
            rec = load_any(kind, path, ann, st.session_state["opts"])
            main, _, _ = pick_leads(rec["sig_names"])
            smp, _ = get_beats(rec, rec["signal"][:, main], rec["fs"])
            sqi = signal_quality(rec["signal"][:, main], rec["fs"], smp, thr=o["min_sqi"])
            fig = plot_sqi(sqi, rec["fs"], o["min_sqi"])
            st.pyplot(fig)
            plt.close(fig)
            bad = [(i, r_) for i, r_ in enumerate(sqi["reasons"]) if r_]
            if bad:
                st.caption("Window issues: " + "; ".join(f"{i * 5}-{i * 5 + 5}s: {r_}" for i, r_ in bad[:12]) + (" ..." if len(bad) > 12 else ""))
        except Exception as e:
            st.error(f"Could not compute SQI plot: {e}")
    else:
        st.caption("SQI is shown for continuous-signal records.")

    st.markdown("### 5. Explainability - *why* did the model decide this?")
    if ml["schema"] != "signal" or not per:
        st.caption("Explanations need continuous-signal records (a beat can be re-cut and re-scored).")
    else:
        sel = st.selectbox("Record", sorted(per.keys()), key="xai_rec")
        kind, path, ann = results["recs"][sel]
        try:
            rec = load_any(kind, path, ann, st.session_state["opts"])
            main, _, _ = pick_leads(rec["sig_names"])
            x = rec["signal"][:, main]
            smp, sym = get_beats(rec, x, rec["fs"])
            seg, rrf, ok = extract_segments(x, rec["fs"], smp)
            truth = np.array([ANN_TO_AAMI[s] for s in sym])[ok] if sym is not None else None
            demo = None
            if bundle["demo_used"]:
                age, sx = (results.get("demo_map") or {}).get(_norm_stem(os.path.basename(sel)), (np.nan, np.nan))
                demo = demo_features(np.full(len(seg), age), np.full(len(seg), sx), bundle["age_fill"])
            z, tab = build_inputs(seg, rrf, demo)
            P, S = bundle["model"].predict(z, tab)
            pred_r = np.array(bundle["labels"])[P.argmax(1)]
            cls_filter = st.selectbox("Show beats the model labelled as", [c for c in bundle["labels"] if (pred_r == c).any()],
                                      index=0)
            idxs = np.where(pred_r == cls_filter)[0]
            j = idxs[st.slider("Beat # within that class", 1, len(idxs), 1) - 1] if len(idxs) > 1 else idxs[0]
            ci = bundle["labels"].index(pred_r[j])
            sal = bundle["model"].explain(z[j], tab[j], ci)
            conf_j = float(P[j].max())
            conflict_j = bool(rule_conflicts(pred_r[j:j + 1], rrf[j:j + 1], seg[j:j + 1])[0])
            fig = plot_saliency(z[j], sal, f"Predicted {pred_r[j]} (confidence {conf_j:.0%})" + (f" - reference label {truth[j]}" if truth is not None else ""))
            st.pyplot(fig)
            plt.close(fig)
            if conf_j < thr:
                st.warning("This beat would be routed to **Uncertain / Refer to Cardiologist** (confidence below threshold).")
            if conflict_j:
                st.warning("The rule engine disagrees: an on-time, narrow beat was labelled ectopic - treat as possible AI error.")
            st.caption(("Grad-CAM on the last convolutional layer" if bundle["kind"] == "cnn" else "Occlusion sensitivity (12-sample window slid over the beat; "
                        "attribution = drop in the predicted-class probability)") +
                       ". If the highlighted region lies away from the QRS/ST-T complex (e.g. at the window edges), suspect an artefact-driven label.")
        except Exception as e:
            st.error(f"Could not compute explanation: {type(e).__name__}: {e}")
    footer()


def run_external(bundle, recs, opts, demo_map, bar):
    model, labels = bundle["model"], bundle["labels"]
    skipped = {}
    if bundle["schema"] == "signal":
        chunks, rec_names = [], {}
        for k, (name, (kind, path, ann)) in enumerate(recs.items()):
            bar.progress((k + 1) / len(recs), text=f"Reading external record {name}")
            try:
                rec = load_any(kind, path, ann, opts)
                if rec.get("kind") == "beats":
                    raise ValueError("beat-matrix file cannot be scored by a signal-schema model")
                b = extract_record_beats(rec, name, k, demo_map, opts["min_sqi"])
                if b is None:
                    raise ValueError("no usable reference annotations")
                chunks.append(b)
                rec_names[k] = name
            except Exception as e:
                skipped[name] = f"{type(e).__name__}: {e}"
        if not chunks:
            return None, skipped
        cat = lambda key: np.concatenate([c[key] for c in chunks])
        seg, rrf, y, clean, groups = np.vstack([c["seg"] for c in chunks]), np.vstack([c["rrf"] for c in chunks]), cat("y"), cat("clean"), cat("groups")
        demo = demo_features(cat("age"), cat("sex"), bundle["age_fill"]) if bundle["demo_used"] else None
    else:
        parts = []
        for name, (kind, path, ann) in recs.items():
            try:
                rec = load_any(kind, path, ann, opts)
                if rec.get("kind") != "beats" or rec["scheme"] != bundle["schema"]:
                    raise ValueError(f"expected a '{bundle['schema']}' beat file")
                parts.append(rec)
            except Exception as e:
                skipped[name] = f"{type(e).__name__}: {e}"
        if not parts:
            return None, skipped
        seg, y = np.vstack([p["X"] for p in parts]), np.concatenate([p["y"] for p in parts])
        rrf, demo, clean, groups, rec_names = None, None, None, None, {}
    keep = np.isin(y, labels)
    unknown = int((~keep).sum())
    sel = np.where(keep)[0]
    sel = sel[_cap_idx(len(sel), opts["max_beats"])]
    seg, y = seg[sel], y[sel]
    rrf = rrf[sel] if rrf is not None else None
    demo = demo[sel] if demo is not None else None
    clean = clean[sel] if clean is not None else None
    groups = groups[sel] if groups is not None else None
    z, tab = build_inputs(seg, rrf, demo)
    P, S = model.predict(z, tab)
    pred_arg = np.array(labels)[P.argmax(1)]
    metrics = metrics_from_pred(y, pred_arg)
    conflict = rule_conflicts(pred_arg, rrf, seg) if rrf is not None else None
    oof = dict(proba=P, spread=S, y=y, labels=labels, use_spread=model.use_spread, clean=clean, conflict=conflict,
               rec=groups, age=None, sex=None)
    return dict(metrics=metrics, oof=oof, unknown=unknown, n=int(len(y)), rec_names=rec_names), skipped


def page_external():
    st.title("🌐 External Validation")
    st.caption("Freeze the model trained on your first dataset and score it on a completely different dataset (different hospital / device / database). "
               "No retraining happens here.")
    results = ml_or_warn()
    if not results:
        return
    title = st.selectbox("Frozen model (trained on)", list(results["bundles"].keys()))
    bundle, ml = results["bundles"][title], results["ml_all"][title]
    st.info(f"Model: **{MODEL_KINDS.get(bundle['kind'], bundle['kind'])}**, schema **{bundle['schema']}**, trained on *{results['dataset']}*. "
            "The external dataset must use the same schema (signal records WITH annotations, or the same kind of beat-matrix file). "
            "Classes are mapped through the same AAMI grouping. Different leads or sampling rates are a real source of performance loss - that is the point of this test.")
    ups = st.file_uploader("External dataset (.zip or loose files)", type=["zip", "csv", "txt", "hea", "dat", "atr", "mat"],
                           accept_multiple_files=True, key="ext_up")
    dm = None
    if bundle["demo_used"]:
        f = st.file_uploader("Metadata CSV for the external records (the model uses age/sex; missing values are imputed)", type=["csv"], key="ext_demo")
        if f is not None:
            try:
                dm = parse_demographics(f)
            except Exception as e:
                st.error(f"Could not read metadata: {e}")
    if ups and st.button("▶ Run external validation", type="primary"):
        wd = tempfile.mkdtemp(prefix="hridaya_ext_")
        try:
            ingest_uploads(ups, wd)
            recs = discover_records(wd)
            if not recs:
                st.error("No ECG records found in the external upload.")
            else:
                bar = st.progress(0.0, text="Scoring external dataset...")
                out, skipped = run_external(bundle, recs, st.session_state["opts"], dm, bar)
                bar.empty()
                st.session_state["ext_result"] = dict(out=out, skipped=skipped, name=ups[0].name if len(ups) == 1 else f"{ups[0].name} (+{len(ups) - 1})",
                                                      internal=ml, title=title)
                if out:
                    audit("external_validation", f"dataset={st.session_state['ext_result']['name']}; beats={out['n']}; macroF1={out['metrics']['macro_f1']:.3f}")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    er = st.session_state.get("ext_result")
    if not er or er["title"] != title:
        footer()
        return
    if er["skipped"]:
        with st.expander(f"⚠️ {len(er['skipped'])} external file(s) skipped"):
            st.json(er["skipped"])
    out = er["out"]
    if not out:
        st.error("Nothing could be scored.")
        footer()
        return
    m = out["metrics"]
    st.header(f"External result - {er['name']}")
    if out["unknown"]:
        st.caption(f"{out['unknown']:,} external beats had classes the model was never trained on and were left out.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("External macro-F1", f"{m['macro_f1'] * 100:.1f}%")
    c2.metric("External accuracy", f"{m['accuracy'] * 100:.1f}%")
    cr = crit_recall(m)
    c3.metric("Critical-class recall", "n/a" if not np.isfinite(cr) else f"{cr * 100:.1f}%")
    c4.metric("External beats", f"{out['n']:,}")
    ref = er["internal"]["patient"]["mean_f1"] if er["internal"]["patient"] else (er["internal"]["beat"]["macro_f1"] if er["internal"]["beat"] else np.nan)
    if np.isfinite(ref):
        gap = ref - m["macro_f1"]
        (st.error if gap > 0.05 else st.success)(
            f"Internal macro-F1 {ref:.1%} vs external {m['macro_f1']:.1%}: generalisation gap **{gap * 100:+.1f} points**. "
            + ("A gap above ~5 points signals sensitivity to site / device / population shift." if gap > 0.05 else "The gap is small on this external set."))
    fig = plot_confusion(m["cm"], m["labels"], "External")
    st.pyplot(fig)
    plt.close(fig)
    rep = pd.DataFrame(m["report"]).T
    st.dataframe(rep.loc[[i for i in rep.index if i in m["labels"] or i in ("macro avg", "weighted avg")]].round(3), width='stretch')
    o = st.session_state["opts"]
    labels = out["oof"]["labels"]
    pred, _, status = apply_decisions(out["oof"]["proba"], out["oof"]["spread"], labels, o["conf_thr"], o["max_spread"],
                                      out["oof"]["use_spread"], out["oof"]["clean"], out["oof"]["conflict"])
    d = decision_metrics(out["oof"]["y"], pred, status, labels)
    st.markdown(f"**Safety layer on the external set (confidence ≥ {o['conf_thr']:.2f}):** auto-decided {d['coverage']:.1%}, accuracy on decided "
                f"{d['acc_accepted']:.1%}.")
    st.dataframe(d["per_class"], width='stretch', hide_index=True)
    if out["oof"]["rec"] is not None:
        st.markdown("**Per external record**")
        st.dataframe(per_record_table(out["oof"], out["rec_names"], o["conf_thr"], o["max_spread"]).round(3), width='stretch', hide_index=True)
    footer()


def page_modelcard():
    st.title("📜 Model Card & Regulatory Evidence")
    st.caption("Traceability artefacts for design-history documentation. This page does not make HRIDAYA compliant with any standard - it records what exists and lists the gaps.")
    results = ml_or_warn()
    if not results:
        return
    card = build_model_card(results)
    st.markdown("### Model card")
    st.markdown(card_markdown(card))
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Model card (Markdown)", card_markdown(card).encode("utf-8"), file_name="HRIDAYA_model_card.md", mime="text/markdown")
    c2.download_button("⬇️ Evidence pack (JSON)", json.dumps(card, indent=2, default=str).encode("utf-8"), file_name="HRIDAYA_evidence_pack.json", mime="application/json")
    st.markdown("### Standards evidence checklist")
    st.dataframe(regulatory_checklist(), width='stretch', hide_index=True)
    st.markdown("### Audit log (this session)")
    log = st.session_state.get("audit_log", [])
    st.dataframe(pd.DataFrame(log) if log else pd.DataFrame(columns=["time", "event", "detail"]), width='stretch', hide_index=True)
    footer()


def page_doctors():
    st.title("🏥 Cardiologists (Bengaluru)")
    st.caption("Listing supplied by the author. Please verify availability, fees and credentials directly with the clinic before visiting.")
    cities = ["All"] + sorted({d["city"] for d in DOCTORS})
    city = st.radio("City", cities, horizontal=True)
    for d in DOCTORS:
        if city != "All" and d["city"] != city:
            continue
        q = quote_plus(f"{d['name'].split(' (')[0]} cardiologist {d['loc']}")
        note = f"<br><i>{html.escape(d['note'])}</i>" if d.get("note") else ""
        extra = "".join(f" &nbsp;|&nbsp; <a href='{html.escape(u)}' target='_blank'>{html.escape(t)}</a>" for t, u in d.get("links", []))
        st.markdown(
            f"<div class='doc-card'><h4 style='margin:0'>{html.escape(d['name'])}</h4>"
            f"<b>Experience:</b> {html.escape(d['exp'])}<br><b>Specialization:</b> {html.escape(d['spec'])}<br>"
            f"<b>Location:</b> {html.escape(d['loc'])}{note}<br>"
            f"<a href='https://www.google.com/maps/search/?api=1&query={q}' target='_blank'>📍 Open in Google Maps</a> &nbsp;|&nbsp; "
            f"<a href='https://www.google.com/search?q={q}' target='_blank'>🔎 Find contact details</a>{extra}</div>",
            unsafe_allow_html=True)
    st.warning("🚨 Chest pain, fainting, breathlessness or palpitations with dizziness are emergencies - call your local emergency number (112 in India) immediately.")
    footer()


def page_reviewer():
    st.title("🩺 Clinical Review")
    st.caption("The HRIDAYA project, its screening logic and its outputs were reviewed by the cardiologist below.")
    ph = reviewer_photo_b64()
    photo_lower = (reviewer_photo_path() or "").lower()
    ext = "webp" if photo_lower.endswith(".webp") else ("jpeg" if photo_lower.endswith((".jpg", ".jpeg")) else "png")
    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        if ph:
            st.markdown(f"<img class='reviewer-photo' src='data:image/{ext};base64,{ph}' alt='{html.escape(REVIEWER['name'])}'>",
                        unsafe_allow_html=True)
        else:
            st.info("Photo not found. Upload **profile.png.webp** beside hridaya.py, or place a PNG/JPG at **assets/dhairya.png**.")
    with c2:
        st.markdown(
            f"<div class='doc-card'><h3 style='margin:0;color:#8B1E23'>{html.escape(REVIEWER['name'])}</h3>"
            f"<b>{html.escape(REVIEWER['role'])}</b><br>"
            f"<b>Practice area:</b> {html.escape(REVIEWER['base'])}<br><br>"
            f"<a href='{html.escape(REVIEWER['profile'])}' target='_blank'>🩺 MyMedisage profile</a></div>",
            unsafe_allow_html=True)
        st.markdown("**Role in this project:** clinical review of the HRIDAYA ECG screening workflow and its results.")
        st.markdown(
            f"<div class='reviewer-badge'>✅ Reviewed by <b>{html.escape(REVIEWER['name'])}</b> - "
            f"project by {html.escape(OWNER['name'])}</div>", unsafe_allow_html=True)
    st.info("A clinical review of the work does not make the app a certified medical device. "
            "HRIDAYA remains a research prototype and its output must be confirmed by a qualified clinician.")
    footer()


def page_review():
    st.title("📝 Final Review & Sign-Off")
    results = results_or_warn()
    if not results:
        return
    per = results["per"]
    sel = st.selectbox("Record to review", sorted(per.keys()))
    res = per[sel]
    ml = results["ml"]
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Screening risk", res["risk"])
    if ml and ml["patient"]:
        c2.metric("Patient-wise macro-F1", f"{ml['patient']['mean_f1']:.1%}")
    elif ml and ml["beat"]:
        c2.metric("Beat-level macro-F1 (optimistic)", f"{ml['beat']['macro_f1']:.1%}")
    else:
        c2.metric("Classifier performance", "n/a")
    c3.metric("Conditions flagged", sum(c["status"] in FLAGGED for c in res["conditions"].values()))
    st.dataframe(record_conditions_df(res), width='stretch', hide_index=True)
    st.markdown("### Reviewer sign-off")
    a, b = st.columns(2)
    with a:
        st.text_input("Patient ID / label", sel)
        st.text_input("Reviewing clinician")
    with b:
        st.selectbox("Final assessment", ["Normal / routine follow-up", "Moderate - Holter / outpatient review", "High - urgent cardiology review"],
                     index={"Low": 0, "Moderate": 1, "High": 2}.get(res["risk"], 0))
        st.date_input("Review date")
    st.text_area("Clinician notes")
    st.checkbox("I have reviewed the automated findings and confirm them against the source ECG.")
    st.download_button("⬇️ Download report (HTML - print to PDF)", report_html(results, sel).encode("utf-8"),
                       file_name=f"HRIDAYA_report_{sel.replace('/', '_')}.html", mime="text/html")
    st.markdown("---")
    st.markdown("##### Copyright & author information")
    st.markdown(f"* **Platform:** HRIDAYA (हृदय) ECG Research Companion\n* **Copyright ©:** All Rights Reserved\n"
                f"* **Lead author:** {OWNER['name']}\n* **Phone:** {OWNER['phone']}\n* **Email:** {OWNER['email']}\n"
                f"* **Project clinically reviewed by:** {REVIEWER['name']}")
    footer()


# ======================================================================
# 12. MAIN
# ======================================================================
def main():
    icon = "💓"
    if LOGO:
        try:
            from PIL import Image

            icon = Image.open(LOGO)
        except Exception:
            icon = "💓"
    st.set_page_config(page_title="HRIDAYA - ECG Research Companion", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<div class='disclaimer'>⚠️ <b>Research prototype.</b> Not a medical device; it does not diagnose, treat, cure or prevent any condition. "
                "Results can be wrong - never use them for emergencies or treatment decisions. Do not upload identifiable patient data.</div>",
                unsafe_allow_html=True)
    if LOGO:
        st.sidebar.image(LOGO, width=230)
    st.sidebar.title("💓 HRIDAYA Controls")
    pages = {
        "🏠 Home & Overview": page_home,
        "📁 ECG Data Upload": page_upload,
        "🔍 Live Waveform & 13-Condition Screener": page_screener,
        "⚠️ Clinical Risk Summary": page_risk,
        "🛡️ AI Safety & Explainability": page_safety,
        "🌐 External Validation": page_external,
        "📜 Model Card & Regulatory Evidence": page_modelcard,
        "🏥 Cardiologists (Bengaluru)": page_doctors,
        "🩺 Clinical Review": page_reviewer,
        "📝 Review & Sign-Off Report": page_review,
    }
    page = st.sidebar.radio("Navigate Workspace:", list(pages.keys()))
    with st.sidebar.expander("Analysis options"):
        csv_fs = st.number_input("Sampling rate for CSV/TXT when it can't be detected (Hz)", 50, 2000, 360)
        force_fs = st.checkbox("Always use this sampling rate (ignore auto-detect)", False)
        csv_units = st.selectbox("CSV amplitude units (if the file doesn't state them)", ["mV", "µV", "arbitrary"])
        max_beats = st.slider("Max beats for classifier", 20_000, 150_000, 80_000, step=10_000)
    with st.sidebar.expander("Model & validation"):
        kinds = list(MODEL_KINDS.keys())
        model_kind = st.selectbox("Classifier", kinds, format_func=lambda k: MODEL_KINDS[k] + ("" if (k == "rf" or TORCH_OK) else " - torch missing"))
        cv_folds = st.slider("Patient-wise CV folds (GroupKFold)", 3, 10, 5)
        beat_level = st.checkbox("Also report beat-level 80/20 (optimistic)", True)
        aug_ratio = st.slider("Augment minority classes up to X% of the majority (0 = off)", 0, 100, 30, step=5)
        n_trees = st.slider("Random-Forest trees", 50, 400, 150, step=50)
        loss = st.selectbox("CNN loss", ["focal", "weighted_ce"], format_func=lambda s: "Focal loss (γ=2)" if s == "focal" else "Class-weighted cross-entropy")
        epochs = st.slider("CNN epochs", 3, 40, 12)
        pretrain = st.checkbox("Contrastive pre-training of the CNN encoder (SimCLR-style, unlabeled training beats)", False)
        pretrain_epochs = st.slider("Pre-training epochs", 1, 15, 3)
    with st.sidebar.expander("Safety layer"):
        conf_thr = st.slider("Auto-decide only if confidence ≥", 0.30, 0.99, 0.85, 0.01)
        mc_passes = st.slider("MC-Dropout passes (CNN)", 5, 50, 20)
        max_spread = st.slider("Max MC-Dropout spread (CNN)", 0.05, 0.50, 0.25, 0.01)
        min_sqi = st.slider("Minimum signal-quality index per 5-s window", 0.2, 0.9, 0.5, 0.05)
    st.session_state["opts"] = dict(csv_fs=csv_fs, csv_units=csv_units, max_beats=max_beats, force_fs=force_fs,
                                    model_kind=model_kind, cv_folds=cv_folds, beat_level=beat_level, aug_ratio=aug_ratio,
                                    n_trees=n_trees, loss=loss, epochs=epochs, pretrain=pretrain, pretrain_epochs=pretrain_epochs,
                                    conf_thr=conf_thr, mc_passes=mc_passes, max_spread=max_spread, min_sqi=min_sqi)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**© Copyright Reserved**  \n**Author:** {OWNER['name']}  \n**Phone:** {OWNER['phone']}  \n**Email:** {OWNER['email']}")
    st.sidebar.markdown(f"**Clinically reviewed by:**  \n{REVIEWER['name']}")

    pages[page]()


if __name__ == "__main__":
    main()
