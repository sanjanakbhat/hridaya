# -*- coding: utf-8 -*-
"""
HRIDAYA (हृदय) - Automated ECG Arrhythmia & Anomaly Diagnostic Hub
Run:  streamlit run app.py --server.maxUploadSize 1000

Folder layout (keep these next to app.py):
    app.py
    logo.png                 <- HRIDAYA emblem (sidebar, home page, favicon, reports)
    assets/dhairya.png       <- photo shown on the "Clinical Review" page

Design rules
 * Every number shown is computed from the uploaded data. Nothing is hard-coded,
   "calibrated" or random-generated.
 * Beat classifier is trained and tested ONLY on labels that come with the upload
   (WFDB .atr, PhysioNet annotation .txt, or the label column of beat-matrix CSVs).
 * The 13 conditions are screened with signal-processing rules on the uploaded ECG.
   Where the leads in the file cannot support a condition (e.g. LVH needs V1 + V5),
   the app says "Not assessable" instead of guessing.
 * Rule outputs are validated against the dataset's own annotations where those exist
   (AFib, 2nd-degree block, LBBB, RBBB) and the agreement table is shown.
 * Screening/research tool only - not a certified medical device.

Accepted uploads (a .zip, or the individual files selected together)
 1. WFDB records: .hea + .dat/.mat (+ optional .atr)
 2. CSV/TXT signal + PhysioNet annotation text, e.g. 100.csv + 100annotations.txt
    (paired automatically; a time column such as 0:00.003 gives the sampling rate)
 3. Pre-segmented beat CSVs (Kaggle mitbih_train.csv / mitbih_test.csv, last column 0-4):
    classifier accuracy + confusion matrix only (no timing data -> no rhythm/PR/QT)
 4. CSV/TXT signal without annotations: 13-condition screening runs, accuracy cannot be computed
 5. Unknown sampling rate: set it in the sidebar
"""
from __future__ import annotations

import base64
import datetime
import html
import os
import re
import shutil
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    import wfdb

    WFDB_OK = True
except ImportError:  # pragma: no cover
    WFDB_OK = False

# ======================================================================
# 0. CONSTANTS
# ======================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

OWNER = {
    "name": "Sanjana K Bhat",
    "phone": "9535309338",
    "email": "sanjanakbhatunitech@gmail.com",
}

REVIEWER = {
    "name": "Dr. Dhairya Thakker",
    "role": "Consultant Cardiologist",
    "base": "Mulund & Northeast Mumbai (visiting consultant)",
    "photo_candidates": ("assets/dhairya.png", "assets/dhairya.jpg", "dhairya.png", "dhairya.jpg"),
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

SIG_TITLE = "Annotated ECG records (waveform + RR features)"
SCHEME_TITLE = {
    "mitbih": "Pre-segmented beat files (labels 0-4 = N, S, V, F, Q)",
    "ptbdb": "Pre-segmented beat files (PTB-DB: 0 = Normal, 1 = Abnormal)",
}
AAMI_LABEL_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
STD_FS = (100, 125, 128, 200, 250, 256, 360, 500, 512, 1000, 2000)
IGNORE_PREFIX = ("readme", "license", "records", "sha256", "annotators", "index", "changelog")

# Dr. Dhairya Thakker now lives on his own page (page_reviewer) as the doctor who reviewed this work.
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


def analyze_record(rec, name, max_beats=300):
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
               ectopic_burden=np.nan, conditions={}, truth=None, risk="Unknown")
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

    # ---- eligible beats for morphology measurements (non-ectopic, plausible RR)
    elig = [i for i in range(1, len(samples) - 1)
            if ne[i] and ne[i - 1] and 0.3 < (samples[i] - samples[i - 1]) / fs < 2.0]
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

    # ---- 1. AFib: RR irregularity in 30-beat windows (ectopic-contaminated windows skipped)
    flags = []
    win = 30
    for s in range(0, len(rr) - win + 1, win // 2):
        w = rr[s:s + win]
        if ne[s:s + win + 1].mean() < 0.9 or w.min() < 0.25 or w.max() > 2.5:
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
# 4. BEAT CLASSIFIER
# ======================================================================
def _stats_block(seg, z):
    g = np.gradient(z, axis=1)
    h = z.shape[1] // 2
    return np.column_stack([seg.mean(1), seg.std(1), seg.max(1), seg.min(1), np.ptp(seg, axis=1),
                            g.mean(1), g.std(1), g.max(1), g.min(1),
                            z[:, :h].std(1), z[:, h:].std(1), np.abs(z[:, h:]).mean(1)])


def beat_feature_matrix(x, fs, samples):
    """Features for beats cut out of a continuous record (waveform + local RR context)."""
    pre, post = int(round(0.25 * fs)), int(round(0.27 * fs))
    L = pre + post
    ok = (samples - pre >= 0) & (samples + post <= len(x))
    s_ok = samples[ok]
    idx = s_ok[:, None] + np.arange(-pre, post)[None, :]
    seg = x[idx]
    pos = np.linspace(0, L - 1, 188)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, L - 1)
    fr = pos - lo
    seg = seg[:, lo] * (1 - fr) + seg[:, hi] * fr
    mean, std = seg.mean(1, keepdims=True), seg.std(1, keepdims=True)
    z = (seg - mean) / np.where(std > 0, std, 1.0)
    rr = np.diff(samples) / fs
    loc = pd.Series(rr).rolling(11, center=True, min_periods=1).median().values
    loc_b = np.r_[loc, loc[-1]]
    prev = np.r_[loc_b[0], rr]
    nxt = np.r_[rr, loc_b[-1]]
    rrf = np.column_stack([prev, nxt, prev / loc_b, nxt / loc_b, loc_b])[ok]
    return np.hstack([z, _stats_block(seg, z), rrf]), ok


def matrix_feature_matrix(M):
    """Features for pre-segmented beat rows (waveform only - no timing context in these files)."""
    M = np.nan_to_num(np.asarray(M, dtype=float))
    mean, std = M.mean(1, keepdims=True), M.std(1, keepdims=True)
    z = (M - mean) / np.where(std > 0, std, 1.0)
    return np.hstack([z, _stats_block(M, z)])


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


def train_and_evaluate(X, y, groups, n_trees=150, patient_wise=True, presplit=None):
    """presplit=(X_test, y_test) uses the files' own train/test split; otherwise a stratified 80/20 split."""
    notes = []
    if presplit is None:
        counts = pd.Series(y).value_counts()
        rare = counts[counts < 10].index.tolist()
        if rare:
            keep = ~np.isin(y, rare)
            X, y, groups = X[keep], y[keep], groups[keep]
            notes.append(f"Classes with fewer than 10 beats were excluded from evaluation: {', '.join(rare)}")
        if len(set(y)) < 2:
            return None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        split_desc = "Beat-level stratified 80/20 split"
    else:
        X_tr, y_tr = X, y
        X_te, y_te = presplit
        counts = pd.Series(y_tr).value_counts()
        rare = counts[counts < 10].index.tolist()
        if rare:
            keep = ~np.isin(y_tr, rare)
            X_tr, y_tr = X_tr[keep], y_tr[keep]
            notes.append(f"Classes with fewer than 10 training beats were excluded: {', '.join(rare)}")
        m = np.isin(y_te, np.unique(y_tr))
        X_te, y_te = X_te[m], y_te[m]
        if len(set(y_tr)) < 2 or len(y_te) == 0:
            return None
        split_desc = "Official train/test files (trained on the train file, scored on the unseen test file)"
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=25, min_samples_split=4,
                                 class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    out = metrics_from_pred(y_te, clf.predict(X_te))
    out.update(n_train=int(len(y_tr)), class_counts={k: int(v) for k, v in pd.Series(np.r_[y_tr, y_te]).value_counts().items()},
               notes=notes, patient=None, split_desc=split_desc)
    if presplit is None and patient_wise and groups is not None and len(set(groups)) >= 4:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        if len(set(y[tr])) >= 2 and len(te) > 0:
            clf2 = RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=4,
                                          class_weight="balanced_subsample", random_state=42, n_jobs=-1)
            clf2.fit(X[tr], y[tr])
            pm = metrics_from_pred(y[te], clf2.predict(X[te]))
            pm["n_test_records"] = int(len(set(groups[te])))
            out["patient"] = pm
    return out


# ======================================================================
# 5. DATASET PIPELINE
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


def _norm_stem(stem, ann=False):
    s = stem.lower()
    if ann:
        for tok in ("annotations", "annotation", "annot", "atr", "ann", "labels", "label"):
            s = s.replace(tok, "")
    for ch in ("_", "-", " ", "."):
        s = s.replace(ch, "")
    return s


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


def run_pipeline(recs, opts, dataset_name):
    per, skipped, beat_files = {}, {}, {}
    X_all, y_all, g_all = [], [], []
    mats = {}
    bar = st.progress(0.0, text="Analysing files...")
    for k, (name, (kind, path, ann)) in enumerate(recs.items()):
        bar.progress((k + 1) / max(1, len(recs)), text=f"Analysing {name} ({k + 1}/{len(recs)})")
        try:
            rec = load_any(kind, path, ann, opts)
            if rec.get("kind") == "beats":
                s = mats.setdefault(rec["scheme"], dict(train=[], test=[], all=[]))
                s[rec["role"]].append((rec["X"], rec["y"]))
                beat_files[name] = {"File": name, "Type": SCHEME_TITLE[rec["scheme"]], "Role": rec["role"],
                                    "Beats": len(rec["y"]), "Class counts": str(dict(Counter(rec["y"])))}
                continue
            if rec["signal"].shape[0] < rec["fs"] * 5:
                raise ValueError("record shorter than 5 s")
            res = analyze_record(rec, name)
            per[name] = res
            if rec["ann_samples"] is not None:
                main, _, _ = pick_leads(rec["sig_names"])
                mask = np.array([s in ANN_TO_AAMI for s in rec["ann_symbols"]])
                smp, sym = rec["ann_samples"][mask], rec["ann_symbols"][mask]
                if len(smp) > 20:
                    feats, ok = beat_feature_matrix(rec["signal"][:, main], rec["fs"], smp)
                    X_all.append(feats)
                    y_all.extend([ANN_TO_AAMI[s] for s in sym[ok]])
                    g_all.extend([k] * int(ok.sum()))
        except Exception as e:  # keep going - one bad file must not kill the run
            skipped[name] = f"{type(e).__name__}: {e}"
    bar.empty()

    ml_all = {}
    if X_all:
        X, y, g = np.vstack(X_all), np.array(y_all), np.array(g_all)
        sel = _cap_idx(len(y), opts["max_beats"])
        X, y, g = X[sel], y[sel], g[sel]
        if len(y) >= 200:
            with st.spinner(f"Training and testing on {len(y):,} annotated beats from the signal records..."):
                res = train_and_evaluate(X, y, g, patient_wise=opts["patient_wise"])
            if res:
                ml_all[SIG_TITLE] = res
    for scheme, parts in mats.items():
        title = SCHEME_TITLE[scheme]
        with st.spinner(f"Training and testing on beat files: {title}..."):
            if parts["train"] and parts["test"]:
                Xtr = np.vstack([p[0] for p in parts["train"]])
                ytr = np.concatenate([p[1] for p in parts["train"]])
                Xte = np.vstack([p[0] for p in parts["test"]])
                yte = np.concatenate([p[1] for p in parts["test"]])
                i1, i2 = _cap_idx(len(ytr), opts["max_beats"]), _cap_idx(len(yte), opts["max_beats"])
                res = train_and_evaluate(matrix_feature_matrix(Xtr[i1]), ytr[i1], None,
                                         presplit=(matrix_feature_matrix(Xte[i2]), yte[i2]), patient_wise=False)
            else:
                allp = parts["all"] + parts["train"] + parts["test"]
                Xm = np.vstack([p[0] for p in allp])
                ym = np.concatenate([p[1] for p in allp])
                i1 = _cap_idx(len(ym), opts["max_beats"])
                res = train_and_evaluate(matrix_feature_matrix(Xm[i1]), ym[i1], np.zeros(len(i1), int), patient_wise=False)
        if res:
            ml_all[title] = res

    ml = next(iter(ml_all.values())) if ml_all else None
    ml_msg = None
    if not ml_all:
        if per:
            ml_msg = ("No reference annotations (.atr, annotation .txt) or labelled beat files were found, so classification accuracy "
                      "cannot be computed. The 13-condition rule screening below still ran on the raw signals.")
        else:
            ml_msg = "Not enough labelled beats / classes to train and test a classifier."
    return dict(per=per, skipped=skipped, ml=ml, ml_all=ml_all, ml_msg=ml_msg, beat_files=beat_files, dataset=dataset_name,
                recs={n: recs[n] for n in per}, time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


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
# 6. PLOTS
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


# ======================================================================
# 7. UI HELPERS
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


def show_ml_one(ml):
    if ml["patient"]:
        p = ml["patient"]
        st.markdown("**Primary result - patient-wise split (test records never seen in training)**")
        q1, q2, q3 = st.columns(3)
        q1.metric("Accuracy (unseen records)", f"{p['accuracy'] * 100:.2f}%")
        q2.metric("Macro F1 (unseen records)", f"{p['macro_f1'] * 100:.2f}%")
        q3.metric("Unseen test records", p["n_test_records"])
        st.caption("This is the figure that reflects generalisation. The beat-level figure below can be optimistic because beats from the same recording appear in both train and test.")
        st.markdown("**Beat-level split (secondary)**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hold-out accuracy", f"{ml['accuracy'] * 100:.2f}%")
    c2.metric("Macro F1", f"{ml['macro_f1'] * 100:.2f}%")
    c3.metric("Macro sensitivity", f"{ml['macro_sens'] * 100:.2f}%")
    c4.metric("Macro specificity", f"{ml['macro_spec'] * 100:.2f}%")
    c5.metric("Test beats", f"{ml['n_test']:,}")
    acc = ml["accuracy"]
    if acc > 0.98:
        st.success(f"✅ Measured accuracy {acc:.2%} exceeds the 98% target.")
    elif acc > 0.95:
        st.info(f"✅ Measured accuracy {acc:.2%} exceeds 95% but is below the 98% stretch target.")
    else:
        st.error(f"⚠️ Measured accuracy {acc:.2%} is below the 95% target on this dataset. "
                 "The figure is reported as measured - see notes below for likely causes.")
    for n in ml["notes"]:
        st.caption(n)
    st.caption(f"{ml['split_desc']} ({ml['n_train']:,} train / {ml['n_test']:,} test). "
               f"Class mix: {ml['class_counts']}. Accuracy is dominated by the majority class - check macro F1 and per-class recall too.")
    st.markdown("**🧮 Confusion matrix (hold-out test beats)**")
    fig = plot_confusion(ml["cm"], ml["labels"], "Beat classifier")
    st.pyplot(fig)
    plt.close(fig)
    rep = pd.DataFrame(ml["report"]).T
    rep = rep.loc[[i for i in rep.index if i in ml["labels"] or i in ("macro avg", "weighted avg")]]
    st.dataframe(rep.round(3), width='stretch')
    if "N" in ml["labels"]:
        st.caption("N = normal/BBB/escape, S = supraventricular ectopic, V = ventricular ectopic, F = fusion, Q = paced/unclassifiable (AAMI EC57 grouping).")


def show_ml_all(results):
    st.markdown("### 🎯 Classification accuracy (computed from the uploaded labels)")
    ml_all = results.get("ml_all") or {}
    if not ml_all:
        st.warning(results.get("ml_msg") or "Classifier could not be evaluated.")
        return
    for title, ml in ml_all.items():
        if len(ml_all) > 1 or title != SIG_TITLE:
            st.markdown(f"#### {title}")
        show_ml_one(ml)


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
    acc = (f"Hold-out accuracy {ml['accuracy']:.2%}, macro F1 {ml['macro_f1']:.2%} "
           f"({ml['n_test']:,} test beats)") if ml else "Not computed (no reference labels)"
    lg = logo_b64()
    logo_tag = f"<img src='data:image/png;base64,{lg}' style='height:110px'><br>" if lg else ""
    return f"""<html><head><meta charset="utf-8"><title>HRIDAYA report</title>
<style>body{{font-family:Arial;margin:30px;color:#3A1F1F}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #D4AF37;padding:6px;font-size:13px}}th{{background:#F3E7C9}}h1{{color:#8B1E23}}</style></head><body>
{logo_tag}<h1>HRIDAYA - ECG Screening Report</h1>
<p><b>Dataset:</b> {html.escape(results['dataset'])} &nbsp; <b>Record:</b> {html.escape(sel)} &nbsp; <b>Generated:</b> {results['time']}</p>
<p><b>Overall screening risk:</b> {res['risk']} &nbsp; <b>Median HR:</b> {res['hr_med']:.0f} bpm &nbsp;
<b>QRS:</b> {res['qrs_ms']:.0f} ms &nbsp; <b>PR:</b> {res['pr_ms']:.0f} ms &nbsp; <b>QTc:</b> {res['qtc_ms']:.0f} ms</p>
<p><b>Beat-classifier performance (dataset level):</b> {html.escape(acc)}</p>
<table><tr><th>#</th><th>Condition</th><th>Result</th><th>Risk tier</th><th>Evidence</th></tr>{rows}</table>
<p style="margin-top:20px"><i>Automated screening output; not a diagnosis. Requires review and sign-off by a qualified clinician.</i></p>
<p>Reviewer: ______________________ &nbsp; Signature: ______________ &nbsp; Date: ____________</p>
<hr><p>Project clinically reviewed by {html.escape(REVIEWER['name'])}, {html.escape(REVIEWER['role'])}.</p>
<p>© {datetime.date.today().year} {OWNER['name']} - All Rights Reserved | {OWNER['phone']} | {OWNER['email']}</p>
</body></html>"""


# ======================================================================
# 8. PAGES
# ======================================================================
def page_home():
    if LOGO:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.image(LOGO, width=360)
    st.title("💓 HRIDAYA (हृदय): ECG Research Companion")
    st.markdown("##### Multi-class heartbeat classification & 13-condition clinical screener")
    st.info("Accuracy and every other figure in this app are computed from the data you upload - nothing is pre-filled. "
            "Go to **📁 ECG Data Upload** to begin.")
    st.markdown("### The 13 conditions screened")
    st.dataframe(pd.DataFrame([{"#": c["id"], "Condition": c["name"], "Description": c["desc"],
                                "Risk tier": TIER_ICON[c["tier"]], "Needs": c["needs"]} for c in CONDITIONS]),
                 width='stretch', hide_index=True)
    st.markdown("### Supported uploads (any ECG database)")
    st.markdown(
        "Upload a **.zip** or select the **individual files together**. Nested folders are searched recursively.\n"
        "- **WFDB records** (`.hea` + `.dat`/`.mat`, optional `.atr`): MIT-BIH, INCART, European ST-T, PTB, PhysioNet challenge sets, etc.\n"
        "- **CSV/TXT signal + annotation text:** a file like `100.csv` next to `100annotations.txt` (PhysioNet text format) is paired automatically. "
        "The text file becomes the reference labels, so accuracy and validation work. If the CSV has a time column like `0:00.003`, the sampling rate is worked out automatically.\n"
        "- **Pre-segmented beat CSVs** such as Kaggle `mitbih_train.csv` / `mitbih_test.csv` (one beat per row, last column 0-4 = N, S, V, F, Q): "
        "you get classifier accuracy and a confusion matrix only - no rhythm, PR or QT screening, because these files contain no timing data. "
        "If both a *train* and a *test* file are uploaded, the model is trained on train and scored on the unseen test file.\n"
        "- **CSV/TXT with no annotations:** the 13-condition screening runs, but accuracy can't be computed and the app says so.\n"
        "- **Unknown sampling rate:** set it in the sidebar (*Analysis options*).")
    st.markdown("### How results are produced")
    st.markdown(
        "- **Accuracy:** a Random-Forest beat classifier is trained/tested on the labels that come with your upload. "
        "Without labels, accuracy can't be computed and the app says so.\n"
        "- **13 conditions:** signal-processing rules (R-peak detection, RR irregularity, PR/QRS/ST/QT delineation). "
        "Conditions that the available leads can't support are marked *Not assessable* rather than guessed.\n"
        "- **Validation:** rule outputs are compared with the dataset's rhythm/beat annotations (AFib, 2nd-degree block, LBBB, RBBB).")
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
    opts = st.session_state["opts"]

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
        found = st.session_state["found"]
        if not found:
            st.error("No ECG data found. Need WFDB files (`.hea`/`.dat`) or `.csv`/`.txt` signals / beat files in the upload.")
        else:
            wf = sum(1 for v in found.values() if v[0] == "wfdb")
            paired = sum(1 for v in found.values() if v[0] == "csv" and v[2])
            st.success(f"Found {len(found)} record file(s): {wf} WFDB, {len(found) - wf} CSV/TXT "
                       f"({paired} paired with an annotation text file).")
            if st.button("▶ Run full analysis", type="primary"):
                st.session_state["results"] = run_pipeline(found, opts, st.session_state["upload_name"])

    results = st.session_state.get("results")
    if not results:
        st.caption("No analysis yet - results (accuracy, confusion matrix, waveform, risk) appear here after you upload data and run the analysis.")
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
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High-risk records", n_high)
    c2.metric("🟡 Moderate-risk records", n_mod)
    c3.metric("🟢 Low-risk records", n_low)
    st.caption("High = any Critical-tier condition Detected/Suspected. Moderate = Moderate-tier flag or ectopic burden >10%. Otherwise Low.")
    sel = st.selectbox("Record", sorted(per.keys()))
    r = per[sel]
    st.subheader(f"Record {sel}: {r['risk']} risk")
    flagged = [(i, c) for i, c in r["conditions"].items() if c["status"] in FLAGGED]
    if flagged:
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
    ext = "jpeg" if (reviewer_photo_path() or "").lower().endswith((".jpg", ".jpeg")) else "png"
    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        if ph:
            st.markdown(f"<img class='reviewer-photo' src='data:image/{ext};base64,{ph}' alt='{html.escape(REVIEWER['name'])}'>",
                        unsafe_allow_html=True)
        else:
            st.info("Photo not found. Place it at **assets/dhairya.png** next to app.py.")
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
    c2.metric("Beat-classifier accuracy", f"{ml['accuracy']:.2%}" if ml else "n/a")
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
# 9. MAIN
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
        patient_wise = st.checkbox("Also run patient-wise validation", True)
    st.session_state["opts"] = dict(csv_fs=csv_fs, csv_units=csv_units, max_beats=max_beats,
                                    patient_wise=patient_wise, force_fs=force_fs)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**© Copyright Reserved**  \n**Author:** {OWNER['name']}  \n**Phone:** {OWNER['phone']}  \n**Email:** {OWNER['email']}")
    st.sidebar.markdown(f"**Clinically reviewed by:**  \n{REVIEWER['name']}")

    pages[page]()


if __name__ == "__main__":
    main()
