# -*- coding: utf-8 -*-
"""
HRIDAYA (हृदय) - ECG Research Companion  |  Website edition
Run:  streamlit run app.py --server.maxUploadSize 1000

What changed in this version
 * Theme matches the HRIDAYA mobile app: maroon app bar, cream background, gold accents,
   tile home screen, chat-bubble assistant screen.
 * New "HRIDAYA Assistant" page: ECG concepts, explanation of the model's ACTUAL results,
   research/dataset help, in-app guidance, and voice (mic input + spoken replies).
 * The assistant is rule-based (built-in knowledge base). It never invents findings: every
   number it quotes is read from the analysis results stored in this session.
 * Analysis engine (signal processing, 13-condition rules, beat classifier) is unchanged.

Screening/research tool only - not a certified medical device.
"""
from __future__ import annotations

import base64
import datetime
import html
import json
import os
import re
import shutil
import tempfile
import zipfile
from urllib.parse import quote_plus

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap
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
OWNER = {
    "name": "Sanjana K Bhat",
    "phone": "9535309338",
    "email": "sanjanakbhatunitech@gmail.com",
}

# Brand palette (from the HRIDAYA mobile app mockup)
MAROON = "#8B1E23"
GOLD = "#D4AF37"
CREAM = "#FAF7F0"
BEIGE = "#E8DCC6"
BROWN = "#3A1F1F"
CMAP = LinearSegmentedColormap.from_list("hridaya", [CREAM, "#D9B8A8", MAROON])

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
AAMI_NAMES = {"N": "normal/bundle-branch/escape", "S": "supraventricular ectopic",
              "V": "ventricular ectopic", "F": "fusion", "Q": "paced/unclassifiable"}

DOCTORS = [
    dict(name="Dr. Nagamani A. C.", exp="23 Years", spec="MBBS, DM - Cardiology",
         loc="Apollo Clinic, Koramangala, Bengaluru"),
    dict(name="Dr. Anusha A. Rao", exp="8 Years", spec="MBBS, MD, DNB (Cardiology)",
         loc="Manipal Hospitals Old Airport Road / Apollo Clinic, Indiranagar, Bengaluru"),
    dict(name="Dr. Shilpa Jayaprakash", exp="12 Years", spec="MBBS, MD General Medicine, DM Cardiology",
         loc="Apollo Clinic, JP Nagar, Bengaluru"),
    dict(name="Dr. K. Nikhitha Reddy (Dr. K. Nikhitha)", exp="12 Years", spec="MBBS, MD General Medicine, DM Cardiology",
         loc="Apollo Clinic, HSR Layout, Bengaluru"),
    dict(name="Dr. Disha R. Shetty", exp="8 Years", spec="MBBS, MD, DM Cardiology",
         loc="Apollo One, Electronic City, Bengaluru"),
]


# ======================================================================
# 1. SIGNAL LOADING
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
    out = dict(signal=sig, fs=float(rec.fs), sig_names=list(rec.sig_name), calibrated=cal,
               ann_samples=None, ann_symbols=None, rhythm=[])
    try:
        ann = wfdb.rdann(base, "atr")
        out["ann_samples"] = np.asarray(ann.sample, dtype=int)
        out["ann_symbols"] = np.asarray(ann.symbol)
        notes = list(ann.aux_note) if ann.aux_note is not None else [""] * len(ann.symbol)
        marks = [(int(s), str(n).replace("\x00", "").strip().lstrip("(")) for s, sym, n in
                 zip(ann.sample, ann.symbol, notes) if sym == "+"]
        for k, (s, lab) in enumerate(marks):
            end = marks[k + 1][0] if k + 1 < len(marks) else sig.shape[0]
            out["rhythm"].append((s, end, lab))
    except Exception:
        pass
    return out


def load_csv_record(path, fs, units):
    df = pd.read_csv(path)
    if all(_is_float(c) for c in df.columns):
        df = pd.read_csv(path, header=None)
    drop = [c for c in df.columns if any(k in str(c).lower() for k in ("time", "sample", "index", "unnamed"))]
    num = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("no numeric ECG columns")
    if num.shape[1] >= 150 and num.shape[0] > 50:
        raise ValueError("looks like a pre-segmented beat matrix (no rhythm/PR/QT analysis possible)")
    num = num.iloc[:, :12]
    sig = np.column_stack([_fill_nans(num.iloc[:, i].values) for i in range(num.shape[1])])
    cal = [units != "arbitrary"] * sig.shape[1]
    if units == "µV":
        sig = sig / 1000.0
    return dict(signal=sig, fs=float(fs), sig_names=[str(c) for c in num.columns], calibrated=cal,
                ann_samples=None, ann_symbols=None, rhythm=[])


def _is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


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

    res = dict(name=name, fs=fs, duration_s=sig.shape[0] / fs, leads=", ".join(map(str, names)),
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
# 4. BEAT CLASSIFIER (AAMI N/S/V/F/Q)  - trained on uploaded annotations
# ======================================================================
def beat_feature_matrix(x, fs, samples):
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
    g = np.gradient(z, axis=1)
    h = z.shape[1] // 2
    rr = np.diff(samples) / fs
    loc = pd.Series(rr).rolling(11, center=True, min_periods=1).median().values
    loc_b = np.r_[loc, loc[-1]]
    prev = np.r_[loc_b[0], rr]
    nxt = np.r_[rr, loc_b[-1]]
    rrf = np.column_stack([prev, nxt, prev / loc_b, nxt / loc_b, loc_b])[ok]
    stats = np.column_stack([mean[:, 0], std[:, 0], seg.max(1), seg.min(1), np.ptp(seg, axis=1),
                             g.mean(1), g.std(1), g.max(1), g.min(1),
                             z[:, :h].std(1), z[:, h:].std(1), np.abs(z[:, h:]).mean(1)])
    return np.hstack([z, stats, rrf]), ok


def metrics_from_pred(y_true, y_pred):
    labels = [c for c in AAMI_ORDER if c in set(y_true) | set(y_pred)]
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


def train_and_evaluate(X, y, groups, n_trees=150, patient_wise=True):
    notes = []
    counts = pd.Series(y).value_counts()
    rare = counts[counts < 10].index.tolist()
    if rare:
        keep = ~np.isin(y, rare)
        X, y, groups = X[keep], y[keep], groups[keep]
        notes.append(f"Classes with fewer than 10 beats were excluded from evaluation: {', '.join(rare)}")
    if len(set(y)) < 2:
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=25, min_samples_split=4,
                                 class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    out = metrics_from_pred(y_te, clf.predict(X_te))
    out.update(n_train=int(len(y_tr)), class_counts={k: int(v) for k, v in pd.Series(y).value_counts().items()},
               notes=notes, patient=None)
    if patient_wise and len(set(groups)) >= 4:
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


def discover_records(root):
    recs = {}
    for d, _, files in os.walk(root):
        if "__MACOSX" in d:
            continue
        for f in sorted(files):
            low = f.lower()
            if low.endswith(".hea"):
                kind, base = "wfdb", os.path.join(d, f[:-4])
            elif low.endswith((".csv", ".txt")):
                kind, base = "csv", os.path.join(d, f)
            else:
                continue
            key = os.path.splitext(f)[0]
            if key in recs:
                key = f"{os.path.basename(d)}/{key}"
            recs[key] = (kind, base)
    return recs


def load_any(kind, path, opts):
    if kind == "wfdb":
        if not WFDB_OK:
            raise RuntimeError("wfdb is not installed")
        return load_wfdb_record(path)
    return load_csv_record(path, opts["csv_fs"], opts["csv_units"])


def run_pipeline(recs, opts, dataset_name):
    per, skipped = {}, {}
    X_all, y_all, g_all = [], [], []
    bar = st.progress(0.0, text="Analysing records...")
    for k, (name, (kind, path)) in enumerate(recs.items()):
        bar.progress((k + 1) / max(1, len(recs)), text=f"Analysing record {name} ({k + 1}/{len(recs)})")
        try:
            rec = load_any(kind, path, opts)
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
        except Exception as e:  # keep going - one bad record must not kill the run
            skipped[name] = f"{type(e).__name__}: {e}"
    bar.empty()

    ml, ml_msg = None, None
    if X_all:
        X = np.vstack(X_all)
        y = np.array(y_all)
        g = np.array(g_all)
        if len(y) > opts["max_beats"]:
            rng = np.random.RandomState(42)
            sel = np.sort(rng.choice(len(y), opts["max_beats"], replace=False))
            X, y, g = X[sel], y[sel], g[sel]
        if len(y) >= 200:
            with st.spinner(f"Training and testing beat classifier on {len(y):,} annotated beats..."):
                ml = train_and_evaluate(X, y, g, patient_wise=opts["patient_wise"])
        if ml is None:
            ml_msg = "Not enough annotated beats / classes to train and test a classifier."
    else:
        ml_msg = ("No reference annotations (.atr) were found, so classification accuracy cannot be computed. "
                  "The 13-condition rule screening below still ran on the raw signals.")
    return dict(per=per, skipped=skipped, ml=ml, ml_msg=ml_msg, dataset=dataset_name,
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
    fig.patch.set_facecolor(CREAM)
    sns.heatmap(cm, annot=True, fmt="d", cmap=CMAP, xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title(f"{title} - counts", color=BROWN)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    sns.heatmap(norm, annot=True, fmt=".2f", cmap=CMAP, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title(f"{title} - row-normalised (sensitivity)", color=BROWN)
    for a in axes:
        a.set_xlabel("Predicted class", color=BROWN)
        a.set_ylabel("True class (reference annotation)", color=BROWN)
    fig.tight_layout()
    return fig


def plot_waveform(rec, start_s, dur_s, samples, symbols):
    fs = rec["fs"]
    a, b = int(start_s * fs), int((start_s + dur_s) * fs)
    n_lead = min(2, rec["signal"].shape[1])
    fig, axes = plt.subplots(n_lead, 1, figsize=(11, 2.8 * n_lead), sharex=True, squeeze=False)
    fig.patch.set_facecolor(CREAM)
    t = np.arange(a, min(b, rec["signal"].shape[0])) / fs
    in_win = (samples >= a) & (samples < b)
    for i in range(n_lead):
        ax = axes[i, 0]
        ax.set_facecolor("#FFFFFF")
        y = rec["signal"][a:b, i]
        ax.plot(t[:len(y)], y, color=MAROON, lw=1.1)
        pk = samples[in_win]
        ax.scatter(pk / fs, rec["signal"][pk, i], s=18, color=GOLD, edgecolor=BROWN, linewidth=0.4, zorder=3)
        if symbols is not None:
            for s_, sym in zip(pk, symbols[in_win]):
                ax.text(s_ / fs, rec["signal"][s_, i], sym, fontsize=8, color=BROWN, ha="center", va="bottom", weight="bold")
        ax.set_ylabel(f"{rec['sig_names'][i]}", color=BROWN)
        ax.grid(True, ls="--", alpha=0.35, color=BEIGE)
    axes[-1, 0].set_xlabel("Time (s)", color=BROWN)
    axes[0, 0].set_title("ECG with beat markers" + (" (reference annotation symbols)" if symbols is not None else " (detected R peaks)"), color=BROWN)
    fig.tight_layout()
    return fig


# ======================================================================
# 7. THEME + APP-BAR (matches the HRIDAYA mobile app)
# ======================================================================
CSS = """
<style>
.stApp { background:#FAF7F0; color:#3A1F1F; }
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp p,.stApp li,.stApp label,.stApp span { color:#3A1F1F; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.2rem; max-width:1150px; }

/* sidebar */
[data-testid="stSidebar"] { background:#E8DCC6; }
[data-testid="stSidebar"] * { color:#3A1F1F; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:6px 10px; border-radius:10px; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#8B1E23; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color:#FAF7F0; }

/* app bar */
.appbar { background:#8B1E23; border-radius:16px; padding:12px 18px; display:flex; align-items:center; gap:12px; margin-bottom:18px; box-shadow:0 2px 10px rgba(58,31,31,.18); }
.appbar .brand { color:#FAF7F0; font-size:1.35rem; font-weight:700; letter-spacing:.04em; }
.appbar .where { color:#E8DCC6; font-size:.95rem; margin-left:auto; }
.appbar svg { flex:none; }

/* hero */
.hero h1 { font-size:2.1rem; line-height:1.15; margin:0 0 6px 0; color:#3A1F1F; }
.hero p { font-size:1.05rem; color:#5b3a3a; max-width:46ch; }
.tagline { color:#B08A1E; font-weight:600; }

/* buttons */
.stButton>button,.stDownloadButton>button { background:#8B1E23; color:#FAF7F0; border:0; border-radius:12px; padding:.55rem 1rem; }
.stButton>button:hover,.stDownloadButton>button:hover { background:#6f171b; color:#FAF7F0; }
.stButton>button p,.stDownloadButton>button p { color:#FAF7F0; }
.stButton>button:focus-visible { outline:3px solid #D4AF37; outline-offset:2px; }
.stButton>button[kind="primary"] { width:100%; min-height:56px; font-size:1.05rem; font-weight:600; border-radius:14px; }

/* home tiles */
div[class*="st-key-tile_"] .stButton>button { min-height:112px; width:100%; background:#FFFFFF; color:#8B1E23; border:1px solid #E8DCC6; border-radius:20px; box-shadow:0 2px 8px rgba(58,31,31,.08); font-weight:600; }
div[class*="st-key-tile_"] .stButton>button p { color:#8B1E23; font-size:1.05rem; }
div[class*="st-key-tile_"] .stButton>button:hover { border-color:#D4AF37; background:#FFF9EA; }

/* chip buttons (assistant suggestions) */
div[class*="st-key-chip_"] .stButton>button { background:#FFFFFF; color:#8B1E23; border:1px solid #D4AF37; border-radius:999px; padding:.25rem .8rem; font-size:.85rem; }
div[class*="st-key-chip_"] .stButton>button p { color:#8B1E23; font-size:.85rem; }
div[class*="st-key-chip_"] .stButton>button:hover { background:#FFF3CF; }

/* metrics, uploader, cards */
div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E8DCC6; border-radius:14px; padding:12px; }
div[data-testid="stMetric"] * { color:#3A1F1F !important; }
[data-testid="stFileUploader"] section { background:#FFFFFF; border:2px dashed #D4AF37; border-radius:14px; }
.doc-card { background:#FFFFFF; border:1px solid #E8DCC6; border-left:5px solid #8B1E23; border-radius:12px; padding:14px 18px; margin-bottom:12px; color:#3A1F1F; }
.doc-card a { color:#8B1E23; }
[data-testid="stExpander"] { background:#FFFFFF; border:1px solid #E8DCC6; border-radius:12px; }

/* chat bubbles */
[data-testid="stChatMessage"] { background:#F3ECDD; border:1px solid #E8DCC6; border-radius:16px; padding:12px 16px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:#8B1E23; border-color:#8B1E23; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color:#FAF7F0 !important; }
[data-testid="stChatInput"] { border:1px solid #8B1E23; border-radius:14px; }

.footer { text-align:center; font-size:0.85rem; color:#3A1F1F; border-top:1px solid #E8DCC6; padding-top:10px; margin-top:30px; }
</style>
"""


def logo_svg(size=40, heart="#FAF7F0", pulse="#D4AF37"):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-label="HRIDAYA logo">'
            f'<path d="M32 56 C10 40 6 24 16 15 C23 9 30 13 32 19 C34 13 41 9 48 15 C58 24 54 40 32 56Z" '
            f'fill="none" stroke="{heart}" stroke-width="4" stroke-linejoin="round"/>'
            f'<polyline points="6,34 21,34 26,23 33,45 38,29 42,34 58,34" fill="none" stroke="{pulse}" '
            f'stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def logo_img_html(width=170):
    """Use hridaya_logo.png (next to app.py) on the home screen if it exists."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hridaya_logo.png")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{width}" alt="HRIDAYA logo">'
    return logo_svg(width // 2, heart=MAROON, pulse=GOLD)


def appbar(page_label):
    st.markdown(
        f"<div class='appbar'>{logo_svg(36)}<span class='brand'>HRIDAYA</span>"
        f"<span class='where'>{html.escape(page_label)}</span></div>", unsafe_allow_html=True)


def footer():
    st.markdown(
        f"<div class='footer'>© {datetime.date.today().year} {OWNER['name']} - All Rights Reserved &nbsp;|&nbsp; "
        f"📞 {OWNER['phone']} &nbsp;|&nbsp; ✉️ {OWNER['email']}<br>"
        "Screening/research tool - not a certified medical device. Always confirm with a qualified cardiologist.</div>",
        unsafe_allow_html=True)


# ======================================================================
# 8. HRIDAYA ASSISTANT  (knowledge base + grounded result explanations + voice)
# ======================================================================
GREETING = (
    "Hi! I'm **HRIDAYA Assistant**, your ECG guide. I can help with:\n\n"
    "- ECG concepts and terminology (P wave, QRS, ST, QT, sampling, filtering)\n"
    "- Explaining your model results from this session\n"
    "- Research and dataset help (MIT-BIH, splits, sensitivity, specificity, F1, quantization, 1D-CNN)\n"
    "- Using the app: uploads, running analysis, reading charts, fixing errors\n"
    "- Voice: tap the mic to speak, and turn on spoken replies in Settings\n\n"
    "How can I assist you today?")

SUGGESTIONS = ["What does the P wave represent?", "Explain my results", "What is MIT-BIH?",
               "How do I upload a file?", "Sensitivity vs specificity", "Why is a condition 'Not assessable'?"]

KB = [
    dict(id="greet_scope", title="What the assistant can and can't do",
         kw=["what can you do", "capabilities", "who are you", "what are you"],
         text="I'm a rule-based assistant built into this app. I answer from a built-in ECG and machine-learning knowledge base, and "
              "when you ask about results I read the numbers straight from the analysis stored in this session. "
              "I don't give diagnoses or treatment advice, and I won't make up findings."),
    dict(id="pwave", title="P wave",
         kw=["p wave", "p waves", "p-wave", "p-waves", "atrial depolarization", "atrial depolarisation"],
         text="The **P wave** is the electrical activation (depolarization) of the atria, the upper chambers. It shows that the sinoatrial "
              "node, the heart's natural pacemaker, has fired.\n\n**Key points**\n- Normal duration: about 0.08-0.12 s\n"
              "- Normal amplitude: up to about 0.25 mV\n- Best seen in Lead II\n- Absent or replaced by chaotic activity in atrial fibrillation"),
    dict(id="pr", title="PR interval and AV blocks",
         kw=["pr interval", "pr segment", "first-degree", "first degree", "av block", "heart block", "mobitz", "wenckebach", "complete heart block", "third-degree", "third degree"],
         text="The **PR interval** runs from the start of the P wave to the start of the QRS. It reflects conduction delay through the AV node. Normal is about 120-200 ms.\n\n"
              "- **First-degree block**: PR above 200 ms, every beat still conducts\n- **Mobitz I (Wenckebach)**: PR lengthens beat by beat until one beat is dropped\n"
              "- **Mobitz II**: beats drop suddenly with a constant PR\n- **Complete block**: atria and ventricles beat independently\n\n"
              "In this app PR is measured with threshold-based delineation, so treat it as approximate."),
    dict(id="qrs", title="QRS complex and bundle branch block",
         kw=["qrs", "qrs complex", "qrs width", "ventricular depolarization", "bundle branch", "lbbb", "rbbb", "wide qrs"],
         text="The **QRS complex** is ventricular depolarization, the main spike on the ECG. Normal width is under about 120 ms (often 80-100 ms).\n\n"
              "A wide QRS suggests slow conduction through the ventricles, for example **bundle branch block** (RBBB has an 'M' shape in V1, LBBB is broad and monophasic) "
              "or a ventricular ectopic beat. This app measures QRS width from the signal slope and uses V1 for side classification when V1 exists; "
              "on a single lead the side is low confidence."),
    dict(id="st", title="ST segment, STEMI and ischemia",
         kw=["st segment", "st elevation", "st depression", "stemi", "ischemia", "ischaemia", "j point", "j-point", "myocardial infarction"],
         text="The **ST segment** is the flat stretch between the end of the QRS and the start of the T wave. Elevation or depression of about 0.1 mV or more can point to ischemia or infarction.\n\n"
              "This app measures ST at J-point + 60 ms against the PR-segment baseline, needs a signal calibrated in mV, skips records with wide QRS (unreliable), "
              "and only 'Suspects' because a single lead can't confirm; at least two contiguous leads are needed clinically."),
    dict(id="qt", title="T wave and QT / QTc",
         kw=["qt", "qtc", "qt interval", "t wave", "t-wave", "bazett", "long qt", "torsades"],
         text="The **T wave** is ventricular repolarization. The **QT interval** runs from QRS start to T-wave end. Because QT shortens at faster heart rates it is corrected: "
              "**QTc (Bazett) = QT / sqrt(RR in seconds)**.\n\nQTc above about 450 ms is flagged as prolonged here. Bazett over-corrects when HR is above 100, and the T-wave end is hard to locate, "
              "so the app reports 'Inconclusive' when it can't find it reliably."),
    dict(id="hr", title="Heart rate, RR interval and R-peaks",
         kw=["heart rate", "rr interval", "rr intervals", "bpm", "bradycardia", "tachycardia", "hrv", "r-r", "r peak", "r peaks", "r-peak", "r-peaks", "pan-tompkins", "pan tompkins"],
         text="**Heart rate = 60 / RR interval (seconds)**, where RR is the time between two R-peaks. Below 60 bpm is bradycardia; above 100 bpm is tachycardia.\n\n"
              "R-peaks are found with a Pan-Tompkins-style detector: band-pass 5-15 Hz, derivative, squaring, moving-window integration, then a refractory period of 250 ms. "
              "If your file has reference annotations (.atr), those beat positions are used instead."),
    dict(id="afib", title="Atrial fibrillation",
         kw=["afib", "atrial fibrillation", "irregularly irregular", "fibrillation"],
         text="**Atrial fibrillation** is an irregularly irregular rhythm with no organised P waves, which raises stroke risk. The app looks at RR irregularity in 30-beat windows "
              "(coefficient of variation and RMSSD/RR both above 0.15), skips ectopic-heavy windows, and checks whether P waves are found. Where the dataset has rhythm annotations, "
              "the result is compared against them in the validation table."),
    dict(id="fs", title="Sampling frequency",
         kw=["sampling", "sampling frequency", "sampling rate", "sample rate", "fs", "hz", "nyquist", "resampling", "360 hz"],
         text="**Sampling frequency (fs)** is how many samples per second are recorded. MIT-BIH uses 360 Hz. By Nyquist, fs must be at least twice the highest frequency you care about; "
              "QRS energy sits roughly below 40 Hz, so 250-500 Hz is typical for diagnostic work.\n\nIf you upload CSV/TXT files, set the correct sampling rate and units in **Settings**: a wrong fs "
              "makes every interval (heart rate, PR, QT) wrong. Models trained at one fs need resampling if the input uses another."),
    dict(id="filter", title="Filtering and noise",
         kw=["filter", "filtering", "baseline wander", "noise", "powerline", "power line", "band-pass", "bandpass", "denoise", "artifact", "artefact", "butterworth", "emg"],
         text="ECG is contaminated by **baseline wander** (below about 0.5 Hz, breathing and movement), **powerline** interference (50 Hz in India), and **muscle noise** (above about 40 Hz).\n\n"
              "This app uses zero-phase Butterworth band-pass filters (filtfilt, so waves are not shifted in time): 5-15 Hz for R-peak detection and 0.5-40 Hz for interval measurements. "
              "Heavy high-pass filtering can distort the ST segment, which is one reason ST findings are only 'Suspected'."),
    dict(id="leads", title="Leads and LVH",
         kw=["lead", "leads", "mlii", "v1", "v5", "12-lead", "12 lead", "sokolow", "lvh", "hypertrophy"],
         text="A **lead** is one view of the heart's electrical activity. MIT-BIH records have two leads (usually MLII plus V1 or V5). Lead II/MLII is best for rhythm and P waves; V1 is best for bundle branch blocks.\n\n"
              "**LVH** uses the Sokolow-Lyon rule (S in V1 + R in V5/V6 of at least 3.5 mV), so it needs both leads in mV. If the file lacks them, the app says 'Not assessable' instead of guessing."),
    dict(id="aami", title="AAMI beat classes (N, S, V, F, Q)",
         kw=["aami", "ectopic", "pvc", "pac", "supraventricular", "fusion", "paced", "beat class", "beat classes", "annotation", "annotations", "atr", "symbols", "ec57"],
         text="The app groups beat annotations into the five **AAMI EC57** classes:\n\n- **N**: normal, bundle-branch beats, escape beats\n- **S**: supraventricular ectopic (atrial premature etc.)\n"
              "- **V**: ventricular ectopic (PVC)\n- **F**: fusion of ventricular and normal\n- **Q**: paced or unclassifiable\n\nThe `.atr` file holds the cardiologist-verified beat labels the classifier learns from."),
    dict(id="mitbih", title="MIT-BIH Arrhythmia Database",
         kw=["mit-bih", "mitbih", "mit bih", "physionet", "dataset", "datasets", "48 records", "wfdb", ".hea", ".dat", "database"],
         text="The **MIT-BIH Arrhythmia Database** (PhysioNet) has 48 half-hour two-channel ambulatory ECG recordings from 47 subjects, sampled at 360 Hz, with beat-by-beat annotations by cardiologists.\n\n"
              "A WFDB record is a set of files: `.hea` (header), `.dat` (signal) and `.atr` (annotations). Common cautions: classes are highly imbalanced (normal beats dominate), and the AAMI protocol usually excludes the four paced records (102, 104, 107, 217)."),
    dict(id="prep", title="Preprocessing pipeline",
         kw=["preprocess", "preprocessing", "pre-processing", "segmentation", "normalise", "normalize", "normalization", "beat segment", "z-score", "feature", "features", "window"],
         text="Typical pipeline: **filter, locate R-peaks, cut a window around each beat, resample to a fixed length, normalise**.\n\n"
              "In this app each beat is cut from 0.25 s before to 0.27 s after the R-peak, resampled to 188 samples, z-score normalised, and combined with summary statistics and RR-interval features "
              "(previous/next RR and their ratio to the local median) before classification."),
    dict(id="split", title="Train/test splits and leakage",
         kw=["train", "test", "split", "splits", "patient-wise", "inter-patient", "leakage", "validation", "overfit", "overfitting", "generalisation", "generalization", "cross-validation", "stratified", "intra-patient"],
         text="A **beat-level split** puts beats from the same patient in both train and test, so the model can memorise a patient's quirks and the score looks too good (data leakage). "
              "A **patient-wise (inter-patient) split** keeps whole records out of training, which measures real generalisation and is normally lower.\n\n"
              "The app reports both: a stratified 80/20 beat-level hold-out and, when enabled in Settings, a patient-wise check on unseen records. Quote the patient-wise number when claiming generalisation."),
    dict(id="metrics", title="Accuracy, sensitivity, specificity, F1",
         kw=["sensitivity", "specificity", "f1", "f1 score", "f1-score", "precision", "recall", "accuracy", "metric", "metrics", "macro", "imbalance", "imbalanced"],
         text="- **Accuracy** = correct / total. Misleading when one class dominates.\n- **Sensitivity (recall)** = TP / (TP + FN): how many true cases you catch.\n"
              "- **Specificity** = TN / (TN + FP): how many non-cases you correctly clear.\n- **Precision** = TP / (TP + FP).\n- **F1** = harmonic mean of precision and recall.\n"
              "- **Macro** averages treat each class equally.\n\nFor arrhythmias, check per-class recall and macro F1: a model can reach 90%+ accuracy by predicting 'normal' almost every time."),
    dict(id="cm", title="Confusion matrix",
         kw=["confusion matrix", "confusion", "heatmap", "row-normalised", "row normalised", "row-normalized"],
         text="In the **confusion matrix**, rows are the true class (from annotations) and columns are the predicted class. The diagonal is correct predictions. "
              "The left panel shows counts; the right panel divides each row by its total, so the diagonal equals per-class **sensitivity**. Off-diagonal cells show which classes get confused, "
              "for example S beats predicted as N."),
    dict(id="quant", title="Quantization",
         kw=["quantization", "quantisation", "quantized", "int8", "tflite", "on-device", "edge deployment", "pruning", "compress", "compression"],
         text="**Quantization** stores weights (and often activations) as 8-bit integers instead of 32-bit floats. Model size drops about 4x and inference gets faster on phones, which suits an Android app.\n\n"
              "Do post-training quantization with a representative calibration set, then **re-measure sensitivity per class**: rare classes (S, F) can degrade more than overall accuracy suggests. "
              "Note: this website's classifier is a Random Forest, which is not quantized."),
    dict(id="cnn", title="1D-CNN and the models used",
         kw=["cnn", "1d-cnn", "1d cnn", "convolution", "convolutional", "deep learning", "neural", "mlp", "random forest", "architecture", "which model", "what model"],
         text="A **1D-CNN** slides learned filters along the time axis of a beat, so it finds waveform shapes (a wide QRS, a missing P) automatically. A typical design is several conv + pooling blocks, then dense layers and a softmax over the beat classes.\n\n"
              "This website's classifier is a **Random Forest** on hand-built beat and RR features. If your paper or mobile app uses an MLP or a 1D-CNN, its numbers are separate: don't mix them with the figures shown here."),
    dict(id="howto", title="Uploading and running an analysis",
         kw=["upload", "zip", "how do i", "how to", "run analysis", "supported", "file format", "csv", "txt", "folder", "getting started", "get started"],
         text="**To analyse an ECG dataset**\n\n1. Open **Upload & Analysis** (or tap *Upload ECG File* on Home).\n2. Choose a `.zip` containing WFDB records (`.hea` + `.dat`/`.mat`, optional `.atr`) or CSV/TXT signals, one column per lead. Subfolders are fine.\n"
              "3. For CSV/TXT, set sampling rate and units in **Settings** first.\n4. Click **Run full analysis**.\n5. Read the accuracy cards, confusion matrix, 13-condition table and risk table, then open **Waveform & Screener** for one record.\n"
              "6. Use **Review & Sign-Off** to download an HTML report you can print to PDF.\n\nPre-segmented 187-column beat files aren't supported: they hold no rhythm, PR or QT information."),
    dict(id="charts", title="Reading the charts",
         kw=["chart", "charts", "waveform", "plot", "graph", "slider", "marker", "markers", "interpret", "read the"],
         text="The **waveform** shows up to two leads over the window you pick. Gold dots mark beats; if the file has annotations, each dot carries its symbol (N, V, A, L, R ...), otherwise the dots are detected R-peaks. "
              "Use the sliders to change window length and start time. The **bar chart** counts beats per AAMI class, and the **risk table** lists HR, QRS, PR, QTc and flagged conditions per record."),
    dict(id="trouble", title="Troubleshooting",
         kw=["error", "troubleshoot", "not working", "problem", "fail", "failed", "skipped", "not assessable", "inconclusive", "wfdb is missing", "no records", "no ecg records", "memory", "slow", "maxuploadsize", "too large", "200mb", "stuck"],
         text="**Common issues**\n\n- **'wfdb is missing'**: run `pip install wfdb` and restart the app.\n- **'No ECG records found'**: the ZIP needs `.hea` + `.dat` pairs (or numeric CSV/TXT files).\n"
              "- **Upload too large**: start with `streamlit run app.py --server.maxUploadSize 1000`.\n- **Records skipped**: open the 'skipped' panel; typical reasons are a missing `.dat`, a record under 5 s, or a beat-matrix CSV.\n"
              "- **'Not assessable'**: the file lacks what the rule needs (for example V1 and V5 for LVH, or mV calibration for ST). This is deliberate.\n"
              "- **'Inconclusive'**: the P or T wave couldn't be located reliably, or AFib/wide QRS made the rule unreliable.\n- **Slow run**: lower 'Max beats for classifier' in Settings.\n"
              "- **CSV intervals look wrong**: check sampling rate and units in Settings."),
    dict(id="voice", title="Voice input and spoken replies",
         kw=["voice", "microphone", "mic", "speech", "speak", "text to speech", "text-to-speech", "tts", "listen", "talk"],
         text="Tap the **mic** on the Assistant page and speak; your words are sent as a message. Turn on **Speak assistant replies** in Settings to hear answers.\n\n"
              "**Compatibility notes**\n- Works best in Chrome and Edge (desktop and Android). Firefox has no speech recognition.\n- The page must be on HTTPS or localhost, and you must allow microphone access.\n"
              "- Chrome's speech recognition sends audio to an online service, so it needs internet.\n- Spoken replies use your device's voices and work offline on most systems.\n"
              "- In an Android WebView, speech recognition is often unavailable and needs the RECORD_AUDIO permission wired in the app. Test it on real devices."),
    dict(id="safety", title="Medical safety",
         kw=["diagnose", "diagnosis", "safe", "medical", "doctor", "trust", "reliable", "emergency", "treatment", "medicine", "should i worry", "cardiologist"],
         text="HRIDAYA is a **screening and research tool, not a certified medical device**. Its flags are not diagnoses and need review by a qualified clinician. "
              "If you have chest pain, fainting, breathlessness, or palpitations with dizziness, treat it as an emergency and call 112 in India. The **Cardiologists** page lists clinics in Bengaluru."),
]

RES_TRIGGERS = ["my result", "my results", "my accuracy", "my model", "my analysis", "my record", "my dataset", "model result",
                "model results", "the results", "these results", "explain results", "explain the results", "explain my",
                "what did the model", "how did the model", "predicted class", "prediction", "predicted", "confidence",
                "this record", "latest analysis", "analysis result", "analysis results", "flagged", "my risk"]

CHIP_ICON = "💬"


def _f(v, unit="", nd=0):
    try:
        return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}{unit}"
    except TypeError:
        return "n/a"


def explain_record(res, name):
    lines = [f"**Record {name}**: screening risk **{res['risk']}**.",
             f"Median HR {_f(res['hr_med'], ' bpm')}, QRS about {_f(res['qrs_ms'], ' ms')}, PR about {_f(res['pr_ms'], ' ms')}, "
             f"QTc about {_f(res['qtc_ms'], ' ms')} (approximate threshold-based measurements; n/a means it could not be measured)."]
    flagged = [(i, c) for i, c in res["conditions"].items() if c["status"] in FLAGGED]
    if flagged:
        lines.append("\n**Flagged conditions**")
        for i, c in flagged:
            lines.append(f"- **{COND_BY_ID[i]['name']}**: {c['status']}. {c['evidence']}")
    else:
        lines.append("\nNo condition was flagged for this record.")
    n_inc = sum(c["status"] == "Inconclusive" for c in res["conditions"].values())
    n_na = sum(c["status"] == "Not assessable" for c in res["conditions"].values())
    lines.append(f"\n{n_inc} condition(s) were Inconclusive and {n_na} Not assessable with the leads in this file.")
    if res["class_counts"]:
        cc = ", ".join(f"{k}={v:,}" for k, v in res["class_counts"].items())
        lines.append(f"Beat classes from the reference annotations: {cc}. Ectopic/non-normal burden: {_f(res['ectopic_burden'] * 100, '%', 1)}.")
    lines.append("\n*Limitations: Detected/Suspected are screening flags, not diagnoses, and the delineation is approximate.*")
    return "\n".join(lines)


def explain_dataset(results):
    per, ml = results["per"], results["ml"]
    lines = [f"**Dataset:** {results['dataset']}, analysed {results['time']}. {len(per)} record(s) analysed"
             + (f", {len(results['skipped'])} skipped." if results["skipped"] else ".")]
    if ml:
        lines.append(f"\n**Beat classifier (Random Forest, AAMI N/S/V/F/Q)**\n- Hold-out accuracy **{ml['accuracy']:.2%}**, macro F1 **{ml['macro_f1']:.2%}**, "
                     f"macro sensitivity {ml['macro_sens']:.2%}, macro specificity {ml['macro_spec']:.2%}, on {ml['n_test']:,} test beats ({ml['n_train']:,} for training).")
        rep = ml["report"]
        per_cls, weakest = [], None
        for lab in ml["labels"]:
            r = rep.get(lab)
            if r and r.get("support", 0) > 0:
                per_cls.append(f"{lab} {r['recall']:.0%} (n={int(r['support']):,})")
                if weakest is None or r["recall"] < weakest[1]:
                    weakest = (lab, r["recall"], int(r["support"]))
        if per_cls:
            lines.append("- Per-class recall: " + ", ".join(per_cls) + ".")
        if weakest and weakest[1] < 0.9:
            lines.append(f"- Weakest class: **{weakest[0]}** ({AAMI_NAMES.get(weakest[0], '')}) with recall {weakest[1]:.0%} on {weakest[2]:,} test beats. "
                         "Accuracy is dominated by normal beats, so this matters more than the headline number.")
        if ml["patient"]:
            p = ml["patient"]
            lines.append(f"- Patient-wise check (unseen records): accuracy {p['accuracy']:.2%}, macro F1 {p['macro_f1']:.2%} on {p['n_test_records']} record(s). "
                         f"Gap versus beat-level accuracy: {(ml['accuracy'] - p['accuracy']) * 100:+.2f} points; this is the better estimate of generalisation.")
        for n in ml["notes"]:
            lines.append(f"- Note: {n}")
        lines.append("\n**Predicted class and confidence:** the app reports aggregate hold-out metrics. It does not keep per-beat predictions or probabilities, "
                     "so there is no confidence score to explain.")
    else:
        lines.append(f"\n**Classifier:** {results['ml_msg']}")
    n_high = sum(r["risk"] == "High" for r in per.values())
    n_mod = sum(r["risk"] == "Moderate" for r in per.values())
    n_low = sum(r["risk"] == "Low" for r in per.values())
    lines.append(f"\n**Screening risk:** {n_high} high, {n_mod} moderate, {n_low} low.")
    counts = {}
    for r in per.values():
        for i, c in r["conditions"].items():
            if c["status"] in FLAGGED:
                counts[COND_BY_ID[i]["name"]] = counts.get(COND_BY_ID[i]["name"], 0) + 1
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        lines.append("Most-flagged conditions: " + "; ".join(f"{n} ({c} record(s))" for n, c in top) + ".")
    lines.append("\nAsk about a specific record by name (for example *explain record 100*) for its measurements and evidence.")
    lines.append("\n*Limitations: one dataset, a single split seed, and rule-based screening. Treat the numbers as a research measurement, not clinical validation.*")
    return "\n".join(lines)


def assistant_reply(q):
    t = re.sub(r"\s+", " ", q.lower().strip())
    if re.fullmatch(r"(hi|hello|hey|hii|namaste|help|menu)[!. ?]*", t):
        return GREETING
    results = st.session_state.get("results")
    per = results["per"] if results and results.get("per") else {}
    rec_name = next((n for n in per if re.search(rf"(?<![\w.]){re.escape(n.lower())}(?![\w.])", t)), None)
    results_intent = any(tr in t for tr in RES_TRIGGERS) or (rec_name is not None and "record" in t)
    if results_intent:
        if not per:
            return ("I can't explain results yet because nothing has been analysed in this session. Go to **Upload & Analysis**, upload a ZIP, "
                    "and click **Run full analysis**. Then ask me again and I'll walk through the actual numbers.")
        if rec_name:
            return explain_record(per[rec_name], rec_name)
        return explain_dataset(results)
    best, best_score = None, 0
    for e in KB:
        score = 0
        for kw in e["kw"]:
            if re.search(rf"(?<![\w]){re.escape(kw)}(?![\w])", t):
                score += 1 + len(kw.split())
        if score > best_score:
            best, best_score = e, score
    if best:
        return best["text"]
    topics = ", ".join(e["title"].lower() for e in KB[1:9])
    return ("I don't have a good answer for that yet. I can help with ECG terms (" + topics + "), dataset and model questions, "
            "how to use this app, and explaining your results. Try rephrasing, or tap one of the suggestions.")


def clean_for_speech(text):
    t = re.sub(r"`([^`]*)`", r"\1", text)
    t = re.sub(r"[*_#>]+", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:900]


def speak(text):
    payload = json.dumps(clean_for_speech(text)).replace("</", "<\\/")
    lang = json.dumps(st.session_state.get("voice_lang", "en-IN"))
    components.html(f"""<script>
(function() {{
  try {{
    const W = window.parent, s = W.speechSynthesis;
    if (!s) return;
    s.cancel();
    const parts = {payload}.match(/[^.!?]+[.!?]*/g) || [];
    parts.forEach(p => {{ const u = new W.SpeechSynthesisUtterance(p.trim()); u.lang = {lang}; s.speak(u); }});
  }} catch (e) {{}}
}})();
</script>""", height=0)


def voice_widget():
    lang = json.dumps(st.session_state.get("voice_lang", "en-IN"))
    components.html(f"""
<style>
 body {{ margin:0; font-family:system-ui,sans-serif; color:#3A1F1F; }}
 .row {{ display:flex; align-items:center; gap:10px; }}
 button {{ border:0; cursor:pointer; }}
 #mic {{ width:46px; height:46px; border-radius:50%; background:#8B1E23; color:#FAF7F0; font-size:20px; }}
 #mic.on {{ box-shadow:0 0 0 6px rgba(212,175,55,.45); }}
 #stop {{ background:#E8DCC6; color:#3A1F1F; border-radius:999px; padding:6px 12px; font-size:13px; }}
 #st {{ font-size:13px; }}
</style>
<div class="row">
  <button id="mic" title="Speak to HRIDAYA" aria-label="Speak to HRIDAYA">🎤</button>
  <button id="stop" title="Stop speaking">🔇 Stop voice</button>
  <span id="st">Tap the mic and speak</span>
</div>
<script>
(function() {{
  const mic = document.getElementById('mic'), stopB = document.getElementById('stop'), st = document.getElementById('st');
  let W;
  try {{ W = window.parent; W.document; }} catch (e) {{ st.textContent = 'Voice is blocked in this browser view. Type instead.'; mic.disabled = true; return; }}
  const SR = W.SpeechRecognition || W.webkitSpeechRecognition;
  if (!SR) {{ st.textContent = 'Speech input is not supported here (try Chrome or Edge). You can still type.'; mic.disabled = true; }}
  let rec = null, on = false;
  mic.onclick = () => {{
    if (on && rec) {{ rec.stop(); return; }}
    try {{ W.speechSynthesis && W.speechSynthesis.cancel(); }} catch (e) {{}}
    rec = new SR(); rec.lang = {lang}; rec.interimResults = false; rec.maxAlternatives = 1;
    rec.onstart = () => {{ on = true; mic.classList.add('on'); st.textContent = 'Listening...'; }};
    rec.onerror = (e) => {{
      let hint = '';
      if (e.error === 'not-allowed') hint = ' - allow microphone access for this site';
      else if (e.error === 'network') hint = ' - the speech service needs internet';
      else if (e.error === 'no-speech') hint = ' - nothing heard, try again';
      st.textContent = 'Mic error: ' + e.error + hint;
    }};
    rec.onend = () => {{ on = false; mic.classList.remove('on'); }};
    rec.onresult = (e) => {{ const text = e.results[0][0].transcript; st.textContent = 'Heard: ' + text; send(text); }};
    try {{ rec.start(); }} catch (e) {{ st.textContent = 'Could not start the microphone.'; }}
  }};
  function send(text) {{
    const P = W.document;
    const ta = P.querySelector('textarea[data-testid="stChatInputTextArea"]') || P.querySelector('[data-testid="stChatInput"] textarea');
    if (!ta) {{ st.textContent += ' (chat box not found; type it instead)'; return; }}
    const setter = Object.getOwnPropertyDescriptor(W.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, text);
    ta.dispatchEvent(new W.Event('input', {{ bubbles: true }}));
    setTimeout(() => {{ const b = P.querySelector('[data-testid="stChatInputSubmitButton"]'); if (b) b.click(); }}, 250);
  }}
  stopB.onclick = () => {{ try {{ W.speechSynthesis.cancel(); }} catch (e) {{}} }};
}})();
</script>""", height=60)


def queue_question(q):
    st.session_state["pending_q"] = q


# ======================================================================
# 9. SHARED UI HELPERS
# ======================================================================
def go(page):
    st.session_state["nav"] = page


def results_or_warn():
    r = st.session_state.get("results")
    if not r or not r["per"]:
        st.warning("⚠️ Upload a dataset ZIP on **Upload & Analysis** and click **Run full analysis** first.")
        footer()
        return None
    return r


def show_ml(ml, ml_msg):
    st.markdown("### 🎯 Classification accuracy (computed from the uploaded readings)")
    if ml is None:
        st.warning(ml_msg or "Classifier could not be evaluated.")
        return
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
    st.caption(f"Beat-level stratified 80/20 split ({ml['n_train']:,} train / {ml['n_test']:,} test). "
               f"Class mix: {ml['class_counts']}. Accuracy is dominated by normal beats - check macro F1 and per-class recall too.")
    if ml["patient"]:
        p = ml["patient"]
        st.markdown("**Stricter check - patient-wise split** (test records never seen in training)")
        d1, d2, d3 = st.columns(3)
        d1.metric("Accuracy (unseen records)", f"{p['accuracy'] * 100:.2f}%")
        d2.metric("Macro F1 (unseen records)", f"{p['macro_f1'] * 100:.2f}%")
        d3.metric("Unseen test records", p["n_test_records"])
        st.caption("Inter-patient accuracy is normally lower than beat-level accuracy; this is the number that reflects generalisation.")
    st.markdown("### 🧮 Confusion matrix (hold-out test beats)")
    fig = plot_confusion(ml["cm"], ml["labels"], "Beat classifier")
    st.pyplot(fig)
    plt.close(fig)
    rep = pd.DataFrame(ml["report"]).T
    rep = rep.loc[[i for i in rep.index if i in ml["labels"] or i in ("macro avg", "weighted avg")]]
    st.dataframe(rep.round(3), use_container_width=True)
    st.caption("N = normal/BBB/escape, S = supraventricular ectopic, V = ventricular ectopic, F = fusion, Q = paced/unclassifiable (AAMI EC57 grouping).")
    st.caption("Want this explained? Ask the **HRIDAYA Assistant**: 'Explain my results'.")


def show_waveform_block(results, key_prefix):
    names = sorted(results["per"].keys())
    sel = st.selectbox("Record", names, key=f"{key_prefix}_rec")
    kind, path = results["recs"][sel]
    try:
        rec = load_any(kind, path, st.session_state["opts"])
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
           f"({ml['n_test']:,} test beats)") if ml else "Not computed (no reference annotations)"
    return f"""<html><head><meta charset="utf-8"><title>HRIDAYA report</title>
<style>body{{font-family:Arial;margin:30px;color:#3A1F1F;background:#FAF7F0}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #E8DCC6;padding:6px;font-size:13px}}th{{background:#8B1E23;color:#FAF7F0}}h1{{color:#8B1E23}}</style></head><body>
<h1>HRIDAYA - ECG Screening Report</h1>
<p><b>Dataset:</b> {html.escape(results['dataset'])} &nbsp; <b>Record:</b> {html.escape(sel)} &nbsp; <b>Generated:</b> {results['time']}</p>
<p><b>Overall screening risk:</b> {res['risk']} &nbsp; <b>Median HR:</b> {res['hr_med']:.0f} bpm &nbsp;
<b>QRS:</b> {res['qrs_ms']:.0f} ms &nbsp; <b>PR:</b> {res['pr_ms']:.0f} ms &nbsp; <b>QTc:</b> {res['qtc_ms']:.0f} ms</p>
<p><b>Beat-classifier performance (dataset level):</b> {html.escape(acc)}</p>
<table><tr><th>#</th><th>Condition</th><th>Result</th><th>Risk tier</th><th>Evidence</th></tr>{rows}</table>
<p style="margin-top:20px"><i>Automated screening output; not a diagnosis. Requires review and sign-off by a qualified clinician.</i></p>
<p>Reviewer: ______________________ &nbsp; Signature: ______________ &nbsp; Date: ____________</p>
<hr><p>© {datetime.date.today().year} {OWNER['name']} - All Rights Reserved | {OWNER['phone']} | {OWNER['email']}</p>
</body></html>"""


# ======================================================================
# 10. PAGES
# ======================================================================
P_HOME = "🏠 Home"
P_UPLOAD = "📁 Upload & Analysis"
P_SCREEN = "🔍 Waveform & Screener"
P_RISK = "⚠️ Risk Summary"
P_CHAT = "💬 HRIDAYA Assistant"
P_RES = "📚 Research & Resources"
P_DOCS = "🏥 Cardiologists"
P_REVIEW = "📝 Review & Sign-Off"
P_SET = "⚙️ Settings"
PAGES = [P_HOME, P_UPLOAD, P_SCREEN, P_RISK, P_CHAT, P_RES, P_DOCS, P_REVIEW, P_SET]

SETTING_DEFAULTS = dict(csv_fs=360, csv_units="mV", max_beats=80_000, patient_wise=True,
                        speak_replies=False, voice_lang="en-IN")


def page_home():
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown(f"<div style='padding:6px 0'>{logo_img_html(190)}</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='hero'><h1>Welcome,<br>to HRIDAYA</h1>"
                    "<p>Upload your ECG, explore the analysis and get insights with our AI assistant.</p>"
                    "<p class='tagline'>Understand signals. Explore research.</p></div>", unsafe_allow_html=True)
        st.button("📄  Upload ECG File", type="primary", on_click=go, args=(P_UPLOAD,), key="home_upload")
    st.write("")
    t1, t2 = st.columns(2)
    with t1:
        st.button("📈  View Analysis", on_click=go, args=(P_SCREEN,), key="tile_analysis", use_container_width=True)
        st.button("📖  Research & Resources", on_click=go, args=(P_RES,), key="tile_research", use_container_width=True)
    with t2:
        st.button("💬  Chat with HRIDAYA", on_click=go, args=(P_CHAT,), key="tile_chat", use_container_width=True)
        st.button("⚙️  Settings", on_click=go, args=(P_SET,), key="tile_settings", use_container_width=True)
    st.write("")
    st.info("Accuracy and every other figure in this app are computed from the dataset you upload - nothing is pre-filled.")
    with st.expander("The 13 conditions screened"):
        st.dataframe(pd.DataFrame([{"#": c["id"], "Condition": c["name"], "Description": c["desc"],
                                    "Risk tier": TIER_ICON[c["tier"]], "Needs": c["needs"]} for c in CONDITIONS]),
                     use_container_width=True, hide_index=True)
    with st.expander("Supported uploads and how results are produced"):
        st.markdown(
            "- **WFDB records** (`.hea` + `.dat`/`.mat`, optional `.atr`): MIT-BIH, INCART, European ST-T, PTB, PhysioNet challenge sets, etc.\n"
            "- **CSV/TXT raw signals** (one column per lead; set sampling rate/units in Settings). Nested folders are searched recursively.\n"
            "- Pre-segmented beat matrices (e.g. 187-column Kaggle files) are *not* supported - they contain no rhythm, PR or QT information.\n\n"
            "- **Accuracy:** a Random-Forest beat classifier is trained/tested on the dataset's own `.atr` annotations. Without annotations, accuracy can't be computed and the app says so.\n"
            "- **13 conditions:** signal-processing rules (R-peak detection, RR irregularity, PR/QRS/ST/QT delineation). Conditions the available leads can't support are *Not assessable*.\n"
            "- **Validation:** rule outputs are compared with the dataset's rhythm/beat annotations (AFib, 2nd-degree block, LBBB, RBBB).")
    footer()


def page_upload():
    if not WFDB_OK:
        st.error("The `wfdb` package is missing. Run `pip install wfdb` and restart.")
    up = st.file_uploader("Upload ECG dataset (.zip)", type=["zip"])
    opts = st.session_state["opts"]

    if up is not None:
        sig = f"{up.name}:{up.size}"
        if st.session_state.get("upload_sig") != sig:
            old = st.session_state.get("workdir")
            if old and os.path.isdir(old):
                shutil.rmtree(old, ignore_errors=True)
            wd = tempfile.mkdtemp(prefix="hridaya_")
            with st.spinner("Extracting archive..."):
                safe_extract(up, wd)
            st.session_state.update(workdir=wd, upload_sig=sig, upload_name=up.name, results=None,
                                    found=discover_records(wd))
        found = st.session_state["found"]
        if not found:
            st.error("No ECG records found (need `.hea`/`.dat` WFDB files or `.csv`/`.txt` signals) in the ZIP.")
        else:
            wf = sum(1 for v in found.values() if v[0] == "wfdb")
            st.success(f"Found {len(found)} record(s): {wf} WFDB, {len(found) - wf} CSV/TXT.")
            if st.button("▶ Run full analysis", type="primary"):
                st.session_state["results"] = run_pipeline(found, opts, st.session_state["upload_name"])

    results = st.session_state.get("results")
    if not results:
        st.caption("No analysis yet - results (accuracy, confusion matrix, waveform, risk) appear here after you upload a dataset and run the analysis.")
        footer()
        return
    per = results["per"]
    st.markdown("---")
    st.header(f"Results - {results['dataset']}")
    if results["skipped"]:
        with st.expander(f"⚠️ {len(results['skipped'])} record(s) skipped"):
            st.json(results["skipped"])
    if not per:
        st.error("No record could be analysed.")
        footer()
        return

    show_ml(results["ml"], results["ml_msg"])

    st.markdown("### 🩺 Abnormalities detected - 13-condition evaluation (all records)")
    st.dataframe(condition_summary(per), use_container_width=True, hide_index=True)
    st.caption("**Records tested** = records where the condition could be evaluated with the available leads. "
               "Detected/Suspected are screening flags, not diagnoses.")

    ag = agreement_table(per)
    if len(ag):
        st.markdown("### ✅ Rule-engine validation against dataset annotations")
        st.dataframe(ag, use_container_width=True, hide_index=True)
        st.caption("Record-level agreement between the rule engine (signal only) and the reference rhythm/beat annotations.")

    st.markdown("### ⚠️ Risk stratification per record")
    df = pd.DataFrame([{
        "Record": n, "Risk": r["risk"], "Beats": r["n_beats"], "HR (bpm)": round(r["hr_med"], 0) if np.isfinite(r["hr_med"]) else None,
        "QRS (ms)": round(r["qrs_ms"], 0) if np.isfinite(r["qrs_ms"]) else None,
        "PR (ms)": round(r["pr_ms"], 0) if np.isfinite(r["pr_ms"]) else None,
        "QTc (ms)": round(r["qtc_ms"], 0) if np.isfinite(r["qtc_ms"]) else None,
        "Ectopic burden": f"{r['ectopic_burden']:.1%}" if np.isfinite(r["ectopic_burden"]) else "n/a",
        "Flagged": ", ".join(COND_BY_ID[i]["name"].split(" (")[0] for i, c in r["conditions"].items() if c["status"] in FLAGGED) or "-",
    } for n, r in per.items()])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 📈 Waveform")
    sel, res = show_waveform_block(results, "up")
    if res:
        st.markdown(f"**13-condition findings for record {sel}**")
        st.dataframe(record_conditions_df(res), use_container_width=True, hide_index=True)
    footer()


def page_screener():
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
    st.caption(f"Leads in file: {res['leads']} - analysis lead: {res['main_lead']}")
    m5, m6, m7, m8 = st.columns(4)
    f = lambda v, u: "n/a" if not np.isfinite(v) else f"{v:.0f} {u}"
    m5.metric("Median HR", f(res["hr_med"], "bpm"))
    m6.metric("QRS (approx.)", f(res["qrs_ms"], "ms"))
    m7.metric("PR (approx.)", f(res["pr_ms"], "ms"))
    m8.metric("QTc Bazett (approx.)", f(res["qtc_ms"], "ms"))
    st.markdown("### All 13 conditions - tested or not, with evidence")
    st.dataframe(record_conditions_df(res), use_container_width=True, hide_index=True)
    if res["class_counts"]:
        st.markdown("### Beat classes in this record (reference annotations)")
        st.bar_chart(pd.Series(res["class_counts"]), color=MAROON)
    footer()


def page_risk():
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
        st.info("Consider cardiology review - see **🏥 Cardiologists**.")
    footer()


def page_chat():
    st.markdown("#### HRIDAYA Assistant · *Your ECG guide*")
    st.caption("Answers come from a built-in ECG and machine-learning knowledge base plus the actual results of your analysis in this session. "
               "No external AI service is used, and no findings are invented.")
    voice_widget()

    chat = st.session_state.setdefault("chat", [{"role": "assistant", "content": GREETING}])

    cols = st.columns(3)
    for k, q in enumerate(SUGGESTIONS):
        with cols[k % 3]:
            st.button(q, key=f"chip_{k}", on_click=queue_question, args=(q,), use_container_width=True)

    for m in chat:
        with st.chat_message(m["role"], avatar="💓" if m["role"] == "assistant" else None):
            st.markdown(m["content"])

    typed = st.chat_input("Type a message...")
    pending = st.session_state.pop("pending_q", None)
    q = typed or pending
    if q:
        chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        reply = assistant_reply(q)
        chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant", avatar="💓"):
            st.markdown(reply)
        if st.session_state.get("speak_replies"):
            speak(reply)

    if len(chat) > 1:
        st.button("🧹 Clear conversation", key="clear_chat",
                  on_click=lambda: st.session_state.update(chat=[{"role": "assistant", "content": GREETING}]))
    footer()


def page_resources():
    st.markdown("Quick explainers from the HRIDAYA knowledge base. Ask the **HRIDAYA Assistant** for anything not covered here.")
    st.button("💬 Ask the assistant", on_click=go, args=(P_CHAT,), key="res_to_chat")
    for e in KB[1:]:
        with st.expander(e["title"]):
            st.markdown(e["text"])
    st.markdown("### Links")
    st.markdown("- [MIT-BIH Arrhythmia Database on PhysioNet](https://physionet.org/content/mitdb/1.0.0/)\n"
                "- [WFDB Python package documentation](https://wfdb.readthedocs.io)")
    footer()


def page_doctors():
    st.caption("Listing supplied by the author. Please verify availability, fees and credentials directly with the clinic before visiting.")
    for d in DOCTORS:
        q = quote_plus(f"{d['name'].split(' (')[0]} cardiologist {d['loc']}")
        st.markdown(
            f"<div class='doc-card'><h4 style='margin:0'>{html.escape(d['name'])}</h4>"
            f"<b>Experience:</b> {d['exp']}<br><b>Specialization:</b> {html.escape(d['spec'])}<br>"
            f"<b>Location:</b> {html.escape(d['loc'])}<br>"
            f"<a href='https://www.google.com/maps/search/?api=1&query={q}' target='_blank'>📍 Open in Google Maps</a> &nbsp;|&nbsp; "
            f"<a href='https://www.google.com/search?q={q}' target='_blank'>🔎 Find contact details</a></div>",
            unsafe_allow_html=True)
    st.warning("🚨 Chest pain, fainting, breathlessness or palpitations with dizziness are emergencies - call your local emergency number (112 in India) immediately.")
    footer()


def page_review():
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
    st.dataframe(record_conditions_df(res), use_container_width=True, hide_index=True)
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
    st.markdown(f"* **Platform:** HRIDAYA (हृदय) Automated ECG Diagnostics\n* **Copyright ©:** All Rights Reserved\n"
                f"* **Lead author:** {OWNER['name']}\n* **Phone:** {OWNER['phone']}\n* **Email:** {OWNER['email']}")
    footer()


def page_settings():
    st.markdown("#### Analysis")
    a, b = st.columns(2)
    with a:
        st.number_input("CSV/TXT sampling rate (Hz)", min_value=50, max_value=2000, step=10, key="csv_fs")
        st.selectbox("CSV amplitude units", ["mV", "µV", "arbitrary"], key="csv_units")
    with b:
        st.slider("Max beats for classifier", 20_000, 150_000, step=10_000, key="max_beats")
        st.checkbox("Also run patient-wise validation", key="patient_wise")
    st.markdown("#### Assistant voice")
    c, d = st.columns(2)
    with c:
        st.checkbox("Speak assistant replies", key="speak_replies")
    with d:
        st.selectbox("Voice language / accent", ["en-IN", "en-US", "en-GB"], key="voice_lang")
    st.caption("Voice features need a browser with the Web Speech API (Chrome or Edge). Speech input needs microphone permission and, in Chrome, an internet connection. "
               "Test on every target browser and on Android WebView before release.")
    footer()


# ======================================================================
# 11. MAIN
# ======================================================================
PAGE_FUNCS = {P_HOME: page_home, P_UPLOAD: page_upload, P_SCREEN: page_screener, P_RISK: page_risk,
              P_CHAT: page_chat, P_RES: page_resources, P_DOCS: page_doctors, P_REVIEW: page_review,
              P_SET: page_settings}


def main():
    st.set_page_config(page_title="HRIDAYA (हृदय) - ECG Research Companion", page_icon="💓", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # keep settings alive even when their widgets are not on screen
    for k, v in SETTING_DEFAULTS.items():
        st.session_state.setdefault(k, v)
        st.session_state[k] = st.session_state[k]
    st.session_state.setdefault("nav", P_HOME)
    if st.session_state["nav"] not in PAGES:
        st.session_state["nav"] = P_HOME

    st.sidebar.markdown(f"<div style='display:flex;align-items:center;gap:8px'>{logo_svg(34, heart=MAROON, pulse=GOLD)}"
                        f"<b style='font-size:1.2rem;color:{MAROON}'>HRIDAYA</b></div>", unsafe_allow_html=True)
    st.sidebar.caption("ECG Research Companion")
    page = st.sidebar.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    st.session_state["opts"] = dict(csv_fs=st.session_state["csv_fs"], csv_units=st.session_state["csv_units"],
                                    max_beats=st.session_state["max_beats"], patient_wise=st.session_state["patient_wise"])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**© Copyright Reserved**  \n**Author:** {OWNER['name']}  \n**Phone:** {OWNER['phone']}  \n**Email:** {OWNER['email']}")

    appbar(page.split(" ", 1)[1])
    PAGE_FUNCS[page]()


if __name__ == "__main__":
    main()
