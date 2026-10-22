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
    photo_path = reviewer_photo_path() or ""
    photo_lower = photo_path.lower()
    if photo_lower.endswith(".webp"):
        ext = "webp"
    elif photo_lower.endswith((".jpg", ".jpeg")):
        ext = "jpeg"
    else:
        ext = "png"
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
