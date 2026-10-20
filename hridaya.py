import os
import zipfile
import tempfile
import streamlit as st
import wfdb
import matplotlib.pyplot as plt

st.title("❤️ HRIDAYA (हृदय): Automated ECG Arrhythmia & Anomaly Diagnostic Hub")
st.subheader("Upload MIT-BIH Arrhythmia ZIP Archive")

uploaded_zip = st.file_uploader("Upload ECG Dataset (.zip)", type=["zip"])

if uploaded_zip is not None:
    # Use a temporary directory to extract files safely
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Recursively scan the extracted folders for .dat files
        record_paths = {}
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.dat'):
                    record_name = os.path.splitext(file)[0]
                    # Check if corresponding .atr file exists in the same folder
                    atr_path = os.path.join(root, f"{record_name}.atr")
                    dat_path = os.path.join(root, file)
                    if os.path.exists(atr_path):
                        # Store the base path without extension
                        record_paths[record_name] = os.path.join(root, record_name)

        if not record_paths:
            st.error("Could not locate valid MIT-BIH .dat and .atr files in the zip directory structure.")
        else:
            st.success(f"Successfully located {len(record_paths)} valid MIT-BIH record pair(s)!")
            
            # Let the user select a record to inspect
            selected_record = st.selectbox("Select Record for Analysis", sorted(list(record_paths.keys())))
            
            if selected_record:
                record_prefix = record_paths[selected_record]
                
                # Read record and annotations using wfdb
                record = wfdb.rdrecord(record_prefix)
                annotation = wfdb.rdann(record_prefix, 'atr')
                
                # Display metadata metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Sampling Rate", f"{record.fs} Hz")
                col2.metric("Signal Shape", str(record.p_signal.shape))
                col3.metric("Annotation Count", len(annotation.symbol))
                
                st.write(f"**Annotation symbols found:** {sorted(list(set(annotation.symbol)))}")
                
                # Plotting the ECG signal
                st.subheader(f"ECG Waveform & Annotations — Record {selected_record}")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Plot first 10 seconds (fs * 10 samples)
                samples_to_plot = int(record.fs * 10)
                ax.plot(record.p_signal[:samples_to_plot, 0], color='#1f77b4', linewidth=1.2, label='MLII')
                ax.set_title(f"Actual MIT-BIH ECG — Record {selected_record}")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude (WFDB units)")
                ax.grid(True, linestyle='--', alpha=0.6)
                
                st.pyplot(fig)
