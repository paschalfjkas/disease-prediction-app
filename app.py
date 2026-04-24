import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import DiseasePredictionModel
import time

st.set_page_config(page_title="AI Health Diagnostician", page_icon="🩺", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
        color: white;
    }
    .glass-panel {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model():
    model = DiseasePredictionModel()
    if not model.load_model():
        model.train()
    return model

model = load_and_train_model()

st.title("🩺 AI Health Diagnostician")
st.markdown("### Advanced Disease Prediction using Machine Learning")
st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
        <div class="glass-panel">
            <h3 style="margin-top:0;">📝 Symptom Entry</h3>
            <p style='color: #94a3b8;'>Select the symptoms you are currently experiencing. The more accurate your selection, the better the prediction.</p>
        </div>
    """, unsafe_allow_html=True)
    
    selected_symptoms = st.multiselect(
        "Choose your symptoms:",
        model.symptoms_list,
        placeholder="Type or select symptoms..."
    )
    
    analyze_btn = st.button("🔍 Analyze Symptoms")

with col2:
    if analyze_btn:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("🧠 Analyzing symptom patterns..."):
                time.sleep(1) # simulate processing
                result = model.predict(selected_symptoms)
                
            st.markdown(f"""
                <div class="glass-panel" style="border-left: 5px solid #10b981;">
                    <h2 style="color: #10b981; margin-top:0;">🎯 Primary Diagnosis: {result['disease']}</h2>
                    <p style="font-size: 1.1em;"><b>Description:</b> {result['description']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display Precautions
            st.markdown("### 🛡️ Recommended Precautions")
            cols = st.columns(min(len(result['precautions']), 4))
            for i, p in enumerate(result['precautions']):
                if p and str(p).lower() != 'nan':
                    with cols[i % len(cols)]:
                        st.info(str(p).replace('_', ' ').title())
            
            # Additional Analytics
            st.write("---")
            st.markdown("### 📊 Diagnostic Confidence")
            
            # Display Top 3 Probabilities
            top3 = result['top3']
            df_probs = pd.DataFrame(top3, columns=['Disease', 'Probability'])
            df_probs['Probability'] = df_probs['Probability'] * 100
            
            # Seaborn/Matplotlib Plot
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            sns.barplot(x='Probability', y='Disease', data=df_probs, palette='viridis', ax=ax, hue='Disease', legend=False)
            ax.set_xlabel('Probability (%)', color='white')
            ax.set_ylabel('')
            ax.tick_params(colors='white')
            ax.set_xlim(0, 100)
            for spine in ax.spines.values():
                spine.set_edgecolor('#ffffff33')
            st.pyplot(fig)
            
            # Symptom Severity Warning
            severity_scores = [model.symptom_severity.get(s, 0) for s in selected_symptoms]
            max_severity = max(severity_scores) if severity_scores else 0
            if max_severity > 5:
                st.error("🚨 **High Severity Symptom Detected!** Please consult a doctor immediately.")

    else:
        st.markdown("""
            <div class="glass-panel" style="text-align: center; padding: 50px;">
                <h1 style="font-size: 60px; margin:0;">🔬</h1>
                <h3 style="color: #94a3b8;">Awaiting Symptom Input</h3>
                <p style="color: #64748b;">Select your symptoms on the left to begin the diagnostic analysis.</p>
            </div>
        """, unsafe_allow_html=True)

st.write("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 12px;'>⚠️ Disclaimer: This application is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.</p>", unsafe_allow_html=True)
