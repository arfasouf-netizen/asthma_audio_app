import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile
import io

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="PneumoScan | Détection Asthme",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Injection de CSS personnalisé ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        width: 100%; background-color: #2563eb; color: white;
        border-radius: 8px; height: 50px; font-weight: bold;
    }
    .stButton>button:hover { background-color: #1d4ed8; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .report-card {
        padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 20px; border-left: 5px solid #2563eb; background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELE_PATH = os.path.join(BASE_DIR, "models", "modele_asthme_rf.pkl")

@st.cache_resource
def charger_modele():
    return joblib.load(MODELE_PATH)

modele = charger_modele()

# ============================================================
# BARRE LATÉRALE
# ============================================================
with st.sidebar:
    st.title("PneumoScan AI")
    st.markdown("---")
    st.markdown("### 👨‍⚕️ Espace Médical")
    st.write("Outil d'aide au diagnostic pour les centres de santé.")
    st.info("**Note :** Analyse des sifflements (wheezes) via Random Forest.")
    st.caption("Projet DTS - IAI Cameroun")

# ============================================================
# CONTENU PRINCIPAL
# ============================================================
st.title("🩺 Analyseur Respiratoire")
st.markdown("Enregistrez le patient en direct ou uploadez un fichier.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1️⃣ Acquisition du signal")
    
    # --- Option 1 : Enregistrement natif Streamlit ---
    st.markdown("**🎤 Enregistrement en direct :**")
    audio_bytes = st.audio_input("Cliquez sur le micro pour enregistrer", key="recorder")
    
    st.markdown("---")
    
    # --- Option 2 : Upload de fichier ---
    st.markdown("**📂 Ou uploader un fichier (.wav) :**")
    fichier_audio = st.file_uploader("", type=["wav"], label_visibility="collapsed")

    # Gestion de l'audio (priorité à l'enregistrement direct)
    audio_source = None
    
    if audio_bytes:
        audio_source = audio_bytes
    elif fichier_audio is not None:
        audio_source = fichier_audio

    # Bouton d'analyse
    analyser = st.button("🚀 Lancer l'analyse IA", type="primary")

with col2:
    st.markdown("### 2️⃣ Résultats du diagnostic")
    
    if audio_source is not None and analyser:
        with st.spinner("Traitement du signal en cours..."):
            
            # CORRECTION ICI : Conversion robuste en octets
            if hasattr(audio_source, 'getvalue'):
                # Si c'est un UploadedFile de Streamlit
                data_bytes = audio_source.getvalue()
            else:
                # Si c'est déjà des bytes (micro)
                data_bytes = audio_source

            # Création du fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(data_bytes)
                tmp_path = tmp.name
            
            # Traitement avec Librosa
            signal, sr = librosa.load(tmp_path, sr=22050)
            
            segment_length = sr * 1 
            segments = []
            
            for i in range(0, len(signal), segment_length):
                segment = signal[i:i+segment_length]
                if len(segment) > sr * 0.2: 
                    segments.append(segment)
            
            predictions = []
            for seg in segments:
                mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
                mfcc_moyen = np.mean(mfcc, axis=1)
                predictions.append(mfcc_moyen)
            
            if len(predictions) > 0:
                predictions = np.array(predictions)
                resultats = modele.predict(predictions)
                pourcentage_sifflement = (np.sum(resultats == 1) / len(resultats)) * 100
                
                if pourcentage_sifflement > 30:
                    st.markdown(f"""
                    <div class="report-card" style="border-left-color: #dc2626;">
                        <h3 style="color: #dc2626; margin-top:0;">⚠️ Anomalie Détectée</h3>
                        <p><strong>Indice de sifflement :</strong> {pourcentage_sifflement:.1f}%</p>
                        <p>Présence significative de sifflements détectée.</p>
                        <p><em>Recommandation : Consultation médicale conseillée.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="report-card" style="border-left-color: #16a34a;">
                        <h3 style="color: #16a34a; margin-top:0;">✅ Signal Normal</h3>
                        <p><strong>Indice de sifflement :</strong> {pourcentage_sifflement:.1f}%</p>
                        <p>Le signal ne présente pas d'anomalies majeures.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Fichier audio trop court ou inaudible.")
            
            # Nettoyage du fichier temporaire
            os.remove(tmp_path)
            
    elif audio_source is None:
        st.info("Les résultats s'afficheront ici.")