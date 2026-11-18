"""Beginner-friendly Streamlit interface for the user's Fake Job Posting AI project."""
from pathlib import Path
import re
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='JobShield AI | Fake Job Detection', page_icon='🛡️', layout='wide')

st.markdown('''
<style>
.stApp {background: #0b1120; color: #e8eefc;}
.block-container {max-width: 1080px; padding-top: 2.5rem;}
h1,h2,h3 {color: #edf3ff !important;}
p, label, .stMarkdown {color: #c3cde2;}
.hero {padding: 2rem 2.2rem; border-radius: 22px; background: linear-gradient(115deg,#142548,#253e76 70%,#34439e); border:1px solid #425a95; margin-bottom:1.6rem;}
.hero h1 {font-size: 2.65rem; color: white !important; margin:0 0 .5rem 0;}
.hero p {color:#e0e9ff; font-size:1.04rem; margin:0;}
.panel {padding: 1.2rem 1.5rem; border: 1px solid #344563; border-radius: 16px; background:#121e35; margin: 1rem 0;}
.eyebrow {letter-spacing: .14em; font-size:.78rem; font-weight:700; color:#a9c3ff;}
.stButton>button {background: #7c92ff; color:#0b1120; border-radius:10px; font-weight:750; border:0; padding: .72rem 1.7rem;}
.stButton>button:hover {background:#a7b5ff; color:#091127;}
.result-alert {padding:1.1rem 1.35rem; border-radius:14px; border:1px solid #f59e0b; background:#3d2b19; color:#ffe9bd;}
.result-safe {padding:1.1rem 1.35rem; border-radius:14px; border:1px solid #22c55e; background:#10352b; color:#d1fae5;}
.small-note {font-size: .86rem; color:#93a4c0;}
</style>
''', unsafe_allow_html=True)


def clean_text(text):
    """Must match the cleaning used to train the model."""
    text = str(text).lower()
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / 'job_model.joblib')
    tfidf = joblib.load(ROOT / 'tfidf_vectorizer.joblib')
    return model, tfidf


st.markdown('''<div class="hero"><div class="eyebrow">AI-POWERED JOB FRAUD DETECTION</div>
<h1>🛡️ JobShield AI</h1><p>Analyze job descriptions to identify potentially fraudulent job postings.</p></div>''', unsafe_allow_html=True)

if not (ROOT / 'job_model.joblib').exists() or not (ROOT / 'tfidf_vectorizer.joblib').exists():
    st.error('Trained model files are not available yet.')
    st.info('First place fake_job_postings.csv in this folder. In your VS Code terminal run:  python train_model.py  . Then refresh this page.')
    st.stop()

try:
    model, tfidf = load_artifacts()
except Exception as exc:
    st.error('Unable to load trained model files. Re-run python train_model.py. Details: ' + str(exc))
    st.stop()

left, right = st.columns([1.8, 1], gap='large')
with left:
    st.subheader('Analyze a job posting')
    with st.form('job_form'):
        title = st.text_input('Job title *', placeholder='e.g. Customer Support Representative')
        company_profile = st.text_area('Company profile (optional)', height=110, placeholder='Tell us about the employer...')
        description = st.text_area('Job description *', height=220, placeholder='Paste the full job description, salary claims and requirements here...')
        submitted = st.form_submit_button('🔎 Analyze Job Posting', use_container_width=True)
with right:
    st.markdown('''<div class="panel"><div class="eyebrow">HOW IT WORKS</div>
    <h3>Three simple steps</h3><p>1. Enter job details</p>
    <p>2. Run the trained ML model</p><p>3. Review the fraud risk estimate</p></div>''', unsafe_allow_html=True)
    st.markdown('**Model:** Logistic Regression')
    st.caption('Text processing: TF-IDF (unigrams and bigrams). Class 1 = fraudulent; class 0 = genuine.')
    st.warning('This is an educational screening tool. It cannot verify a company or guarantee a job is genuine.')

if submitted:
    if not title.strip() or not description.strip():
        st.error('Please enter both a job title and a job description.')
    else:
        combined = f'{title} {description} {company_profile}'
        cleaned = clean_text(combined)
        vector = tfidf.transform([cleaned])
        prediction = int(model.predict(vector)[0])
        probability_by_class = dict(zip(model.classes_, model.predict_proba(vector)[0]))
        fraud_probability = float(probability_by_class.get(1, 0.0))
        genuine_probability = float(probability_by_class.get(0, 0.0))

        st.divider()
        st.subheader('Analysis result')
        if prediction == 1:
            st.markdown('<div class="result-alert"><h3>⚠️ Potentially fraudulent posting</h3>The model classified this job posting as suspicious. Verify it independently before applying.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-safe"><h3>✅ Predicted as genuine</h3>The model classified this posting as genuine, but you should still verify the employer and job details.</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric('Estimated fraud probability', f'{fraud_probability * 100:.1f}%')
        m2.metric('Estimated genuine probability', f'{genuine_probability * 100:.1f}%')
        st.progress(fraud_probability, text='Estimated fraud probability')
        st.caption('Probabilities come from the trained classifier; they are not independently calibrated risk scores.')

st.divider()
st.caption('JobShield AI • First-semester AI App Lab project • Built with Python, Streamlit, TF-IDF and Logistic Regression')
