import streamlit as st
from hf_adapter import HFAdapter
from openai_adapter import OpenAIAdapter
from min_k_prob import min_k_percent_prob
from adaptive_detector import adaptive_min_prob
from evidence_report import html_highlight_tokens, plot_token_prob_trend
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import io
import base64

# --- Page Config ---
st.set_page_config(page_title="CogniSight", page_icon="🧠", layout="wide")

# --- Custom CSS ---
# --- Custom CSS ---
st.markdown(
    """
    <style>
    body, .main, .block-container {background-color: #00B4D8; color: #1e1e1e;}
    
    /* Sidebar background and text */
    section[data-testid="stSidebar"] {background-color: #0096C7 !important; color: #f5f7fa; font-weight: bold; font-size: 16px; }
    section[data-testid="stSidebar"] * {color: #3944bc !important;font-weight: bold;  font-size: 16px; }
    
    /* Input boxes and text areas */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div>div>div {background-color: #cce7ff !important; color: black;}
    
    /* Placeholder text black */
    .stTextInput>div>div>input::placeholder,
    .stTextArea>div>div>textarea::placeholder {color: #000000 !important; opacity: 1 !important;}

    /* Metric box */
    .stMetric {background: #0096C7; color:blue; border-radius: 12px; padding: 12px; box-shadow: 0px 3px 10px rgba(0,0,0,0.08);}

    /* Headings */
    h1, h2, h3 {font-family: 'Inter', sans-serif; color: blue;}

    /* Buttons */
    .stButton>button {background: linear-gradient(90deg,#6a11cb,#2575fc); color:white; border:none; border-radius:10px; padding:10px 22px; font-weight:600;}
    .stButton>button:hover {opacity:0.95;}
    </style>
    """,
    unsafe_allow_html=True,
)



# --- Title ---
st.title(" Cognisight")
st.caption(" Detect whether text may have appeared in training data using min-k% PROB and Adaptive selection methods.")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
backend = st.sidebar.selectbox('Backend', ['HuggingFace', 'OpenAI'])
model_name = st.sidebar.text_input('HF model name', 'EleutherAI/pythia-2.8b')
use_adaptive = st.sidebar.checkbox('Use adaptive selection (zscore)', True)
method = st.sidebar.selectbox('Adaptive method', ['zscore', 'iqr'])

if backend == 'OpenAI':
    st.sidebar.info('🔑 Requires OPENAI_API_KEY')

# --- Input ---
st.subheader("✍️ Input Text")
st.markdown("Provide the text passage you want to analyze. CogniSight will highlight the least probable tokens and estimate how likely the text was part of model training.")
text = st.text_area('Enter text to analyze:', height=200, placeholder="Paste text here...")

# --- Run Detection ---
if st.button('🚀 Run Detection'):
    with st.spinner("Analyzing text with model..."):
        adapter = HFAdapter(model_name) if backend == 'HuggingFace' else OpenAIAdapter()

        toks, lps = adapter.token_log_probs(text)
        if use_adaptive:
            score, idx = adaptive_min_prob(lps, method=method)
        else:
            score = min_k_percent_prob(lps, k_percent=0.5)
            idx = sorted(range(len(lps)), key=lambda j: lps[j])[:max(1, int(0.2*len(lps)))]
        max_score, chunk_scores, chunk_indices = adapter.chunked_token_log_probs(text)
        col1, col2 = st.columns(2)
        col1.metric('Detection score', f" {score:.4f}")
        col2.metric('Max chunk detection score', f"{adapter.chunked_token_log_probs(text)[0]:.4f}")
        st.write('Chunk scores:', chunk_scores)
        st.markdown("### 🔍 Highlighted Tokens")
        st.markdown(html_highlight_tokens(toks, idx), unsafe_allow_html=True)

        st.markdown("### 📈 Token Probability Trend")
        st.image(plot_token_prob_trend(lps, idx))

# --- AUC Evaluation ---
st.divider()
st.subheader("📊 Evaluate AUC on your Dataset")
st.markdown("Run a benchmark test on a standard dataset to evaluate detection performance in distinguishing training members vs. non-members.")
st.markdown("""
**How to interpret this result:**
- The **AUC (Area Under the Curve)** measures how well the detector separates member (seen in training) vs non-member (unseen) texts.
- A score of **0.5** means random guessing, while **1.0** indicates perfect separation.
- Higher AUC means stronger detection capability.
""")
if st.button('📡 Run AUC Evaluation'):
    with st.spinner('Loading dataset and running detection...'):
        ds = load_dataset('swj0419/WikiMIA', split='WikiMIA_length256')
        st.write("Dataset columns:", ds.column_names)

        requested_samples = 100
        n_samples = min(requested_samples, len(ds))
        ds_sample = ds.select(range(n_samples))

        texts = [ex['input'] for ex in ds_sample]
        labels = [ex['label'] for ex in ds_sample]  # Ensure correct membership label

        adapter = HFAdapter(model_name) if backend == 'HuggingFace' else OpenAIAdapter()
        scores = []
        
        progress = st.progress(0)
        for i, txt in enumerate(texts):
            toks, lps = adapter.token_log_probs(txt)
            score, _ = adaptive_min_prob(lps, method=method) if use_adaptive else (min_k_percent_prob(lps, k_percent=0.5), None)
            scores.append(score)
            progress.progress((i+1)/len(texts))

        auc = roc_auc_score(labels, scores)
        st.success(f"✅ AUC score for member vs non-member classification: {auc:.4f}")

        fpr, tpr, _ = roc_curve(labels, scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color="#2575fc")
        ax.plot([0,1], [0,1], linestyle='--', color='grey')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()

        st.pyplot(fig)

        st.markdown("""
        📈 The ROC curve above shows the trade-off between correctly identifying members (**True Positive Rate**) and incorrectly flagging non-members (**False Positive Rate**).
        Ideally, the curve bows towards the top-left corner, showing the detector is highly accurate.
        """)
