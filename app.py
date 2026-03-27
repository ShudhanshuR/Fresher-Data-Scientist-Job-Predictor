import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE ----------------
st.set_page_config(page_title="Job Predictor", page_icon="🚀")

st.title("🚀 Fresher Data Scientist Job Prediction")

st.markdown("Enter your details to predict hiring chances")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = joblib.load("model.pkl")

# manually define features (same as training)
feature_names = [
    "python",
    "sql",
    "ml",
    "projects",
    "internship",
    "algorithm",
    "statistics",
    "project_level"
]

# ================= INPUTS =================

python_skill = st.slider("Python (0-5)", 0, 5, 0)
sql_skill = st.slider("SQL (0-5)", 0, 5, 0)
ml_skill = st.slider("ML (0-5)", 0, 5, 0)
projects = st.slider("Projects (0-10)", 0, 10, 0)
internship = st.selectbox("Internship", [0, 1])

algo = st.selectbox("Algorithm Knowledge", ["None", "Basic", "Good", "Strong"])
stats = st.selectbox("Statistics", ["Low", "Medium", "High"])
project_level = st.selectbox("Project Level", ["Beginner", "Intermediate", "Advanced"])

# ---------------- ENCODING ----------------
algo_val = {"None": 0, "Basic": 1, "Good": 2, "Strong": 3}[algo]
stats_val = {"Low": 0, "Medium": 1, "High": 2}[stats]
project_val = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}[project_level]

# ================= PREDICTION =================

if st.button("🔍 Predict Hiring Chances"):

    input_dict = {
        "python": python_skill,
        "sql": sql_skill,
        "ml": ml_skill,
        "projects": projects,
        "internship": internship,
        "algorithm": algo_val,
        "statistics": stats_val,
        "project_level": project_val
    }

    input_df = pd.DataFrame([input_dict])

    # 🔥 IMPORTANT: ensure same order
    input_df = input_df[feature_names]

    prob = model.predict_proba(input_df)[0][1]
    percent = round(prob * 100, 2)

    st.subheader("📈 Result")
    st.metric("Hiring Chance", f"{percent}%")

    if percent >= 75:
        st.success("🔥 High Chance of Selection")
    elif percent >= 50:
        st.warning("⚡ Moderate Chance")
    else:
        st.error("❌ Low Chance")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(f"Developed with ❤️ by **[Shudhanshu Ranjan](http://linkedin.com/in/shudhanshu-ranjan-b56b76239)**")
