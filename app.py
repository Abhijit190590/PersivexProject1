import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration & Custom CSS Injection for a Modern Look
st.set_page_config(
    page_title="Enterprise People Analytics Dashboard",
    page_icon="💼",
    layout="wide"
)

# Custom CSS to style metrics, headers, and backgrounds nicely
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-size: 16px !important; font-weight: bold; }
    .report-card { padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white; }
    .critical-card { background: linear-gradient(135deg, #ff4b4b 0%, #c1121f 100%); }
    .warning-card { background: linear-gradient(135deg, #ffa500 0%, #cc7a00 100%); }
    .stable-card { background: linear-gradient(135deg, #00cc66 0%, #006633 100%); }
    </style>
""", unsafe_allow_html=True) # <-- Fixed keyword argument here

# App Title & Header Banner
st.title("💼 Enterprise People Analytics Platform")
st.markdown("### Proactive Employee Attrition Risk Management System")
st.write("---")

# 2. Load Saved Assets (Cached for Speed)
@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('attrition_logistic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return scaler, model

try:
    loaded_scaler, loaded_model = load_assets()
except FileNotFoundError:
    st.error("❌ System Error: Prediction engine components missing. Please run your model training pipeline first.")
    st.stop()

# 3. Create a Split Layout: Left Sidebar for Inputs, Right Main Panel for Analytics
with st.sidebar:
    st.header("📋 Employee Profile Setup")
    st.write("Modify employee data variables here to recalculate live system risk scores.")
    st.markdown("---")
    
    # Group inputs cleanly inside the sidebar
    st.subheader("Personal Metrics")
    age = st.slider("Age", 18, 65, 30)
    distance = st.slider("Distance From Home (KM)", 1, 50, 10)
    num_comp = st.slider("Previous Companies Worked", 0, 10, 2)
    
    st.markdown("---")
    st.subheader("Organizational Profile")
    income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
    years_at_co = st.slider("Tenure (Years at Company)", 0, 40, 3)
    job_sat = st.select_slider("Job Satisfaction Rating", options=[1, 2, 3, 4], value=3)
    
    st.markdown("---")
    st.subheader("Department Placement")
    dept = st.selectbox("Department Assigned", ["Research & Development", "Sales", "Human Resources"])
    edu = st.selectbox("Education Specialization", ["Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"])

# 4. Main Panel - Analytics Display
st.subheader("📊 Live Predictive Modeling Assessment")

# Automatically map structural values from selection fields
dept_rd = 1.0 if dept == "Research & Development" else 0.0
dept_sales = 1.0 if dept == "Sales" else 0.0
edu_ls = 1.0 if edu == "Life Sciences" else 0.0
edu_mkt = 1.0 if edu == "Marketing" else 0.0
edu_med = 1.0 if edu == "Medical" else 0.0
edu_tech = 1.0 if edu == "Technical Degree" else 0.0
edu_other = 1.0 if edu == "Other" else 0.0

# Formulate vector row
input_data = pd.DataFrame([{
    'Age': age, 'DistanceFromHome': distance, 'JobSatisfaction': job_sat,
    'MonthlyIncome': income, 'NumCompaniesWorked': num_comp, 'YearsAtCompany': years_at_co,
    'Department_Research & Development': dept_rd, 'Department_Sales': dept_sales,
    'EducationField_Life Sciences': edu_ls, 'EducationField_Marketing': edu_mkt,
    'EducationField_Medical': edu_med, 'EducationField_Other': edu_other,
    'EducationField_Technical Degree': edu_tech
}])

# Pipeline scaling step execution
numerical_cols = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'YearsAtCompany']
input_data[numerical_cols] = loaded_scaler.transform(input_data[numerical_cols])

# Run active prediction mapping
risk_score = loaded_model.predict_proba(input_data)[0][1]
risk_percentage = risk_score * 100

# 5. Render Beautiful Executed Insights Section
col_metric1, col_metric2 = st.columns([1, 2])

with col_metric1:
    st.metric(label="Calculated Exit Probability", value=f"{risk_percentage:.1f}%")
    # Draw a clean native progress indicator bar
    st.progress(risk_score)

with col_metric2:
    if risk_score >= 0.75:
        st.markdown(f"""
            <div class='report-card critical-card'>
                <h3>🔴 ACTION REQUIRED: CRITICAL RISK STATUS</h3>
                <p>This employee profile crosses the safety threshold. High turnover probability detected.</p>
            </div>
        """, unsafe_allow_html=True) # <-- Fixed here
    elif 0.40 <= risk_score < 0.75:
        st.markdown(f"""
            <div class='report-card warning-card'>
                <h3>🟡 MONITORING REQUIRED: MODERATE RISK STATUS</h3>
                <p>Early warning indicators are flaring up. Retention intervention recommended.</p>
            </div>
        """, unsafe_allow_html=True) # <-- Fixed here
    else:
        st.markdown(f"""
            <div class='report-card stable-card'>
                <h3>🟢 STABLE RETENTION STATUS</h3>
                <p>Metrics point to strong workforce engagement. Low probability of churn.</p>
            </div>
        """, unsafe_allow_html=True) # <-- Fixed here

st.write("---")
st.subheader("💡 Tailored HR Intervention Strategy Blueprint")

# Render dynamic data-driven business action cards based on input thresholds
col_strat1, col_strat2 = st.columns(2)

with col_strat1:
    st.markdown("#### 🎯 Core Retention Playbook")
    if job_sat <= 2:
        st.error("**Managerial Intervention Overdue:** Low job satisfaction scores significantly accelerate resignation timetables. Schedule an exploratory structural 1-on-1 interview to realign goals.")
    if distance > 20:
        st.warning("**Commute Strain Detected:** The employee lives significantly far from HQ. Propose a flexible hybrid scheduling arrangement or remote work days to lower burnout.")
    if income < 3500:
        st.warning("**Compensation Adjustment Check:** Salary parameters are tracking lower than standard baseline limits. Coordinate with compensation leads to perform parity evaluation.")
    if job_sat > 2 and distance <= 20 and income >= 3500:
        st.success("No immediate baseline vulnerabilities found for core operational risk triggers.")

with col_strat2:
    st.markdown("#### 📈 Talent Growth Pathway Vector")
    if years_at_co >= 4 and num_comp >= 3:
        st.info("**High Mobility Profile:** Historical patterns indicate this individual switches environments frequently and has reached tenured milestones here. Prioritize clear discussions regarding next-level promotions.")
    else:
        st.write("Standard talent development guidelines apply to this profile structure.")