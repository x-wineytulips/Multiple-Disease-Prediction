import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import imblearn

# model loading
model1 = joblib.load("A:\Project\Multiple-Disease-Prediction\Multiple-Disease-Prediction\models\kidney.pkl")
model2 = joblib.load("A:\Project\Multiple-Disease-Prediction\Multiple-Disease-Prediction\models\liver_model.pkl")
model3 = joblib.load("A:\Project\Multiple-Disease-Prediction\Multiple-Disease-Prediction\models\parkinsons.pkl") 

# --- Page Config ---
st.set_page_config(page_title="Multiple Disease Prediction",page_icon="🏥")

# --- Glassmorphism CSS + helper functions ---
st.markdown(
"""
<style>
/* Background gradient for subtle depth */
html, body {background: linear-gradient(180deg, #d1d5db 0%, #9ca3af 100%); color: #0f172a;} 
.glass {
    background: rgba(255,255,255,0.85);
    color: #0f172a;
    backdrop-filter: blur(10px) saturate(130%);
    -webkit-backdrop-filter: blur(10px) saturate(130%);
    border-radius: 14px;
    border: 1px solid rgba(191, 219, 254, 0.55);
    box-shadow: 0 14px 40px rgba(15, 23, 42, 0.08);
    padding: 16px;
}
.card-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: 12px;}
.tips-title {font-size:18px; font-weight:700; margin-bottom:6px; color: #0ea5e9;}
.tips-list {margin:0 0 8px 18px; color:#0f172a;}
.specialist {font-weight:600; color:#0f172a}
</style>
""",
        unsafe_allow_html=True,
)

def _render_tips_card(title, dos, donts, when_to_consult, specialists):
        dos_html = "".join([f"<li>{d}</li>" for d in dos])
        donts_html = "".join([f"<li>{d}</li>" for d in donts])
        when_html = "".join([f"<li>{w}</li>" for w in when_to_consult])
        specs_html = "".join([f"<div><span class=\"specialist\">• {s}</span></div>" for s in specialists])

        html = f"""
        <div class="glass">
            <div class="tips-title">{title} — Healthy Tips & Suggestions</div>
            <div style="display:flex; gap:20px; flex-wrap:wrap;">
                <div style="flex:1; min-width:220px;">
                    <div style="font-weight:600;">Do's</div>
                    <ul class="tips-list">{dos_html}</ul>
                </div>
                <div style="flex:1; min-width:220px;">
                    <div style="font-weight:600;">Don'ts</div>
                    <ul class="tips-list">{donts_html}</ul>
                </div>
                <div style="flex:1; min-width:220px;">
                    <div style="font-weight:600;">When to consult a doctor</div>
                    <ul class="tips-list">{when_html}</ul>
                </div>
            </div>
            <div style="margin-top:10px;">
                <div style="font-weight:600;">Recommended specialists</div>
                <div style="margin-top:6px;">{specs_html}</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


def _render_motivational_quote(disease, positive):
    quotes = {
        "Kidney": {
            True: "Keep going — small lifestyle steps can make a big difference for kidney health.",
            False: "Great news! Continue healthy habits to protect your kidneys for the long term."
        },
        "Liver": {
            True: "This is a chance to reset; the liver is resilient and responds well to healthier choices.",
            False: "Well done — maintaining balance now can keep your liver strong into the future."
        },
        "Parkinsons": {
            True: "Focus on steady progress — consistent exercise and support matter more than perfection.",
            False: "You are on the right path; keep building strength, balance, and confidence."
        }
    }
    st.info(f"💬 {quotes[disease][positive]}")


def _render_outcome_guidance(disease, positive, dos, donts, specialists):
    title = f"{disease} {'Support Plan' if positive else 'Wellness Advice'}"
    status_text = (
        "You have a higher risk and should follow these steps closely." if positive else
        "Your result is lower risk. Keep up the healthy habits below."
    )
    intro_html = f"<p style=\"margin-bottom:10px; font-size:15px;\">{status_text}</p>"
    dos_html = "".join([f"<li>{d}</li>" for d in dos])
    donts_html = "".join([f"<li>{d}</li>" for d in donts])
    specs_html = "".join([f"<div><span class=\"specialist\">• {s}</span></div>" for s in specialists])
    html = f"""
    <div class="glass">
        <div class="tips-title">{title}</div>
        {intro_html}
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
            <div style="flex:1; min-width:220px;">
                <div style="font-weight:600;">Do's</div>
                <ul class="tips-list">{dos_html}</ul>
            </div>
            <div style="flex:1; min-width:220px;">
                <div style="font-weight:600;">Don'ts</div>
                <ul class="tips-list">{donts_html}</ul>
            </div>
        </div>
        <div style="margin-top:10px;">
            <div style="font-weight:600;">Recommended specialists</div>
            <div style="margin-top:6px;">{specs_html}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# Sidebar for navigation
choice = st.sidebar.selectbox(
    "Choose Disease Prediction", 
    ["📄 Overview", "🩸 Kidney Disease", "🧬 Liver Disease", "🧩 Parkinsons Disease"]
)

if choice == "📄 Overview":
    st.markdown(
    '<h1 style="color:#0ea5e9; font-family:Arial; text-align:center;">💊 Multiple Disease Prediction</h1>',
    unsafe_allow_html=True
    )

    def show_overview():
    # Title
        st.title("Multiple Disease Prediction")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Short intro
    st.header("Overview")
    st.write(
        """
        This application demonstrates the use of **Machine Learning models** in the 
        healthcare domain for predicting the likelihood of certain diseases based on 
        patient medical reports.  

        The project covers three conditions:  
        - **Liver Disease** – prediction based on biochemical and demographic attributes.  
        - **Kidney Disease** – prediction using clinical measurements and test results.  
        - **Parkinson’s Disease** – prediction using voice-based biomedical features.  

        The primary goal is to provide an **early indication** of whether a patient 
        may be at risk, which can help in **awareness and timely consultation with healthcare professionals**.  
        However, this tool is intended only for **educational and research purposes** 
        and should not be used as a replacement for medical diagnosis.  
        """
    )

    # Navigation guide
    st.subheader("🔎 How to Use")
    st.write("""
    - Go to **Liver Disease Prediction** to analyze liver-related medical data.  
    - Go to **Kidney Disease Prediction** to analyze kidney-related medical data.  
    - Go to **Parkinson’s Prediction** to analyze voice-based health data.  
    """)

    # Workflow
    st.subheader("⚙️ Workflow")
    st.markdown(
        """
        1. Enter your health parameters in the form.  
        2. The trained ML model processes your input.  
        3. You will get a prediction result instantly.  
        """
    )

    # Dataset and Models
    with st.expander("📊 Dataset & Models Used"):
        st.write("""
        - **Datasets:** Taken from Kaggle.  
        - **Algorithms Used:** Logistic Regression, Random Forest, SVM.  
        - Models are trained and optimized for accuracy and performance.  
        """)

    
    # Model Performance Chart (Plotly)
    st.subheader("📈 Model Performance (Accuracy)")
    performance = pd.DataFrame({
        "Model": ["Kidney (Random Forest)", "Liver (Decision Tree)", "Parkinson's (SVM)"],
        "Accuracy": [0.99, 0.86, 0.97]   
    })

    fig = px.bar(
        performance, 
        x="Model", 
        y="Accuracy", 
        text="Accuracy",
        color="Model",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"], 
        title="Model Accuracy Comparison"
    )
    fig.update_traces(texttemplate='%{text:.0%}', textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False,tickformat=".0%", range=[0,1],showgrid=False, zeroline=False),
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450, 
        width=900 
    )
    st.plotly_chart(fig, use_container_width=True)


    # Disclaimer
    # Quick healthy tips summary (Overview)
    _render_tips_card(
        "Kidney Health",
        dos=["Stay well hydrated but avoid excessive fluids if advised by doctor", "Control blood pressure and blood sugar", "Maintain a balanced low-salt diet", "Get regular urine and kidney function tests if at risk"],
        donts=["Avoid long-term use of NSAIDs without medical advice", "Don't ignore persistent swelling or reduced urine output"],
        when_to_consult=["Sudden decrease in urine output", "Unexplained swelling in legs/face", "Consistently abnormal kidney function tests"],
        specialists=["Nephrologist", "Urologist", "Renal Dietitian"]
    )

    _render_tips_card(
        "Liver Health",
        dos=["Limit alcohol intake and maintain healthy weight", "Get vaccinated for Hepatitis A and B if not immune", "Avoid unnecessary herbal or OTC hepatotoxic drugs", "Eat a balanced diet and exercise regularly"],
        donts=["Do not mix alcohol with medications that affect the liver", "Avoid unregulated supplements of unknown origin"],
        when_to_consult=["Yellowing of skin/eyes (jaundice)", "Persistent abdominal pain or unexplained weight loss", "Very dark urine or pale stools"],
        specialists=["Hepatologist", "Gastroenterologist", "Clinical Nutritionist"]
    )

    _render_tips_card(
        "Parkinson's Awareness",
        dos=["Stay active with regular exercise and balance training", "Prioritize good sleep and mental health support", "Follow medication schedules and report side effects"],
        donts=["Don't ignore new or worsening tremors, stiffness, or balance problems", "Avoid activities with high fall risk without assessment"],
        when_to_consult=["New or progressive tremor/stiffness", "Difficulty walking or frequent falls", "Speech/swallowing problems"],
        specialists=["Neurologist (Movement Disorders)", "Physiotherapist", "Occupational Therapist"]
    )

    st.warning("⚠️ Disclaimer: This app is for educational/demo purposes only, not for medical diagnosis.")

    # Footer / credits
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 18px;">⚕️<span style="color:#FF5733;">Multiple Disease Prediction</span> | Built by <strong>Sudhamrita</strong></p>
            <a href="https://github.com/x-wineytulips" target="_blank" style="text-decoration: none; margin: 0 10px;">🐙 GitHub</a>
            <a href="https://www.linkedin.com/in/sudhamrita-dey-06455a327/" target="_blank" style="text-decoration: none; margin: 0 10px;">🔗 LinkedIn</a>
            <a href="mailto:sudhamritadey1709@gmail.com" style="text-decoration: none; margin: 0 10px;">📩 Contact</a>
        </div>
    """, unsafe_allow_html=True)



elif choice == "🩸 Kidney Disease":
    st.markdown(
    '<h1 style="color:#0ea5e9; font-family:Arial; text-align:center;">🩺 Kidney Disease Prediction</h1>',
    unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Input Form
    with st.form("kidney_form", clear_on_submit=False):
        st.subheader("Enter Kidney Patient Health Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=48)
            bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=80)
            sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.05, value=1.02)
            al = st.number_input("Albumin", min_value=0, max_value=5, value=1)
            su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
            pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])

        with col2:
            ba = st.selectbox("Bacteria", ["present", "notpresent"])
            bgr = st.number_input("Blood Glucose Random", min_value=0, value=121)
            bu = st.number_input("Blood Urea", min_value=0, value=36)
            sc = st.number_input("Serum Creatinine", min_value=0.0, value=1.2)
            sod = st.number_input("Sodium", min_value=0, value=138)
            pot = st.number_input("Potassium", min_value=0.0, value=4.35)
            hemo = st.number_input("Hemoglobin", min_value=0.0, value=15.4)
            pcv = st.number_input("Packed Cell Volume", min_value=0, value=44)

        with col3:
            wc = st.number_input("White Blood Cell Count", min_value=0, value=7800)
            rc = st.number_input("Red Blood Cell Count", min_value=0.0, value=5.2)
            htn = st.selectbox("Hypertension", ["yes", "no"])
            dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal Edema", ["yes", "no"])
            ane = st.selectbox("Anemia", ["yes", "no"])

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        input_dict = {
            'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc,
            'pcc': pcc, 'ba': ba, 'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
            'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc, 'htn': htn, 'dm': dm, 'cad': cad,
            'appet': appet, 'pe': pe, 'ane': ane
        }

        input_df = pd.DataFrame([input_dict])

        # Prediction
        pred_class = model1.predict(input_df)[0]
        pred_proba = model1.predict_proba(input_df)[0]
        class_labels = model1.classes_

        st.markdown("""---""")
        st.subheader("🎯 Prediction Result")
        st.write(f"**Predicted Result:** {pred_class}")

        if pred_class == "ckd":
            st.error("⚠️ Positive case detected.")
        else:
            st.success("✅ Negative case detected.")

        # Show probabilities as progress bar
        no_prob = pred_proba[class_labels.tolist().index("notckd")]
        yes_prob = pred_proba[class_labels.tolist().index("ckd")]

        progress_html = f"""
        <div style="display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden; font-weight: bold; color: white;">
            <div style="width: {no_prob*100}%; background-color: #2ECC40; display: flex; align-items: center; justify-content: center; font-size: 12px;">
                {no_prob*100:.0f}%
            </div>
            <div style="width: {yes_prob*100}%; background-color: #FF4136; display: flex; align-items: center; justify-content: center; font-size: 12px;">
                {yes_prob*100:.0f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        _render_motivational_quote("Kidney", pred_class == "ckd")
        _render_outcome_guidance(
            "Kidney",
            pred_class == "ckd",
            dos=["Maintain blood pressure within target range", "Reduce dietary salt and processed foods", "Stay hydrated and monitor urine output", "Follow up with regular renal function tests"],
            donts=["Avoid unnecessary NSAIDs and nephrotoxic drugs", "Don't delay evaluation for persistent swelling or fatigue"],
            specialists=["Nephrologist", "Renal Dietitian"]
        )

elif choice == "🧬 Liver Disease":
    st.markdown(
    '<h1 style="color:#0ea5e9; font-family:Arial; text-align:center;">🧪 Liver Disease Prediction</h1>',
    unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Input Form
    with st.form("liver_form", clear_on_submit=False):
        st.subheader("Enter Liver Patient Health Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=40)
            gender = st.radio("Gender", ["Male", "Female"])
            total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1)

        with col2:
            direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.5, step=0.1)
            alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=200, step=1)
            alamine_aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0, value=20, step=1)

        with col3:
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, value=25, step=1)
            albumin = st.number_input("Albumin", min_value=0.0, value=3.5, step=0.1)
            albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0, step=0.1)

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        # Prepare input dictionary
        input_dict = {
            "Age": age,
            "Gender": gender,  # Encode Gender
            "Total_Bilirubin": total_bilirubin,
            "Direct_Bilirubin": direct_bilirubin,
            "Alkaline_Phosphotase": alkaline_phosphotase,
            "Alamine_Aminotransferase": alamine_aminotransferase,
            "Aspartate_Aminotransferase": aspartate_aminotransferase,
            "Albumin": albumin,
            "Albumin_and_Globulin_Ratio": albumin_globulin_ratio
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # ---- Prediction ----
        pred_class = model2.predict(input_df)[0]
        pred_proba = model2.predict_proba(input_df)[0]
        class_labels = model2.classes_

        st.markdown("""---""")
        st.subheader("🎯 Prediction Result")
        st.write(f"**Predicted Result:** {pred_class}")

        if pred_class == "Yes":
            st.error("⚠️ Liver Disease Detected")
        else:
            st.success("✅ No Liver Disease Detected")

        # ---- Probabilities ----
        no_prob = pred_proba[class_labels.tolist().index("No")]
        yes_prob = pred_proba[class_labels.tolist().index("Yes")]

        progress_html = f"""
        <div style="display: flex; width: 100%; height: 22px; border-radius: 10px; overflow: hidden; font-weight: bold; color: white;">
            <div style="width: {no_prob*100}%; background-color: #2ECC40; display: flex; align-items: center; justify-content: center; font-size: 12px;">
                {no_prob*100:.0f}%
            </div>
            <div style="width: {yes_prob*100}%; background-color: #FF4136; display: flex; align-items: center; justify-content: center; font-size: 12px;">
                {yes_prob*100:.0f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        _render_motivational_quote("Liver", pred_class == "Yes")
        _render_outcome_guidance(
            "Liver",
            pred_class == "Yes",
            dos=["Limit alcohol intake", "Maintain a healthy weight and stay active", "Monitor medication effects with your doctor"],
            donts=["Avoid unverified supplements and herbal remedies", "Don't skip routine liver function monitoring"],
            specialists=["Hepatologist", "Gastroenterologist"]
        )


elif choice == "🧩 Parkinsons Disease":
    st.markdown(
    '<h1 style="color:#0ea5e9; font-family:Arial; text-align:center;">🧠 Parkinsons Disease Prediction</h1>',
    unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Input Form
    with st.form("parkinsons_form", clear_on_submit=False):
        st.subheader("Enter Parkinsons Patient Voice Measurements")

        col1, col2, col3 = st.columns(3)

        with col1:
            fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
            fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
            flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
            jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.6f")
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, format="%.6f")
            rap = st.number_input("MDVP:RAP", min_value=0.0, format="%.6f")
            ppq = st.number_input("MDVP:PPQ", min_value=0.0, format="%.6f")
            ddp = st.number_input("Jitter:DDP", min_value=0.0, format="%.6f")

        with col2:
            shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, format="%.6f")
            shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, format="%.6f")
            apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, format="%.6f")
            apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, format="%.6f")
            apq = st.number_input("MDVP:APQ", min_value=0.0, format="%.6f")
            dda = st.number_input("Shimmer:DDA", min_value=0.0, format="%.6f")
            nhr = st.number_input("NHR", min_value=0.0, format="%.6f")
            hnr = st.number_input("HNR", min_value=0.0, format="%.6f")

        with col3:
            rpde = st.number_input("RPDE", min_value=0.0, format="%.6f")
            dfa = st.number_input("DFA", min_value=0.0, format="%.6f")
            spread1 = st.number_input("spread1", format="%.6f")
            spread2 = st.number_input("spread2", format="%.6f")
            d2 = st.number_input("D2", min_value=0.0, format="%.6f")
            ppe = st.number_input("PPE", min_value=0.0, format="%.6f")

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        # Prepare input dataframe
        input_dict = {
            'MDVP:Fo(Hz)': fo, 'MDVP:Fhi(Hz)': fhi, 'MDVP:Flo(Hz)': flo,
            'MDVP:Jitter(%)': jitter_percent, 'MDVP:Jitter(Abs)': jitter_abs,
            'MDVP:RAP': rap, 'MDVP:PPQ': ppq, 'Jitter:DDP': ddp,
            'MDVP:Shimmer': shimmer, 'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': apq3, 'Shimmer:APQ5': apq5, 'MDVP:APQ': apq,
            'Shimmer:DDA': dda, 'NHR': nhr, 'HNR': hnr,
            'RPDE': rpde, 'DFA': dfa, 'spread1': spread1, 'spread2': spread2,
            'D2': d2, 'PPE': ppe
        }

        input_df3 = pd.DataFrame([input_dict])

        # Prediction
        pred_class = model3.predict(input_df3)[0]
        pred_proba = model3.predict_proba(input_df3)[0]
        class_labels = model3.classes_

        st.markdown("""---""")
        st.subheader("🎯 Prediction Result")
        st.write(f"**Predicted Result:** {pred_class}")

        if pred_class == "Yes":
            st.error("⚠️ Positive case detected.")
        else:
            st.success("✅ Negative case detected.")

        # Show probabilities
        no_prob = pred_proba[class_labels.tolist().index("No")]
        yes_prob = pred_proba[class_labels.tolist().index("Yes")]

        progress_html = f"""
        <div style="display: flex; width: 100%; height: 20px; border-radius: 10px; overflow: hidden; font-weight: bold; color: white;">
            <div style="width: {no_prob*100}%; background-color: #2ECC40; display: flex; align-items: center; justify-content: center; font-size: 12px;">
                {no_prob*100:.0f}%
            </div>
            <div style="width: {yes_prob*100}%; background-color: #FF4136; display: flex; align-items: center; justify-content: center; font-size: 12px;">
                {yes_prob*100:.0f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        _render_motivational_quote("Parkinsons", pred_class == "Yes")
        _render_outcome_guidance(
            "Parkinsons",
            pred_class == "Yes",
            dos=["Keep moving with regular balance and strength exercises", "Practice stress-reducing routines and good sleep habits", "Follow your neurologist's treatment plan closely"],
            donts=["Don't avoid physical therapy if it is recommended", "Avoid high-fall-risk activities without support"],
            specialists=["Neurologist (Movement Disorders)", "Physiotherapist", "Occupational Therapist"]
        )
