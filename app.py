# app.py
import streamlit as st
import pandas as pd
import joblib
import time

# PAGE CONFIG 
st.set_page_config(page_title="Know Your Car", page_icon="🏎️", layout="wide")

# GLOBAL CSS 
st.markdown(
    """
<style>

 /* ANIMATED MOVING GRADIENT BACKGROUND */
.stApp {
    background: linear-gradient(120deg, #0f172a, #032b3a, #00414d, #032b3a, #0f172a);
    background-size: 500% 500%;
    animation: gradientMove 22s ease infinite;
    color: #e2e8f0;
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating glow overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: -20%;
    left: -20%;
    width: 150%;
    height: 150%;
    background: radial-gradient(circle at 30% 30%, rgba(34,197,94,0.18), transparent 60%),
                radial-gradient(circle at 70% 70%, rgba(56,189,248,0.15), transparent 60%);
    animation: floatGlow 14s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: -1;
}

@keyframes floatGlow {
    from { transform: translateY(0px) rotate(0deg); }
    to   { transform: translateY(40px) rotate(4deg); }
}

/* Glass effect wrapper */
.block-container {
    backdrop-filter: blur(6px) saturate(160%);
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    padding-top: 20px;
    padding-bottom: 20px;
}

/* Form label visibility */
label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label {
    color: #f6fafc !important;
    font-weight: 900 !important;
    font-size: 1.02rem !important;
}

/* Input box visibility */
input, select, textarea {
    background: rgba(255,255,255,0.96) !important;
    color: #0a1c1c !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.18) !important;
}
::placeholder { color: rgba(0,0,0,0.45) !important; }

/* Bold green buttons (all types) */
.stButton>button,
.stForm button {
    background: linear-gradient(135deg,#22c55e,#16a34a) !important;
    font-weight: 900 !important;
    font-size: 1.05rem !important;
    color: white !important;
    padding: 10px 22px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(16,185,129,0.40) !important;
    transition: 0.18s ease-in-out !important;
}
.stButton>button:hover,
.stForm button:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 14px 40px rgba(16,185,129,0.65) !important;
}

/* Navbar visuals */
.nav-row { display:flex; justify-content:flex-end; gap:8px; margin-bottom:12px; align-items:center; }
.nav-svg { vertical-align: middle; margin-right:6px; }
.nav-label { color: #e7fff0; font-weight:700; font-size:0.95rem; }

/* Result box */
.result-box {
    background: linear-gradient(185deg, rgba(16,185,129,0.16), rgba(16,185,129,0.07));
    padding: 24px;
    border-radius: 16px;
    text-align:center;
    border: 1px solid rgba(16,185,129,0.25);
}

</style>
""",
    unsafe_allow_html=True,
)

# NAVBAR (SVG icons shown, Streamlit buttons used for action) 
# We'll render the SVG icons visually using st.markdown, and place Streamlit buttons next to them for interaction.
nav_cols = st.columns([1, 0.3, 0.3, 0.3])  # spacer + three small columns for buttons

# Left spacer (nav_cols[0]) left intentionally blank for alignment
# Home visual
home_svg_md = """<span class="nav-label"><svg class="nav-svg" width="18" height="18" fill="#22c55e" viewBox="0 0 24 24"><path d="M12 3l9 8h-3v9h-12v-9h-3z"/></svg> Home</span>"""
about_svg_md = """<span class="nav-label"><svg class="nav-svg" width="18" height="18" fill="#22c55e" viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 15h-2v-6h2zm0-8h-2V7h2z"/></svg> About</span>"""
help_svg_md = """<span class="nav-label"><svg class="nav-svg" width="18" height="18" fill="#22c55e" viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 15h-2v-2h2zm1.07-7.75-.9.92A3.479 3.479 0 0 0 12 13h-2v-.5a4.49 4.49 0 0 1 1.34-3.22l1.24-1.26A1.5 1.5 0 1 0 10.5 6H8a4 4 0 1 1 6.07 3.25z"/></svg> Help</span>"""

# Render visuals and buttons
with nav_cols[1]:
    st.markdown(home_svg_md, unsafe_allow_html=True)
    if st.button("Home"):
        st.session_state["page"] = "home"
with nav_cols[2]:
    st.markdown(about_svg_md, unsafe_allow_html=True)
    if st.button("About"):
        st.session_state["page"] = "about"
with nav_cols[3]:
    st.markdown(help_svg_md, unsafe_allow_html=True)
    if st.button("Help"):
        st.session_state["page"] = "help"

# Ensure page in state
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# LOAD MODEL 
MODEL_PATH = "train_model.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("❌ Could not load model. Make sure 'train_model.pkl' exists in the app folder.")
    st.stop()

# Page helpers 
def show_about():
    st.header("About Know Your Car")
    st.write("""
        This app predicts used-car selling prices using a Machine Learning model trained
        on a realistic noisy dataset (20k rows). The model is a linear-regression pipeline.
    """)
    st.info("Tip: use dropdowns for Brand / City / Owner to match the training categories.")

def show_help():
    st.header("Help")
    st.write("""
        - Fill inputs (use dropdowns to avoid unseen categories)
        - Click Predict price (button below)
        - If you see ₹0 or a tiny value, retrain or check categories
    """)

#  MAIN: HOME PAGE 
if st.session_state["page"] == "home":
    st.title("🚘 Know Your Car — Price Estimator")
    st.write("Enter car details to get an estimated selling price.")

    with st.form("car_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_year = st.number_input("Model year", 1990, 2035, 2017)
            km_driven = st.number_input("Kilometers driven", 0, 500000, 30000)
            engine = st.number_input("Engine (CC)", 600, 6000, 1200)
        with col2:
            max_power = st.number_input("Max power (bhp)", 20, 500, 80)
            torque_nm = st.number_input("Torque (Nm)", 50, 1000, 150)
            condition_score = st.slider("Condition score (1–10)", 1.0, 10.0, 7.0)
        with col3:
            brand = st.selectbox("Brand", ["Maruti Suzuki","Hyundai","Honda","Toyota","Tata","Mahindra","Ford","Renault","Volkswagen"])
            car_name = st.text_input("Car name", "Swift")
            fuel = st.selectbox("Fuel", ["Petrol","Diesel","CNG","LPG"])
            transmission = st.selectbox("Transmission", ["Manual","Automatic"])
            owner = st.selectbox("Owner", ["First Owner","Second Owner","Third Owner"])
            city = st.selectbox("City", ["Mumbai","Delhi","Bengaluru","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad"])
            seats = st.selectbox("Seats", [4,5,6,7])
            seller_type = st.selectbox("Seller type", ["Dealer","Individual"])

        submitted = st.form_submit_button("Predict price")

    age = 2025 - model_year
    input_df = pd.DataFrame([{
        "km_driven": km_driven,
        "engine": engine,
        "max_power": max_power,
        "torque_nm": torque_nm,
        "condition_score": condition_score,
        "age": age,
        "brand": brand,
        "car_name": car_name,
        "fuel": fuel,
        "transmission": transmission,
        "owner": owner,
        "city": city,
        "seats": seats,
        "seller_type": seller_type,
    }])

    st.subheader("Preview of Inputs")
    st.dataframe(input_df, use_container_width=True)

    if submitted:
        with st.spinner("Predicting..."):
            time.sleep(0.4)
            price_lakh = model.predict(input_df)[0]
            price_rupees = price_lakh * 1

        st.balloons()
        st.markdown(
            f"""
            <div class="result-box">
                <h3>Estimated Selling Price</h3>
                <div style="font-size:24px; font-weight:900;">₹ {price_lakh:.2f} lakh</div>
                <div style="color:#b5ffd4; font-size:16px;">(₹ {price_rupees:,.0f})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

#  ABOUT / HELP PAGES
elif st.session_state["page"] == "about":
    show_about()
elif st.session_state["page"] == "help":
    show_help()


