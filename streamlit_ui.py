"""
streamlit_ui.py — AutoShield platform v3.0 (Pure Python / Streamlit).
Role-based portal with Hybrid Risk Scoring Engine (XGBoost + Business Rules).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import date, datetime
import json
import os

# ── Project imports ──────────────────────────────────────────────
from app import (
    load_fraud_model, load_part_model, predict, predict_orientation,
    generate_gradcam, plot_gridded_heatmap, generate_damage_explanation,
    generate_lime_explanation, calculate_risk_score, CLASS_NAMES,
)
from database import (
    init_db, get_customer_by_phone, insert_customer,
    insert_claim, get_claims_for_customer, increment_past_claims,
    get_all_claims, get_all_customers, update_claim_analysis
)

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoShield — Insurance Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# Initialize database
init_db()

@st.cache_resource
def _load_models():
    f_model = None
    p_model = None
    try:
        f_model = load_fraud_model()
    except Exception as e:
        st.error(f"Models loading... (Wait for first run)")
    try:
        p_model = load_part_model()
    except:
        pass
    return f_model, p_model

model, part_model = _load_models()

# ─────────────────────────────────────────────────────────────────
# Session State & Navigation
# ─────────────────────────────────────────────────────────────────
if "role" not in st.session_state: st.session_state.role = None
if "user" not in st.session_state: st.session_state.user = None
if "page" not in st.session_state: st.session_state.page = "landing"

def go(page):
    st.session_state.page = page

def logout():
    st.session_state.role = None
    st.session_state.user = None
    st.session_state.page = "landing"
    st.rerun()

# ─────────────────────────────────────────────────────────────────
# Shared AI Helper
# ─────────────────────────────────────────────────────────────────
def _analyze_one_image(pil_image, idx, label="", claim_data=None, risk_score=None, rules_triggered=None):
    with st.spinner(f"Predicting…"):
        score = predict(pil_image, model)
        orientation = predict_orientation(pil_image, part_model)

    is_fraud = score < 0.5
    pred_cls = CLASS_NAMES[0] if is_fraud else CLASS_NAMES[1]
    conf     = (1 - score) if is_fraud else score

    c1, c2 = st.columns(2)
    with c1:
        st.image(pil_image, caption=label, use_column_width=True)
    with c2:
        if is_fraud:
            st.error(f"🚨 **{pred_cls} Detected**")
        else:
            st.success(f"✅ **{pred_cls}**")
        st.metric("AI Confidence", f"{conf:.2%}")
        st.progress(int(conf * 100))

    # Grad-CAM
    show_hm = st.toggle("Show Heatmap", value=True, key=f"hm_{idx}")
    heatmap_arr = None
    if show_hm:
        gradcam_img, heatmap_arr = generate_gradcam(model, pil_image)
        h1, h2 = st.columns(2)
        h1.image(gradcam_img, caption="Heatmap Overlay")
        with h2:
            fig, grid_df = plot_gridded_heatmap(heatmap_arr)
            st.pyplot(fig)
            plt.close(fig)
        
    st.divider()
    
    # Explainable AI Damage Report
    st.markdown("##### 🧠 Explainable AI — Damage Report")
    if heatmap_arr is not None:
        st.markdown(generate_damage_explanation(heatmap_arr, pred_cls, conf, orientation, claim_data, risk_score, rules_triggered))
    else:
        st.info("Damage report unavailable without heatmap analysis.")

    # LIME Visual Explanation
    st.markdown("##### 🔬 LIME Visual Explanation")
    st.caption("Green segments support fraud flag. Red segments argue against fraud.")
    with st.spinner("Running LIME (~20 s)…"):
        try:
            lime_img = generate_lime_explanation(model, pil_image)
            st.image(lime_img, caption="LIME Superpixel", use_column_width=True)
        except Exception as e:
            st.warning(f"LIME error: {e}")
            
    st.divider()
    return pred_cls, conf, orientation

def _save_claim_images(uploaded_files, customer_phone):
    base_dir = os.path.join("data", "claims_images", customer_phone)
    os.makedirs(base_dir, exist_ok=True)
    saved_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, file in enumerate(uploaded_files):
        ext = os.path.splitext(file.name)[1]
        path = os.path.join(base_dir, f"claim_{timestamp}_{i}{ext}")
        with open(path, "wb") as f: f.write(file.getbuffer())
        saved_paths.append(path)
    return json.dumps(saved_paths)

# ─────────────────────────────────────────────────────────────────
# 1. LANDING PAGE
# ─────────────────────────────────────────────────────────────────
def page_landing():
    st.title("🛡️ AutoShield Insurance")
    st.subheader("Smart AI Claims Platform")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.info("### 👤 For Customers")
        st.write("Register, file claims, and track status instantly.")
        if st.button("I am a Customer", use_container_width=True, type="primary"):
            st.session_state.role = "customer"
            go("customer_auth")
            st.rerun()
    with col2:
        st.warning("### 🏢 For Employees")
        st.write("Review claims, run AI diagnostics, and manage approvals.")
        if st.button("I am an Employee", use_container_width=True):
            st.session_state.role = "employee"
            go("employee_auth")
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# 2. CUSTOMER AUTH
# ─────────────────────────────────────────────────────────────────
def page_customer_auth():
    st.title("👤 Customer Access")
    mode = st.radio("Access Level", ["Existing Customer (Login)", "New Customer (Sign Up)"], horizontal=True)
    
    if "Existing" in mode:
        phone = st.text_input("Mobile Number", max_chars=10)
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = get_customer_by_phone(phone)
            if user and user["password"] == password:
                st.session_state.user = user
                go("customer_dashboard")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    else:
        phone = st.text_input("Mobile Number *", max_chars=10)
        password = st.text_input("Set Password *", type="password")
        if st.button("Register & Continue"):
            if len(phone) == 10 and password:
                if get_customer_by_phone(phone):
                    st.warning("Number already exists. Please Login.")
                else:
                    st.session_state.temp_phone = phone
                    st.session_state.temp_pw = password
                    go("customer_dashboard")
                    st.rerun()
            else:
                st.error("Enter valid 10-digit phone and password.")
    
    if st.button("← Back"): go("landing"); st.rerun()

# ─────────────────────────────────────────────────────────────────
# 3. EMPLOYEE AUTH
# ─────────────────────────────────────────────────────────────────
def page_employee_auth():
    st.title("🏢 Employee Login")
    user = st.text_input("ID", value="admin")
    pw = st.text_input("Password", type="password", value="1234")
    if st.button("Login"):
        if user == "admin" and pw == "1234":
            st.session_state.user = {"full_name": "Investigator", "id": "EMP1"}
            go("employee_dashboard")
            st.rerun()
        else:
            st.error("Incorrect details.")
    if st.button("← Back"): go("landing"); st.rerun()

# ─────────────────────────────────────────────────────────────────
# 4. CUSTOMER DASHBOARD
# ─────────────────────────────────────────────────────────────────
def page_customer_dashboard():
    user = st.session_state.user
    st.title("🛡️ Customer Portal")
    st.caption(f"Status: {'Registered' if user else 'Registration Pending'}")
    
    t1, t2, t3 = st.tabs(["🆕 Register", "📝 New Claim", "📈 Track"])
    
    with t1:
        if user:
            st.success(f"Verified Profile: {user['full_name']}")
            st.write(f"**Vehicle:** {user.get('vehicle_model', 'Unknown')} ({user.get('vehicle_type', 'Unknown')} - {user.get('vehicle_usage', 'Private')})")
            
            # Display Vehicle Age calculation
            v_age = user.get('vehicle_age', 0)
            st.write(f"**Vehicle Age:** {v_age} years old (calculated from manufacturing date)")
            st.write(f"**Policy Expiry:** {user.get('policy_end_date', 'N/A')}")
        else:
            phone_tmp = st.session_state.get("temp_phone", "")
            pw_tmp = st.session_state.get("temp_pw", "1234")
            _render_register_form(phone_tmp, pw_tmp)

    with t2:
        if not user: st.warning("Please register first.")
        else: _render_claim_form(user)

    with t3:
        if not user: st.info("No active claims.")
        else:
            claims = get_claims_for_customer(user["id"])
            for c in claims:
                color = "green" if "Approved" in c["status"] else "red" if "Rejected" in c["status"] else "blue"
                with st.expander(f"Claim #{c['id']} — {c.get('accident_type', 'Claim')} ({c.get('status', 'Unknown')})"):
                    # Simple Progress Bar logic
                    status_text = c.get('status', 'Unknown')
                    if "Submitted" in status_text:
                        st.progress(25, text="Track: 📝 1. Claim Submitted")
                    elif "Escalated" in status_text:
                        st.progress(50, text="Track: 🔍 2. Escalated to SIU (Under Review)")
                    elif "Approved" in status_text:
                        st.progress(75, text="Track: ✅ 1. Submitted ➔ 2. Reviewed ➔ 3. Approved (Pending Repair)")
                    elif "Settled" in status_text or "Closed" in status_text:
                        st.progress(100, text="Track: 🎉 4. Final Settlement Complete")
                    elif "Rejected" in status_text:
                        st.progress(100, text="Track: 🚫 Claim Rejected")
                        
                    st.write(f"**Current Status:** :{color}[{status_text}]")
                    if c.get("remarks"): st.info(f"**Employee Remarks:** {c['remarks']}")

def _render_register_form(phone_tmp, pw_tmp):
    with st.form("reg"):
        st.subheader("Registration Form")
        c1, c2 = st.columns(2)
        name = c1.text_input("Full Name *")
        age = c2.number_input("Age", 18, 100, 30)
        
        c3, c4 = st.columns(2)
        # Rename: Type -> What car do you use?
        v_type = c3.selectbox("What car do you use? *", ["SUV's", "Hatchbacks", "Sedans"])
        v_model = c4.text_input("Model Name *")
        
        c_new1, c_new2 = st.columns(2)
        v_usage = c_new1.selectbox("Usage Type *", ["Private", "Taxi / Commercial"])
        v_reg = c_new2.text_input("Registration Number *")
        
        c6, c7 = st.columns(2)
        manu = c6.date_input("Manufacturing Date", value=date(2020, 1, 1))
        p_start = c7.date_input("Policy Start Date")
        
        c8, c9 = st.columns(2)
        p_end = c8.date_input("Expiry Date *", value=date(2026, 1, 1))
        lic = c9.text_input("License Number *")
        
        if st.form_submit_button("Complete Registration", type="primary"):
            if name and v_model and v_reg and lic:
                insert_customer({
                    "full_name": name, "age": age, 
                    "phone": phone_tmp,
                    "password": pw_tmp,
                    "vehicle_type": v_type, "vehicle_model": v_model,
                    "vehicle_usage": v_usage,
                    "registration_number": v_reg.upper(), "manufacturing_date": manu.isoformat(),
                    "policy_start_date": p_start.isoformat(), "policy_end_date": p_end.isoformat(),
                    "license_number": lic.upper(), "vehicle_age": max(0, (date.today() - manu).days // 365)
                })
                st.session_state.user = get_customer_by_phone(phone_tmp)
                st.rerun()
            else: st.error("Fill all required fields.")

def _render_claim_form(user):
    with st.container(border=True):
        st.subheader("File New Claim")
        
        # Display the pre-filled vehicle context here
        with st.container(border=True):
            v1, v2, v3 = st.columns(3)
            v1.selectbox("What car do you use? *", ["SUV's", "Hatchbacks", "Sedans"], index=["SUV's", "Hatchbacks", "Sedans"].index(user.get("vehicle_type", "SUV's")) if user.get("vehicle_type", "SUV's") in ["SUV's", "Hatchbacks", "Sedans"] else 0, key='claim_ctype')
            v2.selectbox("Usage Type *", ["Private", "Taxi / Commercial"], index=["Private", "Taxi / Commercial"].index(user.get("vehicle_usage", "Private")) if user.get("vehicle_usage", "Private") in ["Private", "Taxi / Commercial"] else 0, key='claim_usage')
            v3.text_input("Model Name *", value=user.get("vehicle_model", ""), key="claim_model")
            
        c1, c2 = st.columns(2)
        acc_date = c1.date_input("Accident Date *")
        claim_date = c2.date_input("Claim Date", value=date.today())
        
        acc_type = st.selectbox("Damage Area *", ["Front", "Back", "Left Side", "Right Side", "Multiple Areas", "Glass", "Unknown"])
        c3, c4 = st.columns(2)
        # Severe -> Major
        severity = c3.selectbox("Severity *", ["Minor", "Major"])
        loc = c4.selectbox("Location", ["Urban", "Rural", "Highway"])
        
        fir_opt = "No"
        fir_path = ""
        if severity in ["Major"]:
            st.info("FIR is required for this severity.")
            fir_opt = st.radio("Is FIR Filed? *", ["Yes", "No"], horizontal=True)
            if fir_opt == "Yes":
                f = st.file_uploader("Upload FIR")
                if f: fir_path = f"data/fir/{f.name}"

        files = st.file_uploader("Upload Damage Photos *", accept_multiple_files=True)
        descs = []
        if files:
            for i, f in enumerate(files):
                descs.append(st.text_input(f"Describe {f.name}", key=f"d_{i}"))

        if st.button("Submit Claim to Review", type="primary"):
            if files:
                paths = _save_claim_images(files, user["phone"])
                
                # Calculate policy age safely
                try:
                    p_start = date.fromisoformat(user.get("policy_start_date", str(date.today())))
                    pol_age = max(0, (acc_date - p_start).days)
                except:
                    pol_age = 0
                
                insert_claim({
                    "customer_id": user["id"], "accident_date": acc_date.isoformat(),
                    "claim_date": claim_date.isoformat(), "fir_filed": (fir_opt == "Yes"),
                    "fir_file_path": fir_path, "damage_severity": severity,
                    "accident_type": acc_type, "location_type": loc,
                    "image_descriptions": json.dumps(descs), "image_paths": paths,
                    "status": "Submitted", "claim_delay_days": (claim_date - acc_date).days,
                    "policy_age_days": pol_age
                })
                st.success("Claim submitted successfully!")
                st.rerun()
            else: st.error("Photos are mandatory.")

# ─────────────────────────────────────────────────────────────────
# 5. EMPLOYEE DASHBOARD
# ─────────────────────────────────────────────────────────────────
def page_employee_dashboard():
    st.title("🏢 Business Dashboard")
    claims = get_all_claims()
    pending = [c for c in claims if c["status"] == "Submitted"]
    processed = [c for c in claims if c["status"] != "Submitted"]
    
    tab1, tab2 = st.tabs([f"🔔 Queue ({len(pending)})", f"📁 Processed Claims ({len(processed)})"])
    
    with tab1:
        st.subheader("Pending Requests")
        for c in pending:
            # Correct lookup
            all_c = get_all_customers()
            cust = next((cu for cu in all_c if cu["id"] == c["customer_id"]), None)
            
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{cust['full_name'] if cust else 'User'}** · Claim #{c['id']}")
                col1.caption(f"{c['accident_type']} · {c['damage_severity']} Severity")
                if col2.button("Run AI Analysis", key=f"p_{c['id']}"):
                    st.session_state.analyzing_claim = c
                    st.session_state.analyzing_cust = cust
                    go("employee_analysis")
                    st.rerun()
                
    with tab2:
        st.subheader("Archived & Processed")
        for c in processed:
            all_c = get_all_customers()
            cust = next((cu for cu in all_c if cu["id"] == c["customer_id"]), None)
            
            color = "green" if "Approved" in c.get("status", "") else "red"
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{cust['full_name'] if cust else 'User'}** · Claim #{c['id']}")
                col1.caption(f"Status: :{color}[{c.get('status', 'Unknown')}] · Verdict: {c.get('verdict', 'N/A')}")
                
                # Check status and add appropriate action buttons
                if "Approved" in c.get("status", ""):
                    # Give them two buttons: View Report and Mark Settled
                    v_col, s_col = col2.columns(2)
                    if v_col.button("View AI Report", key=f"view_emp_{c['id']}"):
                        st.session_state.viewing_claim = c
                        st.session_state.from_employee = True
                        go("view_report")
                        st.rerun()
                    if s_col.button("Settle Claim", type="primary", key=f"settle_{c['id']}"):
                        from database import update_claim_status
                        update_claim_status(c['id'], "Settled / Closed")
                        st.success("Claim officially closed.")
                        st.rerun()
                else:
                    if col2.button("View AI Report", key=f"view_emp_{c['id']}"):
                        st.session_state.viewing_claim = c
                        st.session_state.from_employee = True
                        go("view_report")
                        st.rerun()

# ─────────────────────────────────────────────────────────────────
# 6. EMPLOYEE ANALYSIS
# ─────────────────────────────────────────────────────────────────
def page_employee_analysis():
    if st.button("← Back to Dashboard"): 
        go("employee_dashboard")
        st.rerun()
        
    c = st.session_state.analyzing_claim
    cust = st.session_state.analyzing_cust
    st.title(f"🔍 Fraud Analysis: Claim #{c['id']}")
    
    paths = json.loads(c.get("image_paths") or "[]")
    preds, confs, orients = [], [], []
    
    # First calculate the Risk Score logic (we run the orientation model quickly on just the first image to feed the string logic)
    from app import predict_orientation
    base_orientation = "Unknown"
    if paths and os.path.exists(paths[0]):
        base_orientation = predict_orientation(Image.open(paths[0]), part_model)
    
    score, level, rules = calculate_risk_score(c, cust, ai_detected_part=base_orientation)

    for i, p in enumerate(paths):
        if os.path.exists(p):
            p_cls, conf, orient = _analyze_one_image(Image.open(p), i, f"Evidence {i+1}", claim_data=c, risk_score=score, rules_triggered=rules)
            preds.append(p_cls); confs.append(conf); orients.append(orient)
    
    st.divider()
    st.subheader("🚩 Risk Score Matrix (0-100)")
    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Score", f"{score}/100")
    r2.metric("Risk Level", level)
    r3.metric("AI Confidence", f"{np.mean(confs):.1%}")
    
    with st.expander("Show Rule Triggers"):
        for r in rules: st.write(f"- {r}")

    if score < 15: st.success("🌟 STP Eligible: High consistency detected.")

    st.subheader("Decision")
    rem = st.text_area("Adjustment Remarks")
    colA, colB = st.columns(2)
    if colA.button("✅ Approve Claim", type="primary", use_container_width=True):
        update_claim_analysis(c["id"], {
            "fraud_prediction": "Legit", "fraud_confidence": float(np.mean(confs)),
            "image_fraud_count": 0, "image_total_count": len(paths),
            "risk_score": score, "remarks": rem or "Approved.",
            "verdict": "LEGITIMATE", "status": "Approved - Pending Repair"
        })
        go("employee_dashboard"); st.rerun()
    if colB.button("🚫 Reject Claim", use_container_width=True):
        update_claim_analysis(c["id"], {
            "fraud_prediction": "Fraud", "fraud_confidence": float(np.mean(confs)),
            "image_fraud_count": len(paths), "image_total_count": len(paths),
            "risk_score": score, "remarks": rem or "Denied due to risk profile.",
            "verdict": "FRAUDULENT", "status": "Rejected"
        })
        go("employee_dashboard"); st.rerun()

# ─────────────────────────────────────────────────────────────────
# 7. VIEW REPORT
# ─────────────────────────────────────────────────────────────────
def page_view_report():
    c = st.session_state.viewing_claim
    st.title(f"📄 Claim Report #{c['id']}")
    st.write(f"**Final Verdict:** {c['verdict']}")
    st.write(f"**Remarks:** {c['remarks']}")
    st.divider()
    paths = json.loads(c.get("image_paths") or "[]")
    for i, p in enumerate(paths):
        if os.path.exists(p):
            st.image(Image.open(p), caption=f"Evidence {i+1}")
            
    if st.button("← Go Back"): 
        if st.session_state.get("from_employee"):
            st.session_state.from_employee = False
            go("employee_dashboard")
        else:
            go("customer_dashboard")
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR & ROUTER
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ AutoShield")
    if st.session_state.role:
        st.write(f"**Role:** {st.session_state.role}")
        if st.button("Logout"): logout()

pages = {
    "landing": page_landing, "customer_auth": page_customer_auth,
    "employee_auth": page_employee_auth, "customer_dashboard": page_customer_dashboard,
    "employee_dashboard": page_employee_dashboard, "employee_analysis": page_employee_analysis,
    "view_report": page_view_report
}
pages.get(st.session_state.page, page_landing)()
