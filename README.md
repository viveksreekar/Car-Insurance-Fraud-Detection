# 🛡️ AutoShield — Car Insurance Fraud Detection

An AI-powered, end-to-end **Insurance Fraud Detection Platform** built with **Deep Learning**, **XGBoost**, and **Streamlit**. AutoShield assists insurance companies in identifying fraudulent claims by combining computer vision analysis on vehicle damage photos with tabular risk scoring — all through a clean, role-based web interface.

---

## 🚀 Features

### 👤 Customer Portal
- New customer registration with vehicle & policy details
- File damage claims with photo uploads
- Real-time claim tracking with visual progress bar

### 🏢 Employee / Investigator Portal
- Claim queue management dashboard
- One-click **AI Fraud Analysis** per claim
- Approve, Reject, or Settle claims with remarks

### 🤖 AI & ML Engine
- **Deep Learning (CNN / MobileNet)** — Classifies damage photos as Fraud or Legitimate
- **Grad-CAM Heatmaps** — Visual explanation of *where* the model is looking
- **8×8 Attention Grid** — Quantified damage location breakdown
- **Car Orientation Classifier** — Detects Front / Back / Left Side / Right Side
- **LIME Explainability** — Superpixel-based visual explanation
- **XGBoost Risk Scorer** — Tabular fraud scoring from claim metadata
- **Hybrid Risk Score (0–100)** — Combines AI output with business rule triggers

### 📏 Business Rule Engine
| Rule | Penalty |
|------|---------|
| Major damage with no FIR filed | +30 pts |
| Claim delay > 14 days | +30 pts |
| Claim delay 7–14 days | +20 pts |
| Claim delay 3–7 days | +10 pts |
| AI-detected part ≠ claimed damage area | +50 pts |
| XGBoost tabular probability | Up to +50 pts |

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Tabular ML | XGBoost, Scikit-learn |
| Explainability | LIME, Grad-CAM |
| Database | SQLite |
| Image Processing | Pillow, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📂 Project Structure

```
AutoShield/
├── app.py                  # ML core (models, Grad-CAM, LIME, risk scoring)
├── streamlit_ui.py         # Streamlit UI (role-based portal)
├── database.py             # SQLite database operations
├── train_model.py          # CNN model training script
├── train_xgboost.py        # XGBoost risk model training
├── evaluate_model.py       # Model evaluation utilities
├── verify_head2.py         # Model architecture verification
├── requirements.txt        # Python dependencies
├── saved_model/
│   ├── fraud_detector.h5         # Trained CNN model
│   ├── part_classifier.h5        # Car orientation classifier
│   └── xgboost_risk_model.joblib # XGBoost model
├── data/
│   ├── training/
│   │   ├── Fraud/          # Fraud training images
│   │   └── Non-Fraud/      # Legitimate training images
│   └── claims_images/      # Uploaded claim photos (auto-created)
└── clean part/
    ├── classify_and_sort.py
    ├── clean_db.py
    └── train_part_classifier.py
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/viveksreekar/Car-Insurance-Fraud-Detection.git
cd Car-Insurance-Fraud-Detection
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run streamlit_ui.py
```

The app opens at **http://localhost:8501** 🚀

---

## 🖥️ Usage

### As a Customer
1. Click **"I am a Customer"** on the landing page
2. Sign up with your phone number and password
3. Fill in your vehicle and policy details
4. Submit a new damage claim with photos
5. Track your claim status in real-time

### As an Employee / Investigator
1. Click **"I am an Employee"**
2. Login with:
   - **ID:** `admin`
   - **Password:** `1234`
3. View the pending claim queue
4. Click **"Run AI Analysis"** to trigger fraud detection
5. Review heatmaps, risk score, and LIME explanations
6. Approve or Reject the claim

---

## 🧠 How the AI Works

```
📸 Customer uploads damage photo
         ↓
🔍 CNN classifies: Fraud / Not Fraud
         ↓
🗺️ Grad-CAM highlights suspicious regions
         ↓
📊 8×8 grid quantifies damage zones
         ↓
🚗 Orientation model detects car angle
         ↓
📋 XGBoost scores tabular risk (delay, FIR, history)
         ↓
⚖️ Business rules applied (5 penalty checks)
         ↓
🚨 Final Risk Score (0–100) with decision support
```

---

## 📊 Risk Levels

| Score | Level | Action |
|-------|-------|--------|
| 0–30 | 🟢 Low | Auto-approve recommended |
| 31–69 | 🟡 Medium | Manual review suggested |
| 70–100 | 🔴 High | Flag for SIU investigation |

---

## 📦 Dependencies

```
tensorflow
numpy
matplotlib
streamlit
Pillow
scikit-learn
scikit-image
seaborn
lime
pandas
xgboost
joblib
```

---

## 👨‍💻 Author

**Vivek Sreekar**  
GitHub: [@viveksreekar](https://github.com/viveksreekar)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
