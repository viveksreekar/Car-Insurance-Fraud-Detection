"""
app.py — Pure ML core for Car Insurance Fraud Detection.

Contains all model-loading, prediction, Grad-CAM, gridded heatmap,
damage-explanation, and LIME functions.  No Streamlit imports here.
"""

# import tensorflow as tf  <-- Moved to local imports in functions
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import joblib

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "saved_model/fraud_detector.h5"
PART_MODEL_PATH = "saved_model/part_classifier.h5"
XGB_MODEL_PATH = "saved_model/xgboost_risk_model.joblib"
IMG_SIZE    = 224
CLASS_NAMES = ["Fraud", "Non-Fraud"]
GRID_N      = 8

_PART_CLASSES = {0: "Back", 1: "Front", 2: "Left Side", 3: "Right Side", 4: "Unclassified"}

# Enhanced part mapping based on orientation
_LOCATION_MAPPING = {
    "Front": {
        (0, 0): "top-left (hood/A-pillar)", (0, 1): "hood center", (0, 2): "top-right (hood/A-pillar)",
        (1, 0): "front-left (headlight/fender)", (1, 1): "front-center (grille/bumper)", (1, 2): "front-right (headlight/fender)",
        (2, 0): "lower-front-left (air intake/fog light)", (2, 1): "lower-front-center (splitter/undertray)", (2, 2): "lower-front-right"
    },
    "Back": {
        (0, 0): "top-left (rear window/pillar)", (0, 1): "trunk/boot lid", (0, 2): "top-right (rear window/pillar)",
        (1, 0): "rear-left (taillight/quarter panel)", (1, 1): "rear-center (license plate/trunk)", (1, 2): "rear-right (taillight)",
        (2, 0): "lower-rear-left (bumper/exhaust)", (2, 1): "lower-rear-center (diffuser)", (2, 2): "lower-rear-right"
    },
    "Left Side": {
        (0, 0): "front-left roof line", (0, 1): "center roof line", (0, 2): "rear-left roof line",
        (1, 0): "front-left door/fender", (1, 1): "center-left doors (B-pillar)", (1, 2): "rear-left door/quarter panel",
        (2, 0): "front-left wheel/skirt", (2, 1): "center-left side skirt", (2, 2): "rear-left wheel/skirt"
    },
    "Right Side": {
        (0, 0): "front-right roof line", (0, 1): "center roof line", (0, 2): "rear-right roof line",
        (1, 0): "front-right door/fender", (1, 1): "center-right doors (B-pillar)", (1, 2): "rear-right door/quarter panel",
        (2, 0): "front-right wheel/skirt", (2, 1): "center-right side skirt", (2, 2): "rear-right wheel/skirt"
    },
    "Unclassified": {
        (0, 0): "top-left area", (0, 1): "top-center", (0, 2): "top-right",
        (1, 0): "left side", (1, 1): "central body", (1, 2): "right side",
        (2, 0): "bottom-left", (2, 1): "bottom-center", (2, 2): "bottom-right"
    }
}


# ─────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────
def load_fraud_model(path=MODEL_PATH):
    """Load and return the Keras model from disk."""
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def load_part_model(path=PART_MODEL_PATH):
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(path)
    except:
        return None

def load_xgboost_model(path=XGB_MODEL_PATH):
    """Load the tabular risk model."""
    try:
        return joblib.load(path)
    except:
        return None


# ─────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────
def predict(pil_image, model):
    """Return raw sigmoid score (float) for one PIL image."""
    import tensorflow as tf
    img = pil_image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = tf.keras.preprocessing.image.img_to_array(img)
    batch = np.expand_dims(arr, axis=0)
    return float(model.predict(batch, verbose=0)[0][0])


def predict_orientation(pil_image, model):
    """Return string label of car orientation (Front, Back, etc)."""
    if model is None: return "Unclassified"
    import tensorflow as tf
    img = pil_image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = tf.keras.preprocessing.image.img_to_array(img)
    batch = np.expand_dims(arr, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(preds))
    return _PART_CLASSES.get(idx, "Unclassified")


# ─────────────────────────────────────────────────────────────────
# Risk Scoring Engine
# ─────────────────────────────────────────────────────────────────
def calculate_risk_score(claim_data, customer_data, ai_detected_part=None):
    """
    Calculates a 0-100 Fraud Risk Score using a hybrid approach:
    1. XGBoost Tabular probability
    2. Business Penalty Rules
    """
    score = 0
    rules_triggered = []
    
    # Safely handle missing datasets
    claim_data = claim_data or {}
    customer_data = customer_data or {}
    
    # 1. XGBoost Base Score
    xgb_model = load_xgboost_model()
    if xgb_model:
        severity_map = {"Minor": 0, "Moderate": 0, "Major": 1}
        try:
            features = pd.DataFrame([{
                'claim_delay_days': int(claim_data.get('claim_delay_days') or 0),
                'past_claims_count': int(customer_data.get('past_claims_count') or 0),
                'fir_filed': 1 if claim_data.get('fir_filed') else 0,
                'damage_severity': severity_map.get(claim_data.get('damage_severity', "Minor"), 0),
                'policy_age_days': int(claim_data.get('policy_age_days') or 0),
            }])
            prob = xgb_model.predict_proba(features)[0][1]
            xgb_score = int(prob * 50) # Max 50 points
            score += xgb_score
            rules_triggered.append(f"AI Tabular Risk (XGBoost): +{xgb_score} points ({prob:.0%} probability)")
        except Exception as e:
            rules_triggered.append(f"XGBoost Analysis Skipped (Error: {e})")
    
    # 2. Rule: FIR Mismatch
    if claim_data.get('damage_severity') == "Major" and not claim_data.get('fir_filed'):
        score += 30
        rules_triggered.append("Major Damage with NO FIR: +30 points")
        
    # 3. Rule: Claim Delay
    delay = int(claim_data.get('claim_delay_days') or 0)
    if delay > 14:
        score += 30
        rules_triggered.append("Severe reporting delay (>14 days): +30 points")
    elif delay > 7:
        score += 20
        rules_triggered.append("Moderate reporting delay (>7 days): +20 points")
    elif delay > 3:
        score += 10
        rules_triggered.append("Minor reporting delay (>3 days): +10 points")

    # 4. Rule: AI Damage Location Match (The string mismatch bug fix)
    claimed_part = str(claim_data.get('accident_type', '')).lower() 
    if ai_detected_part:
        ai_part_lower = ai_detected_part.lower()
        mismatch = False
        
        # Don't penalize if AI is unsure
        if ai_part_lower != 'unclassified':
            # Safely check intersections
            if 'front' in ai_part_lower and 'front' not in claimed_part: mismatch = True
            elif 'back' in ai_part_lower and 'rear' not in claimed_part and 'back' not in claimed_part: mismatch = True
            elif 'side' in ai_part_lower and 'side' not in claimed_part: mismatch = True
            
            if mismatch:
                score += 50
                rules_triggered.append(f"AI Discrepancy (Claimed: {claimed_part.title()} vs Detected: {ai_detected_part}): +50 points")

    # Final Cap at 100
    final_score = min(score, 100)
    
    # Risk Level
    level = "Low"
    if final_score >= 70: level = "High"
    elif final_score >= 31: level = "Medium"
    
    return final_score, level, rules_triggered

# ─────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────
def _get_last_conv_layer(model):
    import tensorflow as tf
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D,
                               tf.keras.layers.Activation)):
            return layer.name
    raise ValueError("No convolutional layer found.")


def generate_gradcam(model, pil_image):
    """
    Returns (blended_pil, heatmap_full).
    blended_pil  — PIL Image overlay
    heatmap_full — np.ndarray (H, W) in [0, 1]
    """
    import tensorflow as tf
    img = pil_image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    arr = tf.keras.preprocessing.image.img_to_array(img)
    batch = np.expand_dims(arr, axis=0)

    conv_name = _get_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(conv_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        inp = tf.cast(batch, tf.float32)
        conv_out, preds = grad_model(inp)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    hm_np = heatmap.numpy()

    hm_full = np.array(
        Image.fromarray(np.uint8(255 * hm_np)).resize(
            (pil_image.width, pil_image.height), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    cmap = cm.get_cmap("jet")
    hm_color = (cmap(hm_full)[:, :, :3] * 255).astype(np.uint8)
    orig = np.array(pil_image.convert("RGB"))
    blended = (0.45 * hm_color + 0.55 * orig).astype(np.uint8)

    return Image.fromarray(blended), hm_full


# ─────────────────────────────────────────────────────────────────
# Gridded Heatmap
# ─────────────────────────────────────────────────────────────────
def plot_gridded_heatmap(heatmap_arr, grid_n=GRID_N):
    """Return (matplotlib Figure, pandas DataFrame) for the NxN grid."""
    h, w = heatmap_arr.shape
    ch, cw = h // grid_n, w // grid_n

    grid = np.zeros((grid_n, grid_n), dtype=np.float32)
    for i in range(grid_n):
        for j in range(grid_n):
            grid[i, j] = float(
                heatmap_arr[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].mean()
            )

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    im = ax.imshow(grid, cmap="jet", vmin=0, vmax=1, aspect="equal")

    for i in range(grid_n):
        for j in range(grid_n):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if v < 0.55 else "black",
                    fontweight="bold")

    ax.set_xticks(np.arange(-0.5, grid_n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(grid_n))
    ax.set_xticklabels([f"C{j+1}" for j in range(grid_n)], fontsize=8, color="white")
    ax.set_yticks(np.arange(grid_n))
    ax.set_yticklabels([f"R{i+1}" for i in range(grid_n)], fontsize=8, color="white")
    ax.tick_params(colors="white")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention (0=low → 1=high)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title("Grad-CAM Attention Grid", fontsize=10, fontweight="bold",
                 color="white", pad=10)
    plt.tight_layout()

    cols = [f"Col {j+1}" for j in range(grid_n)]
    rows = [f"Row {i+1}" for i in range(grid_n)]
    df = pd.DataFrame(np.round(grid, 3), index=rows, columns=cols)
    return fig, df


# ─────────────────────────────────────────────────────────────────
# Natural-Language Damage Explanation
# ─────────────────────────────────────────────────────────────────
def generate_damage_explanation(heatmap_arr, prediction_class, confidence, orientation="Unclassified", claim_data=None, risk_score=None, rules_triggered=None):
    h, w = heatmap_arr.shape
    ch, cw = h // 3, w // 3
    scores = {}
    for i in range(3):
        for j in range(3):
            scores[(i, j)] = float(
                heatmap_arr[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].mean()
            )
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    high = [(r, v) for r, v in ranked if v >= 0.60]
    mid  = [(r, v) for r, v in ranked if 0.35 <= v < 0.60]

    # Get orientation-specific mapping
    parts_map = _LOCATION_MAPPING.get(orientation, _LOCATION_MAPPING["Unclassified"])

    lines = []
    
    # Header
    if prediction_class == "Fraud":
        lines.append(f"🚨 **Fraud Detected — {confidence:.1%} confidence**\n")
        lines.append("The model identified visual patterns resembling conflicting or fraudulent claims.")
    else:
        lines.append(f"✅ **Legitimate Claim — {confidence:.1%} confidence**\n")
        lines.append("No significant fraud indicators found in the visual analysis.")

    # 1. Customer Form Context (NEW)
    if claim_data:
        claimed_part = claim_data.get('accident_type', 'Unknown')
        severity = claim_data.get('damage_severity', 'Unknown')
        lines.append(f"\n**📝 Customer Claim Form Context:**")
        lines.append(f"- **Claimed Damage Area:** {claimed_part}")
        lines.append(f"- **Reported Severity:** {severity}")

    # 2. AI Visual Match
    lines.append(f"\n**📸 Captured Angle (AI Detected):** {orientation}")
    lines.append("\n**📍 Damage Location Analysis:**")
    if high:
        parts = ", ".join(parts_map.get(r, "unidentified") for r, _ in high[:3])
        lines.append(f"- **Primary damage detected in**: {parts}.")
    if mid:
        parts = ", ".join(parts_map.get(r, "unidentified") for r, _ in mid[:3])
        lines.append(f"- **Secondary damage detected in**: {parts}.")
    if not high and not mid:
        lines.append("- Minor or well-distributed surface damage.")

    # 3. Risk Matrix Integration (NEW)
    if risk_score is not None and rules_triggered is not None:
        lines.append(f"\n**🚩 Risk Scoring Matrix (Score: {risk_score}/100):**")
        if not rules_triggered:
            lines.append("- No penalty rules triggered. High consistency.")
        else:
            for rule in rules_triggered:
                lines.append(f"- {rule}")
                
    # 4. Final Verdict
    lines.append("\n**🔍 Summary:**")
    if prediction_class == "Fraud" or (risk_score and risk_score > 30):
        lines.append(
            "- Inconsistencies found between claim data and physical evidence.\n"
            "- Potential rule violations or AI discrepancies flagged."
        )
        lines.append("\n> 🔴 **Action:** Flag for manual claims investigator review.")
    else:
        lines.append(
            "- Damage aligns with expected impact physics.\n"
            "- Form details match the visual AI evidence.\n"
            "- No visual manipulation detected."
        )
        lines.append("\n> 🟢 **Action:** Standard processing / Auto-approval recommended.")
        
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# LIME Explainability
# ─────────────────────────────────────────────────────────────────
def generate_lime_explanation(model, pil_image, num_samples=300):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import tensorflow as tf
    
    img = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))

    def batch_predict(images):
        imgs = np.array(images, dtype=np.float32)
        s = model.predict(imgs, verbose=0)[:, 0]
        return np.column_stack([1 - s, s])

    exp = lime_image.LimeImageExplainer()
    explanation = exp.explain_instance(
        img, batch_predict, top_labels=1, hide_color=0,
        num_samples=num_samples, random_seed=42,
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False, num_features=6, hide_rest=False,
    )
    bnd = mark_boundaries(temp.astype(np.uint8), mask,
                          color=(0, 1, 0), outline_color=(0, 0, 0))
    return Image.fromarray((bnd * 255).astype(np.uint8)).resize(
        (pil_image.width, pil_image.height), Image.BILINEAR
    )
