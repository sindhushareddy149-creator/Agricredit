"""
AgriCredit — Agricultural Loan Risk Assessment System
Flask backend — MongoDB replaces final_dataset.xlsx
ML model (model.pkl) updated: single rf regressor + threshold-based risk label
SHAP Explainable AI included
"""

from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import joblib
import pandas as pd
import numpy as np
import os
import math
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─────────────────────────────────────────
# MongoDB Connection
# ─────────────────────────────────────────
client   = MongoClient("mongodb://localhost:27017/")
mongo_db = client["farmer_loan_db"]

users_collection   = mongo_db["users"]
farmers_collection = mongo_db["farmers"]
weather_collection = mongo_db["weather"]
history_collection = mongo_db["predictions"]
config_col         = mongo_db["config"]

print("✅ Connected to MongoDB — farmer_loan_db")

# ─────────────────────────────────────────
# Load ML Model
# New model.pkl structure:
#   "model"               → RandomForestRegressor (score)
#   "features"            → list of feature names
#   "high_risk_threshold" → float
#   "low_risk_threshold"  → float
# REMOVED: "rf_reg", "rf_clf", "weights"
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading ML model...")
model_bundle        = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
rf_model            = model_bundle["model"]           # ← single regressor (was rf_reg)
features            = model_bundle["features"]
HIGH_RISK_THRESHOLD = float(model_bundle["high_risk_threshold"])
LOW_RISK_THRESHOLD  = float(model_bundle["low_risk_threshold"])

# WEIGHTS are no longer stored in model.pkl; define locally for SHAP fallback
WEIGHTS = {
    "repayment": 0.35,
    "soil":      0.25,
    "weather":   0.25,
    "market":    0.15,
}

print(f"✅ ML Model loaded — High={HIGH_RISK_THRESHOLD:.2f}, Low={LOW_RISK_THRESHOLD:.2f}")

# ─────────────────────────────────────────
# Market Score from MongoDB config
# ─────────────────────────────────────────
cfg = config_col.find_one({"key": "market_score"})
MARKET_SCORE = float(cfg["value"]) if cfg else 72.5
print(f"Market Score: {MARKET_SCORE}")

# ─────────────────────────────────────────
# Load SHAP (UNCHANGED)
# ─────────────────────────────────────────
print("Loading SHAP explainer...")
try:
    import shap
    shap_explainer = shap.TreeExplainer(rf_model)   # ← uses rf_model (single regressor)
    SHAP_AVAILABLE = True
    print("✅ SHAP loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed — run: pip install shap")


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def safe(val, default=""):
    try:
        if val is None:
            return default
        if isinstance(val, float) and math.isnan(val):
            return default
        return val
    except Exception:
        return default

SOIL_EFFICIENCY = {
    "Clay": 0.95, "Loamy": 1.00, "Peaty": 0.90,
    "Saline": 0.70, "Sandy": 0.85, "Silt": 0.95,
    "Black Cotton Soil": 0.95, "Red Laterite Soil": 0.85,
    "Alluvial Soil": 1.00, "Sandy Loam": 0.85,
    "Clay Loam": 0.95, "Saline Soil": 0.70,
}

def score_nitrogen(n):
    n = float(n)
    return (n / 110) * 100 if n <= 110 else max(0, 100 - (n - 110) * (15 / 11))

def score_phosphorus(p):
    p = float(p)
    return (p / 55) * 100 if p <= 55 else max(0, 100 - (p - 55) * (30 / 11))

def score_potassium(k):
    k = float(k)
    return (k / 45) * 100 if k <= 45 else max(0, 100 - (k - 45) * (10 / 3))

def compute_soil_health(n, p, k, soil_type):
    sn = score_nitrogen(n)
    sp = score_phosphorus(p)
    sk = score_potassium(k)
    nutrient_score = (sn + sp + sk) / 3
    efficiency = SOIL_EFFICIENCY.get(soil_type, 0.90)
    return round(nutrient_score * efficiency, 4), round(sn, 4), round(sp, 4), round(sk, 4)


# ─────────────────────────────────────────
# Weather Score — MongoDB
# ─────────────────────────────────────────
def get_weather_score(year):
    year = int(year)
    record = weather_collection.find_one({"year": year})
    if record:
        return round(float(record["weather_score"]), 4)
    future = list(weather_collection.find(
        {"year": {"$gte": 2026}},
        {"weather_score": 1, "_id": 0}
    ))
    if future:
        return round(float(np.mean([r["weather_score"] for r in future])), 4)
    return 90.0


# ─────────────────────────────────────────
# Past Repayment Score — MongoDB
# ─────────────────────────────────────────
def get_past_repayment_score(farmer_id, year):
    try:
        fid = int(farmer_id)
        yr  = int(year)
        record = farmers_collection.find_one(
            {"farmer_id": fid, "year": yr},
            {"past_repayment_score": 1}
        )
        if record and "past_repayment_score" in record:
            return round(float(record["past_repayment_score"]), 4)
        latest = farmers_collection.find_one(
            {"farmer_id": fid},
            {"past_repayment_score": 1},
            sort=[("year", -1)]
        )
        if latest and "past_repayment_score" in latest:
            return round(float(latest["past_repayment_score"]), 4)
    except Exception as e:
        print(f"Repayment lookup error: {e}")

    all_scores = list(farmers_collection.find(
        {"past_repayment_score": {"$exists": True}},
        {"past_repayment_score": 1, "_id": 0}
    ))
    if all_scores:
        vals = [r["past_repayment_score"] for r in all_scores]
        return round(float(np.median(vals)), 4)
    return 65.0


# ─────────────────────────────────────────
# ML Prediction
# UPDATED: uses rf_model (single regressor) + threshold-based risk label
# REMOVED: rf_clf.predict(), risk_label_pred
# ─────────────────────────────────────────
def run_ml_prediction(repayment, soil, weather, market):
    X = pd.DataFrame(
        [[repayment, soil, weather, market]],
        columns=features
    )
    combined_score = float(rf_model.predict(X)[0])

    # Threshold-based classification (replaces rf_clf)
    if combined_score >= LOW_RISK_THRESHOLD:
        risk_level = "Low"
    elif combined_score >= HIGH_RISK_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return round(combined_score, 2), risk_level


# ─────────────────────────────────────────
# SHAP Explanation
# UPDATED: shap_explainer uses rf_model (single regressor)
# ─────────────────────────────────────────
def get_score_label(feature, value):
    if feature == "Past_Repayment_Score":
        if value >= 75:   return "Strong",   "positive"
        elif value >= 50: return "Moderate", "neutral"
        else:             return "Poor",     "negative"
    elif feature == "Soil_Health_Score":
        if value >= 75:   return "Excellent", "positive"
        elif value >= 50: return "Moderate",  "neutral"
        else:             return "Poor",      "negative"
    elif feature == "Weather_Score":
        if value >= 88:   return "Favorable",   "positive"
        elif value >= 75: return "Moderate",    "neutral"
        else:             return "Unfavorable", "negative"
    elif feature == "Market_Score":
        if value >= 70:   return "Stable",   "positive"
        elif value >= 55: return "Moderate", "neutral"
        else:             return "Weak",     "negative"
    return "N/A", "neutral"

FEATURE_DISPLAY = {
    "Past_Repayment_Score": "Repayment History",
    "Soil_Health_Score":    "Soil Health",
    "Weather_Score":        "Weather Conditions",
    "Market_Score":         "Market Conditions",
}
FEATURE_ICONS = {
    "Past_Repayment_Score": "💳",
    "Soil_Health_Score":    "🌱",
    "Weather_Score":        "🌦️",
    "Market_Score":         "📈",
}

def compute_shap_explanation(repayment, soil, weather, market):
    X = pd.DataFrame([[repayment, soil, weather, market]], columns=features)
    if SHAP_AVAILABLE:
        shap_vals = np.array(shap_explainer.shap_values(X)).flatten()
        base_val  = float(np.array(shap_explainer.expected_value).flatten()[0])
    else:
        weights  = [WEIGHTS["repayment"], WEIGHTS["soil"], WEIGHTS["weather"], WEIGHTS["market"]]
        vals     = [repayment, soil, weather, market]
        avg_vals = [65.0, 70.0, 85.0, MARKET_SCORE]
        shap_vals = [w * (v - a) for w, v, a in zip(weights, vals, avg_vals)]
        base_val  = 65.0

    contributions = []
    for i, feat in enumerate(features):
        val      = [repayment, soil, weather, market][i]
        shap_val = float(shap_vals[i])
        label, sentiment = get_score_label(feat, val)
        if shap_val > 0.3:
            direction, arrow, color = "increased", "↑", "green"
        elif shap_val < -0.3:
            direction, arrow, color = "decreased", "↓", "red"
        else:
            direction, arrow, color = "neutral",   "→", "gray"
        contributions.append({
            "feature":      feat,
            "display_name": FEATURE_DISPLAY[feat],
            "icon":         FEATURE_ICONS[feat],
            "value":        round(val, 1),
            "shap_value":   round(shap_val, 3),
            "abs_impact":   round(abs(shap_val), 3),
            "direction":    direction,
            "arrow":        arrow,
            "color":        color,
            "label":        label,
            "sentiment":    sentiment,
            "sentence": (
                f"{FEATURE_ICONS[feat]} {FEATURE_DISPLAY[feat]} is {label} "
                f"({val:.1f}/100) — {direction} your score by {abs(shap_val):.2f} points {arrow}"
            )
        })
    contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
    return contributions, round(base_val, 2)


def build_xai_summary(contributions, risk_level, combined_score):
    top    = contributions[0]
    second = contributions[1]
    risk_phrases = {"High": "flagged as HIGH RISK", "Medium": "assessed as MEDIUM RISK", "Low": "assessed as LOW RISK"}
    summary = (
        f"Your application has been {risk_phrases.get(risk_level, risk_level)} "
        f"with a credit score of {combined_score:.1f}. "
        f"The most influential factor was {top['display_name']} "
        f"({top['value']:.1f}/100 — {top['label']}), which {top['direction']} "
        f"your score by {top['abs_impact']:.2f} points. "
        f"The second key factor was {second['display_name']} "
        f"({second['value']:.1f}/100 — {second['label']}), which {second['direction']} "
        f"your score by {second['abs_impact']:.2f} points. "
    )
    if risk_level == "High":
        worst = [c for c in contributions if c["direction"] == "decreased"]
        if worst:
            summary += (f"Primary reason: {worst[0]['display_name']} is {worst[0]['label']}. "
                        "Improving this factor will significantly improve your risk level.")
    elif risk_level == "Medium":
        summary += (f"You are close to Low Risk. Small improvements in "
                    f"{contributions[0]['display_name']} could move you to a better risk tier.")
    elif risk_level == "Low":
        summary += "All factors are performing well. You qualify for priority loan processing."
    return summary


def build_improvement_tips(contributions, risk_level):
    tips = []
    for c in contributions:
        if c["feature"] == "Past_Repayment_Score":
            if c["value"] < 50:
                tips.append({"icon":"💳","priority":"High","title":"Improve Repayment History",
                    "text":f"Your repayment score is {c['value']:.1f}/100. Repaying your next 2-3 loans on time will significantly boost your credit score."})
            elif c["value"] < 70:
                tips.append({"icon":"💳","priority":"Medium","title":"Maintain Repayment Consistency",
                    "text":f"Repayment score {c['value']:.1f}/100 — moderate. Avoid late payments in the next loan cycle."})
            else:
                tips.append({"icon":"✅","priority":"Good","title":"Strong Repayment History",
                    "text":f"Excellent repayment score {c['value']:.1f}/100. Keep maintaining timely payments."})
        elif c["feature"] == "Soil_Health_Score":
            if c["value"] < 50:
                tips.append({"icon":"🌱","priority":"High","title":"Improve Soil Nutrients",
                    "text":f"Soil Health {c['value']:.1f}/100 — poor. Apply balanced NPK fertilizers. Ideal: N=110, P=55, K=45 mg/kg."})
            elif c["value"] < 75:
                tips.append({"icon":"🌱","priority":"Medium","title":"Optimize Soil Nutrients",
                    "text":f"Soil Health {c['value']:.1f}/100. Consider soil testing and targeted fertilization."})
            else:
                tips.append({"icon":"✅","priority":"Good","title":"Healthy Soil Conditions",
                    "text":f"Excellent Soil Health {c['value']:.1f}/100. Maintain organic matter practices."})
        elif c["feature"] == "Weather_Score":
            if c["value"] < 75:
                tips.append({"icon":"🌦️","priority":"Medium","title":"Mitigate Weather Risk",
                    "text":f"Weather Score {c['value']:.1f}/100 — unfavorable. Consider drip irrigation or crop insurance."})
            else:
                tips.append({"icon":"☀️","priority":"Good","title":"Favorable Weather Conditions",
                    "text":f"Weather Score {c['value']:.1f}/100. Plan crop calendar carefully to maximize yield."})
    if risk_level == "High":
        tips.append({"icon":"⚠️","priority":"High","title":"Consider Micro-Finance First",
            "text":"Start with a smaller micro-finance loan to rebuild credit before applying for a larger agricultural loan."})
    elif risk_level == "Medium":
        tips.append({"icon":"📋","priority":"Medium","title":"Conditional Loan Available",
            "text":"You may qualify for a loan with reduced limit or additional collateral. Discuss with your bank officer."})
    return tips


def build_decision_justification(combined_score, risk_level, contributions,
                                  repayment_score, soil_health, weather_score, market_score):
    positive_factors = [c for c in contributions if c["direction"] == "increased"]
    negative_factors = [c for c in contributions if c["direction"] == "decreased"]
    top_positive = positive_factors[0] if positive_factors else None
    top_negative = negative_factors[0] if negative_factors else None

    if risk_level == "Low":
        decision, decision_code = "Approved", "APPROVED"
        headline   = "Loan application approved — all risk parameters within acceptable range."
        color_code = "green"
        primary_reason = f"Combined credit score {combined_score:.1f} exceeds the Low Risk threshold of {LOW_RISK_THRESHOLD:.1f}."
        supporting_reasons = []
        if repayment_score >= 75: supporting_reasons.append(f"Strong repayment history ({repayment_score:.1f}/100) demonstrates consistent loan discipline.")
        if soil_health >= 75:     supporting_reasons.append(f"Excellent soil health ({soil_health:.1f}/100) indicates high productivity potential.")
        if weather_score >= 88:   supporting_reasons.append(f"Favorable weather ({weather_score:.1f}/100) reduces crop failure risk.")
        if market_score >= 70:    supporting_reasons.append(f"Stable market ({market_score:.1f}/100) supports adequate crop price realization.")
        if not supporting_reasons: supporting_reasons.append("All assessed parameters are within acceptable thresholds.")
        next_action  = "Application qualifies for priority processing with potential for enhanced loan limits and preferential interest rates."
        risk_warning = None
        margin_to_next = None

    elif risk_level == "Medium":
        decision, decision_code = "Conditionally Approved", "CONDITIONAL"
        headline   = "Loan application is conditionally approved — some risk factors require attention."
        color_code = "amber"
        margin_to_next = round(LOW_RISK_THRESHOLD - combined_score, 1)
        primary_reason = (f"Score {combined_score:.1f} falls between High ({HIGH_RISK_THRESHOLD:.1f}) "
                          f"and Low ({LOW_RISK_THRESHOLD:.1f}) Risk thresholds.")
        supporting_reasons = []
        if top_negative: supporting_reasons.append(f"{top_negative['display_name']} ({top_negative['value']:.1f}/100) is pulling score down by {top_negative['abs_impact']:.2f} points.")
        if top_positive: supporting_reasons.append(f"{top_positive['display_name']} ({top_positive['value']:.1f}/100) is a positive factor supporting partial approval.")
        next_action  = f"Only {margin_to_next:.1f} points away from Low Risk. May qualify for 60-70% of requested amount or need additional collateral."
        risk_warning = f"Improving {contributions[0]['display_name'].lower()} by 10-15 points could move you to Low Risk."

    else:  # High
        decision, decision_code = "Not Approved", "REJECTED"
        headline   = "Loan application could not be approved — significant risk factors identified."
        color_code = "red"
        margin_to_next = round(HIGH_RISK_THRESHOLD - combined_score, 1)
        primary_reason = f"Score {combined_score:.1f} is below minimum threshold of {HIGH_RISK_THRESHOLD:.1f}."
        supporting_reasons = [f"{c['display_name']} is {c['label']} ({c['value']:.1f}/100), reducing score by {c['abs_impact']:.2f} points." for c in negative_factors[:2]]
        if not supporting_reasons: supporting_reasons.append("Multiple factors are below required thresholds for loan approval.")
        next_action  = "Steps: (1) Apply for Kisan Credit Card micro-loan to rebuild credit. (2) Improve factors listed below. (3) Reapply after 6-12 months."
        risk_warning = f"Need score improvement of {margin_to_next:.1f} points to reach minimum approval threshold."

    return {
        "decision": decision, "decision_code": decision_code,
        "headline": headline, "color_code": color_code,
        "primary_reason": primary_reason, "supporting_reasons": supporting_reasons,
        "next_action": next_action, "risk_warning": risk_warning,
        "margin_to_next": margin_to_next, "score_used": round(combined_score, 1),
        "threshold_high": round(HIGH_RISK_THRESHOLD, 1),
        "threshold_low":  round(LOW_RISK_THRESHOLD, 1),
    }


def build_suggestions(soil_score, weather_score, repayment_score, market_score, crop, risk_level):
    tips = []
    tips.append({"icon":"🌍","text":"Soil nutrient levels are below optimal. Apply balanced NPK fertilisers before the next sowing season."} if soil_score < 65
           else {"icon":"✅","text":"Soil nutrients are healthy. Maintain current organic matter practices to sustain productivity."})
    tips.append({"icon":"📉","text":f"Market outlook for {crop or 'current crop'} is moderate. Consider diversifying into higher-value crops."} if market_score < 65
           else {"icon":"📈","text":f"{crop or 'Your crop'} shows a favourable market trend. Explore forward contracts to lock in better prices."})
    tips.append({"icon":"🌦️","text":"Seasonal forecasts indicate irregular rainfall risk. Invest in drip irrigation to reduce weather dependency."} if weather_score < 88
           else {"icon":"☀️","text":"Weather conditions are favourable. Plan your crop calendar carefully to leverage the good season."})
    tips.append({"icon":"💳","text":"Repayment history shows some defaults. Supplementary income from dairy or poultry could strengthen your credit profile."} if repayment_score < 60
           else {"icon":"🤝","text":"Strong repayment history. Timely EMI payments will further boost your agricultural credit score."})
    if risk_level == "High":
        tips.append({"icon":"⚠️","text":"Consider a smaller micro-finance loan first to rebuild your credit profile before reapplying."})
    return tips


def build_explanation(combined_score, risk_level, soil_score, weather_score,
                      repayment_score, market_score, crop, year):
    wt = WEIGHTS
    weather_type = "predicted" if int(year) >= 2026 else "historical"
    if risk_level == "Low":
        return (f"The ML model evaluated four weighted dimensions: repayment ({wt['repayment']*100:.0f}%), "
                f"soil health ({wt['soil']*100:.0f}%), weather ({wt['weather']*100:.0f}%), "
                f"market ({wt['market']*100:.0f}%). Soil Score: {soil_score:.1f}, "
                f"Weather Score: {weather_score:.1f} ({weather_type}), Market: {market_score:.1f}. "
                f"Composite score {combined_score:.1f} exceeds Low Risk threshold {LOW_RISK_THRESHOLD:.1f}. Loan approval recommended.")
    elif risk_level == "Medium":
        return (f"Mixed signals detected. Repayment: {repayment_score:.1f}, Soil: {soil_score:.1f}, "
                f"Weather: {weather_score:.1f} ({weather_type}), Market: {market_score:.1f}. "
                f"Composite score {combined_score:.1f} is between thresholds. "
                f"Conditional approval — consider reduced loan limit or additional collateral.")
    else:
        return (f"Multiple risk factors flagged. Repayment: {repayment_score:.1f}, Soil: {soil_score:.1f}, "
                f"Weather: {weather_score:.1f} ({weather_type}), Market: {market_score:.1f}. "
                f"Composite score {combined_score:.1f} is below High Risk threshold {HIGH_RISK_THRESHOLD:.1f}. "
                f"Loan not recommended. Improve soil quality and resolve repayment issues before reapplying.")


# ═══════════════════════════════════════════════════════
#   ROUTES
# ═══════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json(force=True)
        farmer_id  = data.get("farmer_id", "0")
        year       = int(data.get("year", 2024))
        soil_type  = data.get("soil_type", "Clay")
        nitrogen   = float(data.get("nitrogen",   80))
        phosphorus = float(data.get("phosphorus", 40))
        potassium  = float(data.get("potassium",  35))
        crop       = data.get("crop", "")

        # Scores
        soil_health, score_n, score_p, score_k = compute_soil_health(nitrogen, phosphorus, potassium, soil_type)
        weather_score   = get_weather_score(year)
        weather_type    = "Predicted" if year >= 2026 else "Historical"
        repayment_score = get_past_repayment_score(farmer_id, year)
        market_score    = MARKET_SCORE

        # ML — threshold-based risk label (no rf_clf)
        combined_score, risk_level = run_ml_prediction(repayment_score, soil_health, weather_score, market_score)

        # SHAP
        contributions, base_value = compute_shap_explanation(repayment_score, soil_health, weather_score, market_score)
        xai_summary  = build_xai_summary(contributions, risk_level, combined_score)
        improve_tips = build_improvement_tips(contributions, risk_level)
        justification = build_decision_justification(combined_score, risk_level, contributions,
                                                      repayment_score, soil_health, weather_score, market_score)
        suggestions = build_suggestions(soil_health, weather_score, repayment_score, market_score, crop, risk_level)
        explanation = build_explanation(combined_score, risk_level, soil_health, weather_score,
                                        repayment_score, market_score, crop, year)

        # Save to MongoDB predictions
        history_collection.insert_one({
            "farmer_id":       farmer_id,
            "year":            year,
            "soil_score":      soil_health,
            "weather_score":   weather_score,
            "repayment_score": repayment_score,
            "market_score":    market_score,
            "final_score":     combined_score,
            "risk_level":      risk_level,
            "crop":            crop,
            "soil_type":       soil_type,
            "npk":             {"N": nitrogen, "P": phosphorus, "K": potassium}
        })

        return jsonify({
            "success": True,
            "scores": {
                "soil":      round(soil_health, 1),
                "weather":   round(weather_score, 1),
                "repayment": round(repayment_score, 1),
                "market":    round(market_score, 1),
                "final":     round(combined_score, 1),
            },
            "npk_breakdown": {"score_n": round(score_n,1), "score_p": round(score_p,1), "score_k": round(score_k,1)},
            "risk_level":    risk_level,
            "weather_type":  weather_type,
            "thresholds":    {"high": round(HIGH_RISK_THRESHOLD,2), "low": round(LOW_RISK_THRESHOLD,2)},
            "suggestions":   suggestions,
            "explanation":   explanation,
            "xai": {
                "summary":        xai_summary,
                "base_value":     base_value,
                "contributions":  contributions,
                "improve_tips":   improve_tips,
                "shap_available": SHAP_AVAILABLE,
            },
            "justification": justification,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/farmer/auth", methods=["POST"])
def farmer_login():
    body      = request.get_json(force=True)
    farmer_id = body.get("farmer_id", "")
    password  = str(body.get("password", "")).strip()

    try:
        fid = int(farmer_id)
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Farmer ID must be a number."}), 400

    farmer = farmers_collection.find_one({"farmer_id": fid}, sort=[("year", -1)])
    if not farmer:
        return jsonify({"success": False, "error": f"Farmer ID {fid} not found."}), 404

    expected_pw = str(farmer.get("password", f"farmer{fid}@agri"))
    if password != expected_pw:
        return jsonify({"success": False, "error": "Incorrect password."}), 401

    return jsonify({
        "success":   True,
        "farmer_id": fid,
        "farmer": {
            "Farmer_ID":       fid,
            "name":            farmer.get("name",            f"Farmer #{fid}"),
            "annual_income":   farmer.get("annual_income",   0),
            "land_size_acres": farmer.get("land_size_acres", 0.0),
            "Soil_Type":       farmer.get("soil_type",       ""),
            "Combined_Score":  farmer.get("combined_score",  0.0),
            "Risk_Label":      farmer.get("risk_label",      ""),
        }
    })


@app.route("/api/farmer/<int:farmer_id>", methods=["GET"])
def get_farmer(farmer_id):
    farmer = farmers_collection.find_one({"farmer_id": farmer_id}, {"_id": 0}, sort=[("year", -1)])
    if not farmer:
        return jsonify({"error": f"Farmer ID {farmer_id} not found.",
                        "total_farmers": farmers_collection.count_documents({})}), 404
    return jsonify({
        "Farmer_ID":       farmer_id,
        "name":            farmer.get("name",            f"Farmer #{farmer_id}"),
        "annual_income":   farmer.get("annual_income",   0),
        "land_size_acres": farmer.get("land_size_acres", 0.0),
        "Soil_Type":       farmer.get("soil_type",       ""),
        "Combined_Score":  farmer.get("combined_score",  0.0),
        "Risk_Label":      farmer.get("risk_label",      ""),
    })


@app.route("/api/weather", methods=["GET"])
def get_weather():
    records = list(weather_collection.find({}, {"_id": 0}).sort("year", 1))
    return jsonify({"success": True, "data": records})


@app.route("/api/history/<farmer_id>", methods=["GET"])
def get_history(farmer_id):
    records = list(history_collection.find({"farmer_id": farmer_id}, {"_id": 0}).sort("year", -1).limit(10))
    return jsonify({"success": True, "count": len(records), "history": records})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "database":      "MongoDB connected",
        "farmers_in_db": farmers_collection.count_documents({}),
        "weather_years": weather_collection.count_documents({}),
        "shap":          SHAP_AVAILABLE,
        "model":         "model.pkl loaded"
    })


if __name__ == "__main__":
    print("\n🌾 AgriCredit Backend starting...")
    print("📡 API: http://localhost:5000")
    print("🗄️  Database: MongoDB (farmer_loan_db)\n")
    app.run(debug=True, port=5000)