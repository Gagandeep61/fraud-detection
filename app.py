from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load model artifacts on startup ──────────────────────────────────────────
# These load ONCE when the server starts, not on every request
# Loading on every request would make the API extremely slow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("Loading model artifacts...")
model = joblib.load(os.path.join(MODELS_DIR, 'fraud_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
threshold = joblib.load(os.path.join(MODELS_DIR, 'best_threshold.pkl'))
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
print(f"✅ Model loaded. Optimal threshold: {threshold:.4f}")

# ── Health check endpoint ─────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    """
    Simple endpoint to confirm the API is running.
    Visit http://localhost:5000/health in your browser to test.
    """
    return jsonify({
        "status": "running",
        "model": "XGBoost Fraud Detector",
        "threshold": round(float(threshold), 4)
    })

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a JSON payload of transaction features.
    Returns prediction and fraud probability.
    
    Expected input format:
    {
        "V1": -1.35, "V2": -0.07, ..., "V28": -0.02,
        "Amount": 149.62,
        "Time": 406.0
    }
    """
    try:
        # Step 1: Get the incoming data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Step 2: Convert to DataFrame
        # We use a list with one dictionary to create a single-row DataFrame
        input_df = pd.DataFrame([data])
        
        # Step 3: Apply the same feature engineering as preprocessing
        # CRITICAL: Must match exactly what was done in 02_preprocessing.ipynb
        input_df['Amount_Log'] = np.log1p(input_df['Amount'])
        input_df['Hour'] = (input_df['Time'] / 3600) % 24
        input_df['Hour_Sin'] = np.sin(2 * np.pi * input_df['Hour'] / 24)
        input_df['Hour_Cos'] = np.cos(2 * np.pi * input_df['Hour'] / 24)
        
        # Step 4: Drop columns we dropped during preprocessing
        input_df = input_df.drop(['Time', 'Amount', 'Hour'], axis=1)
        
        # Step 5: Ensure columns are in the exact same order as training
        # Column order matters for the model — wrong order = wrong predictions
        input_df = input_df[feature_names]
        
        # Step 6: Scale using the SAME scaler fitted on training data
        input_scaled = scaler.transform(input_df)
        
        # Step 7: Get fraud probability
        fraud_probability = model.predict_proba(input_scaled)[0][1]
        # [0] gets the first (only) row
        # [1] gets the probability of class 1 (fraud)
        
        # Step 8: Apply tuned threshold to make final decision
        prediction = "fraud" if fraud_probability >= threshold else "legitimate"
        
        # Step 9: Calculate risk level for better UX in Streamlit
        if fraud_probability >= 0.8:
            risk_level = "HIGH"
        elif fraud_probability >= 0.5:
            risk_level = "MEDIUM"
        elif fraud_probability >= 0.3:
            risk_level = "LOW"
        else:
            risk_level = "VERY LOW"
        
        return jsonify({
            "prediction": prediction,
            "fraud_probability": round(float(fraud_probability), 4),
            "fraud_probability_pct": f"{fraud_probability*100:.2f}%",
            "risk_level": risk_level,
            "threshold_used": round(float(threshold), 4),
            "message": (
                "⚠️ Transaction flagged as potentially fraudulent"
                if prediction == "fraud"
                else "✅ Transaction appears legitimate"
            )
        })
    
    except KeyError as e:
        return jsonify({
            "error": f"Missing feature in input: {str(e)}",
            "required_features": feature_names
        }), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Sample data endpoint ──────────────────────────────────────────────────────
@app.route('/sample', methods=['GET'])
def get_sample():
    """
    Returns a sample transaction for testing.
    Call this to get a valid input format for /predict.
    """
    sample = {
        "Time": 406.0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }
    return jsonify({
        "sample_transaction": sample,
        "instructions": "POST this to /predict to get a prediction"
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000)