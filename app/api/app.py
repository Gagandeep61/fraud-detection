from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import logging
import time
from datetime import datetime

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Logging Setup ─────────────────────────────────────────────────────────────
# Logs every request with timestamp — useful for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Load Model Artifacts on Startup ───────────────────────────────────────────
# These load ONCE when server starts, not on every request
# Loading on every request would make API extremely slow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("=" * 50)
print("Loading model artifacts...")
print("=" * 50)

try:
    model = joblib.load(os.path.join(MODELS_DIR, 'fraud_model.pkl'))
    print("✅ Model loaded")

    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("✅ Scaler loaded")

    threshold = joblib.load(os.path.join(MODELS_DIR, 'best_threshold.pkl'))
    print(f"✅ Threshold loaded: {threshold:.4f}")

    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    print(f"✅ Feature names loaded: {len(feature_names)} features")

    print("=" * 50)
    print("🚀 All artifacts loaded. API is ready.")
    print("=" * 50)

except FileNotFoundError as e:
    print(f"❌ Failed to load artifact: {e}")
    print("Make sure all .pkl files exist in the models/ folder")
    raise

# ── Track API Statistics ───────────────────────────────────────────────────────
# Counts total requests, fraud detections, and legitimate transactions
# Resets when server restarts — for persistent stats you'd use a database
api_stats = {
    "total_requests": 0,
    "fraud_detected": 0,
    "legitimate_detected": 0,
    "server_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "errors": 0
}

# ── Helper Functions ───────────────────────────────────────────────────────────
def calculate_risk_level(probability):
    """
    Converts a raw probability into a human-readable risk level.
    These thresholds are business decisions, not mathematical ones.
    """
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "VERY LOW"


def calculate_risk_color(risk_level):
    """
    Maps risk level to a color for frontend display.
    Streamlit can use these color names directly.
    """
    colors = {
        "CRITICAL": "red",
        "HIGH": "orange",
        "MEDIUM": "yellow",
        "LOW": "blue",
        "VERY LOW": "green"
    }
    return colors.get(risk_level, "gray")


def preprocess_input(data):
    """
    Applies the exact same preprocessing as the training pipeline.
    CRITICAL: Any mismatch here causes wrong predictions.
    
    Steps:
    1. Convert to DataFrame
    2. Log transform Amount (handles skewness)
    3. Cyclical encode Time (connects 23:00 to 01:00)
    4. Drop original Time, Amount, Hour columns
    5. Reorder columns to match training order
    6. Scale using fitted scaler
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Log transform Amount — same as preprocessing notebook
    # np.log1p handles zero safely (log(0) is undefined)
    input_df['Amount_Log'] = np.log1p(input_df['Amount'])

    # Convert seconds to hour of day
    input_df['Hour'] = (input_df['Time'] / 3600) % 24

    # Cyclical encoding — so model knows 23:00 is close to 01:00
    input_df['Hour_Sin'] = np.sin(2 * np.pi * input_df['Hour'] / 24)
    input_df['Hour_Cos'] = np.cos(2 * np.pi * input_df['Hour'] / 24)

    # Drop original columns — replaced by engineered versions
    input_df = input_df.drop(['Time', 'Amount', 'Hour'], axis=1)

    # Reorder columns to exactly match training feature order
    # Wrong order = wrong predictions, even with correct values
    input_df = input_df[feature_names]

    # Scale using the SAME scaler fitted on training data
    # Never refit the scaler on new data — that would be data leakage
    input_scaled = scaler.transform(input_df)

    return input_scaled


def validate_input(data):
    """
    Validates incoming request data before processing.
    Returns (is_valid, error_message).
    """
    if not data:
        return False, "No data provided in request body"

    # Check all required features are present
    required_v_features = [f'V{i}' for i in range(1, 29)]
    required_features = required_v_features + ['Amount', 'Time']

    missing = [f for f in required_features if f not in data]
    if missing:
        return False, {
            "error": "Missing required features",
            "missing_features": missing,
            "total_missing": len(missing),
            "hint": "Call GET /sample to see the correct input format"
        }

    # Check all values are numeric
    for feature in required_features:
        if not isinstance(data[feature], (int, float)):
            return False, {
                "error": f"Feature '{feature}' must be a number",
                "received_type": type(data[feature]).__name__,
                "received_value": str(data[feature])
            }

    # Check for NaN or infinite values
    for feature in required_features:
        val = data[feature]
        if val != val:  # NaN check (NaN != NaN is True)
            return False, {
                "error": f"Feature '{feature}' contains NaN (not a number)"
            }
        if abs(val) == float('inf'):
            return False, {
                "error": f"Feature '{feature}' contains infinite value"
            }

    # Business logic validation
    if data['Amount'] < 0:
        return False, {
            "error": "Amount cannot be negative",
            "received": data['Amount']
        }

    if data['Time'] < 0:
        return False, {
            "error": "Time cannot be negative",
            "received": data['Time']
        }

    return True, None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Tells you the API is alive and shows basic stats.
    Visit: http://localhost:5000/health
    """
    return jsonify({
        "status": "running",
        "model": "XGBoost Fraud Detector",
        "optimal_threshold": round(float(threshold), 4),
        "total_features": len(feature_names),
        "server_start_time": api_stats["server_start_time"],
        "uptime_note": "Stats reset on server restart"
    })


@app.route('/stats', methods=['GET'])
def stats():
    """
    Returns live API usage statistics.
    Shows how many predictions made, fraud detected etc.
    Visit: http://localhost:5000/stats
    """
    total = api_stats["total_requests"]
    fraud = api_stats["fraud_detected"]

    return jsonify({
        "total_predictions": total,
        "fraud_detected": fraud,
        "legitimate_detected": api_stats["legitimate_detected"],
        "fraud_detection_rate": f"{(fraud/total*100):.2f}%" if total > 0 else "N/A",
        "errors": api_stats["errors"],
        "server_start_time": api_stats["server_start_time"]
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Accepts JSON with all V1-V28 features plus Amount and Time.
    Returns prediction, probability, risk level and explanation.
    
    Example:
    POST http://localhost:5000/predict
    Body: {"V1": -1.35, ..., "V28": -0.02, "Amount": 149.62, "Time": 406.0}
    """
    # Track request start time for response time logging
    start_time = time.time()
    api_stats["total_requests"] += 1

    try:
        # Step 1: Get incoming JSON data
        data = request.get_json(force=True)
        # force=True means parse JSON even if Content-Type header is missing

        # Step 2: Validate input
        is_valid, error = validate_input(data)
        if not is_valid:
            api_stats["errors"] += 1
            return jsonify(error), 400

        # Step 3: Preprocess exactly like training pipeline
        input_scaled = preprocess_input(data)

        # Step 4: Get fraud probability from model
        # predict_proba returns [[prob_class0, prob_class1]]
        # [0][1] = first row, second class (fraud probability)
        fraud_probability = float(model.predict_proba(input_scaled)[0][1])

        # Step 5: Apply tuned threshold to make binary decision
        # Default 0.5 is arbitrary — our tuned threshold is better
        prediction = "fraud" if fraud_probability >= float(threshold) else "legitimate"

        # Step 6: Calculate risk metadata
        risk_level = calculate_risk_level(fraud_probability)
        risk_color = calculate_risk_color(risk_level)

        # Step 7: Update statistics
        if prediction == "fraud":
            api_stats["fraud_detected"] += 1
        else:
            api_stats["legitimate_detected"] += 1

        # Step 8: Calculate response time
        response_time_ms = round((time.time() - start_time) * 1000, 2)

        # Step 9: Log the prediction
        logger.info(
            f"Prediction: {prediction.upper()} | "
            f"Probability: {fraud_probability:.4f} | "
            f"Risk: {risk_level} | "
            f"Amount: ${data['Amount']:.2f} | "
            f"Response time: {response_time_ms}ms"
        )

        # Step 10: Return comprehensive response
        return jsonify({
            # Core prediction
            "prediction": prediction,
            "is_fraud": prediction == "fraud",

            # Probability details
            "fraud_probability": round(fraud_probability, 4),
            "fraud_probability_pct": f"{fraud_probability * 100:.2f}%",
            "legitimate_probability": round(1 - fraud_probability, 4),

            # Risk assessment
            "risk_level": risk_level,
            "risk_color": risk_color,

            # Threshold info
            "threshold_used": round(float(threshold), 4),
            "threshold_note": "Tuned via Precision-Recall curve to maximize F1",

            # Human readable message
            "message": (
                f"⚠️ FRAUD DETECTED — {risk_level} risk ({fraud_probability*100:.1f}% probability)"
                if prediction == "fraud"
                else f"✅ Transaction appears legitimate ({fraud_probability*100:.1f}% fraud probability)"
            ),

            # Transaction context
            "transaction_amount": data['Amount'],
            "amount_formatted": f"${data['Amount']:.2f}",

            # Performance
            "response_time_ms": response_time_ms
        })

    except KeyError as e:
        api_stats["errors"] += 1
        logger.error(f"KeyError: {str(e)}")
        return jsonify({
            "error": f"Missing feature in input: {str(e)}",
            "required_features": feature_names,
            "hint": "Call GET /sample to see the correct input format"
        }), 400

    except ValueError as e:
        api_stats["errors"] += 1
        logger.error(f"ValueError: {str(e)}")
        return jsonify({
            "error": f"Invalid value: {str(e)}"
        }), 400

    except Exception as e:
        api_stats["errors"] += 1
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "detail": str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint — predict multiple transactions at once.
    Accepts a list of transactions and returns predictions for all.
    
    Example:
    POST http://localhost:5000/predict/batch
    Body: [{"V1": -1.35, ..., "Amount": 149.62, "Time": 406.0},
           {"V1": 1.19, ..., "Amount": 2.69, "Time": 8736.0}]
    """
    try:
        data_list = request.get_json(force=True)

        if not isinstance(data_list, list):
            return jsonify({
                "error": "Batch endpoint expects a JSON array of transactions",
                "example": "[{transaction1}, {transaction2}]"
            }), 400

        if len(data_list) == 0:
            return jsonify({"error": "Empty list provided"}), 400

        if len(data_list) > 100:
            return jsonify({
                "error": "Batch size too large",
                "max_allowed": 100,
                "received": len(data_list)
            }), 400

        results = []
        fraud_count = 0

        for i, transaction in enumerate(data_list):
            # Validate each transaction
            is_valid, error = validate_input(transaction)
            if not is_valid:
                results.append({
                    "index": i,
                    "error": error,
                    "status": "validation_failed"
                })
                continue

            # Preprocess and predict
            input_scaled = preprocess_input(transaction)
            fraud_prob = float(model.predict_proba(input_scaled)[0][1])
            prediction = "fraud" if fraud_prob >= float(threshold) else "legitimate"
            risk_level = calculate_risk_level(fraud_prob)

            if prediction == "fraud":
                fraud_count += 1

            results.append({
                "index": i,
                "prediction": prediction,
                "fraud_probability": round(fraud_prob, 4),
                "risk_level": risk_level,
                "amount": transaction.get('Amount', 'N/A'),
                "status": "success"
            })

        # Update global stats
        api_stats["total_requests"] += len(data_list)
        api_stats["fraud_detected"] += fraud_count
        api_stats["legitimate_detected"] += (len(data_list) - fraud_count)

        return jsonify({
            "total_transactions": len(data_list),
            "fraud_detected": fraud_count,
            "legitimate_detected": len(data_list) - fraud_count,
            "fraud_rate": f"{fraud_count/len(data_list)*100:.2f}%",
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sample', methods=['GET'])
def get_sample():
    """
    Returns a sample legitimate transaction for testing.
    Use this to understand the required input format.
    Visit: http://localhost:5000/sample
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

    fraud_sample = {
        "Time": 406.0,
        "V1": -2.3122265423263,
        "V2": 1.95199201064158,
        "V3": -1.60985073229769,
        "V4": 3.9979055875468,
        "V5": -0.522187864667764,
        "V6": -1.42654531920595,
        "V7": -2.53738730624579,
        "V8": 1.39165724829804,
        "V9": -2.77008927719433,
        "V10": -2.77227214465915,
        "V11": 3.20203320709635,
        "V12": -2.89990738849473,
        "V13": -0.595221881324605,
        "V14": -4.28925378244217,
        "V15": 0.389724120274487,
        "V16": -1.14074717980657,
        "V17": -2.83005567450437,
        "V18": -0.0168224681808257,
        "V19": 0.416955705037907,
        "V20": 0.126910559061474,
        "V21": 0.517232370861764,
        "V22": -0.0350493686052974,
        "V23": -0.465211076944875,
        "V24": 0.320198197836216,
        "V25": 0.0445191674731724,
        "V26": 0.177839798284401,
        "V27": 0.261145002567677,
        "V28": -0.143275874698918,
        "Amount": 1.00
    }

    return jsonify({
        "legitimate_sample": sample,
        "fraud_sample": fraud_sample,
        "required_features": feature_names,
        "total_features_required": len(feature_names) + 2,
        "instructions": {
            "single": "POST either sample to /predict",
            "batch": "POST an array of transactions to /predict/batch",
            "note": "All V1-V28, Amount, and Time are required"
        }
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Returns detailed information about the trained model.
    Useful for understanding what the model is and how it works.
    Visit: http://localhost:5000/model/info
    """
    return jsonify({
        "model_type": "XGBoost Classifier",
        "problem_type": "Binary Classification (Fraud vs Legitimate)",
        "dataset": "Credit Card Fraud Detection (ULB)",
        "training_samples": "~227,845 (80% of dataset after deduplication)",
        "test_samples": "~56,962 (20% of dataset)",
        "class_imbalance": "0.17% fraud — severely imbalanced",
        "imbalance_handling": "scale_pos_weight parameter in XGBoost",
        "optimal_threshold": round(float(threshold), 4),
        "threshold_method": "Precision-Recall curve maximizing F1 score",
        "performance": {
            "roc_auc": 0.9788,
            "f1_score": 0.8409,
            "precision": 0.9136,
            "recall": 0.7789
        },
        "features": {
            "total": len(feature_names),
            "names": feature_names,
            "top_predictors": ["V14", "V10", "V4", "V12"],
            "note": "V1-V28 are PCA-transformed for privacy"
        },
        "preprocessing": {
            "amount": "Log transformed (np.log1p) — handles right skew",
            "time": "Cyclical encoded (sin/cos) — handles circular nature of hours",
            "scaling": "RobustScaler — resistant to outliers"
        }
    })


# ── Error Handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": {
            "GET /health": "API health check",
            "GET /stats": "API usage statistics",
            "GET /sample": "Sample input data",
            "GET /model/info": "Model details",
            "POST /predict": "Single transaction prediction",
            "POST /predict/batch": "Batch transaction prediction"
        }
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method not allowed",
        "hint": "Check if you're using GET vs POST correctly"
    }), 405


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
    # host='0.0.0.0' makes it accessible on your network
    # not just localhost — needed for deployment