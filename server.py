import pickle
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load objects
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    X_vec = vectorizer.transform([text])

    if model:
        prediction = int(model.predict(X_vec)[0])
        return jsonify(
            {
                "status": "success",
                "prediction": prediction,
                "category": "Positive" if prediction == 1 else "Negative",
            }
        )
    else:
        return jsonify(
            {
                "status": "vectorized_only",
                "num_features": int(X_vec.nnz),
            }
        )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
