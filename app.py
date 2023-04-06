from transformers import pipeline
from flask import Flask, request, jsonify

# Load the BERT model and tokenizer using the pipeline shortcut
classifier = pipeline("text-classification", model = "Souvikcmsa/BERT_sentiment_analysis")

# Initialize the Flask app
app = Flask(__name__)

# Define the route for the prediction endpoint
@app.route("/predict", methods=["GET"])
def predict():
    # Get the input sentence from the URL parameter
    sentence = request.args.get("sentence")

    # Make predictions
    result = classifier(sentence)[0]

    # Return the predicted sentiment as a JSON object
    return jsonify({"label": result["label"], "score": result["score"]})

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
