from flask import Flask, request, jsonify
import torch
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
from utils import download_model

app = Flask(__name__)

label2id = {'negatif': 0, 'netral': 1, 'positif': 2}
id2label = {v: k for k, v in label2id.items()}
aspects = ["Fasilitas", "Harga", "Pelayanan"]

models = {}

def load_model(aspect):
    if aspect in models:
        return models[aspect]

    model_path = download_model(aspect)
    data = torch.load(model_path, map_location=torch.device("cpu"))

    config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
    model = BertForSequenceClassification(config)
    model.load_state_dict(data['model'])
    model.eval()
    tokenizer = data['tokenizer']

    models[aspect] = (model, tokenizer)
    return model, tokenizer

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text")
        aspect = data.get("aspect")

        if aspect not in aspects:
            return jsonify({"error": "Aspect tidak valid"}), 400

        model, tokenizer = load_model(aspect)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            sentiment = id2label[pred]

        return jsonify({
            "aspect": aspect,
            "text": text,
            "sentiment": sentiment
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
