"""
API d'analyse de sentiment 
"""

import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import logging

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialisation du modèle
print("=" * 60)
print(" Initialisation de l'API avec CamemBERT")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

try:
    # Chargement du tokenizer et modèle
    print("Chargement du tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    
    print("Chargement du modèle...")
    model = CamembertForSequenceClassification.from_pretrained(
        "camembert-base",
        num_labels=2,
        id2label={0: "NEGATIF", 1: "POSITIF"},
        label2id={"NEGATIF": 0, "POSITIF": 1}
    )
    model.to(device)
    model.eval()
    
    print("--- Modèle CamemBERT chargé avec succès!")
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"--- Erreur de chargement: {e}")
    print("Mode simulation activé")
    tokenizer = None
    model = None

@app.route('/')
def home():
    return jsonify({
        "message": "API d'Analyse de Sentiment - CamemBERT",
        "version": "1.0.0",
        "model": "camembert-base" if model else "simulation",
        "device": str(device),
        "endpoints": ["/health", "/predict", "/predict_batch", "/model_info"]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy" if model else "simulation",
        "model_loaded": model is not None,
        "timestamp": time.time()
    })

@app.route('/model_info')
def model_info():
    if model:
        info = {
            "name": model.config._name_or_path,
            "num_labels": model.config.num_labels,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "id2label": model.config.id2label
        }
    else:
        info = {"mode": "simulation", "reason": "Modèle non chargé"}
    
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Le champ 'text' est requis"}), 400
    
    text = data['text']
    
    start_time = time.time()
    
    if model and tokenizer:
        try:
            # Tokenization
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Post-traitement
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            processing_time = time.time() - start_time
            
            response = {
                "text": text[:200],
                "sentiment": "POSITIF" if predicted_class == 1 else "NEGATIF",
                "confidence": round(float(confidence), 4),
                "label": predicted_class,
                "probabilities": {
                    "NEGATIF": round(float(probabilities[0][0]), 4),
                    "POSITIF": round(float(probabilities[0][1]), 4)
                },
                "processing_time_ms": round(processing_time * 1000, 2),
                "mode": "real_model",
                "tokens": len(tokenizer.encode(text))
            }
            
            logger.info(f"Prédiction réussie: {response['sentiment']} ({response['confidence']})")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Erreur modèle: {str(e)}")
            # Fallback 
            return _predict_simulation(text, start_time)
    
    else:
        # Mode simulation
        return _predict_simulation(text, start_time)

def _predict_simulation(text, start_time):
    """Fallback simulation"""
    text_lower = text.lower()
    
    positive_words = ['excellent', 'super', 'génial', 'magnifique']
    negative_words = ['mauvais', 'nul', 'horrible', 'décevant']
    
    positive_score = sum(1 for w in positive_words if w in text_lower)
    negative_score = sum(1 for w in negative_words if w in text_lower)
    
    if positive_score > negative_score:
        sentiment = "POSITIF"
        confidence = 0.8 + (positive_score * 0.05)
    elif negative_score > positive_score:
        sentiment = "NEGATIF"
        confidence = 0.8 + (negative_score * 0.05)
    else:
        sentiment = "NEUTRE"
        confidence = 0.5
    
    processing_time = time.time() - start_time
    
    return jsonify({
        "text": text[:200],
        "sentiment": sentiment,
        "confidence": round(min(0.99, confidence), 4),
        "label": 1 if sentiment == "POSITIF" else 0,
        "processing_time_ms": round(processing_time * 1000, 2),
        "mode": "simulation",
        "note": "Modèle CamemBERT non disponible"
    })

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({"error": "Le champ 'texts' est requis"}), 400
    
    texts = data['texts']
    
    if not isinstance(texts, list) or len(texts) > 20:
        return jsonify({"error": "'texts' doit être une liste de maximum 20 textes"}), 400
    
    start_time = time.time()
    results = []
    
    for text in texts:
        result = predict_single(text)
        results.append(result)
    
    total_time = time.time() - start_time
    
    return jsonify({
        "results": results,
        "summary": {
            "total_texts": len(texts),
            "total_time_ms": round(total_time * 1000, 2),
            "avg_time_ms": round((total_time / len(texts)) * 1000, 2) if texts else 0,
            "model_mode": "real" if model else "simulation"
        }
    })

def predict_single(text):
    """Prédiction pour un seul texte (utilisée par batch)"""
    if model and tokenizer:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": "POSITIF" if predicted_class == 1 else "NEGATIF",
                "confidence": round(float(confidence), 4),
                "label": predicted_class
            }
        except:
            # Fallback
            pass
    
    # Simulation fallback
    text_lower = text.lower()
    if any(w in text_lower for w in ['excellent', 'super', 'génial']):
        return {"text": text[:100], "sentiment": "POSITIF", "confidence": 0.85, "label": 1}
    elif any(w in text_lower for w in ['mauvais', 'nul', 'horrible']):
        return {"text": text[:100], "sentiment": "NEGATIF", "confidence": 0.85, "label": 0}
    else:
        return {"text": text[:100], "sentiment": "NEUTRE", "confidence": 0.5, "label": 1}

if __name__ == '__main__':
    print("\n--- Démarrage du serveur Flask...")
    print("--- API disponible sur: http://localhost:5000")
    print("--- Endpoints:")
    print("   GET  /          - Page d'accueil")
    print("   GET  /health    - Vérification santé")
    print("   GET  /model_info- Informations modèle")
    print("   POST /predict   - Analyse de texte")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)