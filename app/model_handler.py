import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import numpy as np
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise l'analyseur de sentiment avec CamemBERT
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilisation du device: {self.device}")
        
        # Chemin du modèle
        if model_path is None:
            model_path = os.getenv('MODEL_PATH', '/app/models/camembert_sentiment')
        
        logger.info(f"Chargement du modèle depuis: {model_path}")
        
        try:
            # Chargement du tokenizer et le modèle
            self.tokenizer = CamembertTokenizer.from_pretrained(model_path)
            self.model = CamembertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            logger.info("Chargement de camembert-base comme fallback")
            self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
            self.model = CamembertForSequenceClassification.from_pretrained(
                'camembert-base',
                num_labels=2,
                id2label={0: "NEGATIF", 1: "POSITIF"},
                label2id={"NEGATIF": 0, "POSITIF": 1}
            )
            self.model.to(self.device)
            self.model.eval()
    
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Prétraitement du texte pour CamemBERT
        """
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyse le sentiment d'un texte
        """
        if not text or not isinstance(text, str):
            raise ValueError("Le texte doit être une chaîne non vide")
        
        # Prétraitement
        inputs = self.preprocess(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Post-traitement
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            "sentiment": "POSITIF" if predicted_class == 1 else "NÉGATIF",
            "label": predicted_class,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "NEGATIF": round(float(probabilities[0][0]), 4),
                "POSITIF": round(float(probabilities[0][1]), 4)
            }
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Analyse plusieurs textes en batch
        """
        results = []
        for text in texts:
            try:
                result = self.analyze(text)
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    **result
                })
            except Exception as e:
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "error": str(e),
                    "sentiment": None,
                    "confidence": None
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Retourne des informations sur le modèle
        """
        return {
            "model_name": self.model.config._name_or_path,
            "model_type": "CamemBERT",
            "num_labels": self.model.config.num_labels,
            "id2label": self.model.config.id2label,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters())
        }