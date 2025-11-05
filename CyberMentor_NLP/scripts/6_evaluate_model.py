import pandas as pd
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class ModelEvaluator:
    def __init__(self, model_path='./models/cybermentor_nlp_model_fast'):
        self.model_path = model_path
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Mode évaluation
        print(f"✅ Modèle chargé depuis: {model_path}")
        print(f"🖥️  Device: {self.device}")
        
    def load_test_data(self):
        """Charge les données de test"""
        print("📥 Chargement des données de test...")
        test_df = pd.read_csv('./data/nlp_test.csv')
        print(f"📊 Test samples: {len(test_df)}")
        
        # Vérifier la distribution
        label_dist = test_df['Label'].value_counts()
        print(f"🎯 Distribution test: Normal {label_dist[0]}, Attack {label_dist[1]}")
        
        return test_df
    
    def predict_batch(self, texts, batch_size=32):
        """Prédit par batch pour plus d'efficacité"""
        print("🔮 Prédictions en cours...")
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokeniser le batch
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Traité {min(i + batch_size, len(texts))}/{len(texts)} échantillons...")
        
        return predictions
    
    def evaluate_model(self):
        """Évalue le modèle sur les données de test"""
        print("🧪 Évaluation du modèle...")
        
        test_df = self.load_test_data()
        
        # Préparer les données
        test_texts = test_df['text_features'].tolist()
        true_labels = test_df['Label'].tolist()
        
        # Prédictions
        predictions = self.predict_batch(test_texts)
        
        # Calcul des métriques
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        print(f"\n📊 RÉSULTATS DE L'ÉVALUATION:")
        print("=" * 50)
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"🎯 F1-Score: {report['1']['f1-score']:.4f}")
        print(f"🎯 Precision: {report['1']['precision']:.4f}")
        print(f"🎯 Recall: {report['1']['recall']:.4f}")
        print(f"📈 Support: {report['1']['support']} échantillons d'attaque")
        
        # Rapport détaillé
        print(f"\n📋 RAPPORT DÉTAILLÉ:")
        print(classification_report(true_labels, predictions, target_names=['Normal', 'Attack']))
        
        return predictions, true_labels, report, accuracy
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Affiche la matrice de confusion"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'], 
                   yticklabels=['Normal', 'Attack'])
        plt.title('Matrice de Confusion - CyberMentor NLP', fontsize=14, fontweight='bold')
        plt.ylabel('Vrai Label', fontweight='bold')
        plt.xlabel('Prédiction', fontweight='bold')
        plt.tight_layout()
        plt.savefig('./results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Matrice de confusion sauvegardée")
    
    def plot_metrics_comparison(self, report):
        """Affiche la comparaison des métriques"""
        metrics = ['precision', 'recall', 'f1-score']
        normal_scores = [report['0'][metric] for metric in metrics]
        attack_scores = [report['1'][metric] for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, normal_scores, width, label='Normal', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, attack_scores, width, label='Attack', color='red', alpha=0.7)
        
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Scores')
        ax.set_title('Comparaison des Métriques par Classe', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('./results/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_report(self, report, accuracy, predictions, true_labels):
        """Sauvegarde le rapport d'évaluation complet"""
        evaluation_report = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'macro_avg_f1': float(report['macro avg']['f1-score']),
                'weighted_avg_f1': float(report['weighted avg']['f1-score'])
            },
            'class_metrics': {
                'normal': {
                    'precision': float(report['0']['precision']),
                    'recall': float(report['0']['recall']),
                    'f1_score': float(report['0']['f1-score']),
                    'support': int(report['0']['support'])
                },
                'attack': {
                    'precision': float(report['1']['precision']),
                    'recall': float(report['1']['recall']),
                    'f1_score': float(report['1']['f1-score']),
                    'support': int(report['1']['support'])
                }
            },
            'test_set_info': {
                'total_samples': len(true_labels),
                'normal_samples': true_labels.count(0),
                'attack_samples': true_labels.count(1)
            },
            'model_info': {
                'model_name': 'DistilBERT CyberMentor',
                'training_samples': 2000,
                'test_samples': len(true_labels)
            }
        }
        
        with open('./results/evaluation_report.json', 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        print("📊 Rapport d'évaluation sauvegardé")
        
        # Afficher un résumé
        print(f"\n📋 RÉSUMÉ DES PERFORMANCES:")
        print(f"   Exactitude globale: {accuracy:.1%}")
        print(f"   F1-Score Attaques: {report['1']['f1-score']:.1%}")
        print(f"   Précision Attaques: {report['1']['precision']:.1%}")
        print(f"   Rappel Attaques: {report['1']['recall']:.1%}")

def main():
    """
    ÉVALUATION DU MODÈLE ENTRAÎNÉ
    """
    print("=" * 60)
    print("ÉTAPE 6: ÉVALUATION DU MODÈLE DISTILBERT")
    print("=" * 60)
    
    # Créer le dossier results
    os.makedirs('./results', exist_ok=True)
    
    try:
        # Évaluer le modèle
        evaluator = ModelEvaluator()
        predictions, true_labels, report, accuracy = evaluator.evaluate_model()
        
        # Visualisations
        evaluator.plot_confusion_matrix(true_labels, predictions)
        evaluator.plot_metrics_comparison(report)
        
        # Sauvegarder le rapport
        evaluator.save_evaluation_report(report, accuracy, predictions, true_labels)
        
        print(f"\n{'🎉'*20}")
        print("ÉTAPE 6 TERMINÉE AVEC SUCCÈS!")
        print("📊 Modèle évalué sur le set de test")
        print("📁 Résultats sauvegardés dans ./results/")
        print(f"{'🎉'*20}")
        
    except Exception as e:
        print(f"❌ Erreur pendant l'évaluation: {e}")
        print("💡 Vérifiez que le modèle a bien été entraîné")

if __name__ == "__main__":
    main()