import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os
import time

def train_fast():
    """Version RAPIDE de l'entraînement - 5-10 minutes max"""
    print("🚀 ENTRAÎNEMENT RAPIDE DISTILBERT")
    print("⏱️  Durée estimée: 5-10 minutes")
    
    # Configuration optimisée
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Charger UNIQUEMENT un sous-échantillon pour aller vite
    print("📥 Chargement des données (sous-échantillon)...")
    train_df = pd.read_csv('./data/nlp_train.csv')
    
    # Prendre seulement 2000 échantillons pour aller vite
    sample_size = 2000
    train_df = train_df.sample(n=sample_size, random_state=42)
    
    print(f"📊 Utilisation de {sample_size} échantillons pour entraînement rapide")
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Préparer les données rapidement
    def prepare_fast_data(texts, labels):
        print("🔤 Tokenisation rapide...")
        encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=64,  # Longueur réduite
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }
    
    # Préparer les données
    train_texts = train_df['text_features'].tolist()[:sample_size]
    train_labels = train_df['Label'].tolist()[:sample_size]
    
    train_data = prepare_fast_data(train_texts, train_labels)
    
    # Dataset simple
    class FastDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.encodings['labels'][idx]
            }
            
        def __len__(self):
            return len(self.encodings['labels'])
    
    train_dataset = FastDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Batch plus grand
    
    # Modèle
    print("🤖 Chargement du modèle...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)
    model.train()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)  # Learning rate plus élevé
    
    # Entraînement RAPIDE - 2 epochs seulement
    print("🎯 Début entraînement RAPIDE (2 epochs)...")
    start_time = time.time()
    
    for epoch in range(2):  # SEULEMENT 2 EPOCHS
        epoch_start = time.time()
        total_loss = 0
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Afficher la progression
            if batch_idx % 10 == 0:
                elapsed = time.time() - epoch_start
                batches_done = batch_idx + 1
                total_batches = len(train_loader)
                progress = (batches_done / total_batches) * 100
                
                print(f"  Epoch {epoch+1}: {progress:.1f}% ({batches_done}/{total_batches}), "
                      f"Loss: {loss.item():.4f}, Temps: {elapsed:.0f}s")
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1}/2 terminée - Loss: {avg_loss:.4f}, Temps: {epoch_time:.0f}s")
    
    total_time = time.time() - start_time
    print(f"⏱️  Temps total d'entraînement: {total_time:.0f} secondes")
    
    # Sauvegarde rapide
    print("💾 Sauvegarde du modèle...")
    model.save_pretrained('./models/cybermentor_nlp_model_fast')
    tokenizer.save_pretrained('./models/cybermentor_nlp_model_fast')
    
    print("🎉 ENTRAÎNEMENT RAPIDE TERMINÉ!")
    return model

def main():
    """
    ENTRAÎNEMENT RAPIDE - 5-10 MINUTES MAX
    """
    print("=" * 60)
    print("ÉTAPE 5: ENTRAÎNEMENT RAPIDE DISTILBERT")
    print("=" * 60)
    
    # Créer les dossiers
    os.makedirs('./models', exist_ok=True)
    
    try:
        model = train_fast()
        
        print(f"\n{'🎉'*20}")
        print("ÉTAPE 5 TERMINÉE AVEC SUCCÈS!")
        print("🤖 Modèle DistilBERT entraîné RAPIDEMENT!")
        print("📁 Modèle sauvegardé dans: ./models/cybermentor_nlp_model_fast/")
        print("⏱️  Prêt pour l'évaluation!")
        print(f"{'🎉'*20}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Essayez avec encore moins de données...")

if __name__ == "__main__":
    main()