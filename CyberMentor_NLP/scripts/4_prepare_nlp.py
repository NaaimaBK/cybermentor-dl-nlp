import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def prepare_nlp_features(df):
    """
    Prépare les features texte pour l'entraînement NLP
    """
    print("=" * 60)
    print("ÉTAPE 4: PRÉPARATION DES FEATURES NLP")
    print("=" * 60)
    
    print("📝 Création des features texte...")
    
    # Créer des features texte combinées pour DistilBERT
    text_features = []
    
    for idx, row in df.iterrows():
        text_parts = []
        
        # Features principales pour la détection d'attaques
        if 'proto' in df.columns:
            text_parts.append(f"protocol_{int(row['proto'])}")
        if 'service' in df.columns:
            text_parts.append(f"service_{int(row['service'])}")
        if 'state' in df.columns:
            text_parts.append(f"state_{int(row['state'])}")
        if 'srcip' in df.columns:
            # Extraire seulement le premier octet de l'IP pour éviter le bruit
            try:
                first_octet = str(row['srcip']).split('.')[0]
                text_parts.append(f"srcip_{first_octet}")
            except:
                pass
        if 'dstip' in df.columns:
            try:
                first_octet = str(row['dstip']).split('.')[0]
                text_parts.append(f"dstip_{first_octet}")
            except:
                pass
        
        # Ajouter des informations de trafic
        if 'sbytes' in df.columns and row['sbytes'] > 0:
            text_parts.append("has_sent_bytes")
        if 'dbytes' in df.columns and row['dbytes'] > 0:
            text_parts.append("has_received_bytes")
        if 'dur' in df.columns and row['dur'] > 1.0:
            text_parts.append("long_duration")
        
        text_features.append(" ".join(text_parts))
    
    df['text_features'] = text_features
    
    print(f"✅ Features texte créées. Exemple:")
    print(f"   '{text_features[0][:80]}...'")
    print(f"   Longueur moyenne: {np.mean([len(text) for text in text_features]):.0f} caractères")
    
    return df

def prepare_train_test_split(df):
    """
    Prépare la division train/validation/test
    """
    print("\n🎯 Préparation des splits train/validation/test...")
    
    # Vérifier la distribution des labels
    label_dist = df['Label'].value_counts()
    print(f"Distribution des labels: {label_dist.to_dict()}")
    
    # Division stratifiée
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['Label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['Label']
    )
    
    print(f"✅ Division terminée:")
    print(f"   Train:      {len(train_df)} échantillons")
    print(f"   Validation: {len(val_df)} échantillons")
    print(f"   Test:       {len(test_df)} échantillons")
    
    # Vérifier la distribution dans chaque split
    print(f"\n📊 Distribution dans chaque split:")
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        dist = split_df['Label'].value_counts()
        print(f"   {split_name:12} - Normal: {dist[0]:>5}, Attack: {dist[1]:>5}")
    
    return train_df, val_df, test_df

def save_nlp_data(train_df, val_df, test_df):
    """
    Sauvegarde les données préparées pour NLP
    """
    print("\n💾 Sauvegarde des données NLP...")
    
    # Sauvegarder les splits
    train_df[['text_features', 'Label']].to_csv('./data/nlp_train.csv', index=False)
    val_df[['text_features', 'Label']].to_csv('./data/nlp_val.csv', index=False)
    test_df[['text_features', 'Label']].to_csv('./data/nlp_test.csv', index=False)
    
    print("✅ Données NLP sauvegardées:")
    print(f"   nlp_train.csv: {len(train_df)} échantillons")
    print(f"   nlp_val.csv:   {len(val_df)} échantillons")
    print(f"   nlp_test.csv:  {len(test_df)} échantillons")
    
    # Sauvegarder les métadonnées
    metadata = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': {
            'train': train_df['Label'].value_counts().to_dict(),
            'val': val_df['Label'].value_counts().to_dict(),
            'test': test_df['Label'].value_counts().to_dict()
        },
        'text_feature_stats': {
            'average_length': np.mean([len(text) for text in train_df['text_features']]),
            'vocabulary_size': len(set(' '.join(train_df['text_features']).split()))
        }
    }
    
    with open('./results/nlp_preparation_report.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("📊 Métadonnées NLP sauvegardées")

def main():
    """
    EXÉCUTION PRINCIPALE - PRÉPARATION NLP
    """
    # Charger les données équilibrées
    print("📥 Chargement des données équilibrées...")
    df = pd.read_csv('./data/UNSW-NB15_undersampled.csv')
    print(f"📊 Dataset équilibré: {df.shape}")
    
    # Préparer les features NLP
    df_nlp = prepare_nlp_features(df)
    
    # Préparer les splits
    train_df, val_df, test_df = prepare_train_test_split(df_nlp)
    
    # Sauvegarder les données NLP
    save_nlp_data(train_df, val_df, test_df)
    
    # Sauvegarder le dataset complet NLP-ready
    df_nlp.to_csv('./data/UNSW-NB15_nlp_ready.csv', index=False)
    print("💾 Dataset NLP-ready sauvegardé: UNSW-NB15_nlp_ready.csv")
    
    print(f"\n{'✅'*20}")
    print("ÉTAPE 4 TERMINÉE AVEC SUCCÈS!")
    print("🤖 Données prêtes pour l'entraînement DistilBERT!")
    print(f"{'✅'*20}")

if __name__ == "__main__":
    main()