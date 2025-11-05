# 🛡️ CyberMentor - Système Intelligent de Détection d'Attaques

**CyberMentor** est un système avancé de détection d'attaques réseau utilisant l'apprentissage profond (NLP) pour analyser et classifier le trafic réseau en temps réel.

## 🎯 Performances Exceptionnelles

| Métrique | Score |
|----------|-------|
| **Accuracy** | 99.51% ✅ |
| **F1-Score** | 99.51% ✅ |
| **Precision** | 99.03% ✅ |
| **Recall** | 100.00% 🎯 |
| **Support** | 4,284 échantillons |

> ⚡ **Détecte 100% des attaques avec seulement 0.97% de faux positifs**

## 🚀 Fonctionnalités

### 🔍 Détection Intelligente
- **Classification binaire** : Normal vs Attaque
- **9 types d'attaques** détectés : Generic, Exploits, Fuzzers, Reconnaissance, DoS, etc.
- **Analyse en temps réel** des logs réseau
- **Features NLP avancées** avec DistilBERT

### 📊 Préprocessing Avancé
- **Nettoyage automatique** des données UNSW-NB15
- **Équilibrage des classes** (Under-sampling)
- **Feature engineering** pour l'analyse NLP
- **Split temporel** sans fuite de données

### 🤖 Modèle State-of-the-Art
- **Architecture** : DistilBERT fine-tuné
- **Entraînement optimisé** : 2 epochs, 2,000 échantillons
- **Inférence rapide** : Prédictions en millisecondes
- **Modèle léger** : 268MB, adapté production

## 📊 Dataset UNSW-NB15

### Caractéristiques
- **📏 Taille** : 2,540,047 échantillons originaux
- **🎯 Labels** : 9 types d'attaques différentes
- **⚖️ Équilibrage** : 50% Normal, 50% Attack après traitement
- **🕒 Période** : Données réseau réalistes

### Types d'attaques détectés
- **Generic** - Attaques génériques
- **Exploits** - Exploitation de vulnérabilités
- **Fuzzers** - Tests de fuzzing
- **Reconnaissance** - Reconnaissance réseau
- **DoS** - Déni de service
- **Backdoors** - Portes dérobées
- **Analysis** - Analyse malveillante
- **Shellcode** - Code d'exploitation
- **Worms** - Vers réseau
