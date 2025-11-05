import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_label_distribution():
    """
    Analyse détaillée de la distribution des labels
    """
    print("🔍 Analyse détaillée des labels...")
    
    # Charger le dataset nettoyé
    df = pd.read_csv('./data/UNSW-NB15_cleaned.csv')
    
    # Analyse de la colonne Label
    label_counts = df['Label'].value_counts()
    
    print(f"📊 Statistiques des labels:")
    print(f"   Total des catégories: {len(label_counts)}")
    print(f"   Valeur la plus fréquente: {label_counts.index[0]} ({(label_counts.iloc[0]/len(df))*100:.2f}%)")
    print(f"   Valeur la moins fréquente: {label_counts.index[-1]} ({(label_counts.iloc[-1]/len(df))*100:.4f}%)")
    
    # Top 20 labels
    print(f"\n🏆 Top 20 labels:")
    for i, (label, count) in enumerate(label_counts.head(20).items()):
        percentage = (count / len(df)) * 100
        print(f"   {i+1:2d}. Label {label}: {count:>8} ({percentage:6.2f}%)")
    
    # Analyser si c'est un problème de classification binaire ou multi-classe
    unique_labels = sorted(df['Label'].unique())
    print(f"\n🎯 Type de problème:")
    print(f"   Labels uniques: {unique_labels[:10]}...")  # Afficher les 10 premiers
    
    if len(unique_labels) == 2:
        print("   → Classification BINAIRE")
    elif 3 <= len(unique_labels) <= 10:
        print(f"   → Classification MULTI-CLASSE ({len(unique_labels)} classes)")
    else:
        print(f"   → Classification MULTI-CLASSE COMPLEXE ({len(unique_labels)} classes)")
        print("   💡 Recommandation: Regrouper les classes rares")
    
    # Visualisation détaillée
    plt.figure(figsize=(15, 10))
    
    # 1. Top 30 labels
    plt.subplot(2, 2, 1)
    top_30 = label_counts.head(30)
    plt.bar(range(len(top_30)), top_30.values, color='steelblue')
    plt.title('Top 30 Labels (Distribution Complète)', fontweight='bold')
    plt.xlabel('Label ID')
    plt.ylabel('Nombre d\'occurrences')
    plt.xticks(rotation=45)
    
    # 2. Distribution cumulative
    plt.subplot(2, 2, 2)
    cumulative_percentage = (label_counts.cumsum() / len(df)) * 100
    plt.plot(range(len(cumulative_percentage)), cumulative_percentage.values, 
             color='red', linewidth=2)
    plt.title('Distribution Cumulative des Labels', fontweight='bold')
    plt.xlabel('Nombre de Labels')
    plt.ylabel('Pourcentage Cumulé (%)')
    plt.grid(True, alpha=0.3)
    
    # 3. Top 10 labels détaillés
    plt.subplot(2, 2, 3)
    top_10 = label_counts.head(10)
    bars = plt.bar(range(len(top_10)), top_10.values, color=['green' if x == 0 else 'red' for x in top_10.index])
    plt.title('Top 10 Labels (Détail)', fontweight='bold')
    plt.xlabel('Label')
    plt.ylabel('Occurrences')
    plt.xticks(range(len(top_10)), top_10.index, rotation=45)
    
    # Ajouter les pourcentages sur les barres
    for i, (label, count) in enumerate(top_10.items()):
        percentage = (count / len(df)) * 100
        plt.text(i, count, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Distribution des classes rares
    plt.subplot(2, 2, 4)
    rare_labels = label_counts[label_counts < 1000]  # Labels avec moins de 1000 occurrences
    if len(rare_labels) > 0:
        plt.hist(rare_labels.values, bins=30, color='orange', alpha=0.7, edgecolor='black')
        plt.title(f'Distribution des {len(rare_labels)} Labels Rares', fontweight='bold')
        plt.xlabel('Nombre d\'occurrences')
        plt.ylabel('Nombre de Labels')
    else:
        plt.text(0.5, 0.5, 'Aucun label rare\n(tous > 1000 occurrences)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Distribution des Labels Rares', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./results/label_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    if len(label_counts) > 50:
        print("   - Trop de classes! Regrouper les labels rares en 'Other'")
        print("   - Considérer une approche de classification hiérarchique")
        print("   - Focus sur les top 10-20 labels les plus fréquents")
    else:
        print("   - Classification multi-classes gérable")
        print("   - Vérifier l'équilibre des classes")

if __name__ == "__main__":
    analyze_label_distribution()