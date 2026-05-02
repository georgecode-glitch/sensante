import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")
from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

from sklearn.model_selection import train_test_split

# 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entraîner
model.fit(X_train, y_train)

print("Modèle entraîné !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# Prédire sur les données de test
y_pred = model.predict(X_test)

# Comparer les 10 premières prédictions
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(comparison)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualisation (optionnel)
import os
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardée dans figures/confusion_matrix.png")

import joblib
import os

# Créer le dossier models/
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modèle sauvegardé : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardés.")

# Recharger le modèle depuis le fichier
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Simuler un nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

# Encoder
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# Construire le vecteur de features
features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

# Prédire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print(f"\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%} {bar}")

# EXERCICE 1 : Importance des features
importances = model.feature_importances_

print("\nClassement des features par importance :\n")
for name, imp in sorted(zip(feature_cols, importances),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 50)
    print(f"  {name:20s} : {imp:.3f}  {bar}")

# Exercice 2 - tester avec 3 patients differents

# patient 1 : jeune sans symptomes
p1 = [20, le_sexe_loaded.transform(['M'])[0], 37.0, 120, 0, 0, 0, le_region_loaded.transform(['Dakar'])[0]]
diag1 = model_loaded.predict([p1])[0]
prob1 = model_loaded.predict_proba([p1])[0]
print("\n--- Patient 1 : jeune 20 ans sans symptomes ---")
print(f"Diagnostic : {diag1} ({prob1.max():.1%})")
for c, p in zip(model_loaded.classes_, prob1):
    print(f"  {c} : {p:.1%}")

# patient 2 : adulte avec forte fievre
p2 = [35, le_sexe_loaded.transform(['F'])[0], 40.2, 105, 1, 1, 1, le_region_loaded.transform(['Dakar'])[0]]
diag2 = model_loaded.predict([p2])[0]
prob2 = model_loaded.predict_proba([p2])[0]
print("\n--- Patient 2 : adulte 35 ans forte fievre ---")
print(f"Diagnostic : {diag2} ({prob2.max():.1%})")
for c, p in zip(model_loaded.classes_, prob2):
    print(f"  {c} : {p:.1%}")

# patient 3 : personne agee avec toux
p3 = [65, le_sexe_loaded.transform(['M'])[0], 38.5, 140, 1, 1, 0, le_region_loaded.transform(['Dakar'])[0]]
diag3 = model_loaded.predict([p3])[0]
prob3 = model_loaded.predict_proba([p3])[0]
print("\n--- Patient 3 : age 65 ans avec toux ---")
print(f"Diagnostic : {diag3} ({prob3.max():.1%})")
for c, p in zip(model_loaded.classes_, prob3):
    print(f"  {c} : {p:.1%}")