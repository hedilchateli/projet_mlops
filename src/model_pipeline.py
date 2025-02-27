import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Définition des chemins des fichiers CSV sous WSL
file_path_1 = "/home/hedil_ch_4DS3_mlproject/data/churn-bigml-80.csv"
file_path_2 = "/home/hedil_ch_4DS3_mlproject/data/churn-bigml-20.csv"


def prepare_data(file_path_1, file_path_2):
    """Charge et prétraite les données depuis deux fichiers CSV."""
    # Charger les données depuis les fichiers CSV
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    # Concaténer les deux datasets
    df = pd.concat([df1, df2], ignore_index=True)

    # Vérifier si la colonne cible est la dernière (modifie si nécessaire)
    target_column = df.columns[-1]  # Dernière colonne comme cible
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encodage des variables catégoriques en nombres
    X = pd.get_dummies(
        X
    )  # Convertit les variables catégoriques en indicateurs numériques

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraîne un modèle Random Forest."""
    # Initialisation et entraînement du modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle avec un rapport complet."""
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Affichage des résultats
    print("\n📊 **Rapport d'évaluation du modèle**")
    print(f"✅ Précision du modèle : {accuracy:.4f}")
    print("\n🔹 Rapport de classification :\n", report)
    print("\n🔹 Matrice de confusion :\n", matrix)

    return accuracy, report, matrix


MODELS_DIR = "/home/hedil/hedil_ch_4DS3_mlproject/models/"
MODEL_FILENAME = os.path.join(MODELS_DIR, "random_forest.pkl")


def save_model(model, filename=MODEL_FILENAME):
    """Sauvegarde le modèle entraîné."""
    # Créer le dossier models/ s'il n'existe pas
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Sauvegarder le modèle
    joblib.dump(model, filename)
    print(f"✅ Modèle sauvegardé sous {filename}")


def load_model(filename=MODEL_FILENAME):
    """Charge un modèle sauvegardé."""
    # Charger le modèle
    model = joblib.load(filename)
    print(f"✅ Modèle chargé depuis {filename}")
    return model
