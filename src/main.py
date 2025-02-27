# Import des fonctions depuis le fichier principal
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# Chemins des fichiers CSV sous WSL
file_path_1 = "/home/hedil/hedil_ch_4DS3_mlproject/data/churn-bigml-80.csv"
file_path_2 = "/home/hedil/hedil_ch_4DS3_mlproject/data/churn-bigml-20.csv"


def main():
    try:
        # 1. Préparation des données
        print("🚀 Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(file_path_1, file_path_2)
        print("✅ Données préparées avec succès !")

        # 2. Entraînement du modèle
        print("\n🚀 Entraînement du modèle Random Forest...")
        model = train_model(X_train, y_train)
        print("✅ Modèle entraîné avec succès !")

        # 3. Évaluation du modèle
        print("\n🚀 Évaluation du modèle...")
        accuracy, report, matrix = evaluate_model(model, X_test, y_test)

        # 4. Sauvegarde du modèle
        print("\n🚀 Sauvegarde du modèle...")
        save_model(
            model,
            filename="/home/hedil/hedil_ch_4DS3_mlproject/models/random_forest.pkl",
        )
        print("✅ Modèle sauvegardé avec succès !")

    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
    except Exception as e:
        print(f"❌ Une erreur s'est produite : {e}")


if __name__ == "__main__":
    main()
