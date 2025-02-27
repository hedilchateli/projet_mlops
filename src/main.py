import mlflow
import mlflow.sklearn
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
)


def setup_mlflow():
    """Configure MLflow avec l'URI de suivi et le nom de l'expérience."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Serveur MLflow local
    mlflow.set_experiment("Churn Prediction")  # Nom de l'expérience


# Chemins des fichiers CSV sous WSL
file_path_1 = "/home/hedil/hedil_ch_4DS3_mlproject/data/churn-bigml-80.csv"
file_path_2 = "/home/hedil/hedil_ch_4DS3_mlproject/data/churn-bigml-20.csv"


def main():
    """Fonction principale pour préparer les données, entraîner et évaluer le modèle."""
    try:
        # 1️⃣ Configuration MLflow
        setup_mlflow()

        # 2️⃣ Préparation des données
        print("🚀 Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(file_path_1, file_path_2)
        print("✅ Données préparées avec succès !")

        # 3️⃣ Entraînement du modèle
        print("\n🚀 Entraînement du modèle Random Forest...")
        with mlflow.start_run():  # Début du tracking MLflow

            # 🔹 Enregistrement des hyperparamètres
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)  # Exemple d'hyperparamètre

            # 🔹 Entraînement du modèle
            model = train_model(X_train, y_train)
            print("✅ Modèle entraîné avec succès !")

            # 4️⃣ Évaluation du modèle
            print("\n🚀 Évaluation du modèle...")
            accuracy, report, matrix = evaluate_model(model, X_test, y_test)

            # 🔹 Enregistrement des métriques
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", report["weighted avg"]["precision"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

            # 🔹 Création de l'exemple d'entrée (on utilise une ligne d'entrées)
            input_example = X_test.iloc[0].to_dict()

            # 🔹 Enregistrement du modèle dans MLflow
            signature = mlflow.models.signature.infer_signature(X_train, model.predict(X_train))
            print("✅ Signature du modèle créée avec succès !")

            # Enregistrement du modèle avec l'exemple d'entrée et la signature
            mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                signature=signature,
                input_example=input_example
            )
            print("✅ Modèle enregistré dans MLflow avec signature et exemple d'entrée !")

        # 5️⃣ Sauvegarde du modèle en local
        print("\n🚀 Sauvegarde du modèle...")
        save_model(
            model,
            filename="/home/hedil/hedil_ch_4DS3_mlproject/models/random_forest.pkl"
        )
        print("✅ Modèle sauvegardé avec succès !")

    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
    except Exception as e:
        print(f"❌ Une erreur s'est produite : {e}")


if __name__ == "__main__":
    main()
