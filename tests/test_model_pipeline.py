import pytest
from src.model_pipeline import prepare_data

def test_prepare_data():
    """
    Teste la fonction prepare_data pour s'assurer qu'elle charge et prétraite correctement les données.
    """
    # Chemins des fichiers de données
    file_path_1 = "data/churn-bigml-80.csv"
    file_path_2 = "data/churn-bigml-20.csv"

    # Appel de la fonction prepare_data
    X_train, X_test, y_train, y_test = prepare_data(file_path_1, file_path_2)

    # Vérifications de base
    assert X_train is not None, "X_train ne doit pas être None"
    assert X_test is not None, "X_test ne doit pas être None"
    assert y_train is not None, "y_train ne doit pas être None"
    assert y_test is not None, "y_test ne doit pas être None"

    # Vérification des formes des données
    assert X_train.shape[0] > 0, "X_train doit contenir des lignes"
    assert X_test.shape[0] > 0, "X_test doit contenir des lignes"
    assert len(y_train) > 0, "y_train doit contenir des éléments"
    assert len(y_test) > 0, "y_test doit contenir des éléments"

    # Vérification que les données sont bien séparées
    assert X_train.shape[0] == len(y_train), "X_train et y_train doivent avoir le même nombre d'échantillons"
    assert X_test.shape[0] == len(y_test), "X_test et y_test doivent avoir le même nombre d'échantillons"
