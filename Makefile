# Déclaration des cibles phony (non-fichier)
.PHONY: prepare_data install_requirements check_venv format_code check_code_quality check_code_security train test clean

# Installation des dépendances
install_requirements:
	@echo "Installation des dépendances..."
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Vérification de l'environnement virtuel
check_venv:
	@echo "Vérification de l'environnement virtuel..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Erreur : L'environnement virtuel n'est pas activé."; \
		exit 1; \
	else \
		echo "L'environnement virtuel est actif."; \
	fi

# Formatage automatique du code avec black
format_code:
	@echo "Formatage du code avec black..."
	@echo "Vérification de l'environnement virtuel..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "Erreur : L'environnement virtuel n'est pas activé."; \
		exit 1; \
	else \
		echo "L'environnement virtuel est actif : $$VIRTUAL_ENV"; \
	fi
	. venv/bin/activate && black src/
# Vérification de la qualité du code avec flake8
check_code_quality:
	@echo "Vérification de la qualité du code avec flake8..."
	. venv/bin/activate && flake8 --config .flake8 src/

# Vérification de la sécurité du code avec bandit
check_code_security:
	@echo "Vérification de la sécurité du code avec bandit..."
	. venv/bin/activate && bandit -r src/

# Préparation des données
prepare_data:
	@echo "Préparation des données..."
	. venv/bin/activate && python3 src/main.py prepare_data

# Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	. venv/bin/activate && python3 src/main.py train

# Exécution des tests
test:
	@echo "Exécution des tests..."
	. venv/bin/activate && python3 -m pytest tests/

# Nettoyage des fichiers générés
clean:
	@echo "Nettoyage des fichiers générés..."
	rm -rf models/random_forest.pkl
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/

# CI/CD : Exécution de toutes les étapes
ci: install_requirements check_venv format_code check_code_quality check_code_security prepare_data train test
	@echo "CI/CD terminée avec succès !"
