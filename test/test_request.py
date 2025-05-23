import requests

# URL de base de l'API
url_base = 'http://127.0.0.1:8000'

# Test du endpoint d'accueil
response = requests.get(f"{url_base}/")
print("Réponse du endpoint d'accueil:", response.text)

# Données d'exemple pour la prédiction
donnees_predire = {
    "MARQUE": "RENAULT",
    "TYPE": "KERAX 440",
    "GENRE": "C",
    "DESI_GENR": "CAMION PORTE-PALETTES",  # exemple de valeur
    "STATUT": "OPR",
    "month": 4,
    "dayofweek": 7,
    "vehicle_age": 25,
    "Kilometrage_Total": 2500000.0,
    "text_description": "Véhicule présente un bruit moteur anormal"
}

# Test du endpoint de prédiction
response = requests.post(f"{url_base}/predire", json=donnees_predire)  # Removed the trailing slash
print("Réponse du endpoint de prédiction:", response.text)