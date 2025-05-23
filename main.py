from pydantic import BaseModel  # Utilisé pour la validation des données
import numpy as np
import pandas as pd  # Utilisé pour la manipulation de données
import joblib  # Utilisé pour charger le modèle sauvegardé
from flask import Flask, request, jsonify  # Flask est un micro-framework pour les applications web

# Charger le modèle de forêt aléatoire depuis le disque
modele = joblib.load('modele/maintenance_model.joblib')

# Charger les label encoders
label_encoders = {}
for feature in ['MARQUE', 'TYPE', 'GENRE', 'STATUT','DESI_GENR']:
    label_encoders[feature] = joblib.load(f"label_encoder_{feature}.joblib")

# Définition du schéma des données d'entrée avec Pydantic
# Cela garantit que les données reçues correspondent aux attentes du modèle
class DonneesEntree(BaseModel):
    text_description: str
    MARQUE: str
    TYPE: str
    GENRE: str
    DESI_GENR: str
    STATUT: str
    month: int
    dayofweek: int
    vehicle_age: float
    Kilometrage_Total: float


# Création de l'instance de l'application Flask
app = Flask(__name__)

# Définition de la route racine qui retourne un message de bienvenue
@app.route("/", methods=["GET"])
def accueil():
    """ Endpoint racine qui fournit un message de bienvenue. """
    return jsonify({"message": "Bienvenue sur l'API de prédiction pour la prédiction de pannes"})

# Définition de la route pour les prédictions de pannes
@app.route("/predire", methods=["POST"])
def predire():
    """
    Endpoint pour les prédictions en utilisant le modèle chargé.
    Les données d'entrée sont validées et transformées en DataFrame pour le traitement par le modèle.
    """
    if not request.json:
        return jsonify({"erreur": "Aucun JSON fourni"}), 400
    
    try:
        # Extraction et validation des données d'entrée en utilisant Pydantic
        donnees = DonneesEntree(**request.json)
        donnees_df = pd.DataFrame([donnees.model_dump()])  # Conversion en DataFrame

        data_dict = donnees_df.iloc[0].to_dict()

        # Transformer les colonnes catégorielles en int avec les label encoders
        for feature in ['MARQUE', 'TYPE', 'GENRE', 'STATUT','DESI_GENR']:
            val = data_dict[feature]
            le = label_encoders[feature]
            # Vérifier que la valeur existe dans le vocabulaire du label encoder
            if val in le.classes_:
                data_dict[feature] = int(le.transform([val])[0])
            else:
                return jsonify({"erreur": f"Valeur inconnue pour {feature}: {val}"}), 400
            
        # Reconstruction du DataFrame encodé   
        donnees_df = pd.DataFrame([data_dict])


         # Utilisation du modèle pour prédire et obtenir les probabilités
        predictions = modele.predict(donnees_df)
        probabilities = modele.predict_proba(donnees_df)[:, 1]  # Probabilité de la classe positive 

        # Compilation des résultats dans un dictionnaire
        resultats =  donnees.model_dump()
        resultats['prediction'] = int(predictions[0])
        resultats['probabilite_panne'] = probabilities[0]

        # Renvoie les résultats sous forme de JSON
        return jsonify({"resultats": resultats})
    except Exception as e:
        # Gestion des erreurs et renvoi d'une réponse d'erreur
        return jsonify({"erreur": str(e)}), 400
    # Point d'entrée pour exécuter l'application
if __name__ == "__main__":
    app.run(debug=True, port=8000)  # Lancement de l'application sur le port 8000 avec le mode debug activé