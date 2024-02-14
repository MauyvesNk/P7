# Import des bibliothèques nécessaires
from os import system
from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import numpy as np
import pickle
import json 

# Création d'une instance de l'application Flask
app = Flask(__name__)

# Charger le modèle à l'aide d'un gestionnaire de contexte
with open("model_LGBM.pkl", 'rb') as model_file:
    lgbm = pickle.load(model_file)

# Route Flask ("GET") renvoie simplement le template initial
@app.route('/')
def home():
    """
    Cette fonction prend en charge la route principale de l'application, généralement la page d'accueil.
    Elle renvoie le contenu de la page HTML associée à " Akwaba Mauyves NKONDO ! ".

    Returns
    -------
    HTML : str
        Le contenu de la page HTML à afficher en tant que page d'accueil de l'application.
    """
    return " Akwaba Mauyves NKONDO ! "

# Route Flask ("POST") est utilisée pour gérer la soumission d'un formulaire
@app.route('/predict', methods=['POST'])
def predict():
    """
    Cette fonction prend en charge une route pour obtenir la liste des clients présents dans le fichier.
    """
    # Supposez que vous ayez une liste d'identifiants de clients définie ici
    num_client = [100002, 100003, 100004, 100005, 100006]
    # Renvoyer une réponse JSON contenant des informations sur le modèle et la liste des identifiants de clients
    return jsonify({
        "model": "model_LGBM.pkl",
        "list_client_id": [str(item) for item in num_client]
    })


@app.route('/predictByClientId/<int:sk_id>', methods=['POST'])
def predictByClientId(sk_id):
    # Chargement de l'ensemble de données pour obtenir les caractéristiques du client spécifié
    #try:
       
    data_set = pd.read_csv("X_train.csv")
    client = data_set[data_set['sk_id_curr'] ==  sk_id].drop(['sk_id_curr'], axis=1)
    print(client)
    X_transformed = client  # Vous pouvez ajuster cette partie en fonction de vos besoins
    y_pred = lgbm.predict(X_transformed)
    y_proba = lgbm.predict_proba(X_transformed)
    # Retournez la réponse en JSON correctement
    return jsonify({
        'prediction': str(y_pred[0]),
        'prediction_proba': str(y_proba[0][0])
    })

@app.route('/predictNewClient', methods=['POST'])
def predictNewClient(): 
    data_client_dict = request.json
    X = pd.DataFrame([data_client_dict])
    y_pred = lgbm.predict(X)
    y_proba = lgbm.predict_proba(X)
    return jsonify({
        'prediction': str(y_pred[0]),
        'prediction_proba': str(y_proba[0][0])
    })

    #except Exception as e:
        # Gérer les exceptions ici
        #print(f"Une erreur s'est produite : {str(e)}")
        #return jsonify({
           # 'message':'Aucun modèle disponible à utiliser'
      #  })
    
# Exécution de l'application
if __name__ == "__main__":
    port = 5000
    app.run(port=port, debug=True, threaded=True)