# Import des bibliothèques nécessaires
import pickle
import pandas as pd
import numpy as np

# Fonction pour charger l'ensemble de données
def load_dataset(sample_size):
    # Charger le fichier CSV contenant l'ensemble de données d'entraînement
    data = pd.read_csv("X_train.csv",nrows=sample_size)
    return data


# Fonction pour charger le modèle LightGBM préalablement entraîné
'''Charger le modèle LightGBM entraîné '''
def load_lgbm_model():
    # Charger le modèle LightGBM à partir du fichier pickle
    model = pickle.load(open("model_LGBM.pkl", 'rb'))
    return model


# Fonction pour prédire un client à partir de ses caractéristiques avec le modèle LightGBM
'''Prédire un client avec le modèle LightGBM '''
def predict_client_lgbm(X):
    # Supprimer la colonne 'sk_id_curr' (identifiant du client)
    X_processed = X.drop(['sk_id_curr'], axis=1)
    # Charger le modèle LightGBM
    model = load_lgbm_model()
    # Prédire la classe et les probabilités associées
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)
    return y_pred, y_proba


# Fonction pour prédire un client par son ID dans le dataset avec le modèle LightGBM
'''Prédire un client par son ID dans le dataset avec le modèle LightGBM '''
def predict_client_par_ID_lgbm(id_client):
    # Taille d'échantillon limite pour charger l'ensemble de données
    sample_size = 20000
    # Charger l'ensemble de données avec la taille d'échantillon spécifiée
    data_set = load_dataset(sample_size)
    # Sélectionner le client spécifié par son ID et supprimer la colonne 'sk_id_curr'
    client = data_set[data_set['sk_id_curr'] == id_client].drop(['sk_id_curr'], axis=1)
    # Afficher le client (optionnel)
    print(client)
    # Charger le modèle LightGBM
    model = load_lgbm_model()
    # Prédire la classe et les probabilités associées pour le client spécifié
    y_pred = model.predict(client)
    y_proba = model.predict_proba(client)
    return y_pred, y_proba
