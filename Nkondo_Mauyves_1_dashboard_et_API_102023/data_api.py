# Importation des bibliothèques
# pandas et numpy sont utilisés pour la manipulation des données.
import pandas as pd
import numpy as np
# streamlit est la bibliothèque principale pour la création de l'application web.
# components de Streamlit est utilisé pour incorporer des composants HTML personnalisés dans l'application.
import streamlit as st
import streamlit.components.v1 as components


###------------------------- load data ---------------------------------------------------
def load_all_data(sample_size):
    # Charger les données à partir de fichiers CSV
    data = pd.read_csv("dataset_exported.csv",nrows=sample_size)
    y_pred_test_export = pd.read_csv("y_pred_test_export.csv")
    train_set = pd.read_csv('application_train.csv',nrows=sample_size)

    # Prétraitement des données sur l'âge
    data['DAYS_BIRTH']= data['DAYS_BIRTH']/-365
    bins= [0,10,20,30,40,50,60,70,80]
    data['age_bins'] = (pd.cut(data['DAYS_BIRTH'], bins=bins)).astype(str)       
        
    return data ,y_pred_test_export,train_set


#------------- Affichage des infos client en HTML------------------------------------------
def display_client_info(id,revenu,age,nb_ann_travail):
   # Affichage des informations du client dans un composant HTML personnalisé
    components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div class="card" style="width: 500px; margin:10px;padding:0">
        <div class="card-body">
            <h5 class="card-title">Info Client</h5>
            
            <ul class="list-group list-group-flush">
                <li class="list-group-item"> <b>ID                           : </b>"""+id+"""</li>
                <li class="list-group-item"> <b>Revenu                       : </b>"""+revenu+"""</li>
                <li class="list-group-item"> <b>Age                          : </b>"""+age+"""</li>
                <li class="list-group-item"> <b>Nombre d'années travaillées  : </b>"""+nb_ann_travail+"""</li>
            </ul>
        </div>
    </div>
    """,
    height=300
    
    )