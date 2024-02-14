# Importation des bibliothèques
import streamlit as st
import pickle
# pandas et numpy sont utilisés pour la manipulation des données.
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
import seaborn as sns
import plotly.figure_factory as ff
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import time
# streamlit est la bibliothèque principale pour la création de l'application web.
# components de Streamlit est utilisé pour incorporer des composants HTML personnalisés dans l'application.
import streamlit.components.v1 as components
import requests
import json
import shap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import os
from prediction_api import *
from data_api import *


sample_size = 10000
data, y_pred_test_export, train_set = load_all_data(sample_size)


###---- Data----------
def show_data():
    st.write(data.head(10))


### -----------------------------------Solvency-----------------------------
def pie_chart(thres):
    percent_sup_seuil = 100 * (data['TARGET'] > thres).sum() / data.shape[0]
    percent_inf_seuil = 100 - percent_sup_seuil
    d = {'col1': [percent_sup_seuil, percent_inf_seuil], 'col2': ['% Non Solvable', '% Solvable', ]}
    df = pd.DataFrame(data=d)
    fig = px.pie(df, values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
    st.plotly_chart(fig)


def show_overview():
    st.title("Risque")
    risque_threshold = st.slider(label='Seuil de risque', min_value=0.0,
                                 max_value=1.0,
                                 value=0.5,
                                 step=0.1)
    pie_chart(risque_threshold)


###-------------------------------------------- Graphs------------------------------------------------------
def filter_graphs():
    st.subheader("Filtre des Graphes")
    col1, col2, col3 = st.columns(3)
    is_educ_selected = col1.radio("Graph Education", ('non', 'oui'))
    is_statut_selected = col2.radio('Graph Statut', ('non', 'oui'))
    is_income_selected = col3.radio('Graph Revenu', ('non', 'oui'))

    return is_educ_selected, is_statut_selected, is_income_selected


def hist_graph():
    # Remplacez la partie où vous utilisez st.pyplot() sans argument par le code suivant :
    fig, ax = plt.subplots()
    ax.bar_chart(data['DAYS_BIRTH'])
    df = pd.DataFrame(data[:200], columns=['DAYS_BIRTH', 'AMT_CREDIT'])
    df.hist(ax=ax)
    st.pyplot(fig)

# Remplacez l'ancienne ligne qui désactive l'avertissement par la suivante :
st.set_option('deprecation.showPyplotGlobalUse', False)


def education_type():
    ed = train_set.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = train_set.NAME_EDUCATION_TYPE.unique()
    fig = go.Figure(data=[go.Bar(
        x=u_ed,
        y=ed
    )])
    fig.update_layout(title_text='Data education')
    st.plotly_chart(fig)

    ed_solvable = train_set[train_set['TARGET'] == 0].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    ed_non_solvable = train_set[train_set['TARGET'] == 1].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = train_set.NAME_EDUCATION_TYPE.unique()
    fig = go.Figure(data=[
        go.Bar(name='Solvable', x=u_ed, y=ed_solvable),
        go.Bar(name='Non Solvable', x=u_ed, y=ed_non_solvable)
    ])
    fig.update_layout(title_text='Solvabilité Vs education')
    st.plotly_chart(fig)


def statut_plot():
    ed = train_set.groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = train_set.NAME_FAMILY_STATUS.unique()
    fig = go.Figure(data=[go.Bar(
        x=u_ed,
        y=ed
    )])
    fig.update_layout(title_text='Data situation familiale')
    st.plotly_chart(fig)

    ed_solvable = train_set[train_set['TARGET'] == 0].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    ed_non_solvable = train_set[train_set['TARGET'] == 1].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
    u_ed = train_set.NAME_FAMILY_STATUS.unique()
    fig = go.Figure(data=[
        go.Bar(name='Solvable', x=u_ed, y=ed_solvable),
        go.Bar(name='Non Solvable', x=u_ed, y=ed_non_solvable)
    ])
    fig.update_layout(title_text='Solvabilité Vs situation familiale')
    st.plotly_chart(fig)


def income_type():
    ed = train_set.groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = train_set.NAME_INCOME_TYPE.unique()
    fig = go.Figure(data=[go.Bar(
        x=u_ed,
        y=ed
    )])
    fig.update_layout(title_text='Data Type de Revenu')
    st.plotly_chart(fig)

    ed_solvable = train_set[train_set['TARGET'] == 0].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    ed_non_solvable = train_set[train_set['TARGET'] == 1].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
    u_ed = train_set.NAME_INCOME_TYPE.unique()
    fig = go.Figure(data=[
        go.Bar(name='Solvable', x=u_ed, y=ed_solvable),
        go.Bar(name='Non Solvable', x=u_ed, y=ed_non_solvable)
    ])
    fig.update_layout(title_text='Solvabilité Vs Type de Revenu')
    st.plotly_chart(fig)


###------------------------------------------ Distribution -------------------------------------------------
def filter_distribution():
    st.subheader("Filtre des Distribution")
    col1, col2 = st.columns(2)
    is_age_selected = col1.radio("Distribution Age ", ('non', 'oui'))
    is_incomdis_selected = col2.radio('Distribution Revenus ', ('non', 'oui'))


    return is_age_selected, is_incomdis_selected


def age_distribution():
    df = pd.DataFrame({'Age': data['DAYS_BIRTH'],
                       'Solvabilite': data['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}
    df = df.replace({"Solvabilite": dic})

    fig = px.histogram(df, x="Age", color="Solvabilite", nbins=40)
    st.subheader("Distribution des ages selon la sovabilité")
    st.plotly_chart(fig)


def revenu_distribution():
    df = pd.DataFrame({'Revenus': data['AMT_INCOME_TOTAL'],
                       'Solvabilite': data['TARGET']})

    dic = {0: "solvable", 1: "non solvable"}
    df = df.replace({"Solvabilite": dic})

    fig = px.histogram(df, x="Revenus", color="Solvabilite", nbins=40)
    st.subheader("Distribution des revenus selon la sovabilité")
    st.plotly_chart(fig)


# --------------------------- Client Predection -------------------------------------------------

def show_client_predection():
    client_id = st.number_input("Donnez Id du Client", 100002)
    if st.button('Voir Client'):
        client = data[data['SK_ID_CURR'] == client_id]

        display_client_info(
            str(client['SK_ID_CURR'].squeeze()),
            str(client['AMT_INCOME_TOTAL'].squeeze()),
            str(round(client['DAYS_BIRTH'].squeeze())),
            str(round(client['DAYS_EMPLOYED'].squeeze() / -365))
        )

        # Bar Chart
        age_bins = data['age_bins'].value_counts(sort=False)

        d = {'Ages par Decennie': age_bins.index, 'Nombre de clients par Decennie': age_bins.values}
        ages_decennie = pd.DataFrame(data=d)

        ages_decennie['Ages par Decennie'] = ages_decennie['Ages par Decennie'].astype(str)

        # Vérifiez si la colonne 'age_bins' dans le DataFrame client n'est pas vide
        if not client['age_bins'].empty:
            # Vérifiez si la décennie du client existe dans le DataFrame
            if client['age_bins'].values[0] in ages_decennie['Ages par Decennie'].values:
                idx_decennie = ages_decennie[ages_decennie['Ages par Decennie'] == client['age_bins'].values[0]].index

                colors = ['lightslategray', ] * len(ages_decennie['Nombre de clients par Decennie'])
                colors[idx_decennie[0]] = 'crimson'

                fig = go.Figure(data=[go.Bar(
                    x=ages_decennie['Ages par Decennie'],
                    y=ages_decennie['Nombre de clients par Decennie'],
                    marker_color=colors
                )])
                fig.update_layout(title_text='Nombre de Clients par Décennie')

                st.plotly_chart(fig)
            else:
                print("La décennie du client n'a pas été trouvée dans les données.")
        else:
            print("La colonne 'age_bins' dans le DataFrame client est vide.")

        # Line Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data['pred_prob'], x=data['AMT_INCOME_TOTAL'],
                                 mode='markers', name='Revenus des autres clients'))
        fig.add_trace(go.Scatter(y=client['pred_prob'], x=client['AMT_INCOME_TOTAL'],
                                 mode='markers', name='Revenu de ce Client', marker=dict(size=[25])))
        st.plotly_chart(fig)


# --------------------------- model analysis -------------------------
### Confusion matrixe
def matrix_confusion(X, y):
    cm = confusion_matrix(X, y)
    print('\nTrue Positives(TP) = ', cm[0, 0])
    print('\nTrue Negatives(TN) = ', cm[1, 1])
    print('\nFalse Positives(FP) = ', cm[0, 1])
    print('\nFalse Negatives(FN) = ', cm[1, 0])
    return cm


def show_model_analysis():
    conf_mtx = matrix_confusion(y_pred_test_export['y_test'], y_pred_test_export['y_predicted'])
    # st.write(conf_mtx)
    fig = go.Figure(data=go.Heatmap(
        z=conf_mtx,
        x=['Actual Negative:0', 'Actual Positive:1'],
        y=['Predict Negative:0', 'Predict Positive:1'],
        hoverongaps=False))
    st.plotly_chart(fig)

    fpr, tpr, thresholds = roc_curve(y_pred_test_export['y_test'], y_pred_test_export['y_probability'])

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    st.plotly_chart(fig)


### ----------------------- Prédiction d'un client ----------------
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Sélectionner un fichier client', filenames)
    return os.path.join(folder_path, selected_filename)

# Fonction pour charger le modèle LightGBM
def load_lgbm_model():
    model = pickle.load(open("model_LGBM.pkl", 'rb'))
    return model

# Fonction pour prédire un client
def predict_client(model, nouveau_client):
    # Chargez le modèle LightGBM ici ou passez-le en tant qu'argument
    model = load_lgbm_model()

    url = "https://app-flask-5efe138da57d.herokuapp.com/predictNewClient"
    payload = json.dumps(nouveau_client.to_dict('records')[0])
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload)
    prediction = response.json()

    # Ajoutez le code pour calculer les SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(nouveau_client)

    return int(prediction['prediction']), float(prediction['prediction_proba']), shap_values

# Fonction pour afficher la prédiction d'un client existant
def show_existing_client_prediction(client_id):
    client = data[data['SK_ID_CURR'] == client_id]

    display_client_info(
        str(client['SK_ID_CURR'].squeeze()),
        str(client['AMT_INCOME_TOTAL'].squeeze()),
        str(round(client['DAYS_BIRTH'].squeeze())),
        str(round(client['DAYS_EMPLOYED'].squeeze() / -365))
    )

    # Bar Chart
    age_bins = data['age_bins'].value_counts(sort=False)

    d = {'Ages par Decennie': age_bins.index, 'Nombre de clients par Decennie': age_bins.values}
    ages_decennie = pd.DataFrame(data=d)

    ages_decennie['Ages par Decennie'] = ages_decennie['Ages par Decennie'].astype(str)

    # Vérifiez si la colonne 'age_bins' dans le DataFrame client n'est pas vide
    if not client['age_bins'].empty:
        # Vérifiez si la décennie du client existe dans le DataFrame
        if client['age_bins'].values[0] in ages_decennie['Ages par Decennie'].values:
            idx_decennie = ages_decennie[ages_decennie['Ages par Decennie'] == client['age_bins'].values[0]].index

            colors = ['lightslategray', ] * len(ages_decennie['Nombre de clients par Decennie'])
            colors[idx_decennie[0]] = 'crimson'

            fig = go.Figure(data=[go.Bar(
                x=ages_decennie['Ages par Decennie'],
                y=ages_decennie['Nombre de clients par Decennie'],
                marker_color=colors
            )])
            fig.update_layout(title_text='Nombre de Clients par Décennie')

            st.plotly_chart(fig)
        else:
            print("La décennie du client n'a pas été trouvée dans les données.")
    else:
        print("La colonne 'age_bins' dans le DataFrame client est vide.")

    # Line Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data['pred_prob'], x=data['AMT_INCOME_TOTAL'],
                             mode='markers', name='Revenus des autres clients'))
    fig.add_trace(go.Scatter(y=client['pred_prob'], x=client['AMT_INCOME_TOTAL'],
                             mode='markers', name='Revenu de ce Client', marker=dict(size=[25])))
    st.plotly_chart(fig)


# --------------------------- Client Prediction --------------------------
def show_client_prediction():
    selected_choice = st.radio("", ('Client existant dans le dataset', 'Nouveau client'))
    
    if selected_choice == 'Client existant dans le dataset':
        client_id = st.number_input("Donnez Id du Client", 100002)
        if st.button('Voir Client'):
            show_existing_client_prediction(client_id)

    if selected_choice == 'Nouveau client':
        filename = st.file_uploader("Sélectionnez le fichier du nouveau client (CSV)", type=["csv"])
        if filename is not None:
            st.write('Fichier du nouveau client sélectionné `%s`' % filename.name)

            if st.button('Prédire Client'):
                nouveau_client = pd.read_csv(filename)
                y_pred, y_proba, shap_values = predict_client("lgbm", nouveau_client)

                st.info('Probabilité de solvabilité du client : ' + str(100 * y_proba) + ' %')
                st.info("Notez que 100% => Client non solvable ")

                # Utilisez la variable seuil_risque ici (définissez-la ou passez-la en tant qu'argument)
                if y_proba < seuil_risque:
                    st.success('Client prédit comme solvable')
                else:
                    st.error('Client prédit comme non solvable !')

                # Affichez les SHAP values
                st.subheader('SHAP Values')
                shap.summary_plot(shap_values, nouveau_client, show=False)
                st.pyplot()

                    
 # ------------------------------------------------------- Sidebar --------------------------

### Title
st.title('Home Credit Default Risk')

### Sidebar
st.sidebar.title("Menus")
sidebar_selection = st.sidebar.radio(
    'Select Menu:',
    ['Overview', 'Data Analysis', 'Model & Prediction', 'Prédire solvabilité client'],
)

if sidebar_selection == 'Overview':
    selected_item = ""
    with st.spinner('Data load in progress...'):
        time.sleep(2)
    st.success('Data loaded')
    show_data()
    show_overview()

if sidebar_selection == 'Data Analysis':
    selected_item = st.sidebar.selectbox('Select Menu:',
                                         ('Graphs', 'Distributions'))

if sidebar_selection == 'Model & Prediction':
    selected_item = st.sidebar.selectbox('Select Menu:',
                                         ('Prediction', 'Model'))

if sidebar_selection == 'Prédire solvabilité client':
    selected_item = "predire_client"

seuil_risque_key = "seuil_risque_sidebar"  # Utilisez une clé unique pour le slider de la barre latérale
seuil_risque = st.sidebar.slider(seuil_risque_key, min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if selected_item == 'Data':
    show_data()

if selected_item == 'Solvency':
    show_overview()

if selected_item == 'Graphs':
    # hist_graph()
    is_educ_selected, is_statut_selected, is_income_selected = filter_graphs()
    if is_educ_selected == "oui":
        education_type()
    if is_statut_selected == "oui":
        statut_plot()
    if is_income_selected == "oui":
        income_type()

if selected_item == 'Distributions':
    is_age_selected, is_incomdis_selected = filter_distribution()
    if is_age_selected == "oui":
        age_distribution()
    if is_incomdis_selected == "oui":
        revenu_distribution()

if selected_item == 'Prediction':
    show_client_predection()

if selected_item == 'Model':
    show_model_analysis()

if selected_item == 'predire_client':
    seuil_risque_container = st.empty()  # Crée un conteneur vide avec une clé unique
    seuil_risque_key = "seuil_risque_" + str(id(seuil_risque_container))  # Utilisez l'ID du conteneur comme clé unique
    seuil_risque = seuil_risque_container.slider(seuil_risque_key, min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    show_client_prediction()
    
    
    
    
    
    
    



#print(response.text)
