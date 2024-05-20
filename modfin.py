# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:53:17 2024

@author: aicha
"""

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Fonction pour charger et prétraiter les données
def load_data(file_path, sheet_name='Feuil1'):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# Charger le jeu de données par défaut
default_file_path = 'C:/Users/aicha/OneDrive/Documents/A MASERATI/M2 MASERATI/Mémoire 2023-2024/BDD/AGREGEE.xlsx'
df = load_data(default_file_path)

# Interface Streamlit
st.title("Prédiction de Médailles aux JO 2024")
st.image("C:/Users/aicha/OneDrive/Documents/A MASERATI/M2 MASERATI/Mémoire 2023-2024/BDD/JO.jpg", caption='Image des Jeux Olympiques', width=300)


# Permettre aux utilisateurs de télécharger leur propre jeu de données
uploaded_file = st.file_uploader("Téléchargez votre propre jeu de données (fichier Excel)", type="xlsx")
if uploaded_file is not None:
    df = load_data(uploaded_file)


# Afficher les premières lignes du DataFrame
st.write("Aperçu du jeu de données :")
st.write(df.head())

# Sélection des caractéristiques et de la cible
features = ['entity', 'population_t_4', 'GDP_T_4', 'number_of_athletes', 'number_of_sports',
            'total_t_4', 'Athlete_Category', 'host']
target_options = ['Gold', 'silver', 'bronze', 'total']

categorical_features = ['entity', 'Athlete_Category', 'host']
numeric_features = ['population_t_4', 'GDP_T_4', 'number_of_athletes', 'number_of_sports', 'total_t_4']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

def train_model(target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Régression Linéaire': Pipeline(steps=[('preprocessor', preprocessor),
                                               ('regressor', LinearRegression())]),
        'Forêt Aléatoire': Pipeline(steps=[('preprocessor', preprocessor),
                                           ('regressor', RandomForestRegressor(random_state=42))]),
        'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', GradientBoostingRegressor(random_state=42))]),
        'SVR': Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', SVR())]),
        'MLPRegressor': Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', MLPRegressor(random_state=42))])
    }
    
    model_performance = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        model_performance[model_name] = {'model': model, 'mse': mse, 'r2': r2, 'mae': mae}
        print(f"Modèle : {model_name}, Cible : {target} - MSE : {mse}, R² : {r2}, MAE : {mae}")
    
    return model_performance

# Entraîner les modèles pour chaque cible
models = {target: train_model(target) for target in target_options}


# Sélectionner le pays et le type de médaille
selected_country = st.selectbox("Sélectionnez un pays", df['entity'].unique())
selected_medal_type = st.selectbox("Sélectionnez le type de médaille", target_options)
selected_model = st.selectbox("Sélectionnez le modèle", ['Régression Linéaire', 'Forêt Aléatoire', 'Gradient Boosting', 'SVR', 'MLPRegressor'])

if st.button("Prédire"):
    # Convertir en DataFrame
    country_data = df[df['entity'] == selected_country][features].iloc[0].to_frame().T
    model_info = models[selected_medal_type][selected_model]
    model = model_info['model']
    prediction = model.predict(country_data)
    rounded_prediction = max(0, int(round(prediction[0])))  # Arrondir la prédiction au nombre entier le plus proche et s'assurer qu'elle est non négative
    st.write(f"Nombre prédit de médailles {selected_medal_type.lower()} pour {selected_country} : {rounded_prediction}")
    
    # Afficher les performances du modèle
    st.write(f"Performances du modèle {selected_model} ({selected_medal_type}) :")
    st.write(f"Erreur Quadratique Moyenne : {model_info['mse']}")
    st.write(f"Score R² : {model_info['r2']}")
    st.write(f"Erreur Absolue Moyenne : {model_info['mae']}")

# Comparaison des modèles
st.title("Comparaison des Modèles")

# Sélectionner la cible pour la comparaison
selected_comparison_target = st.selectbox("Sélectionnez la cible pour la comparaison des modèles", target_options)

# Extraire les performances des modèles pour la cible sélectionnée
comparison_data = []
for model_name, metrics in models[selected_comparison_target].items():
    comparison_data.append({
        'Modèle': model_name,
        'MSE': metrics['mse'],
        'R²': metrics['r2'],
        'MAE': metrics['mae']
    })

# Convertir en DataFrame pour l'affichage
comparison_df = pd.DataFrame(comparison_data)

# Afficher le tableau de comparaison des modèles
st.write("Comparaison des performances des modèles :")
st.dataframe(comparison_df)

# Visualisation des performances des modèles
fig, ax = plt.subplots(figsize=(10, 6))
comparison_df.set_index('Modèle')[['MSE', 'R²', 'MAE']].plot(kind='bar', ax=ax)
plt.title(f"Comparaison des Modèles pour {selected_comparison_target}")
plt.ylabel("Score")
plt.xlabel("Modèle")
plt.xticks(rotation=45)
st.pyplot(fig)

# Visualisation des données historiques
st.title("Performance Historique")
country_historical_data = df[df['entity'] == selected_country]
fig, ax = plt.subplots()
for medal in target_options:
    ax.plot(country_historical_data['year'], country_historical_data[medal], label=medal)
ax.set_xlabel('Année')
ax.set_ylabel('Nombre de Médailles')
ax.set_title(f'Performance Historique de {selected_country}')
ax.legend()
st.pyplot(fig)
