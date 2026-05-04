import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


# Chargement des données
data = pd.read_excel("RK1_fulldataset.xlsx")

# Sélection des features et de la cible
features = ["[I2]", "[LiI]", "[BMII]", "[TBP]", "[GuSCN]"]
target = "Voc"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Séparation features / target
X = data[features]
y = data[target]

# Normalisation des données
scaler = StandardScaler()
scaled = scaler.fit_transform(X)

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    scaled, y, test_size=0.2, random_state=42
)

# ----- Modèle XGBoost optimisé -----
best_xgb_model = XGBRegressor(
    learning_rate=0.01,
    max_depth=4,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric='rmse'
)

# Entraînement du modèle
best_xgb_model.fit(X_train, y_train)

# Prédictions
y_pred = best_xgb_model.predict(X_test)

# Évaluation des performances
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Modèle: XGBoost optimisé")
print(f"Paramètres utilisés: {{'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.8}}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
import itertools
import numpy as np
import pandas as pd
import plotly.express as px

# Liste des paramètres
params = ["[I2]", "[LiI]", "[BMII]", "[TBP]", "[GuSCN]"]

# Générer toutes les combinaisons possibles de trois paramètres
combinations = list(itertools.combinations(params, 3))

# Parcourir chaque combinaison et tracer le graphique
for combo in combinations:
    param1, param2, param3 = combo

    # Définir les plages de valeurs pour les trois paramètres
    param1_range = np.linspace(start=min(data[param1]), stop=max(data[param1]), num=20)
    param2_range = np.linspace(start=min(data[param2]), stop=max(data[param2]), num=20)
    param3_range = np.linspace(start=min(data[param3]), stop=max(data[param3]), num=20)

    # Générer une grille de paramètres pour trois dimensions
    param1_grid, param2_grid, param3_grid = np.meshgrid(param1_range, param2_range, param3_range, indexing='ij')

    # Aplatir les grilles pour les utiliser comme entrées du modèle
    flat_param1 = param1_grid.ravel()
    flat_param2 = param2_grid.ravel()
    flat_param3 = param3_grid.ravel()

    # Créer un DataFrame avec les paramètres aplatis
    # Utiliser des valeurs moyennes pour les autres paramètres
    remaining_params = [p for p in params if p not in combo]
    mean_values = {p: data[p].mean() for p in remaining_params}

    grid_df = pd.DataFrame({
        param1: flat_param1,
        param2: flat_param2,
        param3: flat_param3,
    })

    for p, mean_value in mean_values.items():
        grid_df[p] = mean_value

    # Réorganiser les colonnes pour correspondre à l'ordre d'origine
    grid_df = grid_df[params]

    # Normaliser les données de la grille
    grid_scaled = scaler.transform(grid_df)

    # Prédire les rendements avec XGBoost optimisé
    grid_y_pred = best_xgb_model.predict(grid_scaled)

    # Remodeler les prédictions pour correspondre à la grille
    yield_map = grid_y_pred.reshape(param1_grid.shape)

    # Créer un DataFrame pour Plotly
    plot_df = pd.DataFrame({
        param1: flat_param1,
        param2: flat_param2,
        param3: flat_param3,
        'Voc': grid_y_pred
    })

    # Tracer les cartographies en 3D avec Plotly
    fig = px.scatter_3d(
        plot_df,
        x=param1, y=param2, z=param3,
        color='Voc',
        title=f'Cartographie des rendements prédits en 3D: {param1}, {param2}, {param3}',
        labels={'color': 'Rendement (PCE)'}
    )

    fig.update_traces(marker=dict(size=3))
    fig.show()

