# opti_Indoor
Optimisation of an Electrolyte for Indoor DSSCs

# Dye sensitized solar cells Optimization with XGBoost for Indoor application

This repository contains the python script for predicting and visualizing the **open-circuit voltage (Voc)** of dye sensitized solar cells based on the concentration of various additives. The model uses **XGBoost** to map the relationship between additive concentrations and Voc.

---
The script:
1. Trains an **optimized XGBoost regressor** on experimental data.
2. Evaluates model performance using **R², RMSE, and MAE** metrics.
3. Generates **3D yield maps** to visualize how combinations of three additives affect Voc.


## **Model Details**
### Hyperparameters
The XGBoost model is configured with the following optimized hyperparameters:
```python
{
    "learning_rate": 0.01,
    "max_depth": 4,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "eval_metric": "rmse"
}
```
### Performance Metrics
The script outputs:
- **R² Score**: Coefficient of determination (closer to 1 is better).
- **RMSE**: Root Mean Squared Error (lower is better).
- **MAE**: Mean Absolute Error (lower is better).

---
## 📊 **Visualization**
The script generates **3D yield maps** for every combination of three additives (e.g., `[I2]`, `[LiI]`, `[BMII]`). Each plot:
- Uses **Plotly** for interactive exploration.
- Colors points by predicted **Voc**.
- Fixes the remaining two additives at their **mean values** from the dataset.

Example:

<img width="390" height="312" alt="image" src="https://github.com/user-attachments/assets/9c1ca4e6-65b1-4916-a4a1-78b87685ef6f" />


</canvaentity
---

