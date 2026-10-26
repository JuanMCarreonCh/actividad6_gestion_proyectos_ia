"""
Script de Experimentación y Validación de Modelos de Machine Learning
Curso: Gestión de Proyectos de Inteligencia Artificial - Actividad 6
Problema: Evaluación y Validación de Modelos de Riesgo Crediticio (Credit Default)

Autor: Maestría en Inteligencia Artificial
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score, brier_score_loss
)

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
np.random.seed(42)

os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("="*75)
print("1. GENERACIÓN Y PREPARACIÓN DEL CONJUNTO DE DATOS (CREDIT RISK PORTFOLIO)")
print("="*75)

# Dataset realista de 12,500 solicitudes con no-linealidades e interacciones de riesgo financiero
n_samples = 12500
np.random.seed(42)

income = np.random.lognormal(mean=8.25, sigma=0.55, size=n_samples) # Mediana ~$3,800 USD
loan_amount = np.random.lognormal(mean=9.2, sigma=0.6, size=n_samples) # Mediana ~$10,000 USD
credit_score = np.clip(np.random.normal(loc=655, scale=75, size=n_samples), 350, 850)
dti = np.clip(np.random.beta(a=2.5, b=5.0, size=n_samples) * 0.75, 0.02, 0.85)
revolving_util = np.clip(np.random.beta(a=3.0, b=3.5, size=n_samples), 0.01, 1.15)
delinq_2yrs = np.random.poisson(lam=0.35, size=n_samples)
inquiries_last_6mths = np.random.poisson(lam=0.8, size=n_samples)
emp_length = np.clip(np.random.exponential(scale=6.0, size=n_samples), 0, 35)
credit_history_years = np.clip(np.random.normal(loc=12, scale=5, size=n_samples), 1, 40)
loan_to_income = loan_amount / (income * 12)

# Interacciones financieras reales y no-linealidades:
# 1. Estrés de deuda: DTI alto combinado con ingresos bajos
debt_stress = (dti > 0.40) & (income < 3000)
# 2. Hambre de crédito: múltiples consultas recientes con alta utilización rotativa
credit_hunger = (inquiries_last_6mths >= 2) & (revolving_util > 0.70)
# 3. Sobreendeudamiento: préstamo alto respecto al ingreso con score bajo
high_burden = (loan_to_income > 0.35) & (credit_score < 620)
# 4. Historial adverso: moras recientes con score deteriorado
severe_delinq = (delinq_2yrs >= 1) & (credit_score < 600)

logit_p = (
    -0.014 * (credit_score - 650)
    + 4.5 * (dti - 0.35)
    + 3.2 * (revolving_util - 0.45)
    + 0.85 * delinq_2yrs
    + 0.55 * inquiries_last_6mths
    - 0.08 * emp_length
    + 2.2 * (loan_to_income - 0.25)
    - 0.04 * credit_history_years
    + 2.0 * debt_stress.astype(float)
    + 1.8 * credit_hunger.astype(float)
    + 2.2 * high_burden.astype(float)
    + 2.4 * severe_delinq.astype(float)
    - 2.85 # Calibración a prevalencia ~16%
)

prob_default = 1 / (1 + np.exp(-logit_p))
y = (np.random.binomial(1, prob_default)).astype(int)

df = pd.DataFrame({
    'income': np.round(income, 2),
    'loan_amount': np.round(loan_amount, 2),
    'credit_score': np.round(credit_score, 1),
    'dti': np.round(dti, 4),
    'revolving_util': np.round(revolving_util, 4),
    'delinq_2yrs': delinq_2yrs,
    'inquiries_last_6mths': inquiries_last_6mths,
    'emp_length': np.round(emp_length, 1),
    'credit_history_years': np.round(credit_history_years, 1),
    'loan_to_income': np.round(loan_to_income, 4),
    'default': y
})

df.to_csv('data/credit_risk_dataset.csv', index=False)
actual_default_rate = df['default'].mean()
print(f"Dataset generado exitosamente: {df.shape[0]} registros, {df.shape[1]-1} variables predictoras.")
print(f"Tasa de Incumplimiento (Default Prevalencia): {actual_default_rate*100:.2f}% (Casos positivos: {df['default'].sum()})")

# División Train / Test (80% / 20%) estratificada
X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train.shape[0]} muestras | Test set: {X_test.shape[0]} muestras")

print("\n" + "="*75)
print("2. ENTRENAMIENTO DE MODELOS Y COMPARACIÓN CON BASELINES")
print("="*75)

models = {
    'Dummy_Most_Frequent': DummyClassifier(strategy='most_frequent'),
    'Dummy_Stratified': DummyClassifier(strategy='stratified', random_state=42),
    'Regresion_Logistica_Base': LogisticRegression(max_iter=1000, random_state=42),
    'Regresion_Logistica_Balanceada': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=8, random_state=42, n_jobs=-1),
    'Gradient_Boosting': GradientBoostingClassifier(n_estimators=220, learning_rate=0.06, max_depth=5, subsample=0.85, random_state=42)
}

# Validación Cruzada Estratificada 5-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_summary = []
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print("\nResultados de Validación Cruzada (Stratified 5-Fold) en Train:")
for name, model in models.items():
    if 'Logistica' in name:
        scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    else:
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    res = {
        'Modelo': name,
        'CV_Accuracy_Mean': scores['test_accuracy'].mean(),
        'CV_Accuracy_Std': scores['test_accuracy'].std(),
        'CV_Precision_Mean': scores['test_precision'].mean(),
        'CV_Precision_Std': scores['test_precision'].std(),
        'CV_Recall_Mean': scores['test_recall'].mean(),
        'CV_Recall_Std': scores['test_recall'].std(),
        'CV_F1_Mean': scores['test_f1'].mean(),
        'CV_F1_Std': scores['test_f1'].std(),
        'CV_ROC_AUC_Mean': scores['test_roc_auc'].mean(),
        'CV_ROC_AUC_Std': scores['test_roc_auc'].std(),
    }
    cv_results_summary.append(res)
    print(f"-> {name:32s} | ROC-AUC: {res['CV_ROC_AUC_Mean']:.4f} (+/- {res['CV_ROC_AUC_Std']:.4f}) | Recall: {res['CV_Recall_Mean']:.4f} | F1: {res['CV_F1_Mean']:.4f}")

df_cv = pd.DataFrame(cv_results_summary)
df_cv.to_csv('reports/cross_validation_results.csv', index=False)

# Ajuste Final y Evaluación en Test Holdout (Umbral 0.50)
test_evaluations = []
fitted_models = {}
y_probs = {}

for name, model in models.items():
    if 'Logistica' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
    fitted_models[name] = model
    y_probs[name] = y_prob
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    except:
        auc = 0.5
        pr_auc = actual_default_rate
        
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    test_evaluations.append({
        'Modelo': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'F2_Score': f2,
        'ROC_AUC': auc,
        'PR_AUC': pr_auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    })

df_test = pd.DataFrame(test_evaluations)
df_test.to_csv('reports/test_holdout_evaluations_threshold_050.csv', index=False)
print("\n" + "="*75)
print("EVALUACIÓN EN TEST HOLDOUT (UMBRAL = 0.50):")
print("="*75)
print(df_test[['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC']].to_string(index=False))

print("\n" + "="*75)
print("3. ANÁLISIS Y AJUSTE DE UMBRAL (THRESHOLD TUNING)")
print("="*75)

# Modelo principal seleccionado: Gradient Boosting Classifier
target_model_name = 'Gradient_Boosting'
y_test_probs = y_probs[target_model_name]

# Matriz de Costo-Beneficio Financiero Real en Banca de Consumo:
# Préstamo promedio: $10,000 USD
# - False Negative (FN): Préstamo aprobado a cliente insolvente. Pérdida severa del principal = $7,500 USD (EAD * LGD = $10,000 * 75%).
# - False Positive (FP): Solicitud solvente rechazada por falsa alarma. Costo de oportunidad = margen financiero neto perdido = $1,500 USD.
# - True Negative (TN): Cliente solvente aprobado exitosamente. Ganancia de intereses neta = +$1,500 USD.
# - True Positive (TP): Cliente riesgoso detectado y declinado a tiempo. Evita pérdida de $7,500 USD (beneficio = $0 ganancia directa pero ahorro de pérdida).

COST_FN = 7500.0  # Pérdida directa por default no anticipado
COST_FP = 1500.0  # Costo de oportunidad por cliente solvente rechazado
GAIN_TN = 1500.0  # Margen financiero neto ganado por cliente solvente aprobado

thresholds = np.linspace(0.01, 0.99, 100)
threshold_metrics = []

for t in thresholds:
    preds = (y_test_probs >= t).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
    
    expected_loss = (fn * COST_FN) + (fp * COST_FP)
    net_profit = (tn * GAIN_TN) - (fn * COST_FN) - (fp * COST_FP)
    
    threshold_metrics.append({
        'threshold': t,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'f2': f2,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'expected_loss': expected_loss,
        'net_profit': net_profit,
        'approval_rate': (tn + fn) / len(y_test)
    })

df_thresholds = pd.DataFrame(threshold_metrics)
df_thresholds.to_csv('reports/threshold_tuning_analysis.csv', index=False)

idx_opt_f1 = df_thresholds['f1'].idxmax()
t_opt_f1 = df_thresholds.loc[idx_opt_f1, 'threshold']

idx_opt_f2 = df_thresholds['f2'].idxmax()
t_opt_f2 = df_thresholds.loc[idx_opt_f2, 'threshold']

idx_opt_profit = df_thresholds['net_profit'].idxmax()
t_opt_profit = df_thresholds.loc[idx_opt_profit, 'threshold']

row_050 = df_thresholds.iloc[(df_thresholds['threshold']-0.50).abs().argsort()[:1]].iloc[0]
row_f1 = df_thresholds.loc[idx_opt_f1]
row_f2 = df_thresholds.loc[idx_opt_f2]
row_profit = df_thresholds.loc[idx_opt_profit]

print(f"Umbral Estándar (t=0.50):")
print(f"  Precision: {row_050['precision']:.4f} | Recall: {row_050['recall']:.4f} | F1: {row_050['f1']:.4f} | Utilidad Neta: ${row_050['net_profit']:,.2f} | Pérdida: ${row_050['expected_loss']:,.2f}")

print(f"\nUmbral Óptimo por F1-Score (t={t_opt_f1:.2f}):")
print(f"  Precision: {row_f1['precision']:.4f} | Recall: {row_f1['recall']:.4f} | F1: {row_f1['f1']:.4f} | Utilidad Neta: ${row_f1['net_profit']:,.2f}")

print(f"\nUmbral Óptimo por F2-Score (t={t_opt_f2:.2f}):")
print(f"  Precision: {row_f2['precision']:.4f} | Recall: {row_f2['recall']:.4f} | F1: {row_f2['f1']:.4f} | Utilidad Neta: ${row_f2['net_profit']:,.2f}")

print(f"\nUmbral Óptimo de Maximización Financiera (t={t_opt_profit:.2f}):")
print(f"  Precision: {row_profit['precision']:.4f} | Recall: {row_profit['recall']:.4f} | F1: {row_profit['f1']:.4f} | Utilidad Neta: ${row_profit['net_profit']:,.2f} | Pérdida: ${row_profit['expected_loss']:,.2f}")

profit_improvement = row_profit['net_profit'] - row_050['net_profit']
print(f"-> Ganancia incremental en muestra test al optimizar umbral: +${profit_improvement:,.2f} USD")

# Guardar matrices de confusión antes y después
cm_050 = confusion_matrix(y_test, (y_test_probs >= 0.50).astype(int))
cm_opt = confusion_matrix(y_test, (y_test_probs >= t_opt_profit).astype(int))

print("\n" + "="*75)
print("4. SIMULACIÓN DE PRUEBA A/B Y ANÁLISIS DE SIGNIFICANCIA ESTADÍSTICA")
print("="*75)

# Escenario A/B Testing:
# Cohorte de 6,000 nuevas solicitudes en producción (período de prueba de 60 días)
# Asignación aleatoria balanceada 1:1
# Grupo A (Control, 3000): Regresión Logística Estándar (t=0.50)
# Grupo B (Tratamiento, 3000): Gradient Boosting con Umbral Optimizado por Utilidad (t=t_opt_profit)

n_ab_total = 6000
np.random.seed(888)

income_ab = np.random.lognormal(mean=8.25, sigma=0.55, size=n_ab_total)
loan_amount_ab = np.random.lognormal(mean=9.2, sigma=0.6, size=n_ab_total)
credit_score_ab = np.clip(np.random.normal(loc=655, scale=75, size=n_ab_total), 350, 850)
dti_ab = np.clip(np.random.beta(a=2.5, b=5.0, size=n_ab_total) * 0.75, 0.02, 0.85)
revolving_util_ab = np.clip(np.random.beta(a=3.0, b=3.5, size=n_ab_total), 0.01, 1.15)
delinq_2yrs_ab = np.random.poisson(lam=0.35, size=n_ab_total)
inquiries_last_6mths_ab = np.random.poisson(lam=0.8, size=n_ab_total)
emp_length_ab = np.clip(np.random.exponential(scale=6.0, size=n_ab_total), 0, 35)
credit_history_years_ab = np.clip(np.random.normal(loc=12, scale=5, size=n_ab_total), 1, 40)
loan_to_income_ab = loan_amount_ab / (income_ab * 12)

debt_stress_ab = (dti_ab > 0.40) & (income_ab < 3000)
credit_hunger_ab = (inquiries_last_6mths_ab >= 2) & (revolving_util_ab > 0.70)
high_burden_ab = (loan_to_income_ab > 0.35) & (credit_score_ab < 620)
severe_delinq_ab = (delinq_2yrs_ab >= 1) & (credit_score_ab < 600)

logit_p_ab = (
    -0.014 * (credit_score_ab - 650)
    + 4.5 * (dti_ab - 0.35)
    + 3.2 * (revolving_util_ab - 0.45)
    + 0.85 * delinq_2yrs_ab
    + 0.55 * inquiries_last_6mths_ab
    - 0.08 * emp_length_ab
    + 2.2 * (loan_to_income_ab - 0.25)
    - 0.04 * credit_history_years_ab
    + 2.0 * debt_stress_ab.astype(float)
    + 1.8 * credit_hunger_ab.astype(float)
    + 2.2 * high_burden_ab.astype(float)
    + 2.4 * severe_delinq_ab.astype(float)
    - 2.85
)
prob_default_ab = 1 / (1 + np.exp(-logit_p_ab))
ground_truth_default_ab = (np.random.binomial(1, prob_default_ab)).astype(int)

df_ab = pd.DataFrame({
    'income': np.round(income_ab, 2),
    'loan_amount': np.round(loan_amount_ab, 2),
    'credit_score': np.round(credit_score_ab, 1),
    'dti': np.round(dti_ab, 4),
    'revolving_util': np.round(revolving_util_ab, 4),
    'delinq_2yrs': delinq_2yrs_ab,
    'inquiries_last_6mths': inquiries_last_6mths_ab,
    'emp_length': np.round(emp_length_ab, 1),
    'credit_history_years': np.round(credit_history_years_ab, 1),
    'loan_to_income': np.round(loan_to_income_ab, 4),
    'ground_truth_default': ground_truth_default_ab
})

# Asignación aleatoria 50/50
ab_group = np.random.choice(['A_Control', 'B_Treatment'], size=n_ab_total, p=[0.5, 0.5])
df_ab['group'] = ab_group

# Evaluación Grupo A (Control: Logística Base con corte 0.50)
idx_A = (df_ab['group'] == 'A_Control')
X_A = df_ab.loc[idx_A, X.columns]
X_A_scaled = scaler.transform(X_A)
model_A = fitted_models['Regresion_Logistica_Base']
probs_A = model_A.predict_proba(X_A_scaled)[:, 1]
predict_default_A = (probs_A >= 0.50).astype(int)
df_ab.loc[idx_A, 'predicted_default'] = predict_default_A
df_ab.loc[idx_A, 'approved'] = (predict_default_A == 0).astype(int)

# Evaluación Grupo B (Tratamiento: Gradient Boosting con corte óptimo t_opt_profit)
idx_B = (df_ab['group'] == 'B_Treatment')
X_B = df_ab.loc[idx_B, X.columns]
model_B = fitted_models['Gradient_Boosting']
probs_B = model_B.predict_proba(X_B)[:, 1]
predict_default_B = (probs_B >= t_opt_profit).astype(int)
df_ab.loc[idx_B, 'predicted_default'] = predict_default_B
df_ab.loc[idx_B, 'approved'] = (predict_default_B == 0).astype(int)

# Resumen de Métricas de Negocio A/B
ab_summary = []
for grp in ['A_Control', 'B_Treatment']:
    sub = df_ab[df_ab['group'] == grp]
    n_solicitudes = len(sub)
    n_aprobados = sub['approved'].sum()
    pct_aprobacion = n_aprobados / n_solicitudes
    
    n_defaults_observados = sub[sub['approved'] == 1]['ground_truth_default'].sum()
    tasa_mora_cartera = n_defaults_observados / n_aprobados
    n_defaults_evitados = sub[(sub['approved'] == 0) & (sub['ground_truth_default'] == 1)]['ground_truth_default'].sum()
    
    # Utilidad neta generada
    utilidad = ((n_aprobados - n_defaults_observados) * GAIN_TN) - (n_defaults_observados * COST_FN)
    
    ab_summary.append({
        'Grupo': grp,
        'Modelo_Asignado': 'Regresión Logística (t=0.50)' if grp == 'A_Control' else f'Gradient Boosting (t={t_opt_profit:.2f})',
        'Solicitudes_Totales': n_solicitudes,
        'Creditos_Aprobados': n_aprobados,
        'Tasa_Aprobacion': pct_aprobacion,
        'Defaults_en_Cartera': n_defaults_observados,
        'Tasa_Mora_Cartera': tasa_mora_cartera,
        'Defaults_Evitados': n_defaults_evitados,
        'Utilidad_Financiera': utilidad
    })

df_ab_metrics = pd.DataFrame(ab_summary)
df_ab_metrics.to_csv('reports/ab_test_metrics_summary.csv', index=False)
print(df_ab_metrics.to_string(index=False))

# Test de Significancia Estadística: Two-sample proportion Z-test
count_defaults = [
    df_ab[(df_ab['group'] == 'A_Control') & (df_ab['approved'] == 1)]['ground_truth_default'].sum(),
    df_ab[(df_ab['group'] == 'B_Treatment') & (df_ab['approved'] == 1)]['ground_truth_default'].sum()
]
nobs_approved = [
    df_ab[(df_ab['group'] == 'A_Control')]['approved'].sum(),
    df_ab[(df_ab['group'] == 'B_Treatment')]['approved'].sum()
]

z_stat, p_val_two_sided = proportions_ztest(count_defaults, nobs_approved, alternative='two-sided')
_, p_val_one_sided = proportions_ztest(count_defaults, nobs_approved, alternative='larger')

p_A = count_defaults[0] / nobs_approved[0]
p_B = count_defaults[1] / nobs_approved[1]
diff = p_A - p_B
se_diff = np.sqrt((p_A * (1 - p_A) / nobs_approved[0]) + (p_B * (1 - p_B) / nobs_approved[1]))
ci_lower = diff - 1.96 * se_diff
ci_upper = diff + 1.96 * se_diff

print("\nResultados del Test de Hipótesis (Two-sample proportion Z-test):")
print(f"  Tasa de Mora Grupo A (Control): {p_A*100:.2f}% ({count_defaults[0]}/{int(nobs_approved[0])})")
print(f"  Tasa de Mora Grupo B (Tratamiento): {p_B*100:.2f}% ({count_defaults[1]}/{int(nobs_approved[1])})")
print(f"  Diferencia Absoluta de Mora: {diff*100:.2f}% puntos porcentuales")
print(f"  Reducción Relativa de Incumplimiento: {((p_A - p_B) / p_A)*100:.2f}%")
print(f"  Intervalo de Confianza al 95%: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
print(f"  Estadístico Z: {z_stat:.4f}")
print(f"  p-value (unilateral, H1: Mora_A > Mora_B): {p_val_one_sided:.6e}")
print(f"  p-value (bilateral): {p_val_two_sided:.6e}")
print(f"  Conclusión Estadística (alfa=0.01): {'RECHAZA H0 (Efecto Altamente Significativo)' if p_val_two_sided < 0.01 else 'NO RECHAZA H0'}")

ab_stats_results = {
    'group_A_defaults': int(count_defaults[0]),
    'group_A_approved': int(nobs_approved[0]),
    'group_A_default_rate': float(p_A),
    'group_B_defaults': int(count_defaults[1]),
    'group_B_approved': int(nobs_approved[1]),
    'group_B_default_rate': float(p_B),
    'diff_default_rate': float(diff),
    'relative_reduction_pct': float(((p_A - p_B) / p_A)*100),
    'ci_95_lower': float(ci_lower),
    'ci_95_upper': float(ci_upper),
    'z_statistic': float(z_stat),
    'p_value_one_sided': float(p_val_one_sided),
    'p_value_two_sided': float(p_val_two_sided),
    'utility_A': float(df_ab_metrics.loc[0, 'Utilidad_Financiera']),
    'utility_B': float(df_ab_metrics.loc[1, 'Utilidad_Financiera']),
    'profit_delta': float(df_ab_metrics.loc[1, 'Utilidad_Financiera'] - df_ab_metrics.loc[0, 'Utilidad_Financiera'])
}

with open('reports/ab_testing_statistical_results.json', 'w') as f:
    json.dump(ab_stats_results, f, indent=4)

print("\n" + "="*75)
print("5. GENERACIÓN DE VISUALIZACIONES GRÁFICAS DE ALTA DEFINICIÓN")
print("="*75)

# Gráfico 1: Matrices de Confusión Comparativas
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cm_050, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False, annot_kws={"size": 13, "weight": "bold"})
axes[0].set_title(f'Matriz de Confusión: Umbral Estándar (t = 0.50)\nRecall: {row_050["recall"]*100:.1f}% | Precision: {row_050["precision"]*100:.1f}% | FN: {cm_050[1,0]}', fontsize=11, fontweight='bold', pad=12)
axes[0].set_xlabel('Predicción del Modelo', fontsize=11)
axes[0].set_ylabel('Realidad Observada', fontsize=11)
axes[0].set_xticklabels(['Solvente (0)', 'Default (1)'])
axes[0].set_yticklabels(['Solvente (0)', 'Default (1)'])

sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False, annot_kws={"size": 13, "weight": "bold"})
axes[1].set_title(f'Matriz de Confusión: Umbral Óptimo Costo (t = {t_opt_profit:.2f})\nRecall: {row_profit["recall"]*100:.1f}% | Precision: {row_profit["precision"]*100:.1f}% | FN: {cm_opt[1,0]}', fontsize=11, fontweight='bold', pad=12)
axes[1].set_xlabel('Predicción del Modelo', fontsize=11)
axes[1].set_ylabel('Realidad Observada', fontsize=11)
axes[1].set_xticklabels(['Solvente (0)', 'Default (1)'])
axes[1].set_yticklabels(['Solvente (0)', 'Default (1)'])

plt.tight_layout()
plt.savefig('figures/1_confusion_matrices_comparison.png', dpi=300)
plt.close()

# Gráfico 2: ROC y Precision-Recall
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors = {
    'Dummy_Most_Frequent': '#9E9E9E',
    'Dummy_Stratified': '#757575',
    'Regresion_Logistica_Base': '#FF9800',
    'Regresion_Logistica_Balanceada': '#FF5722',
    'Random_Forest': '#4CAF50',
    'Gradient_Boosting': '#1976D2'
}

labels_clean = {
    'Dummy_Most_Frequent': 'Baseline Mayoría',
    'Dummy_Stratified': 'Baseline Aleatorio Estratificado',
    'Regresion_Logistica_Base': 'Regresión Logística Estándar',
    'Regresion_Logistica_Balanceada': 'Regresión Logística Balanceada',
    'Random_Forest': 'Random Forest Classifier',
    'Gradient_Boosting': 'Gradient Boosting Classifier'
}

for name in models.keys():
    prob = y_probs[name]
    if len(np.unique(prob)) > 1:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        axes[0].plot(fpr, tpr, label=f"{labels_clean[name]} (AUC = {auc_val:.3f})", color=colors[name], lw=2.2)
        
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, prob)
        pr_auc_val = average_precision_score(y_test, prob)
        axes[1].plot(rec_curve, prec_curve, label=f"{labels_clean[name]} (PR-AUC = {pr_auc_val:.3f})", color=colors[name], lw=2.2)

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Línea Aleatoria (AUC = 0.50)')
axes[0].set_title('Curvas ROC (Receiver Operating Characteristic)', fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
axes[0].set_ylabel('Tasa de Verdaderos Positivos (Recall / Sensibilidad)', fontsize=11)
axes[0].legend(loc='lower right', frameon=True, fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].axhline(actual_default_rate, color='k', linestyle='--', lw=1.5, alpha=0.7, label=f'Prevalencia Base ({actual_default_rate*100:.1f}%)')
axes[1].set_title('Curvas Precision-Recall (PRC)', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('Recall (Sensibilidad)', fontsize=11)
axes[1].set_ylabel('Precisión', fontsize=11)
axes[1].legend(loc='lower left', frameon=True, fontsize=9)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/2_roc_and_pr_curves.png', dpi=300)
plt.close()

# Gráfico 3: Optimización de Umbral y Curva Financiera
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(df_thresholds['threshold'], df_thresholds['recall'], label='Recall (Sensibilidad a Defaults)', color='#E91E63', lw=2.5)
ax1.plot(df_thresholds['threshold'], df_thresholds['precision'], label='Precisión', color='#3F51B5', lw=2.5)
ax1.plot(df_thresholds['threshold'], df_thresholds['f1'], label='F1-Score (Balance Armónico)', color='#009688', lw=2.2, linestyle='--')
ax1.plot(df_thresholds['threshold'], df_thresholds['f2'], label='F2-Score (Sensibilidad Asimétrica)', color='#FF9800', lw=2.2, linestyle='-.')
ax1.axvline(0.50, color='gray', linestyle=':', lw=1.8, label='Umbral Estándar (0.50)')
ax1.axvline(t_opt_profit, color='#D32F2F', linestyle='-', lw=2.5, label=f'Umbral Óptimo Costo-Beneficio ({t_opt_profit:.2f})')
ax1.set_ylabel('Valor de Métrica (0 - 1)', fontsize=11, fontweight='bold')
ax1.set_title('Curvas de Trade-Off de Métricas Técnicas por Umbral', fontsize=13, fontweight='bold')
ax1.legend(loc='center right', frameon=True, fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(df_thresholds['threshold'], df_thresholds['net_profit'] / 1e6, label='Utilidad Neta de Cartera ($M USD)', color='#2E7D32', lw=2.8)
ax2.plot(df_thresholds['threshold'], df_thresholds['expected_loss'] / 1e6, label='Pérdida Financiera Esperada ($M USD)', color='#C62828', lw=2.2, linestyle='--')
ax2.axvline(0.50, color='gray', linestyle=':', lw=1.8)
ax2.axvline(t_opt_profit, color='#D32F2F', linestyle='-', lw=2.5)
ax2.scatter([0.50], [row_050['net_profit'] / 1e6], color='gray', s=90, zorder=5)
ax2.scatter([t_opt_profit], [row_profit['net_profit'] / 1e6], color='#2E7D32', s=130, zorder=5)

ax2.annotate(f"Estándar (0.50): ${row_050['net_profit']/1e6:.2f}M",
             xy=(0.50, row_050['net_profit'] / 1e6),
             xytext=(0.55, (row_050['net_profit'] / 1e6) - 0.35),
             arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6),
             fontweight='bold', fontsize=10)
ax2.annotate(f"Óptimo ({t_opt_profit:.2f}): ${row_profit['net_profit']/1e6:.2f}M\n(Beneficio Adicional: +${profit_improvement:,.0f} USD)",
             xy=(t_opt_profit, row_profit['net_profit'] / 1e6),
             xytext=(t_opt_profit - 0.22, (row_profit['net_profit'] / 1e6) + 0.15),
             arrowprops=dict(facecolor='#2E7D32', shrink=0.05, width=1.5, headwidth=7),
             fontweight='bold', fontsize=10, color='#1B5E20')

ax2.set_xlabel('Umbral de Probabilidad de Corte (Decision Threshold)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Impacto Financiero ($M USD)', fontsize=11, fontweight='bold')
ax2.set_title('Función de Utilidad Financiera Neta y Pérdida por Incumplimiento', fontsize=13, fontweight='bold')
ax2.legend(loc='lower left', frameon=True, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/3_threshold_optimization_and_financial_loss.png', dpi=300)
plt.close()

# Gráfico 4: Validación Cruzada (Boxplots)
cv_data_boxplot = []
for name, model in models.items():
    if 'Dummy' in name:
        continue
    if 'Logistica' in name:
        scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=['roc_auc', 'recall', 'f1', 'precision'])
    else:
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=['roc_auc', 'recall', 'f1', 'precision'])
    for fold_idx in range(5):
        cv_data_boxplot.append({
            'Modelo': labels_clean[name],
            'ROC_AUC': scores['test_roc_auc'][fold_idx],
            'Recall': scores['test_recall'][fold_idx],
            'F1_Score': scores['test_f1'][fold_idx],
            'Precision': scores['test_precision'][fold_idx]
        })

df_cv_box = pd.DataFrame(cv_data_boxplot)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, metric, pal in zip(
    axes.flat, 
    ['ROC_AUC', 'Recall', 'F1_Score', 'Precision'], 
    ['Blues_r', 'Purples_r', 'Greens_r', 'Oranges_r']
):
    sns.boxplot(data=df_cv_box, x='Modelo', y=metric, hue='Modelo', ax=ax, palette=pal, legend=False)
    ax.set_title(f'Distribución {metric} (Stratified 5-Fold CV)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/4_cross_validation_distributions.png', dpi=300)
plt.close()

# Gráfico 5: Experimento A/B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

groups = ['Grupo A (Control)\nRegresión Logística (t=0.50)', f'Grupo B (Tratamiento)\nGradient Boosting (t={t_opt_profit:.2f})']
mora_vals = [p_A * 100, p_B * 100]
bars1 = ax1.bar(groups, mora_vals, color=['#EF5350', '#66BB6A'], edgecolor=['#B71C1C', '#1B5E20'], width=0.55, linewidth=2)
ax1.set_ylabel('Tasa de Mora en Cartera Otorgada (%)', fontsize=11, fontweight='bold')
ax1.set_title(f'Tasa de Mora Observada en Producción\n(Reducción: {((p_A-p_B)/p_A)*100:.1f}%, p-value={p_val_one_sided:.2e})', fontsize=12, fontweight='bold')
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.25, f"{yval:.2f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(mora_vals) * 1.3)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

util_vals = [df_ab_metrics.loc[0, 'Utilidad_Financiera'] / 1e6, df_ab_metrics.loc[1, 'Utilidad_Financiera'] / 1e6]
bars2 = ax2.bar(groups, util_vals, color=['#90CAF9', '#4CAF50'], edgecolor=['#1565C0', '#1B5E20'], width=0.55, linewidth=2)
ax2.set_ylabel('Utilidad Financiera Neta ($ Millones USD)', fontsize=11, fontweight='bold')
ax2.set_title(f'Margen Financiero Neto Generado\n(Impacto Adicional Grupo B: +${(util_vals[1]-util_vals[0])*1e6:,.0f} USD)', fontsize=12, fontweight='bold')
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"${yval:.2f}M", ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(util_vals) * 1.25)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/5_ab_test_experiment_results.png', dpi=300)
plt.close()

# Gráfico 6: Importancia de Variables
gb_model = fitted_models['Gradient_Boosting']
feature_importances = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='#1976D2', edgecolor='#0D47A1', alpha=0.85)
plt.title('Importancia de Variables (Feature Importance) - Gradient Boosting', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Importancia Relativa (Gini Impurity Reduction)', fontsize=11, fontweight='bold')
plt.ylabel('Variable de Solicitud de Crédito', fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)
for idx, val in enumerate(feature_importances):
    plt.text(val + 0.005, idx, f"{val*100:.1f}%", va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/6_feature_importance.png', dpi=300)
plt.close()

print("\n" + "="*75)
print("EXPERIMENTACIÓN COMPLETADA SATISFACTORIAMENTE")
print("="*75)
