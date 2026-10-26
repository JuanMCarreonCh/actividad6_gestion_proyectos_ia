# Actividad 6: Evaluación y Validación de Modelos Mediante Métricas y Pruebas

**Curso:** Gestión de Proyectos de Inteligencia Artificial  
**Programa:** Maestría en Inteligencia Artificial  
**Contexto de Aplicación:** Evaluación de Riesgo y Prevención de Default Crediticio en Banca y Fintech  

---

## 📁 Estructura del Entregable

El repositorio contiene todos los componentes técnicos, analíticos y visuales exigidos en la rúbrica de la actividad:

```text
Actividad6/
│
├── Reporte_Tecnico_Evaluacion_Validacion_Modelos.md   # Reporte técnico exhaustivo (6-8 págs.) con las 10 secciones requeridas
├── evaluacion_validacion_modelos.ipynb                # Jupyter Notebook interactivo con celdas ejecutadas y salidas pre-renderizadas
├── run_experiments.py                                # Script modular completo de experimentación en Python
├── tablero_visual_resultados.html                    # Tablero visual interactivo (HTML responsive con simulador dinámico de umbral)
│
├── data/
│   └── credit_risk_dataset.csv                       # Dataset de riesgo crediticio (12,500 solicitudes con no-linealidades financieras)
│
├── figures/                                          # Gráficos analíticos de alta resolución (300 DPI)
│   ├── 1_confusion_matrices_comparison.png           # Comparación de matrices de confusión (t=0.50 vs t=0.29)
│   ├── 2_roc_and_pr_curves.png                       # Curvas ROC y Precision-Recall de todos los modelos
│   ├── 3_threshold_optimization_and_financial_loss.png# Trade-off de métricas y función de utilidad financiera neta
│   ├── 4_cross_validation_distributions.png          # Boxplots de validación cruzada estratificada (5-Fold CV)
│   ├── 5_ab_test_experiment_results.png              # Resultados de la prueba A/B y significancia estadística
│   └── 6_feature_importance.png                      # Importancia de atributos (Gini feature importance)
│
└── reports/                                          # Evidencia cuantitativa y registros de experimentos
    ├── ab_testing_statistical_results.json           # Estadístico Z, p-valores, intervalos de confianza del test A/B
    ├── ab_test_metrics_summary.csv                   # Resumen comparativo de métricas de negocio A/B
    ├── cross_validation_results.csv                  # Medias y desviaciones estándar de 5-Fold CV
    ├── test_holdout_evaluations_threshold_050.csv    # Métricas de todos los modelos en test holdout (t=0.50)
    └── threshold_tuning_analysis.csv                 # Barrido de 100 umbrales con métricas técnicas y utilidad neta
```

---

## 🚀 Instrucciones de Ejecución

### 1. Visualizar el Tablero Interactivo
Abra el archivo `tablero_visual_resultados.html` directamente en cualquier navegador web (Google Chrome, Microsoft Edge, Firefox, Safari). Cuenta con navegación por pestañas, tarjetas de KPIs en tiempo real y un simulador interactivo de costo-beneficio por umbral de probabilidad.

### 2. Abrir el Jupyter Notebook
Abra `evaluacion_validacion_modelos.ipynb` en VS Code, Jupyter Lab o Google Colab. El cuaderno ya se encuentra completamente pre-ejecutado con todas sus tablas, análisis narrativos y gráficos embebidos. Si desea re-ejecutarlo:
```bash
jupyter notebook evaluacion_validacion_modelos.ipynb
```

### 3. Re-ejecutar el Script de Experimentación
Para reproducir de principio a fin los experimentos, la validación cruzada, el barrido de umbrales y la prueba A/B:
```bash
python run_experiments.py
```

---

## 📊 Resumen de Resultados Principales

* **Modelo Ganador:** Gradient Boosting Classifier con umbral calibrado por costo en $t^* = 0.29$.
* **Capacidad de Discriminación:** $ROC-AUC = 0.9212$ | $PR-AUC = 0.7590$ | Índice de Gini = $84.24\%$.
* **Sensibilidad a la Mora:** $Recall = 71.65\%$ en el umbral óptimo (frente a $46.33\%$ del baseline de Regresión Logística a $t=0.50$).
* **Rendimiento en Prueba A/B (6,000 solicitudes):**
  * Tasa de mora Grupo A (Control - Logística Base): **$8.75\%$**
  * Tasa de mora Grupo B (Tratamiento - Gradient Boosting $t=0.29$): **$5.28\%$**
  * Reducción relativa de morosidad: **$39.71\%$**
  * Estadístico de prueba: **$Z = 4.8814$**
  * Significancia estadística: **$p\text{-valor} = 5.27 \times 10^{-7} \ll 0.01$** (Rechazo contundente de $H_0$).
  * Ganancia financiera neta incremental: **+\$600,000 USD** generados en la cohorte experimental.
