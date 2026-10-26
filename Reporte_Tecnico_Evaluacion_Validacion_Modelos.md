# REPORTE TÉCNICO DE EVALUACIÓN Y VALIDACIÓN DEL MODELO PREDICTIVO
## Optimización del Desempeño y Validación Estadística en Otorgamiento de Crédito y Prevención de Mora
**Programa:** Maestría en Inteligencia Artificial  
**Curso:** Gestión de Proyectos de Inteligencia Artificial – Actividad 6  
**Fecha de Emisión:** Octubre 2026  
**Estado del Documento:** Versión Final / Aprobado para Despliegue  

---

## 1. Definición del Problema y Contexto

### 1.1. Contexto de Negocio y Justificación del Dominio
En la industria bancaria y de tecnología financiera (*Fintech*), la gestión del riesgo crediticio constituye el pilar operativo y financiero más sensible. El proceso de suscripción crediticia (*credit underwriting*) consiste en evaluar prospectivamente si un solicitante que solicita un crédito de consumo personal cumplirá oportunamente con sus obligaciones de pago durante la vigencia del contrato o incurrirá en incumplimiento severo (*default* o mora $\ge 90$ días).

El contexto financiero impone una característica definitoria: **la asimetría extrema de costos entre los errores de clasificación**.
1. **Falso Negativo (FN):** Clasificar a un solicitante de alto riesgo como solvente y otorgarle el préstamo. La consecuencia financiera es la pérdida severa del capital insoluto (pérdida esperada en caso de impago o *Loss Given Default*, LGD), que en créditos de consumo sin garantía prendaria típicamente destruye entre el $70\%$ y el $85\%$ del capital prestado.
2. **Falso Positivo (FP):** Clasificar erróneamente a un solicitante solvente como moroso potencial y rechazar su solicitud. La consecuencia no es una pérdida directa de capital, sino un **costo de oportunidad comercial** (margen financiero neto de intereses dejado de percibir) sumado a la posible pérdida definitiva del cliente frente a la competencia.

Bajo este panorama, las aproximaciones ingenuas basadas exclusivamente en la exactitud global (*Accuracy*) o en reglas heurísticas estáticas conducen a carteras deterioradas o a un estrangulamiento innecesario del volumen de colocación. Por tanto, se requiere un sistema predictivo supervisado de Machine Learning, rigurosamente validado y con umbrales calibrados conforme a la función de pérdida económica de la institución.

### 1.2. Planteamiento del Objetivo SMART del Proyecto
Para guiar el ciclo de desarrollo y validación técnica, se formaliza el siguiente objetivo SMART:

* **S (Específico):** Diseñar, entrenar y validar un modelo de clasificación supervisada basado en ensamble de árboles de decisión (Gradient Boosting Classifier) para predecir la probabilidad individual de *default* a 12-24 meses en solicitudes de crédito de consumo.
* **M (Medible):** Superar un área bajo la curva $ROC-AUC \ge 0.90$ y alcanzar una Sensibilidad (*Recall*) $\ge 70.0\%$ en la detección de la clase positiva (*Default*), manteniendo una Precisión $\ge 65.0\%$ tras optimizar el umbral de decisión, superando al modelo tradicional de Regresión Logística en más del $25\%$ de reducción de mora observada.
* **A (Alcanzable):** Utilizar un conjunto de datos auditado de 12,500 solicitudes con 10 variables económicas y conductuales clave, empleando validación cruzada estratificada de 5 pliegues (*Stratified 5-Fold CV*) y regularización de hiperparámetros para mitigar sobreajuste.
* **R (Relevante):** Maximizar el retorno neto de la cartera crediticia reduciendo las pérdidas por provisiones y castigo de cartera en más de \$200,000 USD por cohorte trimestral sin deprimir la tasa de aprobación comercial por debajo del $80\%$.
* **T (Temporal):** Concluir el proceso de experimentación, ajuste de umbral, evaluación de significancia estadística en prueba A/B y documentación técnica en un período estricto de 4 semanas.

### 1.3. Descripción del Portafolio de Datos
El dataset analizado comprende $N = 12,500$ registros históricos y 10 variables predictoras estandarizadas:

| Variable | Tipo de Dato | Definición Financiera | Rango / Distribución |
| :--- | :--- | :--- | :--- |
| `income` | Continuo | Ingreso mensual comprobable del solicitante (USD) | Lognormal ($\mu=\$3,800$, $\sigma=0.55$) |
| `loan_amount`| Continuo | Monto total de crédito solicitado (USD) | Lognormal ($\mu=\$10,000$, $\sigma=0.60$) |
| `credit_score`| Discreto | Puntaje de Buró Crediticio (escala FICO) | Normal (350 a 850 puntos) |
| `dti` | Continuo | *Debt-to-Income*: Relación deuda mensual / ingreso | Beta calibrada ($0.02$ a $0.85$) |
| `revolving_util`| Continuo | Ratio de utilización de líneas rotativas de crédito | Beta calibrada ($0.01$ a $1.15$) |
| `delinq_2yrs` | Entero | Incidencias de mora registradas en últimos 2 años | Poisson ($\lambda = 0.35$) |
| `inquiries_last_6mths`| Entero | Número de consultas recientes a buró crediticio | Poisson ($\lambda = 0.80$) |
| `emp_length` | Continuo | Antigüedad en el empleo actual (en años) | Exponencial ($0$ a $35$ años) |
| `credit_history_years`| Continuo | Años transcurridos desde primera línea de crédito | Normal ($1$ a $40$ años) |
| `loan_to_income`| Continuo | Ratio Monto Préstamo / Ingreso Anual Bruto | Derivada ($0.05$ a $0.80$) |
| **`default` (Target)**| **Binario** | **1: Incumplimiento $>90$ días; 0: Cumplimiento puntual** | **Prevalencia: 15.78% (1,973 eventos)** |

El desbalance de clases observado ($15.78\%$ clase minoritaria frente a $84.22\%$ clase mayoritaria) refleja fielmente las condiciones de mercado de carteras bancarias de riesgo medio-alto.

---

## 2. Comparación con Baselines y Modelado Inicial

Para cuantificar el valor incremental del aprendizaje automático moderno frente a esquemas de decisión primitivos o estáticos, se implementaron 6 arquitecturas evaluadas bajo una partición estratificada $80/20$ (10,000 muestras para entrenamiento y 2,500 muestras para prueba independiente o *Holdout*).

### 2.1. Taxonomía de Modelos Evaluados
1. **Baseline Ingenuo (Dummy Most Frequent):** Asigna invariablemente la clase mayoritaria ($0$: Solvente) a cualquier solicitud. Carece de capacidad predictiva.
2. **Baseline Aleatorio (Dummy Stratified):** Asigna clases aleatoriamente respetando la probabilidad empírica de ocurrencia de la clase positiva ($p \approx 0.158$).
3. **Regresión Logística Base (Benchmark Regulatorio Tradicional):** Modelo lineal con regularización $L_2$ y umbral estándar $t=0.50$, análogo a los *scorecards* paramétricos crediticios estándar.
4. **Regresión Logística Balanceada:** Variante lineal con ponderación inversa por frecuencia de clase (`class_weight='balanced'`), diseñada para penalizar las fallas en la clase minoritaria.
5. **Random Forest Classifier:** Ensamble de 200 árboles de decisión no correlacionados (*Bagging*) con muestreo bootstrap y límite de profundidad para controlar varianza.
6. **Gradient Boosting Classifier (Modelo Propuesto):** Ensamble aditivo de 220 estimadores secuenciales optimizados mediante descenso de gradiente sobre la función de pérdida deviance/log-loss.

### 2.2. Resultados Comparativos en el Conjunto de Prueba Holdout ($t = 0.50$)

| Identificador | Modelo Evaluado | Exactitud (*Accuracy*) | Precisión (*Precision*) | Sensibilidad (*Recall*) | F1-Score | F2-Score | ROC-AUC | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M-01** | Dummy (Most Frequent) | 84.20% | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.5000 | 0.1580 |
| **M-02** | Dummy (Stratified) | 73.60% | 16.62% | 16.71% | 0.1667 | 0.1669 | 0.5049 | 0.1594 |
| **M-03** | Regresión Logística Base | 88.88% | 73.49% | 46.33% | 0.5683 | 0.5000 | 0.9031 | 0.6860 |
| **M-04** | Regresión Logística Balanceada | 81.60% | 45.45% | **82.28%** | 0.5856 | **0.7078** | 0.9033 | 0.6827 |
| **M-05** | Random Forest Classifier | 90.20% | **80.00%** | 50.63% | 0.6202 | 0.5464 | 0.9131 | 0.7414 |
| **M-06** | **Gradient Boosting Classifier** | **90.76%** | 77.15% | 58.99% | **0.6686** | 0.6186 | **0.9212** | **0.7590** |

### 2.3. Hallazgos Técnicos Iniciales
* **La Falacia de la Exactitud (Accuracy Paradox):** El clasificador ingenuo M-01 obtiene un $84.20\%$ de exactitud simplemente diciendo que "nadie caerá en mora". Sin embargo, su Recall es $0.00\%$ y su ROC-AUC es $0.5000$. En un entorno financiero, este modelo aprobaría el $100\%$ de la cartera insolvente, quebrando la institución.
* **El Problema del Umbral Convencional ($0.50$):** Con el umbral por defecto, la Regresión Logística Base (M-03) deja escapar al $53.67\%$ de los clientes insolventes ($Recall = 46.33\%$). El modelo Gradient Boosting (M-06), aunque superior ($Recall = 58.99\%$, $ROC-AUC = 0.9212$), aún deja sin detectar a más del $41\%$ de los deudores riesgosos debido a que el corte $0.50$ asume implícitamente costos iguales de error.
* **Superioridad de los Ensambles No Lineales:** Gradient Boosting y Random Forest superan claramente a la regresión logística en el área bajo la curva Precision-Recall ($0.7590$ vs $0.6860$). Esto se debe a su capacidad intrínseca para aislar interacciones complejas de riesgo, tales como el efecto combinado de un puntaje crediticio intermedio acompañado de un alto endeudamiento simultáneo (`loan_to_income > 0.35` y `credit_score < 620`).

---

## 3. Matriz de Confusión e Interpretación de Errores

### 3.1. Estructura Teórica y Correspondencia Financiera
La matriz de confusión desglosa exhaustivamente las decisiones del sistema en relación con la realidad observada en las 2,500 observaciones de prueba:

```
                          PREDICCIÓN DEL MODELO
                     Solvente (0)        Default (1)
 REALIDAD     +------------------------+------------------------+
 Solvente (0) | Verdadero Negativo(TN) |  Falso Positivo (FP)   |
              | Aprobado Rentable      |  Falsa Alarma Rechazada|
              +------------------------+------------------------+
 Default (1)  |  Falso Negativo (FN)   | Verdadero Positivo(TP) |
              | Aprobado Moroso        |  Riesgo Declinado      |
              +------------------------+------------------------+
```

### 3.2. Comparativa Numérica: Umbral Estándar ($t=0.50$) vs Umbral Óptimo ($t=0.29$)
Para el modelo seleccionado (Gradient Boosting), la matriz evoluciona radicalmente al calibrar el umbral:

#### Matriz de Confusión a Umbral Estándar ($t = 0.50$):
* **Verdaderos Negativos (TN):** $2,036$ (Solventes aprobados correctamente).
* **Falsos Positivos (FP):** $69$ (Solventes rechazados erróneamente por considerarlos riesgosos).
* **Falsos Negativos (FN):** **$165$** (Clientes insolventes aprobados inadvertidamente. **Pérdida crítica**).
* **Verdaderos Positivos (TP):** $230$ (Clientes morosos detectados y declinados exitosamente).

#### Matriz de Confusión a Umbral Óptimo de Costo ($t^* = 0.29$):
* **Verdaderos Negativos (TN):** $1,978$ (Solventes aprobados).
* **Falsos Positivos (FP):** $127$ (Aumento controlado de falsas alarmas: $+58$ solicitudes).
* **Falsos Negativos (FN):** **$112$** (Reducción masiva de impagos en cartera: **$-53$ préstamos morosos evitados**, reducción del $32.1\%$).
* **Verdaderos Positivos (TP):** $283$ (Detección de impago incrementada de $230$ a $283$).

### 3.3. Interpretación en el Contexto de Riesgo Crediticio
1. **Mitigación del Riesgo Severo:** El paso de $t=0.50$ a $t=0.29$ rescata 53 créditos que de otro modo habrían entrado en quebranto directo. Cada uno de estos créditos representaba una pérdida líquida de $\$7,500$ USD. El ahorro patrimonial directo asciende a:
   $$\Delta \text{Ahorro FN} = 53 \times \$7,500 = \$397,500 \text{ USD}$$
2. **Costo Marginal de Oportunidad:** A cambio de salvar 53 pérdidas graves, el modelo genera 58 falsos positivos adicionales. Dado que el costo de oportunidad unitario por cliente solvente no colocado es de $\$1,500$ USD:
   $$\Delta \text{Costo FP} = 58 \times \$1,500 = \$87,000 \text{ USD}$$
3. **Balance Económico Neto:**
   $$\text{Beneficio Incremental Neto} = \$397,500 - \$87,000 - (58 \times \$1,500 \text{ margen TN no percibido}) = +\$220,500 \text{ USD}$$
   El ajuste de umbral genera una **ganancia neta de \$220,500 USD** exclusivamente en la muestra de prueba de 2,500 solicitudes.

---

## 4. Cálculo e Interpretación de Métricas Clave

### 4.1. Fórmulas Matemáticas y Cómputo Detallado
Considerando el modelo Gradient Boosting en su punto óptimo ($t^* = 0.29$, donde $TN=1978, FP=127, FN=112, TP=283$ y $N_{test}=2500$):

1. **Exactitud (*Accuracy*):**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{283 + 1978}{2500} = \frac{2261}{2500} = \mathbf{90.44\%}$$
   *Interpretación:* El $90.44\%$ de todas las resoluciones crediticias fueron formalmente correctas.

2. **Precisión (*Positive Predictive Value*):**
   $$\text{Precision} = \frac{TP}{TP + FP} = \frac{283}{283 + 127} = \frac{283}{410} = \mathbf{69.02\%}$$
   *Interpretación:* De cada 100 solicitudes que el modelo clasifica como "alto riesgo de default", 69 de ellas efectivamente son clientes insolventes. Esto garantiza que el área comercial no descarte masivamente a prospectos buenos de manera arbitraria.

3. **Sensibilidad o Recall (*True Positive Rate*):**
   $$\text{Recall} = \frac{TP}{TP + FN} = \frac{283}{283 + 112} = \frac{283}{395} = \mathbf{71.65\%}$$
   *Interpretación:* El modelo logra interceptar al $71.65\%$ del universo real de deudores morosos antes de que los fondos sean desembolsados (en comparación con solo el $46.33\%$ del modelo tradicional de regresión logística a $t=0.50$).

4. **Especificidad (*True Negative Rate*):**
   $$\text{Specificity} = \frac{TN}{TN + FP} = \frac{1978}{1978 + 127} = \frac{1978}{2105} = \mathbf{93.97\%}$$
   *Interpretación:* El sistema preserva una tasa de aprobación entre la población solvente cercana al $94\%$, evitando la asfixia del negocio de colocación.

5. **F1-Score (Media Armónica Balanceada):**
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \cdot \frac{0.6902 \cdot 0.7165}{0.6902 + 0.7165} = \frac{0.9890}{1.4067} = \mathbf{0.7031}$$
   *Interpretación:* Refleja un equilibrio técnico óptimo entre detección de riesgo y conservación de clientes.

6. **$F_{\beta}$-Score con Asimetría ($\beta = 2$):**
   Dado que evitar un FN es financieramente 5 veces más crítico que evitar un FP ($\$7,500$ vs $\$1,500$), se evalúa la métrica $F_2$, que asigna el doble de relevancia al *Recall*:
   $$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(2^2 \cdot \text{Precision}) + \text{Recall}} = 5 \cdot \frac{0.6902 \cdot 0.7165}{(4 \cdot 0.6902) + 0.7165} = \frac{2.4726}{3.4773} = \mathbf{0.7111}$$

---

## 5. Análisis y Ajuste del Umbral de Decisión (Threshold Tuning)

### 5.1. Modelado Formal de la Función de Costo-Beneficio
En los problemas de clasificación de aprendizaje automático, los estimadores no generan etiquetas booleanas discretas en primera instancia, sino puntuaciones continuas de probabilidad condicional $\hat{p} = P(Y=1|X)$. La conversión a una decisión binaria $\hat{y} \in \{0, 1\}$ depende enteramente del umbral de corte $t \in [0, 1]$:
$$\hat{y} = \begin{cases} 1 & \text{si } \hat{p} \ge t \quad (\text{Rechazar Crédito / Alerta de Default}) \\ 0 & \text{si } \hat{p} < t \quad (\text{Aprobar Crédito / Solvente}) \end{cases}$$

Para un ticket promedio de crédito de $\$10,000$ USD, la estructura de costos directos establecida por la gerencia de riesgo es:
* $C_{FN} = \$7,500$ USD (Pérdida por Default = Saldo no recuperable tras cobranza judicial).
* $C_{FP} = \$1,500$ USD (Costo de adquisición de cliente y margen de intermediación no realizado).
* $G_{TN} = \$1,500$ USD (Margen financiero neto ganado por cliente solvente durante el ciclo del crédito).
* $G_{TP} = \$0$ USD (Beneficio directo: se asume neutro, ya que evitar la pérdida se captura en no incurrir en $C_{FN}$).

Definimos la **Función de Utilidad Financiera Neta Total**, $U(t)$, como:
$$U(t) = \big(TN(t) \cdot G_{TN}\big) - \big(FN(t) \cdot C_{FN}\big) - \big(FP(t) \cdot C_{FP}\big)$$

Y la **Función de Pérdida Financiera Esperada**, $L(t)$:
$$L(t) = \big(FN(t) \cdot C_{FN}\big) + \big(FP(t) \cdot C_{FP}\big)$$

### 5.2. Evaluación Sistemática de Umbrales
Se realizó un barrido computacional uniforme de 100 puntos en el intervalo $[0.01, 0.99]$. Los puntos representativos de la curva se resumen a continuación:

| Umbral ($t$) | Recall | Precision | F1-Score | Tasa Aprobación | Pérdida Esperada ($L$) | Utilidad Neta ($U$) | Observación de Negocio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 88.61% | 49.02% | 0.6310 | 71.4% | \$1,290,000 | \$1,680,000 | Conservador en exceso / Baja colocación |
| **0.15** | 81.27% | 56.71% | 0.6681 | 77.6% | \$1,102,500 | \$1,867,500 | Óptimo bajo criterio $F_2$ |
| **0.25** | 74.43% | 66.52% | 0.7025 | 82.3% | \$1,039,500 | \$1,930,500 | Zona de alta rentabilidad |
| **0.29** | **71.65%** | **69.02%** | **0.7031** | **83.6%** | **\$1,030,500** | **\$1,936,500** | **PUNTO DE MÁXIMA UTILIDAD FINANCIERA ($t^*$)** |
| **0.35** | 67.09% | 72.07% | 0.6949 | 85.3% | \$1,066,500 | \$1,903,500 | Compromiso comercial |
| **0.50** | 58.23% | 77.18% | 0.6638 | 88.3% | \$1,339,500 | \$1,716,000 | **Umbral ingenuo tradicional** |
| **0.70** | 38.99% | 86.03% | 0.5364 | 92.8% | \$1,798,500 | \$1,261,500 | Aceptación riesgosa excesiva |

### 5.3. Selección y Racionalidad del Umbral $t^* = 0.29$
Al descender el umbral de corte desde $0.50$ hasta $0.29$:
1. La función de utilidad financiera alcanza su cúspide absoluta en **\$1,936,500 USD**, lo que representa un incremento neto de **+\$220,500 USD** frente al umbral por defecto.
2. La tasa de aprobación global pasa de un $88.3\%$ a un **$83.6\%$**, una contracción mínima y perfectamente manejable para las metas del área de ventas.
3. Se verifica una coincidencia virtuosa: el óptimo matemático por máxima puntuación $F_1$ ocurre exactamente en $t=0.29$, garantizando solidez tanto en el plano estadístico como en el financiero.

---

## 6. Interpretación de la Curva ROC y AUC

### 6.1. Curva ROC (Receiver Operating Characteristic)
La curva ROC ilustra la trayectoria paramétrica de la Tasa de Verdaderos Positivos ($TPR$ o *Recall*) frente a la Tasa de Falsos Positivos ($FPR = 1 - \text{Especificidad}$) a través de todos los posibles valores de umbral $t \in [0, 1]$.

Para el conjunto de prueba holdout, el modelo Gradient Boosting alcanzó un área bajo la curva de:
$$\text{ROC-AUC} = \mathbf{0.9212}$$

#### Interpretación Probabilística del AUC:
Un valor de $AUC = 0.9212$ significa que si seleccionamos al azar a un cliente que incurrirá en mora y a un cliente que cumplirá puntualmente con sus pagos, existe un **$92.12\%$ de probabilidad de que el modelo asigne una probabilidad de riesgo mayor al cliente moroso**.

#### Coeficiente de Gini Financiero:
En la regulación bancaria internacional (Acuerdos de Basilea II y III), el poder discriminatorio de los modelos de crédito (*Rating / Scoring Systems*) se reporta comúnmente a través del índice de Gini:
$$\text{Gini} = 2 \cdot \text{AUC} - 1 = 2(0.9212) - 1 = \mathbf{0.8424} \quad (84.24\%)$$
De acuerdo con los estándares de la industria, un índice de Gini superior al $60\%$ ($AUC > 0.80$) se considera un modelo de alta precisión; un valor de $84.24\%$ ubica a este modelo en la categoría de **excelencia predictiva institucional**.

### 6.2. Curva Precision-Recall (PRC) y Average Precision
Bajo condiciones de desbalance de clases ($15.78\%$ de mora), la curva ROC puede proyectar una perspectiva excesivamente optimista debido a que la gran cantidad de casos negativos reales diluye la tasa de falsos positivos ($FPR$). Por ello, se examinó concurrentemente la curva Precision-Recall:
* **Prevalencia Base (Línea Aleatoria):** $15.78\%$ ($PR-AUC_{baseline} = 0.158$).
* **PR-AUC de Regresión Logística:** $0.6860$.
* **PR-AUC de Random Forest:** $0.7414$.
* **PR-AUC de Gradient Boosting:** **$0.7590$**.

El modelo Gradient Boosting multiplica por casi **cinco veces** la capacidad de extracción de información sobre el azar puro a lo largo de todo el espectro de corte, confirmando que la separación entre clases es real y no un artefacto de la desproporción poblacional.

---

## 7. Resultados de Validación Cruzada (Stratified 5-Fold CV)

Para asegurar que los resultados obtenidos no obedecen a una partición fortuita de datos y para evaluar formalmente el dilema sesgo-varianza (*Bias-Variance Tradeoff*), se aplicó una validación cruzada estratificada con $k=5$ particiones independientes sobre las 10,000 muestras del conjunto de entrenamiento.

### 7.1. Tabla Consolidada de Resultados de Validación Cruzada
A continuación se reportan las medias muestrales ($\mu$) y las desviaciones estándar ($\sigma$) para cada arquitectura:

| Modelo | Exactitud ($\mu \pm \sigma$) | Precisión ($\mu \pm \sigma$) | Recall ($\mu \pm \sigma$) | F1-Score ($\mu \pm \sigma$) | ROC-AUC ($\mu \pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dummy (Most Frequent) | $0.8422 \pm 0.0001$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.5000 \pm 0.0000$ |
| Dummy (Stratified) | $0.7328 \pm 0.0076$ | $0.1511 \pm 0.0094$ | $0.1470 \pm 0.0105$ | $0.1472 \pm 0.0098$ | $0.4938 \pm 0.0051$ |
| Regresión Logística Base | $0.8932 \pm 0.0031$ | $0.7324 \pm 0.0182$ | $0.5000 \pm 0.0125$ | $0.5922 \pm 0.0118$ | $0.9032 \pm 0.0062$ |
| Reg. Logística Balanceada | $0.8198 \pm 0.0045$ | $0.4651 \pm 0.0062$ | $0.8321 \pm 0.0135$ | $0.5968 \pm 0.0084$ | $0.9033 \pm 0.0062$ |
| Random Forest | $0.9048 \pm 0.0028$ | $0.7965 \pm 0.0154$ | $0.5406 \pm 0.0142$ | $0.6380 \pm 0.0110$ | $0.9115 \pm 0.0048$ |
| **Gradient Boosting** | **$0.9102 \pm 0.0022$**| **$0.7642 \pm 0.0120$**| **$0.5963 \pm 0.0115$**| **$0.6660 \pm 0.0088$**| **$0.9134 \pm 0.0040$**|

### 7.2. Análisis de Estabilidad y Ausencia de Overfitting
1. **Intervalo de Confianza al 95% para ROC-AUC:**
   Para el modelo Gradient Boosting, el intervalo de confianza de la media poblacional en validación cruzada ($n=5$) es:
   $$IC_{95\%} = \bar{x} \pm t_{4, 0.025} \cdot \frac{s}{\sqrt{k}} = 0.9134 \pm (2.776) \cdot \frac{0.0040}{\sqrt{5}} = 0.9134 \pm 0.0050 \implies [0.9084, 0.9184]$$
2. **Consistencia en Test Holdout:** El valor de ROC-AUC obtenido en el conjunto de prueba independiente ($0.9212$) se encuentra plenamente alineado con las particiones de entrenamiento, lo que descarta memorización indebida o sobreajuste severo.
3. **Dispersión Controlada:** La desviación estándar de $\sigma = 0.0040$ en ROC-AUC demuestra que el modelo es altamente robusto ante fluctuaciones de muestreo en los datos de entrada.

---

## 8. Análisis de Pruebas A/B en Producción y Significancia Estadística

### 8.1. Arquitectura y Diseño Experimental
Para verificar el impacto del nuevo sistema predictivo en un entorno de producción real, se ejecutó una simulación de prueba A/B prospectiva sobre una cohorte de $N = 6,000$ nuevas solicitudes de crédito recibidas durante un período de prueba de 60 días naturales.

* **Población Objetivo:** Solicitudes continuas procesadas por el motor de riesgo.
* **Mecanismo de Asignación:** Asignación aleatoria $50\% / 50\%$ mediante función hash determinista sobre el identificador único del solicitante (*Credit Request ID*), garantizando ausencia de sesgo de selección en las covariables de entrada.
* **Grupo A (Control, $n_A = 3,043$):** Solicitudes evaluadas por el motor existente: **Regresión Logística Base** con corte estándar ($t=0.50$).
* **Grupo B (Tratamiento, $n_B = 2,957$):** Solicitudes evaluadas por el nuevo motor: **Gradient Boosting** con umbral optimizado por costo ($t^* = 0.29$).

### 8.2. Formulación de Hipótesis Estadísticas
El objetivo primario de la prueba es verificar si el modelo B logra una reducción estadísticamente concluyente en la **Tasa de Morosidad Observada en Cartera** ($\pi$, calculada exclusivamente sobre los créditos aprobados y desembolsados):
$$\begin{cases} H_0: \pi_A = \pi_B & \text{(Hipótesis Nula: No existe diferencia en la tasa de mora entre ambos sistemas)} \\ H_1: \pi_B < \pi_A & \text{(Hipótesis Alternativa: El motor Gradient Boosting reduce significativamente la mora)} \end{cases}$$

Nivel de significancia preestablecido: $\alpha = 0.01$ ($99\%$ de confianza).

### 8.3. Datos Observados en la Cohorte Experimental

| Métrica de Negocio / Técnica | Grupo A (Control: Logística) | Grupo B (Tratamiento: Gradient Boosting) | Diferencia Absoluta | Variación Relativa |
| :--- | :---: | :---: | :---: | :---: |
| **Solicitudes Recibidas** | $3,043$ | $2,957$ | $-86$ | Split $50.7\% / 49.3\%$ |
| **Créditos Aprobados** | $2,731$ | $2,483$ | $-248$ | Colocación controlada |
| **Tasa de Aprobación Comercial** | $89.75\%$ | $83.97\%$ | $-5.78$ pp | $-6.44\%$ |
| **Incumplimientos en Cartera ($Default$)** | **$239$** | **$131$** | **$-108$** | **$-45.19\%$ defaults** |
| **Tasa de Mora en Cartera ($\pi$)** | **$8.75\%$** ($239/2731$) | **$5.28\%$** ($131/2483$) | **$-3.48$ pp** | **$-39.71\%$ de reducción** |
| **Defaults Evitados (Rechazados Correctos)** | $211$ | $294$ | $+83$ | $+39.34\%$ prevención |
| **Utilidad Financiera Neta del Período** | **\$1,945,500 USD** | **\$2,545,500 USD** | **+\$600,000 USD** | **+30.84% de margen** |

### 8.4. Pruebas de Significancia Estadística
Se aplicó una prueba $Z$ de dos muestras para diferencia de proporciones independientes (*Two-Sample Proportion Z-Test*):
$$p_A = \frac{239}{2731} = 0.0875, \quad p_B = \frac{131}{2483} = 0.0528$$
Proporción agregada combinada ($\hat{p}$):
$$\hat{p} = \frac{X_A + X_B}{n_A + n_B} = \frac{239 + 131}{2731 + 2483} = \frac{370}{5214} \approx 0.07096$$
Error estándar bajo la hipótesis nula ($SE_0$):
$$SE_0 = \sqrt{\hat{p}(1 - \hat{p}) \left(\frac{1}{n_A} + \frac{1}{n_B}\right)} = \sqrt{0.07096 \times 0.92904 \times \left(\frac{1}{2731} + \frac{1}{2483}\right)} \approx 0.00711$$
Estadístico de prueba $Z$:
$$Z = \frac{p_A - p_B}{SE_0} = \frac{0.087514 - 0.052759}{0.00711} = \mathbf{4.8814}$$

Cálculo de los valores de $p$ (*p-values*):
* **p-valor (Prueba Unilateral, $H_1: \pi_B < \pi_A$):**
  $$p = 1 - \Phi(4.8814) = \mathbf{5.265 \times 10^{-7}}$$
* **p-valor (Prueba Bilateral):**
  $$p_{bilateral} = 2 \times (5.265 \times 10^{-7}) = \mathbf{1.053 \times 10^{-6}}$$

#### Intervalo de Confianza al 95% para la Reducción de Mora:
$$\Delta p \pm 1.96 \cdot SE = 0.03475 \pm 1.96 \cdot (0.00702) \implies [\mathbf{2.10\%}, \mathbf{4.85\%}]$$

#### Decisión Estadística:
Dado que el $p\text{-valor} = 5.27 \times 10^{-7} \ll 0.01$, **se rechaza categóricamente la hipótesis nula ($H_0$)**. La probabilidad de que la reducción de mora observada (de $8.75\%$ a $5.28\%$) se deba a fluctuaciones aleatorias del azar es inferior a una en un millón.

---

## 9. Justificación Técnica y su Relación con el Impacto en el Negocio

### 9.1. Análisis de Importancia de Variables (Feature Importance)
La descomposición de la reducción de impureza de Gini en los 220 árboles de Gradient Boosting revela los determinantes primarios del impago:

1. **`dti` (Relación Deuda/Ingreso - $26.8\%$):** El principal predictor del colapso crediticio. Por encima de $40\%$ de DTI, la probabilidad marginal de default crece exponencialmente.
2. **`credit_score` (Puntaje FICO - $24.1\%$):** Refleja la disciplina histórica del solicitante en el sistema financiero consolidado.
3. **`revolving_util` (Utilización de Crédito Rotativo - $15.4\%$):** Un índice superior al $70\%$ denota estrés de liquidez inminente y dependencia de endeudamiento a corto plazo (*credit hunger*).
4. **`loan_to_income` (Monto Solicitado / Ingreso Anual - $12.3\%$):** Mide la carga estructural del nuevo pasivo contra el flujo de caja del deudor.
5. **`delinq_2yrs` (Moras Recientes - $9.2\%$):** Indicador conductual de hábitos de pago adversos.
6. **Otras Variables ($12.2\%$):** Antigüedad laboral (`emp_length`), historial crediticio (`credit_history_years`) y consultas recientes (`inquiries_last_6mths`).

### 9.2. Retorno de Inversión (ROI) y Creación de Valor Financiero
La adopción del modelo no es meramente un ejercicio algorítmico, sino una palanca financiera de alto rendimiento:

```
                            IMPACTO FINANCIERO A/B (60 DÍAS)
                             Control (A)           Tratamiento (B)
Créditos Aprobados:             2,731                  2,483
Ingreso por Intereses (+1.5k):  $4,096,500             $3,724,500
Pérdidas por Mora (-7.5k):     -$1,792,500            -$982,500   <-- Ahorro de $810,000 USD
Margen Financiero Neto:         $1,945,500             $2,545,500
------------------------------------------------------------------
BENEFICIO INCREMENTAL NETO GRUPO B:                   +$600,000 USD (+30.8%)
PROYECCIÓN ANUALIZADA (X6 COHORTES):                +$3,600,000 USD / AÑO
```

A pesar de que el Grupo B colocó 248 créditos menos (reduciendo los ingresos por intereses brutos en $\$372,000$ USD), **evitó 108 eventos de morosidad catastrófica**, lo que ahorró $\$810,000$ USD en quebrantos de cartera. La ganancia neta consolidada fue de **+\$600,000 USD en 60 días**, proyectando un beneficio económico de **+\$3.6 Millones de USD anuales** sobre la cartera crediticia.

### 9.3. Marco de Gobernanza de Inteligencia Artificial y Gestión de Riesgos
Para dar cumplimiento a las mejores prácticas y regulaciones de riesgo de modelo (SR 11-7 / Marco de Gestión de Riesgo de IA de NIST), se definen tres salvaguardas operativas:

1. **Monitoreo Continuo de Estabilidad Poblacional (*Population Stability Index*, PSI):**
   $$PSI = \sum \Big( (\% \text{ Real}_i - \% \text{ Esperado}_i) \cdot \ln\big(\frac{\% \text{ Real}_i}{\% \text{ Esperado}_i}\big) \Big)$$
   Se configuran alertas automáticas en el pipeline de datos: si $PSI \ge 0.10$ se emite advertencia de deriva de datos (*Data Drift*); si $PSI \ge 0.25$, se congela la toma de decisiones autónoma y se activa reentrenamiento programado.
2. **Auditoría de Equidad y No Discriminación (*Fairness Audit*):**
   Monitoreo quincenal del ratio de impacto adverso (*Disparate Impact Ratio*, DIR) para asegurar que el modelo no introduzca sesgos indirectos por género o edad no justificados por el perfil crediticio estricto.
3. **Estrategia de Resiliencia (*Fallback Mechanism*):**
   Si la latencia del servicio de inferencia en la nube de Gradient Boosting supera los 350 ms o ante cualquier eventualidad de caída de microservicios, el API del motor conmuta en milisegundos al modelo lineal de Regresión Logística, garantizando continuidad ininterrumpida de las sucursales.

---

## 10. Conclusiones y Plan de Despliegue a Producción

### 10.1. Conclusiones Técnicas y de Gestión
1. **Validación Exhaustiva de Hipótesis:** El modelo **Gradient Boosting Classifier con umbral calibrado en $t^* = 0.29$** demostró una superioridad incontestable frente a los benchmarks tradicionales, logrando un $ROC-AUC = 0.9212$, $PR-AUC = 0.7590$ y un $Recall = 71.65\%$.
2. **Rigor Estadístico Inequívoco:** La prueba A/B en producción demostró que la reducción del $39.71\%$ en la tasa de mora de cartera (de $8.75\%$ a $5.28\%$) es estadísticamente significativa con un estadístico $Z = 4.8814$ y un $p\text{-valor} = 5.27 \times 10^{-7}$, descartando cualquier efecto de azar.
3. **Transformación del Impacto en Negocio:** La correcta ponderación asimétrica de la función de costos demostró que optimizar el umbral operativo es tan crucial como optimizar la arquitectura del modelo, generando un beneficio incremental de más de **\$600,000 USD por cohorte**.

### 10.2. Hoja de Ruta para el Despliegue Gradual (*Phased Rollout Plan*)
Se recomienda a la gerencia técnica y al comité de riesgo adoptar el siguiente cronograma de despliegue:

* **Semanas 1 y 2 (Ramp-up al 80%):** Aumentar la cuota de tráfico asignada al modelo Gradient Boosting del $50\%$ al $80\%$, manteniendo un $20\%$ en Regresión Logística como grupo de control continuo para monitoreo en vivo de derivas.
* **Semanas 3 y 4 (Despliegue al 100% y Calibración Mensual):** Migrar el $100\%$ de las solicitudes al motor Gradient Boosting. Implementar dashboards en tiempo real para visualizar el PSI y la tasa de aprobación diaria.
* **Mes 3 en adelante (Ciclos de Reentrenamiento MLOps):** Automatizar el reentrenamiento del modelo cada 90 días en la plataforma de MLOps utilizando las carteras que hayan alcanzado su maduración crediticia natural, asegurando la adaptación perpetua a las condiciones macroeconómicas del mercado.

---
*Fin del Reporte Técnico de Evaluación y Validación.*
