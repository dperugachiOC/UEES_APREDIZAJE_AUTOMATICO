# Taller: Modelos Supervisados (SVM vs Árbol de decisión)

## Problema
Entrenar y comparar clasificadores supervisados (**Árbol de decisión**, **SVM**, **Random Forest** y **Regresión Logística**) para detectar patrones de compra (`Purchased`) a partir de variables demográficas y salario estimado.

## Dataset
**Social_Network_Ads.csv**  
Variables principales:
- `Gender` (categórica)
- `Age` (numérica)
- `EstimatedSalary` (numérica)
- `Purchased` (clase / etiqueta)

> `User ID` se elimina por ser un identificador.

## Metodología
1. **EDA**: inspección, estadísticas, balance de clases, nulos, histogramas, boxplots y correlación.
2. **Preprocesamiento**:
   - One-Hot para `Gender`
   - Escalado (StandardScaler) **solo dentro del pipeline** para modelos sensibles (SVM / regresión logística).
   - Split 80/20 con `stratify`.
3. **Modelos**:
   - Decision Tree
   - SVM con GridSearch (`kernel`, `C`, `gamma`)
   - Random Forest
   - Regresión Logística
4. **Evaluación**:
   - Accuracy, Precision, Recall, F1-score
   - Matriz de confusión
   - Tabla resumen + gráfico comparativo

## Estructura sugerida del repositorio
```
.
├── data/
│   └── Social_Network_Ads.csv
├── notebooks/
│   └── 01_modelos_supervisados.ipynb
├── figures/               # (opcional) exportar gráficas
├── src/                   # (opcional) scripts
├── requirements.txt
└── README.md
```

## Ejecución
1. Crear entorno:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```
2. Abrir notebook:
```bash
jupyter notebook
```

## Conclusiones (guía)
- SVM requiere escalado; su desempeño depende de hiperparámetros.
- Árbol es interpretable pero puede sobreajustar.
- RandomForest suele mejorar estabilidad y generalización.

