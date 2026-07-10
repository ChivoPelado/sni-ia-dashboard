---
title: SNI Ecuador IA Dashboard
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: false
license: mit
---

# Sistema IA — SNI Ecuador

Dashboard interactivo del **Sistema de modelado explicativo del Sistema Nacional Interconectado del Ecuador**, desarrollado como parte del Trabajo de Fin de Máster (TFM) en Inteligencia Artificial Aplicada — UIDE / EIG.

**Autor:** Andrés Herrera
**Periodo analizado:** 2009 – 2025 (6,028 días, 198 meses)
**Fuente de datos:** CENACE — SMEC (Sistema de Medición de Energía Comercial)

---

## Qué hace el sistema

El dashboard integra cuatro técnicas de aprendizaje automático para analizar y explicar el comportamiento operativo del Sistema Nacional Interconectado:

| Modelo | Rol |
|--------|-----|
| **K-Means** (k=5) | Identifica los 5 regímenes operativos recurrentes del sistema |
| **Random Forest** | Evalúa si las variables exógenas (caudal, ONI, precipitación, población) explican los regímenes |
| **XGBoost + SHAP** | Explica día a día las causas de la generación e intercambio (4 modelos) |
| **KNN** | Recupera precedentes históricos con condiciones similares |

El sistema **no predice el futuro**. Es un sistema de análisis explicativo y soporte a decisiones: caracteriza qué ocurrió históricamente bajo condiciones equivalentes.

---

## Secciones del dashboard

1. **Evaluación** — Contexto general: composición de la matriz, estacionalidad, correlaciones.
2. **Sistema Inteligente** — Los cuatro modelos integrados con visualización interactiva.
3. **Recomendador** — Soporte a decisiones basado en régimen detectado + precedentes.
4. **Conclusiones** — Hallazgos principales del análisis.

---

## Ejecutar localmente

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Levantar la app
streamlit run app.py
```

La app abre en `http://localhost:8501`.

---

## Estructura del repo

```
sni-ia-dashboard/
├── app.py                                  # Dashboard Streamlit (4 secciones)
├── recomendador.py                         # Módulo de soporte a decisiones
├── requirements.txt                        # Dependencias fijadas
├── data/
│   ├── dataset_analitico_diario.csv        # 6,028 días × 16 variables
│   └── dataset_analitico_mensual_regimenes.csv  # 198 meses + régimen K-Means
└── models/
    ├── metadata.joblib                     # Metadata general
    ├── metadata_diario.joblib              # Metadata modelos XGBoost
    ├── rf_clasificador_regimen.joblib      # Random Forest
    ├── xgb_diario_gen_hidro.joblib         # XGBoost generación hidroeléctrica
    ├── xgb_diario_gen_termica.joblib       # XGBoost generación térmica
    ├── xgb_diario_importacion.joblib       # XGBoost importación
    ├── xgb_diario_exportacion.joblib       # XGBoost exportación
    └── shap_diario_*.joblib                # 4 SHAP explainers
```

---

## Deploy — Streamlit Community Cloud

1. Empujar este repo a GitHub (público).
2. Ir a [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Seleccionar el repo, rama `main`, archivo `app.py`.
4. **Deploy**.

En pocos minutos queda disponible en `https://<nombre>.streamlit.app`.

---

## Notas metodológicas

- **Split temporal**: entrenamiento 2009–2022, validación/test 2023–2025.
- **K-Means**: k=5 seleccionado por Calinski-Harabasz (máximo).
- **Random Forest**: 200 árboles, 5-Fold CV, F1-macro ≈ 0.396.
- **XGBoost**: 4 modelos independientes sobre 6,028 días con 7 features.
- **SHAP**: TreeExplainer sobre cada XGBoost para descomposición día a día.
- **KNN**: 5 vecinos, distancia euclidiana normalizada.

---

## Licencia

Uso académico. Todos los datos originales provienen de fuentes públicas (CENACE, ARCERNNR).
