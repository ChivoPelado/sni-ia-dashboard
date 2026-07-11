# SNI Ecuador — Modelado explicativo

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
├── Dockerfile                              # Imagen para deploys que requieran contenedor
├── .streamlit/
│   └── config.toml                         # Tema light forzado
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

## Deploy actual

Publicado temporalmente en:

**http://134.56.159.55:8890/sni/**

Corre como servicio systemd en un servidor Linux propio (Ubuntu, Python 3.12), detrás de nginx como reverse proxy en el puerto 8890 (redirige `location /sni/` a `127.0.0.1:8595`).

### Actualizar el deploy

```bash
ssh andres@<server>
cd /opt/sni-dashboard
git pull
sudo systemctl restart sni-dashboard
```

### Alternativas de hosting

- **Docker** — usar el `Dockerfile` incluido en la raíz (Python 3.11-slim, puerto 7860). Compatible con Hugging Face Spaces (SDK Docker), Google Cloud Run, Fly.io, etc.
- **Streamlit Community Cloud** — requiere `>1 GB RAM` porque los 4 SHAP explainers + XGBoost se cargan en memoria. El free tier (1 GB) es insuficiente.

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
