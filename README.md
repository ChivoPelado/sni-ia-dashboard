# SNI Ecuador — Modelado explicativo

Sistema de modelado explicativo del Sistema Nacional Interconectado (SNI) del Ecuador para el análisis del comportamiento operativo entre 2009 y 2025. Desarrollado como componente del Trabajo de Fin de Máster en Inteligencia Artificial Aplicada, Universidad Internacional del Ecuador (UIDE) en convenio con la Escuela Internacional de Postgrados (EIG).

**Autor:** Andrés Herrera
**Periodo:** enero 2009 – marzo 2025 (6 028 días; 198 meses)
**Fuentes primarias:** CENACE — Sistema de Medición de Energía Comercial (SMEC)

## Resumen

La aplicación provee un entorno interactivo para la exploración de las relaciones entre generación eléctrica, intercambio internacional con Colombia y Perú, y variables exógenas del sistema (hidrología, índice ENSO, precipitación en embalses, calendario y demografía). El enfoque es descriptivo–explicativo: los modelos se emplean para caracterizar patrones históricos y cuantificar la contribución relativa de cada variable a los resultados observados.

## Modelos incluidos

| Componente | Función | Detalles técnicos |
|-----------|---------|-------------------|
| K-Means | Segmentación no supervisada de regímenes operativos mensuales | k=5; selección por índice de Calinski–Harabasz |
| Random Forest | Clasificación supervisada de regímenes a partir de variables exógenas | 200 árboles; validación cruzada 5-Fold; F1-macro ≈ 0,396 |
| XGBoost + SHAP | Regresión explicativa de generación e intercambio en resolución diaria | Cuatro modelos independientes: generación hidroeléctrica, generación térmica, importación y exportación; descomposición de contribuciones vía TreeExplainer |
| K-Nearest Neighbors | Recuperación de precedentes históricos con condiciones similares | k=5; distancia euclidiana sobre variables normalizadas |

Los modelos operan sobre segmentos temporales disjuntos: 2009–2022 para entrenamiento y 2023–2025 para validación y prueba.

## Secciones de la aplicación

1. Evaluación: contexto general de la matriz energética, estacionalidad y correlaciones entre variables.
2. Sistema Inteligente: visualización integrada de los cuatro modelos y sus salidas.
3. Recomendador: caracterización estadística del régimen detectado y precedentes históricos comparables.
4. Conclusiones: síntesis de hallazgos principales.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

## Estructura del proyecto

```
sni-ia-dashboard/
├── app.py                                          Aplicación Streamlit
├── recomendador.py                                 Módulo de soporte a decisiones
├── requirements.txt                                Dependencias
├── Dockerfile                                      Definición de imagen para contenedores
├── .streamlit/config.toml                          Configuración de interfaz
├── notebooks/
│   └── entrenamiento_modelos.ipynb                 Pipeline de entrenamiento
├── data/
│   ├── dataset_analitico_diario.csv                Registros diarios (6 028 filas × 16 variables)
│   └── dataset_analitico_mensual_regimenes.csv     Registros mensuales con régimen asignado (198 filas)
└── models/
    ├── metadata.joblib                             Configuración del clasificador supervisado
    ├── metadata_diario.joblib                      Configuración de los regresores diarios
    ├── rf_clasificador_regimen.joblib              Random Forest entrenado
    ├── xgb_diario_gen_hidro.joblib                 XGBoost — generación hidroeléctrica
    ├── xgb_diario_gen_termica.joblib               XGBoost — generación térmica
    ├── xgb_diario_importacion.joblib               XGBoost — importación
    ├── xgb_diario_exportacion.joblib               XGBoost — exportación
    └── shap_diario_*.joblib                        Explainers SHAP asociados a cada regresor
```

## Reproducibilidad

El notebook `notebooks/entrenamiento_modelos.ipynb` contiene el pipeline completo de entrenamiento. Ejecutado sobre los datasets bajo `data/`, regenera los artefactos serializados de la carpeta `models/`. Los archivos `.joblib` corresponden a un estado congelado del entrenamiento con semilla fija.

## Datos y licencia

Los datos empleados provienen de fuentes públicas del sector eléctrico ecuatoriano, principalmente CENACE y la Agencia de Regulación y Control de Energía y Recursos Naturales No Renovables (ARCERNNR), complementados con series de reanálisis climático y variables calendáricas.

Código publicado bajo licencia MIT.
