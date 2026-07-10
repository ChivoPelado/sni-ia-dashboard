"""
Recomendador de escenarios de intercambio energetico - SNI Ecuador
==================================================================

Sistema de soporte a decisiones que combina:
1. Deteccion de regimen operativo (K-Means, ya entrenado)
2. Caracterizacion estadistica historica del regimen
3. Recuperacion de precedentes similares (KNN)
4. Recomendacion textual prescriptiva basada en patrones historicos

Este modulo NO predice valores futuros. Reporta lo que historicamente
ha ocurrido bajo condiciones similares, lo cual es analisis de escenarios
basado en datos reales (no forecasting).
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Variables canonicas usadas para definir condiciones del sistema
FEATURES_CONDICIONES = [
    'caudal_ponderado',
    'oni_mensual',
    'precip_embalses',
    'demanda_gwh',
    'mes',
]

# Plantillas de recomendacion por regimen (5 estados k-means)
PLANTILLAS_RECOMENDACION = {
    'Superavit Energetico': {
        'titulo': 'Condiciones favorables para EXPORTACION',
        'accion': 'Maximizar venta de excedentes a Colombia y/o Peru',
        'racional': 'Alta disponibilidad hidrica y generacion superior a demanda interna.',
        'severidad': 'oportunidad',
    },
    'Equilibrio Hidrico': {
        'titulo': 'Sistema en equilibrio - Intercambio oportunista',
        'accion': 'Exportacion menor segun disponibilidad. Sin presion de importacion.',
        'racional': 'Generacion cubre demanda. Margen reducido pero positivo.',
        'severidad': 'normal',
    },
    'Estres Leve': {
        'titulo': 'PRECAUCION - Estres hidrico leve',
        'accion': 'Monitorear caudales. Mantener disponibilidad termica de respaldo.',
        'racional': 'Caudales bajo lo normal pero el sistema mantiene autosuficiencia.',
        'severidad': 'precaucion',
    },
    'Estres Severo': {
        'titulo': 'ALERTA - Estres hidrico severo',
        'accion': 'Activar respaldo termico completo. Considerar importacion preventiva.',
        'racional': 'Caudales muy por debajo de lo normal. Alta probabilidad de deficit.',
        'severidad': 'alerta',
    },
    'Crisis Energetica': {
        'titulo': 'CRITICO - Importacion necesaria',
        'accion': 'Maximizar importacion desde Colombia. Activar todo el parque termico.',
        'racional': 'Sequia severa. Generacion hidroelectrica insuficiente para cubrir demanda.',
        'severidad': 'critico',
    },
}


def normalizar_regimen(nombre):
    """Normaliza nombres de regimen removiendo acentos para matching robusto."""
    if pd.isna(nombre):
        return None
    s = str(nombre)
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N',
        '\ufffd': '',  # caracter de reemplazo unicode
    }
    for k, v in reemplazos.items():
        s = s.replace(k, v)
    # Limpiar dobles espacios y caracteres residuales
    s = ' '.join(s.split())
    # Mapeo a nombres canonicos
    mapeo = {
        'Superavit Energetico': 'Superavit Energetico',
        'Super vit Energ tico': 'Superavit Energetico',
        'Equilibrio Hidrico': 'Equilibrio Hidrico',
        'Equilibrio H drico': 'Equilibrio Hidrico',
        'Estres Leve': 'Estres Leve',
        'Estr s Leve': 'Estres Leve',
        'Estres Severo': 'Estres Severo',
        'Estr s Severo': 'Estres Severo',
        'Estres Hidrico': 'Estres Hidrico',
        'Estr s H drico': 'Estres Hidrico',
        'Crisis Energetica': 'Crisis Energetica',
        'Crisis Energ tica': 'Crisis Energetica',
    }
    return mapeo.get(s, s)


def cargar_dataset(ruta='data/sni_mensual_2009_2025.csv'):
    """Carga el dataset mensual y normaliza nombres de regimen."""
    df = pd.read_csv(ruta, encoding='utf-8')
    df['regimen_norm'] = df['regimen'].apply(normalizar_regimen)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


def caracterizar_regimen(df, regimen_norm):
    """
    Calcula la caracterizacion estadistica historica de un regimen.

    Returns dict con: count, exportacion (mean/median/std/p25/p75),
    importacion (idem), generacion, demanda, condiciones promedio.
    """
    sub = df[df['regimen_norm'] == regimen_norm]
    if len(sub) == 0:
        return None

    def stats(serie):
        return {
            'mean': float(serie.mean()),
            'median': float(serie.median()),
            'std': float(serie.std()),
            'p25': float(serie.quantile(0.25)),
            'p75': float(serie.quantile(0.75)),
            'min': float(serie.min()),
            'max': float(serie.max()),
        }

    return {
        'regimen': regimen_norm,
        'n_meses_historicos': int(len(sub)),
        'pct_periodo': round(100 * len(sub) / len(df), 1),
        'exportacion_gwh': stats(sub['total_exportacion_gwh']),
        'importacion_gwh': stats(sub['total_importacion_gwh']),
        'generacion_gwh': stats(sub['total_generacion_gwh']),
        'demanda_gwh': stats(sub['demanda_gwh']),
        'pct_hidro': stats(sub['pct_hidro']),
        'caudal_ponderado': stats(sub['caudal_ponderado']),
        'oni_mensual': stats(sub['oni_mensual']),
        'precip_embalses': stats(sub['precip_embalses']),
        'meses_distribucion': sub['mes'].value_counts().sort_index().to_dict(),
        'anios_cubiertos': sorted(sub['anio'].unique().tolist()),
    }


def precedentes_knn(df, condiciones, k=5):
    """
    Encuentra los k meses historicos mas similares a las condiciones dadas.

    condiciones: dict con keys de FEATURES_CONDICIONES
    Returns: DataFrame con los k precedentes ordenados por similitud
    """
    feats = FEATURES_CONDICIONES
    X = df[feats].values
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    consulta = np.array([[condiciones[f] for f in feats]])
    consulta_norm = scaler.transform(consulta)

    knn = NearestNeighbors(n_neighbors=min(k, len(df)), metric='euclidean')
    knn.fit(X_norm)
    dist, idx = knn.kneighbors(consulta_norm)

    precedentes = df.iloc[idx[0]].copy()
    precedentes['distancia'] = dist[0]
    precedentes['similitud_pct'] = np.round(
        100 * (1 - dist[0] / dist[0].max()) if dist[0].max() > 0 else 100, 1
    )
    return precedentes[[
        'fecha', 'anio', 'mes', 'regimen_norm',
        'caudal_ponderado', 'oni_mensual', 'precip_embalses', 'demanda_gwh',
        'total_generacion_gwh', 'total_importacion_gwh', 'total_exportacion_gwh',
        'distancia', 'similitud_pct'
    ]]


def analizar_precedentes(precedentes_df):
    """
    Resume estadisticas de los precedentes KNN: % que importaron, exportaron,
    magnitudes promedio, distribucion de regimenes.
    """
    n = len(precedentes_df)
    n_export = int((precedentes_df['total_exportacion_gwh'] > 10).sum())
    n_import = int((precedentes_df['total_importacion_gwh'] > 10).sum())

    return {
        'n_precedentes': n,
        'n_exportaron': n_export,
        'pct_exportaron': round(100 * n_export / n, 1),
        'n_importaron': n_import,
        'pct_importaron': round(100 * n_import / n, 1),
        'export_promedio': round(float(precedentes_df['total_exportacion_gwh'].mean()), 1),
        'import_promedio': round(float(precedentes_df['total_importacion_gwh'].mean()), 1),
        'regimenes_distribucion': precedentes_df['regimen_norm'].value_counts().to_dict(),
    }


def generar_recomendacion(regimen_norm, caracterizacion, analisis_precedentes=None):
    """
    Genera la recomendacion textual prescriptiva basada en el regimen
    detectado y la caracterizacion historica.
    """
    plantilla = PLANTILLAS_RECOMENDACION.get(regimen_norm)
    if plantilla is None:
        return {
            'titulo': f'Regimen no reconocido: {regimen_norm}',
            'accion': 'Sin recomendacion disponible',
            'severidad': 'desconocido',
            'evidencia_historica': '',
        }

    exp = caracterizacion['exportacion_gwh']
    imp = caracterizacion['importacion_gwh']
    n = caracterizacion['n_meses_historicos']

    evidencia = (
        f"Basado en {n} meses historicos clasificados como '{regimen_norm}' "
        f"(periodo 2009-2025): la exportacion promedio fue de {exp['mean']:.1f} GWh "
        f"(mediana {exp['median']:.1f}, rango p25-p75: {exp['p25']:.1f}-{exp['p75']:.1f}) "
        f"y la importacion promedio fue de {imp['mean']:.1f} GWh "
        f"(mediana {imp['median']:.1f}, rango p25-p75: {imp['p25']:.1f}-{imp['p75']:.1f})."
    )

    if analisis_precedentes:
        ap = analisis_precedentes
        evidencia += (
            f" Adicionalmente, los {ap['n_precedentes']} meses con condiciones mas "
            f"similares muestran que en el {ap['pct_exportaron']}% de los casos hubo "
            f"exportacion (promedio {ap['export_promedio']} GWh) y en el "
            f"{ap['pct_importaron']}% hubo importacion (promedio {ap['import_promedio']} GWh)."
        )

    return {
        'regimen': regimen_norm,
        'titulo': plantilla['titulo'],
        'accion': plantilla['accion'],
        'racional': plantilla['racional'],
        'severidad': plantilla['severidad'],
        'evidencia_historica': evidencia,
        'caracterizacion': caracterizacion,
        'precedentes_resumen': analisis_precedentes,
    }


def recomendar_por_fecha(df, fecha):
    """
    Pipeline completo: dada una fecha del dataset, genera la recomendacion
    completa con regimen, caracterizacion, precedentes y texto.
    """
    fila = df[df['fecha'] == pd.to_datetime(fecha)]
    if len(fila) == 0:
        return {'error': f'Fecha {fecha} no encontrada en dataset'}

    regimen = fila['regimen_norm'].iloc[0]
    caracterizacion = caracterizar_regimen(df, regimen)

    condiciones = {f: float(fila[f].iloc[0]) for f in FEATURES_CONDICIONES}
    precedentes = precedentes_knn(df, condiciones, k=10)
    # Excluir la propia fecha de los precedentes
    precedentes = precedentes[precedentes['fecha'] != pd.to_datetime(fecha)].head(5)
    analisis = analizar_precedentes(precedentes)

    rec = generar_recomendacion(regimen, caracterizacion, analisis)
    rec['fecha_consulta'] = str(fecha)
    rec['condiciones_actuales'] = condiciones
    rec['precedentes_detalle'] = precedentes
    return rec


def recomendar_por_condiciones(df, condiciones):
    """
    Pipeline para condiciones hipoteticas (no asociadas a una fecha real).
    Asigna regimen via el precedente mas cercano (KNN k=1).
    """
    precedentes = precedentes_knn(df, condiciones, k=10)
    # Regimen = el del vecino mas cercano (k=1)
    regimen = precedentes['regimen_norm'].iloc[0]

    caracterizacion = caracterizar_regimen(df, regimen)
    analisis = analizar_precedentes(precedentes.head(5))

    rec = generar_recomendacion(regimen, caracterizacion, analisis)
    rec['condiciones_actuales'] = condiciones
    rec['precedentes_detalle'] = precedentes.head(5)
    rec['regimen_inferido_via'] = 'KNN (vecino mas cercano)'
    return rec


if __name__ == '__main__':
    # Test rapido
    df = cargar_dataset()
    print(f"Dataset cargado: {len(df)} meses")
    print(f"Regimenes: {df['regimen_norm'].value_counts().to_dict()}")
    print()

    # Test 1: recomendacion por fecha (octubre 2024 - crisis conocida)
    print("=" * 70)
    print("TEST 1: Recomendacion para 2024-10-01 (crisis energetica conocida)")
    print("=" * 70)
    rec = recomendar_por_fecha(df, '2024-10-01')
    if 'error' in rec:
        print(rec['error'])
    else:
        print(f"Regimen detectado: {rec['regimen']}")
        print(f"Titulo: {rec['titulo']}")
        print(f"Accion: {rec['accion']}")
        print(f"Severidad: {rec['severidad']}")
        print(f"\nEvidencia:\n{rec['evidencia_historica']}")
