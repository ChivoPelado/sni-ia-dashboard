"""
Sistema de Modelado basado en IA - SNI Ecuador (v3.1)
======================================================
4 secciones: Evaluacion | Sistema Inteligente | Recomendador | Conclusiones
Ejecutar: streamlit run v3.1_sistema_recomendador/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
import recomendador as rec_mod

st.set_page_config(page_title="SNI Ecuador — Modelado explicativo", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root { --ink:#0f172a; --ink2:#475569; --muted:#94a3b8; --raised:#f8fafc; --border:#e2e8f0; --accent:#1e3a5f; }
    html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif !important; color:var(--ink); }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a,#1e293b); }
    section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
    section[data-testid="stSidebar"] hr { border-color:#334155 !important; }
    h1 { font-family:'DM Serif Display',serif !important; font-size:1.8rem !important; color:var(--accent) !important;
         border-bottom:3px solid var(--accent) !important; padding-bottom:10px !important; }
    h2 { font-family:'IBM Plex Sans' !important; font-size:0.72rem !important; font-weight:600 !important;
         color:var(--muted) !important; letter-spacing:0.12em !important; text-transform:uppercase !important;
         margin-top:2rem !important; margin-bottom:0.2rem !important; }
    h3 { font-family:'DM Serif Display',serif !important; font-size:1.3rem !important; color:var(--ink) !important;
         margin-top:0.1rem !important; margin-bottom:0.8rem !important; }
    .lead { font-size:1rem; color:var(--ink2); line-height:1.7; margin:4px 0 24px 0; max-width:720px; }
    .narrative { font-size:0.9rem; color:var(--ink2); line-height:1.7; margin:10px 0 16px 0;
                 padding-left:14px; border-left:2px solid var(--border); }
    .card { background:var(--raised); border:1px solid var(--border); border-radius:6px; padding:12px 16px; margin:6px 0; }
    .card .cl { font-family:'IBM Plex Mono'; font-size:0.68rem; color:var(--muted); text-transform:uppercase;
                letter-spacing:0.08em; margin:0 0 3px 0; }
    .card .cv { font-family:'DM Serif Display'; font-size:1.2rem; color:var(--ink); margin:0; }
    .card .cu { font-size:0.72rem; color:var(--muted); }
    .state-box { display:flex; align-items:center; gap:16px; padding:16px 22px; border-radius:8px; margin:10px 0; }
    .state-box .dot { width:40px; height:40px; border-radius:50%; flex-shrink:0; }
    .divider { border:none; height:1px; background:linear-gradient(90deg,var(--border),transparent); margin:28px 0 6px 0; }
    .footer { background:var(--raised); border-top:2px solid var(--border); padding:18px 22px; margin-top:36px;
              font-size:0.8rem; color:var(--ink2); line-height:1.7; }
    .footer strong { color:var(--ink); }
    [data-testid="stMetric"] { background:var(--raised); border:1px solid var(--border); border-radius:6px; padding:10px 14px; }
    [data-testid="stMetricLabel"] { font-family:'IBM Plex Mono' !important; font-size:0.62rem !important;
                                    text-transform:uppercase; letter-spacing:0.08em; }
    [data-testid="stMetricValue"] { font-family:'DM Serif Display' !important; font-size:1.3rem !important; }
    .stDeployButton { display:none; } footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── DATA & MODELS ──────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
MOD  = BASE / "models"

MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

@st.cache_data
def load_data():
    dd = pd.read_csv(DATA / "dataset_analitico_diario.csv", parse_dates=["fecha"])
    dd["tipo_dia_num"] = dd["tipo_dia"].map({"Laboral":0,"Sabado":1,"Sábado":1,"Domingo":2,"Festivo":3})
    dm = pd.read_csv(DATA / "dataset_analitico_mensual_regimenes.csv", parse_dates=["fecha"])
    dm["estado"] = dm["regimen"]
    dm["anio"] = dm["fecha"].dt.year
    dm["mes"] = dm["fecha"].dt.month
    return dd, dm

@st.cache_resource
def load_models():
    mm = joblib.load(MOD / "metadata.joblib")
    rf = joblib.load(MOD / "rf_clasificador_regimen.joblib")
    md = joblib.load(MOD / "metadata_diario.joblib")
    mo, ex = {}, {}
    for n in md["targets"].values():
        mo[n] = joblib.load(MOD / f"xgb_diario_{n}.joblib")
        ex[n] = joblib.load(MOD / f"shap_diario_{n}.joblib")
    return mm, rf, md, mo, ex

dd, dm = load_data()
mm, rf, md, mo, ex = load_models()
NI = md["label_map"]
SHAP_SHORT = {"Demanda (GWh)":"Demanda","Fraccion hidroelectrica":"Fracc. hidro",
              "Caudal ponderado (m3/s)":"Caudal","Indice ONI (ENSO)":"ONI",
              "Precip. embalses (mm)":"Precip. emb.","Tipo de dia":"Tipo dia","Poblacion":"Poblacion"}

STATES = {
    "Superavit Energetico":{"c":"#3498db","bg":"#eff6ff","fg":"#1e40af","t":"Superavit Energetico"},
    "Equilibrio Hidrico":  {"c":"#2ecc71","bg":"#f0fdf4","fg":"#166534","t":"Equilibrio Hidrico"},
    "Estres Leve":         {"c":"#f39c12","bg":"#fffdf0","fg":"#7d6608","t":"Estres Leve"},
    "Estres Severo":       {"c":"#e67e22","bg":"#fffbeb","fg":"#92400e","t":"Estres Severo"},
    "Crisis Energetica":   {"c":"#e74c3c","bg":"#fef2f2","fg":"#991b1b","t":"Crisis Energetica"},
}
RO = list(STATES.keys())

# KNN
NN_F = ["pct_hidro","dependencia_importacion","caudal_ponderado","excedente_exportacion"]
X_nn = dm[NN_F].values; nn_mu = X_nn.mean(0); nn_sd = X_nn.std(0)+1e-9
knn = NearestNeighbors(n_neighbors=6, metric="euclidean").fit((X_nn-nn_mu)/nn_sd)

def _lay():
    return dict(font=dict(family="IBM Plex Sans",size=11,color="#475569"),
                paper_bgcolor="white",plot_bgcolor="#fafbfc",
                margin=dict(l=16,r=16,t=36,b=12),
                xaxis=dict(gridcolor="#f1f5f9",zeroline=False),
                yaxis=dict(gridcolor="#f1f5f9",zeroline=False))

def sl(fig,title="",height=400,**kw):
    fig.update_layout(**_lay(),title=dict(text=title,font=dict(size=12,color="#0f172a")),height=height,**kw)

# ── SIDEBAR ────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='padding:20px 0 8px'>"
    "<p style='font-family:DM Serif Display,serif;font-size:1.15rem;color:#f1f5f9;margin:0'>"
    "SNI Ecuador</p>"
    "<p style='font-family:IBM Plex Mono;font-size:0.6rem;color:#64748b;margin:4px 0 0;"
    "letter-spacing:0.1em;text-transform:uppercase'>Modelado explicativo · 2009-2025</p></div>",
    unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("Seccion", [
    "Evaluacion",
    "Sistema Inteligente",
    "Recomendador",
    "Conclusiones",
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.75rem;color:#94a3b8;line-height:1.9'>"
    "K-Means · regimenes<br>Random Forest · clasificacion<br>"
    "XGBoost x4 · explicabilidad<br>SHAP · contribucion por variable<br>"
    "KNN · precedentes historicos<br>Recomendador · soporte decision<br><br>"
    "<span style='font-size:0.65rem;color:#64748b'>6,028 dias · 16 variables<br>"
    "CENACE / SMEC</span></div>",
    unsafe_allow_html=True)


# =====================================================================
# SECCION 1: EVALUACION
# =====================================================================
if page == "Evaluacion":
    st.title("Evaluacion del Sistema Electrico Nacional")
    st.markdown("<p class='lead'>Contexto general de los datos: como se compone la generacion, "
                "como evoluciono en 16 anos, que patrones existen.</p>", unsafe_allow_html=True)

    # KPIs generales
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Periodo", f"{dd['fecha'].min().strftime('%Y')} — {dd['fecha'].max().strftime('%Y')}")
    c2.metric("Observaciones", f"{len(dd):,} dias")
    c3.metric("Demanda promedio", f"{dd['demanda_gwh'].mean():.1f} GWh/dia")
    c4.metric("Fraccion hidro promedio", f"{dd['pct_hidro'].mean():.0%}")

    # Composicion de la matriz
    st.markdown("## Composicion")
    st.markdown("### Matriz energetica del Ecuador")
    mt = dd.set_index("fecha").resample("M").sum(numeric_only=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mt.index, y=mt["gen_hidraulica_gwh"], name="Hidro",
        fill="tozeroy", line=dict(width=0), fillcolor="rgba(21,101,192,0.7)"))
    fig.add_trace(go.Scatter(x=mt.index,
        y=mt["gen_hidraulica_gwh"]+mt["gen_termica_total_gwh"],
        name="Termica", fill="tonexty", line=dict(width=0), fillcolor="rgba(255,160,0,0.7)"))
    fig.add_trace(go.Scatter(x=mt.index,
        y=mt["gen_hidraulica_gwh"]+mt["gen_termica_total_gwh"]+mt["total_importacion_gwh"],
        name="Importacion", fill="tonexty", line=dict(width=0), fillcolor="rgba(192,57,43,0.6)"))
    fig.add_trace(go.Scatter(x=mt.index, y=mt["demanda_gwh"], name="Demanda",
        line=dict(color="black",width=2)))
    fig.add_shape(type="line",x0=pd.Timestamp("2016-04-01"),x1=pd.Timestamp("2016-04-01"),
                  y0=0,y1=1,yref="paper",line=dict(dash="dash",color="green",width=1.5))
    fig.add_annotation(x=pd.Timestamp("2016-06-01"),y=0.95,yref="paper",text="CCS 2016",
                       showarrow=False,font=dict(color="green",size=10))
    sl(fig,"Generacion por fuente vs demanda (GWh/mes)",height=400)
    fig.update_yaxes(title="GWh/mes")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<p class='narrative'>Azul: hidroelectrica. Amarillo: termica. Rojo: importacion. "
                "Linea negra: demanda. En 2016 entra Coca Codo Sinclair y el azul crece, "
                "pero la demanda sigue subiendo.</p>", unsafe_allow_html=True)

    # Estacionalidad
    st.markdown("## Patrones")
    st.markdown("### Estacionalidad mensual")
    var_sel = st.selectbox("Variable", [
        ("demanda_gwh","Demanda GWh/mes"),("pct_hidro","Fraccion hidro"),
        ("caudal_ponderado","Caudal m3/s"),("total_importacion_gwh","Importacion GWh/mes")],
        format_func=lambda x: x[1])
    fig2 = go.Figure()
    for m in range(1,13):
        fig2.add_trace(go.Box(y=dm[dm["mes"]==m][var_sel[0]].values, name=MESES[m-1],
            marker_color="#1e3a5f", opacity=0.7))
    sl(fig2,f"Distribucion mensual: {var_sel[1]}",height=350,showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Tipo de dia
    st.markdown("### Efecto del tipo de dia")
    tipos = sorted(dd["tipo_dia"].unique())
    cols = st.columns(3)
    for col, tgt, lb in [(cols[0],"demanda_gwh","Demanda"),(cols[1],"gen_hidraulica_gwh","Gen. hidro"),
                          (cols[2],"gen_termica_total_gwh","Gen. termica")]:
        with col:
            fig3 = go.Figure()
            for t in tipos:
                fig3.add_trace(go.Box(y=dd[dd["tipo_dia"]==t][tgt].values, name=t,
                    marker_color="#1e3a5f", opacity=0.7))
            sl(fig3,lb,height=280,showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    # Correlaciones
    st.markdown("### Correlaciones entre variables clave")
    CV=["demanda_gwh","pct_hidro","caudal_ponderado","oni_mensual",
        "total_importacion_gwh","total_exportacion_gwh","gen_termica_total_gwh"]
    CL=["Demanda","Fracc.hidro","Caudal","ONI","Import.","Export.","Gen.term."]
    co = dm[CV].corr()
    fig4 = go.Figure(go.Heatmap(z=co.values,x=CL,y=CL,colorscale="RdBu_r",zmin=-1,zmax=1,
        text=np.round(co.values,2),texttemplate="%{text}"))
    sl(fig4,"Correlaciones mensuales",height=420)
    st.plotly_chart(fig4, use_container_width=True)


# =====================================================================
# SECCION 2: SISTEMA INTELIGENTE
# =====================================================================
elif page == "Sistema Inteligente":
    st.title("Sistema Inteligente de Modelado Explicativo")
    st.markdown("<p class='lead'>K-Means identifica los regimenes, Random Forest evalua "
                "las variables exogenas, XGBoost + SHAP explica las causas dia a dia, "
                "KNN encuentra precedentes historicos.</p>", unsafe_allow_html=True)

    # ── K-MEANS ──
    st.markdown("## K-Means · Aprendizaje no supervisado")
    st.markdown("### 5 regimenes del sistema electrico")
    st.markdown("<p class='narrative'>K-Means analizo 198 meses y descubrio 5 estados operativos "
                "recurrentes. k=5 seleccionado por Calinski-Harabasz (maximo). Nombres asignados "
                "por score compuesto normalizado de 4 variables.</p>", unsafe_allow_html=True)

    # Timeline
    fig_tl = go.Figure()
    for r in RO:
        s = dm[dm["estado"]==r]
        if len(s)>0:
            fig_tl.add_trace(go.Bar(x=s["fecha"],y=[1]*len(s),name=r,
                marker_color=STATES[r]["c"],width=30*86400000))
    sl(fig_tl,"Regimen por mes (2009-2025)",height=200,barmode="stack",showlegend=True)
    fig_tl.update_yaxes(visible=False)
    st.plotly_chart(fig_tl, use_container_width=True)

    # Tabla de regimenes
    reg_data = []
    for r in RO:
        s = dm[dm["estado"]==r]
        reg_data.append({"Regimen":r,"Meses":len(s),"% del total":f"{len(s)/len(dm)*100:.0f}%",
            "Hidro":f"{s['pct_hidro'].mean()*100:.0f}%",
            "Import.":f"{s['total_importacion_gwh'].mean():.0f} GWh",
            "Caudal":f"{s['caudal_ponderado'].mean():.0f} m3/s"})
    st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)

    # Validacion
    st.markdown("<p class='narrative'><strong>Validacion:</strong> 73% de los meses con racionamiento "
                "oficial (decretos ejecutivos) fueron clasificados en estados graves "
                "(Crisis o Estres Severo) sin que K-Means recibiera esa informacion.</p>",
                unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── RANDOM FOREST ──
    st.markdown("## Random Forest · Clasificacion supervisada")
    st.markdown("### Las condiciones externas explican los regimenes?")
    st.markdown("<p class='narrative'>RF intenta reproducir los 5 regimenes de K-Means usando "
                "solo 4 variables exogenas: caudal ponderado, ONI, precipitacion embalses, poblacion. "
                "F1=0.396. La precision moderada es un hallazgo: las condiciones externas explican "
                "parcialmente el estado del sistema.</p>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        feat_rf = mm["feature_vars_clf"]
        imp = rf.feature_importances_
        idx = np.argsort(imp)
        lbl_rf = {"caudal_ponderado":"Caudal","oni_mensual":"ONI",
                  "precip_embalses":"Precip. emb.","poblacion":"Poblacion"}
        fig_fi = go.Figure(go.Bar(y=[lbl_rf.get(feat_rf[i],feat_rf[i]) for i in idx],
            x=imp[idx],orientation="h",marker_color="#1e3a5f"))
        sl(fig_fi,"Importancia de variables (Gini)",height=250)
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        X_rf = dm[feat_rf].dropna().values
        y_rf = dm.loc[dm[feat_rf].dropna().index,"estado"].values
        yp = cross_val_predict(rf, X_rf, y_rf, cv=StratifiedKFold(5,shuffle=False))
        cm = confusion_matrix(y_rf, yp, labels=RO)
        fig_cm = go.Figure(go.Heatmap(z=cm,x=RO,y=RO,colorscale="Blues",text=cm,texttemplate="%{text}"))
        sl(fig_cm,"Matriz de confusion (5-Fold CV)",height=320)
        fig_cm.update_xaxes(tickangle=45)
        fig_cm.update_yaxes(title="Real")
        fig_cm.update_xaxes(title="RF")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── XGBOOST + SHAP ──
    st.markdown("## XGBoost + SHAP · Modelos explicativos")
    st.markdown("### Que factores determinaron la generacion e intercambio?")
    st.markdown("<p class='narrative'>4 modelos entrenados sobre 6,028 dias. "
                "Seleccione un modelo y una fecha. SHAP descompone cuanto contribuyo "
                "cada variable al resultado de ese dia. "
                "<span style='color:#c0392b'>Rojo</span> = aumento. "
                "<span style='color:#2980b9'>Azul</span> = redujo.</p>",
                unsafe_allow_html=True)

    # Performance
    perf = []
    for n,r in md["results"].items():
        perf.append({"Modelo":{"gen_termica":"Gen. termica","gen_hidro":"Gen. hidro",
            "importacion":"Importacion","exportacion":"Exportacion"}.get(n,n),
            "R2 Train":f"{r['r2_train']:.3f}","R2 Test":f"{r['r2_test']:.3f}",
            "MAE Test":f"{r['mae_test']:.2f} GWh"})
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    with c1:
        ms = st.selectbox("Modelo", list(md["targets"].values()),
            format_func=lambda x:{"importacion":"Importacion","gen_hidro":"Gen. hidro",
                "gen_termica":"Gen. termica","exportacion":"Exportacion"}.get(x,x))
    with c2:
        fd = st.date_input("Fecha", value=dd["fecha"].max().date(),
            min_value=dd["fecha"].min().date(), max_value=dd["fecha"].max().date())

    row = dd[dd["fecha"].dt.date==fd]
    if len(row)==0:
        st.warning("Fecha no disponible.")
    else:
        row = row.iloc[0]
        Xi = np.array([[row[f] for f in md["features"]]])
        tc = [k for k,v in md["targets"].items() if v==ms][0]
        rv = float(row[tc]); pv = max(0,float(mo[ms].predict(Xi)[0]))
        sv = np.array(ex[ms].shap_values(Xi)).flatten()

        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("Valor real",f"{rv:.2f} GWh")
        mc2.metric("Modelo",f"{pv:.2f} GWh")
        mc3.metric("Error",f"{abs(rv-pv):.2f} GWh")

        fn = [SHAP_SHORT.get(NI.get(f,f),NI.get(f,f)) for f in md["features"]]
        # Valores reales del dia y promedios historicos
        feats = md["features"]
        valores_dia = [float(row[f]) for f in feats]
        promedios_hist = [float(dd[f].mean()) for f in feats]
        sdf = pd.DataFrame({
            "v": fn, "s": sv, "valor": valores_dia, "promedio": promedios_hist
        }).reindex(pd.DataFrame({"s":sv}).s.abs().sort_values().index).tail(5)

        fig_sh = go.Figure(go.Bar(y=sdf["v"],x=sdf["s"],orientation="h",
            marker_color=["#c0392b" if v>0 else "#2980b9" for v in sdf["s"]],
            text=[f"{v:+.2f}" for v in sdf["s"]],textposition="outside",
            textfont=dict(size=10,family="IBM Plex Mono")))
        sl(fig_sh,f"SHAP — {fd.strftime('%d/%m/%Y')}",height=240)
        fig_sh.update_xaxes(title="GWh")
        fig_sh.update_layout(margin=dict(l=100,r=40,t=36,b=20))
        st.plotly_chart(fig_sh, use_container_width=True)

        # Tabla con valores reales, promedios y SHAP para interpretacion
        st.markdown("##### Detalle por variable")
        tabla_shap = sdf.iloc[::-1].copy()  # invertir orden (mayor SHAP arriba)
        tabla_data = []
        for _, r in tabla_shap.iterrows():
            valor = r["valor"]; prom = r["promedio"]; shap_val = r["s"]
            # Indicador relativo al promedio
            if abs(valor - prom) < 0.01 * abs(prom):
                comp = "≈ promedio"
            elif valor > prom:
                pct = (valor / prom - 1) * 100 if prom > 0 else 0
                comp = f"↑ {pct:+.0f}% vs promedio"
            else:
                pct = (valor / prom - 1) * 100 if prom > 0 else 0
                comp = f"↓ {pct:+.0f}% vs promedio"
            efecto = "aumentó" if shap_val > 0 else "redujo"
            tabla_data.append({
                "Variable": r["v"],
                "Valor del día": f"{valor:.2f}",
                "Promedio histórico": f"{prom:.2f}",
                "Estado": comp,
                "Contribución SHAP": f"{shap_val:+.2f} GWh",
                "Efecto": efecto,
            })
        st.dataframe(pd.DataFrame(tabla_data), use_container_width=True, hide_index=True)

        top = sdf.iloc[-1]
        d = "incremento" if top["s"]>0 else "redujo"
        note = ""
        if ms=="exportacion" and "oblacion" in top["v"]:
            note = (" *(Poblacion actua como proxy temporal del quiebre de "
                    "Coca Codo Sinclair en 2016, no como causa directa.)*")
        st.markdown(f"Factor principal: **{top['v']}** {d} en {abs(top['s']):.2f} GWh.{note}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── KNN ──
    st.markdown("## KNN · Precedentes historicos")
    st.markdown("### Meses con condiciones similares")

    fecha_knn = st.date_input("Mes de consulta", value=dd["fecha"].max().date(),
        min_value=dd["fecha"].min().date(), max_value=dd["fecha"].max().date(), key="knn_date")
    mes_ts = pd.Timestamp(fecha_knn.year, fecha_knn.month, 1)
    row_m = dm[dm["fecha"]==mes_ts]

    if len(row_m)==0:
        st.warning("Mes no disponible.")
    else:
        row_m = row_m.iloc[0]
        x_q = np.array([[row_m[f] for f in NN_F]])
        dists, indices = knn.kneighbors((x_q-nn_mu)/nn_sd)
        vecinos = dm.iloc[indices[0][1:]]

        st.markdown(f"**{mes_ts.strftime('%B %Y')}** — Estado: **{row_m['estado']}** — "
                    f"Hidro: {row_m['pct_hidro']:.0%} — Import: {row_m['total_importacion_gwh']:.0f} GWh")

        vd = []
        for _,v in vecinos.iterrows():
            vd.append({"Periodo":v["fecha"].strftime("%Y-%m"),"Estado":v["estado"],
                "Hidro":f"{v['pct_hidro']*100:.0f}%",
                "Importacion":f"{v['total_importacion_gwh']:.0f} GWh",
                "Caudal":f"{v['caudal_ponderado']:.0f} m3/s"})
        st.dataframe(pd.DataFrame(vd), use_container_width=True, hide_index=True)

        st.markdown("<p class='narrative'>Precedentes identificados por distancia euclidiana "
                    "en 4 variables normalizadas del sistema. No es prediccion: "
                    "muestra meses historicos con condiciones similares.</p>",
                    unsafe_allow_html=True)


# =====================================================================
# SECCION 3: RECOMENDADOR DE ESCENARIOS
# =====================================================================
elif page == "Recomendador":
    st.title("Recomendador de Escenarios de Intercambio")
    st.markdown("<p class='lead'>Sistema de soporte a decisiones basado en patrones historicos. "
                "Identifica el regimen actual mediante K-Means, recupera precedentes similares con KNN, "
                "y caracteriza el comportamiento historico del sistema en condiciones equivalentes.</p>",
                unsafe_allow_html=True)

    st.markdown("<p class='narrative'>No es prediccion futura. Es prescripcion basada en evidencia: "
                "<em>en condiciones similares historicamente, el sistema se comporto de esta manera</em>. "
                "Esto es analisis de escenarios, no forecasting.</p>", unsafe_allow_html=True)

    # Preparar dataframe normalizado para el recomendador
    dm_rec = dm.copy()
    dm_rec['regimen_norm'] = dm_rec['estado'].apply(rec_mod.normalizar_regimen)

    st.markdown("## Consulta")
    st.markdown("### Seleccione un mes para analizar")

    cc1, cc2 = st.columns([1,2])
    with cc1:
        fecha_rec = st.date_input("Mes a evaluar", value=dm["fecha"].max().date(),
            min_value=dm["fecha"].min().date(), max_value=dm["fecha"].max().date(),
            key="rec_date")
    mes_rec = pd.Timestamp(fecha_rec.year, fecha_rec.month, 1)
    fila_rec = dm_rec[dm_rec["fecha"]==mes_rec]

    if len(fila_rec)==0:
        st.warning("Mes no disponible.")
    else:
        fila_rec = fila_rec.iloc[0]
        regimen_actual = fila_rec['regimen_norm']

        # Caracterizacion historica del regimen
        carac = rec_mod.caracterizar_regimen(dm_rec, regimen_actual)

        # Precedentes KNN sobre dm_rec
        # Adaptamos las features a las columnas disponibles
        feats_disp = ['caudal_ponderado','oni_mensual','precip_embalses','demanda_gwh','mes']
        dm_for_knn = dm_rec.copy()
        condiciones = {f: float(fila_rec[f]) for f in feats_disp}

        # Necesitamos recomendador con features compatibles
        # Llamamos directamente con las features disponibles
        from sklearn.preprocessing import StandardScaler
        X_knn = dm_for_knn[feats_disp].values
        sc = StandardScaler()
        Xn = sc.fit_transform(X_knn)
        consulta_n = sc.transform(np.array([[condiciones[f] for f in feats_disp]]))
        knn_rec = NearestNeighbors(n_neighbors=11, metric='euclidean').fit(Xn)
        d, i = knn_rec.kneighbors(consulta_n)
        # Excluir self
        idx_vec = i[0][1:6]
        precedentes = dm_for_knn.iloc[idx_vec].copy()
        precedentes['distancia'] = d[0][1:6]

        # Analisis precedentes
        n_p = len(precedentes)
        n_e = int((precedentes['total_exportacion_gwh']>10).sum())
        n_i = int((precedentes['total_importacion_gwh']>10).sum())
        ana = {
            'n_precedentes': n_p,
            'n_exportaron': n_e,
            'pct_exportaron': round(100*n_e/n_p,1),
            'n_importaron': n_i,
            'pct_importaron': round(100*n_i/n_p,1),
            'export_promedio': round(float(precedentes['total_exportacion_gwh'].mean()),1),
            'import_promedio': round(float(precedentes['total_importacion_gwh'].mean()),1),
        }

        recomendacion = rec_mod.generar_recomendacion(regimen_actual, carac, ana)

        # ── PANEL 1: Estado actual ──
        st.markdown("## Estado detectado")
        st.markdown(f"### {mes_rec.strftime('%B %Y')}")

        sev_colors = {
            'oportunidad': ('#3498db','#eff6ff','#1e40af'),
            'normal': ('#2ecc71','#f0fdf4','#166534'),
            'precaucion': ('#f1c40f','#fefce8','#854d0e'),
            'alerta': ('#f39c12','#fffdf0','#7d6608'),
            'critico': ('#e74c3c','#fef2f2','#991b1b'),
            'desconocido': ('#94a3b8','#f8fafc','#475569'),
        }
        c, bg, fg = sev_colors.get(recomendacion['severidad'], sev_colors['desconocido'])

        st.markdown(
            f"<div class='state-box' style='background:{bg};border-left:6px solid {c}'>"
            f"<div class='dot' style='background:{c}'></div>"
            f"<div><p style='font-family:DM Serif Display;font-size:1.4rem;color:{fg};margin:0'>"
            f"{recomendacion['titulo']}</p>"
            f"<p style='font-size:0.85rem;color:{fg};margin:4px 0 0'>"
            f"<strong>Regimen:</strong> {regimen_actual}</p></div></div>",
            unsafe_allow_html=True
        )

        # KPIs del mes consultado
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Caudal", f"{fila_rec['caudal_ponderado']:.0f} m³/s")
        k2.metric("Hidro", f"{fila_rec['pct_hidro']:.0%}")
        k3.metric("Importacion", f"{fila_rec['total_importacion_gwh']:.0f} GWh")
        k4.metric("Exportacion", f"{fila_rec['total_exportacion_gwh']:.0f} GWh")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── PANEL 2: Recomendacion ──
        st.markdown("## Recomendacion")
        st.markdown("### Accion sugerida y racional")
        st.markdown(
            f"<div class='card'>"
            f"<p class='cl'>Accion recomendada</p>"
            f"<p style='font-size:1rem;color:#0f172a;margin:0 0 8px'>{recomendacion['accion']}</p>"
            f"<p class='cl'>Racional</p>"
            f"<p style='font-size:0.9rem;color:#475569;margin:0'>{recomendacion['racional']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── PANEL 3: Caracterizacion historica del regimen ──
        st.markdown("## Caracterizacion historica del regimen")
        st.markdown(f"### Comportamiento tipico cuando el sistema esta en \"{regimen_actual}\"")
        st.markdown(
            f"<p class='narrative'>Basado en <strong>{carac['n_meses_historicos']} meses</strong> "
            f"({carac['pct_periodo']}% del periodo 2009-2025) clasificados en este regimen.</p>",
            unsafe_allow_html=True
        )

        # Boxplot comparativo: import/export por regimen, marcando el actual
        fig_box = make_subplots(rows=1, cols=2,
            subplot_titles=["Importacion historica por regimen","Exportacion historica por regimen"])
        regs_orden = ['Superavit Energetico','Equilibrio Hidrico','Estres Leve','Estres Severo','Crisis Energetica']
        for r in regs_orden:
            sub_r = dm_rec[dm_rec['regimen_norm']==r]
            if len(sub_r)==0:
                continue
            color = '#e74c3c' if r==regimen_actual else '#94a3b8'
            fig_box.add_trace(go.Box(y=sub_r['total_importacion_gwh'].values, name=r,
                marker_color=color, showlegend=False), row=1, col=1)
            fig_box.add_trace(go.Box(y=sub_r['total_exportacion_gwh'].values, name=r,
                marker_color=color, showlegend=False), row=1, col=2)
        sl(fig_box,"",height=320)
        fig_box.update_yaxes(title="GWh/mes")
        st.plotly_chart(fig_box, use_container_width=True)

        # Tabla resumen estadistico del regimen actual
        exp = carac['exportacion_gwh']
        imp = carac['importacion_gwh']
        gen = carac['generacion_gwh']
        tabla = pd.DataFrame([
            {"Variable":"Exportacion (GWh/mes)","Mediana":f"{exp['median']:.1f}",
             "Promedio":f"{exp['mean']:.1f}","p25-p75":f"{exp['p25']:.1f} – {exp['p75']:.1f}",
             "Min-Max":f"{exp['min']:.1f} – {exp['max']:.1f}"},
            {"Variable":"Importacion (GWh/mes)","Mediana":f"{imp['median']:.1f}",
             "Promedio":f"{imp['mean']:.1f}","p25-p75":f"{imp['p25']:.1f} – {imp['p75']:.1f}",
             "Min-Max":f"{imp['min']:.1f} – {imp['max']:.1f}"},
            {"Variable":"Generacion total (GWh/mes)","Mediana":f"{gen['median']:.0f}",
             "Promedio":f"{gen['mean']:.0f}","p25-p75":f"{gen['p25']:.0f} – {gen['p75']:.0f}",
             "Min-Max":f"{gen['min']:.0f} – {gen['max']:.0f}"},
        ])
        st.dataframe(tabla, use_container_width=True, hide_index=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── PANEL 4: Precedentes KNN ──
        st.markdown("## Precedentes historicos similares")
        st.markdown("### 5 meses con condiciones mas parecidas (KNN)")

        st.markdown(
            f"<p class='narrative'>De los 5 meses con condiciones mas similares: "
            f"<strong>{ana['pct_importaron']}%</strong> tuvieron importacion (promedio "
            f"{ana['import_promedio']} GWh) y <strong>{ana['pct_exportaron']}%</strong> "
            f"tuvieron exportacion (promedio {ana['export_promedio']} GWh).</p>",
            unsafe_allow_html=True
        )

        prec_tabla = []
        for _, p in precedentes.iterrows():
            prec_tabla.append({
                "Periodo": p['fecha'].strftime("%Y-%m"),
                "Regimen": p['regimen_norm'],
                "Caudal": f"{p['caudal_ponderado']:.0f}",
                "ONI": f"{p['oni_mensual']:+.2f}",
                "Importacion": f"{p['total_importacion_gwh']:.0f} GWh",
                "Exportacion": f"{p['total_exportacion_gwh']:.0f} GWh",
            })
        st.dataframe(pd.DataFrame(prec_tabla), use_container_width=True, hide_index=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── PANEL 5: Evidencia integrada ──
        st.markdown("## Evidencia integrada")
        st.markdown("### Sintesis para soporte a decision")
        st.markdown(
            f"<div class='card'>"
            f"<p style='font-size:0.92rem;color:#0f172a;line-height:1.7;margin:0'>"
            f"{recomendacion['evidencia_historica']}</p></div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p class='narrative' style='font-size:0.78rem'><strong>Fuentes de la recomendacion:</strong> "
            "(1) K-Means clasifica el regimen del mes consultado a partir de variables del sistema. "
            "(2) Estadisticas historicas del regimen calculadas sobre 198 meses (2009-2025). "
            "(3) KNN identifica los 5 meses con condiciones mas similares. "
            "(4) La accion sugerida deriva de la caracterizacion documentada del regimen.</p>",
            unsafe_allow_html=True
        )


# =====================================================================
# SECCION 4: CONCLUSIONES
# =====================================================================
elif page == "Conclusiones":
    st.title("Conclusiones del Analisis")
    st.markdown("<p class='lead'>Hallazgos principales del modelado explicativo del "
                "Sistema Nacional Interconectado del Ecuador.</p>", unsafe_allow_html=True)

    # Hallazgo 1: dependencia hidrica
    st.markdown("## Hallazgo 1")
    st.markdown("### Ecuador depende criticamente de la hidroelectricidad")
    c1,c2 = st.columns([2,1])
    with c1:
        mt = dd.set_index("fecha")["pct_hidro"].resample("M").mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mt.index,y=mt.values*100,fill="tozeroy",
            fillcolor="rgba(21,101,192,0.3)",line=dict(color="#1565C0",width=2)))
        fig.add_hline(y=80,line_dash="dash",line_color="gray",opacity=0.5)
        sl(fig,"Fraccion hidroelectrica mensual (%)",height=280)
        fig.update_yaxes(title="%",range=[20,100])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"<div class='card'><p class='cl'>Promedio historico</p>"
                    f"<p class='cv'>{dd['pct_hidro'].mean():.0%}</p>"
                    f"<p class='cu'>de la generacion es hidroelectrica</p></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='card'><p class='cl'>Minimo registrado</p>"
                    f"<p class='cv'>{dd['pct_hidro'].min():.0%}</p>"
                    f"<p class='cu'>en el peor dia del dataset</p></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='card'><p class='cl'>Meses en crisis</p>"
                    f"<p class='cv'>{len(dm[dm['estado']=='Crisis Energetica'])}</p>"
                    f"<p class='cu'>de 198 meses totales</p></div>",
                    unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Hallazgo 2: estacionalidad vs demanda
    st.markdown("## Hallazgo 2")
    st.markdown("### El caudal es estacional pero la demanda no")
    st.markdown("<p class='narrative'>La demanda crece cada ano sin patron estacional. "
                "El caudal sube y baja con las lluvias. Ese desacople es lo que genera "
                "estres en los meses secos (septiembre-diciembre).</p>", unsafe_allow_html=True)

    fig2 = make_subplots(rows=1,cols=2,subplot_titles=["Demanda por mes","Caudal por mes"])
    for m in range(1,13):
        fig2.add_trace(go.Box(y=dm[dm["mes"]==m]["demanda_gwh"].values,name=MESES[m-1],
            marker_color="#37474F",opacity=0.7,showlegend=False),row=1,col=1)
        fig2.add_trace(go.Box(y=dm[dm["mes"]==m]["caudal_ponderado"].values,name=MESES[m-1],
            marker_color="#00838F",opacity=0.7,showlegend=False),row=1,col=2)
    sl(fig2,"",height=300)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Hallazgo 3: SHAP drivers
    st.markdown("## Hallazgo 3")
    st.markdown("### Los factores que impulsan la generacion e intercambio")

    drivers = {
        "Gen. termica (R2=0.947)": "Se activa cuando la fraccion hidro baja y la demanda sube. Es la respuesta directa del sistema al deficit hidroelectrico.",
        "Gen. hidro (R2=0.680)": "Depende del caudal ponderado (disponibilidad de agua) y de la demanda. El ONI (El Nino/La Nina) afecta indirectamente via los caudales.",
        "Importacion (R2=0.384)": "Ocurre cuando la fraccion hidro es baja y la demanda es alta. La precision moderada refleja que las importaciones dependen tambien de la disponibilidad de Colombia/Peru.",
        "Exportacion (R2=0.216)": "El modelo mas debil. Las exportaciones dependen de acuerdos bilaterales y de la existencia de Coca Codo Sinclair (2016), capturada como proxy por la variable poblacion.",
    }
    for titulo, desc in drivers.items():
        st.markdown(f"**{titulo}**")
        st.markdown(f"<p class='narrative'>{desc}</p>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Hallazgo 4: K-Means valida crisis
    st.markdown("## Hallazgo 4")
    st.markdown("### K-Means detecto las crisis sin que se lo dijeran")
    val_data = [
        {"Periodo":"Crisis 2009","Meses":4,"Crisis":0,"Estres Severo":4,"Total graves":"4/4 (100%)"},
        {"Periodo":"Crisis 2023-F1","Meses":5,"Crisis":3,"Estres Severo":0,"Total graves":"3/5 (60%)"},
        {"Periodo":"Crisis 2024-F2","Meses":2,"Crisis":0,"Estres Severo":0,"Total graves":"0/2 (0%)"},
        {"Periodo":"Crisis 2024-F3","Meses":4,"Crisis":3,"Estres Severo":1,"Total graves":"4/4 (100%)"},
    ]
    st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)
    st.markdown("<p class='narrative'>73% de los meses con racionamiento oficial fueron detectados "
                "como graves. Los 4 no detectados tenian hidro 78-83%: el sistema funcionaba, "
                "el racionamiento fue preventivo.</p>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Nota sobre renovables
    st.markdown("## Implicacion")
    st.markdown("### Diversificacion de la matriz energetica")
    st.markdown(
        "<p class='narrative'>El analisis SHAP muestra consistentemente que la baja fraccion "
        "hidroelectrica es el factor principal que impulsa tanto la generacion termica como las "
        "importaciones. Esto lleva a una conclusion directa: la diversificacion de la matriz "
        "energetica con fuentes complementarias — como la solar, cuya generacion es mayor "
        "en los meses de epoca seca cuando la hidro falla — reduciria la vulnerabilidad "
        "del sistema ante deficit hidricos.</p>", unsafe_allow_html=True)

    # Nota metodologica
    st.markdown(
        "<div class='footer'>"
        "<strong>Nota metodologica</strong><br><br>"
        "<strong>K-Means:</strong> No supervisado, 198 meses, 4 variables, k=5 (Calinski-Harabasz). "
        "Score compuesto para asignacion de nombres.<br>"
        "<strong>Random Forest:</strong> 200 arboles, 4 features exogenas, F1=0.396 (5-Fold CV).<br>"
        "<strong>XGBoost:</strong> 4 modelos sobre 6,028 dias, 7 features. Train 2009-2022, Test 2023-2025.<br>"
        "<strong>SHAP:</strong> TreeExplainer sobre cada XGBoost.<br>"
        "<strong>KNN:</strong> 5 vecinos, distancia euclidiana normalizada, 4 variables del sistema.<br>"
        "<strong>Recomendador:</strong> Soporte a decision basado en caracterizacion historica "
        "del regimen detectado + precedentes KNN. Prescripcion sin forecasting.<br><br>"
        "<em>El sistema analiza, explica, identifica condiciones del comportamiento energetico "
        "historico, y recomienda acciones basadas en patrones documentados. No realiza predicciones.</em></div>",
        unsafe_allow_html=True)
