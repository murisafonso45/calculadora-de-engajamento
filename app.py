"""
Preditor de Alcance de Vídeo (Viral x Flopado)
------------------------------------------------
App Streamlit + TensorFlow para prever o número de visualizações de um vídeo
com base na categoria do conteúdo e na quantidade de hashtags utilizadas.

Rodar localmente:
    streamlit run app.py

Deploy no Render:
    Start Command -> streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    (ver requirements.txt em anexo)
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# ==============================================================================
# Configuração da página
# ==============================================================================
st.set_page_config(
    page_title="Preditor de Alcance de Vídeo",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Preditor de Alcance de Vídeo")
st.caption(
    "Ferramenta interna — estimativa de visualizações com base em categoria "
    "e quantidade de hashtags, usando um modelo de rede neural (TensorFlow)."
)

CATEGORIAS = [
    "Educacional",
    "Entretenimento",
    "Moda",
    "Tecnologia",
    "Humor",
    "Fitness",
]

# Parâmetros de negócio (ajustáveis pela área de marketing/dados)
LIMITE_VIRAL = 0.85   # percentil acima do qual o post é classificado como Viral
LIMITE_FLOP = 0.20    # percentil abaixo do qual o post é classificado como Flopado

# ==============================================================================
# 1. Geração de dados sintéticos (substituir por dados históricos reais)
# ==============================================================================
@st.cache_data
def gerar_dados(n_por_categoria=300, seed=7):
    """
    Simula histórico de posts: cada categoria tem um patamar de alcance base
    e uma resposta diferente ao número de hashtags (com saturação e ruído).
    """
    rng = np.random.default_rng(seed)
    registros = []

    perfil_categoria = {
        "Educacional":     {"base": 8_000,  "ganho": 600,  "saturacao": 15},
        "Entretenimento":  {"base": 40_000, "ganho": 2500, "saturacao": 12},
        "Moda":            {"base": 25_000, "ganho": 1800, "saturacao": 10},
        "Tecnologia":      {"base": 15_000, "ganho": 900,  "saturacao": 18},
        "Humor":           {"base": 60_000, "ganho": 3000, "saturacao": 8},
        "Fitness":         {"base": 20_000, "ganho": 1200, "saturacao": 14},
    }

    for categoria, perfil in perfil_categoria.items():
        hashtags = rng.integers(0, 30, n_por_categoria)
        # Curva com saturação (efeito marginal decrescente após o ponto ótimo)
        efeito = perfil["ganho"] * np.minimum(hashtags, perfil["saturacao"]) \
                 - 150 * np.maximum(hashtags - perfil["saturacao"], 0)
        ruido_multiplicativo = rng.lognormal(mean=0, sigma=0.35, size=n_por_categoria)
        views = np.clip((perfil["base"] + efeito), 500, None) * ruido_multiplicativo

        for h, v in zip(hashtags, views):
            registros.append({"categoria": categoria, "hashtags": h, "views": v})

    return pd.DataFrame(registros)


df = gerar_dados()

with st.expander("Ver amostra dos dados de treino"):
    st.dataframe(df.sample(10).reset_index(drop=True))

# ==============================================================================
# 2. Pré-processamento
# ==============================================================================
@st.cache_resource
def preparar_pipeline(df):
    # One-hot das categorias (ordem fixa para garantir consistência na inferência)
    dummies = pd.get_dummies(df["categoria"], columns=CATEGORIAS)
    dummies = dummies.reindex(columns=CATEGORIAS, fill_value=0)

    hashtags_mean, hashtags_std = df["hashtags"].mean(), df["hashtags"].std()
    hashtags_norm = (df["hashtags"] - hashtags_mean) / hashtags_std

    X = pd.concat([dummies, hashtags_norm.rename("hashtags_norm")], axis=1).values.astype("float32")

    # Log-transform no alvo: views variam em ordens de grandeza entre categorias
    y_log = np.log1p(df["views"]).values.astype("float32")
    y_mean, y_std = y_log.mean(), y_log.std()
    y_norm = (y_log - y_mean) / y_std

    stats = {
        "hashtags_mean": hashtags_mean,
        "hashtags_std": hashtags_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return X, y_norm, stats


X, y_norm, stats = preparar_pipeline(df)

# ==============================================================================
# 3. Treinamento do modelo (cacheado)
# ==============================================================================
@st.cache_resource
def treinar_modelo(X, y, epochs=150, lr=0.01):
    tf.random.set_seed(42)
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    historico = modelo.fit(X, y, epochs=epochs, verbose=0)
    return modelo, historico.history["loss"]


with st.spinner("Treinando modelo preditivo..."):
    modelo, loss_hist = treinar_modelo(X, y_norm)

# Thresholds de negócio calculados sobre a distribuição real de views
limite_viral_views = df["views"].quantile(LIMITE_VIRAL)
limite_flop_views = df["views"].quantile(LIMITE_FLOP)


def prever_views(categoria, hashtags):
    dummies = pd.DataFrame([[1 if c == categoria else 0 for c in CATEGORIAS]], columns=CATEGORIAS)
    hashtags_norm = (hashtags - stats["hashtags_mean"]) / stats["hashtags_std"]
    entrada = np.concatenate([dummies.values, [[hashtags_norm]]], axis=1).astype("float32")

    pred_norm = modelo.predict(entrada, verbose=0)[0][0]
    pred_log = pred_norm * stats["y_std"] + stats["y_mean"]
    return float(np.expm1(pred_log))


# ==============================================================================
# 4. Interface de previsão
# ==============================================================================
st.subheader("🎯 Simular previsão")

col1, col2 = st.columns(2)
with col1:
    categoria_input = st.selectbox("Categoria do vídeo", CATEGORIAS)
with col2:
    hashtags_input = st.slider("Quantidade de hashtags", min_value=0, max_value=30, value=10)

views_previstas = prever_views(categoria_input, hashtags_input)

st.metric("Alcance estimado (visualizações)", f"{views_previstas:,.0f}".replace(",", "."))

# Alerta visual — classificação Viral / Flopado
if views_previstas >= limite_viral_views:
    st.success(f"🔥 **VIRAL** — acima do percentil {int(LIMITE_VIRAL*100)}% de alcance histórico da base.")
elif views_previstas <= limite_flop_views:
    st.error(f"📉 **FLOPADO** — abaixo do percentil {int(LIMITE_FLOP*100)}% de alcance histórico da base.")
else:
    st.warning("➖ **DESEMPENHO NEUTRO** — dentro da faixa mediana esperada.")

st.progress(min(views_previstas / df["views"].max(), 1.0))

# ==============================================================================
# 5. Gráfico: curva de alcance por hashtags para a categoria selecionada
# ==============================================================================
faixa_hashtags = np.arange(0, 31)
curva_views = [prever_views(categoria_input, h) for h in faixa_hashtags]

fig, ax = plt.subplots()
subset = df[df["categoria"] == categoria_input]
ax.scatter(subset["hashtags"], subset["views"], alpha=0.3, label="Histórico (simulado)")
ax.plot(faixa_hashtags, curva_views, color="red", linewidth=2, label="Previsão do modelo")
ax.axhline(limite_viral_views, color="green", linestyle="--", linewidth=1, label="Limite Viral")
ax.axhline(limite_flop_views, color="orange", linestyle="--", linewidth=1, label="Limite Flopado")
ax.scatter([hashtags_input], [views_previstas], color="black", zorder=5, s=100, label="Simulação atual")
ax.set_xlabel("Quantidade de hashtags")
ax.set_ylabel("Visualizações")
ax.set_title(f"Curva de alcance — {categoria_input}")
ax.legend()
st.pyplot(fig)

# ==============================================================================
# 6. Informações do modelo (transparência para uso interno)
# ==============================================================================
with st.expander("Detalhes técnicos do modelo"):
    st.write(f"Loss final (MSE, escala normalizada): {loss_hist[-1]:.4f}")
    st.write(f"Limite Viral (percentil {int(LIMITE_VIRAL*100)}%): {limite_viral_views:,.0f} views")
    st.write(f"Limite Flopado (percentil {int(LIMITE_FLOP*100)}%): {limite_flop_views:,.0f} views")
    st.caption(
        "Dados de treino são sintéticos e servem como placeholder. "
        "Substitua `gerar_dados()` por uma consulta ao histórico real de posts "
        "(ex.: banco de dados interno ou export do CRM de marketing) antes de "
        "usar em decisões de negócio."
    )
