# app.py
import os
from typing import List, Dict, Any

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# API KEY Loader
#   - Streamlit Cloud: st.secrets に保存したキーを環境変数へコピー
#   - ローカル: 既に export した環境変数をそのまま使用
#   どちらも未設定の場合は下位モジュールでエラーとなります。
# ─────────────────────────────────────────────────────────────────────────────
for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_GEMINI_API_KEY"):
    if not os.getenv(var) and var in st.secrets:
        os.environ[var] = st.secrets[var]

from llm_openai import generate_ad_openai
from llm_claude import generate_ad_claude
from llm_gemini import generate_ad_gemini

# ───────────────────────────────────────────────────────────
# Streamlit – ページ設定
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Ad Script Generator",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 マルチ LLM 動画広告台本ジェネレーター")
st.caption("OpenAI / Claude / Gemini をワンタップ比較")

# ───────────────────────────────────────────────────────────
# サイドバー – 入力パラメータ
# ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 商品情報 & パラメータ")

    product_name = st.text_input("商品名", "雲ごこちプレミアムピロー")
    problem = st.text_input("悩み・課題", "首こりで熟睡できない")
    promise = st.text_input("解決後のベネフィット、アピールポイント", "ホテル級の深睡眠")
    
    tones = st.multiselect(
        "動画トーン",
        ["コミカル", "サイエンス", "ASMR", "ドラマ", "インタビュー"],
        default=["コミカル"],
    )
    
    duration_sec = st.slider("動画尺 (秒)", 30, 300, 300, step=30)
    offer_price = st.text_input("価格訴求 (例: 5,900円)", "5,900円")
    audience_age = st.text_input("ターゲット年齢", "30-50")
    n_variations = st.slider("生成する広告案数", 1, 5, 1)
    temperature = st.slider("Temperature: AIの自由度(大きいほどランダム性が高い)", 0.0, 1.5, 0.9, 0.1)

    generate_btn = st.button("台本を生成")

# ───────────────────────────────────────────────────────────
# 生成処理
# ───────────────────────────────────────────────────────────
def build_request_dict() -> Dict[str, Any]:
    return {
        "product_name": product_name,
        "problem": problem,
        "promise": promise,
        "tone": tones,
        "audience_age": audience_age,
        "duration_sec": duration_sec,
        "offer_price": offer_price,
        "n_variations": n_variations,
        "temperature": temperature,
    }

if generate_btn:
    req = build_request_dict()

    with st.spinner("各 LLM で生成中…"):
        try:
            scripts_openai = generate_ad_openai(req)
        except Exception as e:
            scripts_openai = [f"❌ OpenAI Error: {e}"]

        try:
            scripts_claude = generate_ad_claude(req)
        except Exception as e:
            scripts_claude = [f"❌ Claude Error: {e}"]

        try:
            scripts_gemini = generate_ad_gemini(req)
        except Exception as e:
            scripts_gemini = [f"❌ Gemini Error: {e}"]

    # ───────────────────────────────────────────────────────
    # タブ表示 (CSS でブランドカラーを強調)
    # ───────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab"] span {font-size:1.05rem}
        /* ブランド別色分け */
        .stTabs div[data-baseweb="tab"]:nth-child(1) button {color:#00a67e;}  /* OpenAI 緑 */
        .stTabs div[data-baseweb="tab"]:nth-child(2) button {color:#6f4cff;}  /* Claude 紫 */
        .stTabs div[data-baseweb="tab"]:nth-child(3) button {color:#4285F4;}  /* Gemini 青 */
        </style>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["OpenAI GPT-4o", "Claude 3 Haiku", "Gemini 1.5"])

    for tab, scripts, provider in zip(
        tabs,
        [scripts_openai, scripts_claude, scripts_gemini],
        ["OpenAI", "Claude", "Gemini"],
    ):
        with tab:
            for i, script in enumerate(scripts, 1):
                with st.expander(f"{provider} バリエーション {i}", expanded=(i == 1)):
                    st.markdown(f"```markdown\n{script}\n```")
            st.success(f"{provider} で {len(scripts)} 本生成完了 ✅")
