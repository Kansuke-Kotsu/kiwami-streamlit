import os
from typing import List, Dict, Any

import google.generativeai as genai

# ────────────────────────────────────────────────────────────────────────────
# Google Gemini 1.5 Flash / Pro – 広告台本生成モジュール
# ────────────────────────────────────────────────────────────────────────────

_GEMINI_API_KEY: str | None = os.getenv("GOOGLE_GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    raise EnvironmentError("GOOGLE_GEMINI_API_KEY is not set.")

genai.configure(api_key=_GEMINI_API_KEY)

# 利用モデル（必要に応じて flash / pro を切り替え）
_MODEL_NAME = "gemini-1.5-flash-latest"

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "あなたはバズる動画広告台本のプロコピーライターです。\n"
    "TikTok や YouTube Shorts で最後まで視聴され、\n"
    "購買意欲を高める構成 (Hook→Problem→Solution→Offer→CTA) を厳守し、\n"
    "日本語でテンポ良く、各行 1 セリフで書いてください。\n"
)

def _build_user_prompt(req: Dict[str, Any]) -> str:
    """ユーザー入力を 1 つのテキストプロンプトへ整形"""

    tones = ", ".join(req.get("tone", [])) or "コミカル"

    prompt = (
        f"商品名: {req['product_name']}\n"
        f"悩み: {req['problem']}\n"
        f"約束: {req['promise']}\n"
        f"トーン候補: {tones}\n"
        f"ターゲット年齢: {req['audience_age']}\n"
        f"尺(秒): {req['duration_sec']}\n"
        f"価格訴求: {req['offer_price']}\n"
        f"バリエーション数: {req['n_variations']}\n"
    )
    return prompt

# ---------------------------------------------------------------------------
# Public API – generate_ad_gemini
# ---------------------------------------------------------------------------

def generate_ad_gemini(req: Dict[str, Any]) -> List[str]:
    """Gemini で広告台本を n_variations 生成し、Markdown 文字列を返す。"""

    model = genai.GenerativeModel(_MODEL_NAME)

    user_prompt = _build_user_prompt(req)

    scripts: List[str] = []
    for _ in range(req.get("n_variations", 1)):
        try:
            response = model.generate_content(
                [_SYSTEM_PROMPT, "\n", user_prompt],
                generation_config={
                    "temperature": req.get("temperature", 0.9),
                    "top_p": 0.95,
                    "max_output_tokens": 2048,
                },
            )
            scripts.append(response.text.strip())
        except Exception as e:
            scripts.append(f"❌ Gemini Error: {e}")

    return scripts
