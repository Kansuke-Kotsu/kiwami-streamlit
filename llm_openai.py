import os
from typing import List, Dict, Any

from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────
# OpenAI GPT‑4o / GPT‑4o‑mini – 動画広告台本生成モジュール
# ────────────────────────────────────────────────────────────────────────────

_OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

_client = OpenAI(api_key=_OPENAI_API_KEY)

# デフォルトモデルはコスト効率の良い gpt‑4o‑mini
_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_SYSTEM_PROMPT = (
    "あなたはバズる動画広告台本のプロコピーライターです。"
    "Hook→Problem→Solution→Offer→CTA の 5 段構成を守り、"
    "指定のトーン・尺でテンポの良い日本語台本を生成してください。"
)


def _build_user_prompt(req: Dict[str, Any]) -> str:
    """Claude / Gemini と同フォーマットでユーザープロンプトを構築"""
    return (
        f"商品名: {req['product_name']}\n"
        f"悩み: {req['problem']}\n"
        f"約束: {req['promise']}\n"
        f"トーン候補: {', '.join(req['tone'])}\n"
        f"ターゲット年齢: {req['audience_age']}\n"
        f"尺(秒): {req['duration_sec']}\n"
        f"価格訴求: {req['offer_price']}\n"
        f"バリエーション数: {req['n_variations']}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def generate_ad_openai(req: Dict[str, Any]) -> List[str]:
    """OpenAI で広告台本を n_variations 分一括生成して返す

    Parameters
    ----------
    req : Dict[str, Any]
        app.py から渡されるリクエスト辞書。必須キーは:
        - product_name, problem, promise, tone (List[str]), audience_age,
          duration_sec, offer_price, n_variations, temperature

    Returns
    -------
    List[str]
        生成された台本文字列リスト (長さ = n_variations)
    """

    n_variations: int = int(req.get("n_variations", 1))
    temperature: float = float(req.get("temperature", 0.9))

    user_prompt: str = _build_user_prompt(req)

    completion = _client.chat.completions.create(
        model=_MODEL_NAME,
        temperature=temperature,
        top_p=0.95,
        n=n_variations,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    # OpenAI のレスポンスは choices に message.content が入っている
    return [choice.message.content.strip() for choice in completion.choices]
