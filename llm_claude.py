import os
from typing import List, Dict, Any

import anthropic

# ────────────────────────────────────────────────────────────────────────────
# Anthropic Claude 3.x – 広告台本生成モジュール
# ────────────────────────────────────────────────────────────────────────────

_ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
if not _ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

_client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

# デフォルトモデル（環境変数で上書き可）
_MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest")

_SYSTEM_PROMPT = (
    "あなたはバズる動画広告台本のプロコピーライターです。"
    "Hook→Problem→Solution→Offer→CTA の 5 段構成を守り、"
    "指定のトーン・尺でテンポの良い日本語台本を生成してください。"
)


def _build_user_prompt(req: Dict[str, Any]) -> str:
    """OpenAI / Gemini と同フォーマットでユーザープロンプトを構築"""
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

def _extract_text_from_message(message_obj: anthropic.types.Message) -> str:
    """Message.content は TextBlock のリスト。すべて結合して 1 文字列へ"""

    if isinstance(message_obj.content, str):
        # 将来仕様変更で直接 str が来たときに備える
        return message_obj.content.strip()

    texts: List[str] = []
    for block in message_obj.content:
        # TextBlock 型: `block.text` が本文
        text_part = getattr(block, "text", str(block))
        texts.append(text_part)
    return "".join(texts).strip()


def generate_ad_claude(req: Dict[str, Any]) -> List[str]:
    """Claude で広告台本を n_variations 分生成して返す

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

    user_prompt = _build_user_prompt(req)

    scripts: List[str] = []
    for _ in range(n_variations):
        message = _client.messages.create(
            model=_MODEL_NAME,
            max_tokens=1024,
            temperature=temperature,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        scripts.append(_extract_text_from_message(message))

    return scripts
