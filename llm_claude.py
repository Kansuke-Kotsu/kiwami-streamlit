import os
from typing import List, Dict, Any

import anthropic
import tiktoken
# "prompt_utils.pyからテンプレートをインポート"
from prompt_utils import build_prompt

# ────────────────────────────────────────────────────────────────────────────
# Anthropic Claude 3.x – 広告台本生成モジュール
# ────────────────────────────────────────────────────────────────────────────

_ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
if not _ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

_client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

# デフォルトモデル（環境変数で上書き可）
_MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest")

# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def _count_tokens(text: str, model: str) -> int:
    """指定モデルでのトークン数をカウント"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

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
    duration_sec = int(req.get("duration_sec", 30))
    min_tokens = int(duration_sec * 8)  # 1秒あたり8文字を目安に計算

    # プロンプト組み立て
    prompt_text = build_prompt(req)

    scripts: List[str] = []
    for _ in range(n_variations):
        message = _client.messages.create(
            model=_MODEL_NAME,
            max_tokens=4096,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt_text},
            ],
        )
        
        text = _extract_text_from_message(message)
        # 生成された台本のトークン数をカウント
        tokens = _count_tokens(text, _MODEL_NAME)

        if tokens < min_tokens:
            refine_instructions = (
                f"あなたは動画広告のプロコピーライターです。"
                f"以下の台本を添削して、より再生回数が増えるような台本にしてください。"
                f"また、{min_tokens}文字以上になるように、内容を増やしてください。"
                f"出力は、添削後の台本のみを返してください。\n\n"
                f"１分ごとに改行してください。\n\n"
            )
            refine_prompt = f"{refine_instructions}\n\n{text}"  # 元文 + 指示
            refine_message = _client.messages.create(
                model=_MODEL_NAME,
                max_tokens=4096,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": refine_prompt},
                ],
            )
            scripts.append(_extract_text_from_message(refine_message))

    return scripts
