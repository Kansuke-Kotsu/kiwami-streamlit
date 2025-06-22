import os
from typing import List, Dict, Any

from openai import OpenAI
import tiktoken

# "prompt_utils.pyからテンプレートをインポート"
from prompt_utils import build_prompt

# ────────────────────────────────────────────────────────────────────────────
# OpenAI GPT‑4o / GPT‑4o‑mini – 動画広告台本生成モジュール
# ────────────────────────────────────────────────────────────────────────────

_OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

_client = OpenAI(api_key=_OPENAI_API_KEY)

# デフォルトモデルはコスト効率の良い gpt‑4o‑mini
_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _count_tokens(text: str, model: str) -> int:
    """指定モデルでのトークン数をカウント"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def generate_ad_openai(req: Dict[str, Any]) -> List[str]:
    """OpenAI で広告台本を n_variations 分生成して返す"""
    n_variations = int(req.get("n_variations", 1))
    temperature = float(req.get("temperature", 0.9))
    duration_sec = int(req.get("duration_sec", 30))
    min_tokens = int(duration_sec * 8)  # 1秒あたり8文字を目安に計算

    # プロンプト組み立て
    prompt_text = build_prompt(req)

    initial_resp = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        top_p=0.95,
        n=n_variations,
        max_tokens=4096,
    )
    
    # 2. 各バリエーションをチェック & ブラッシュアップ
    for choice in initial_resp.choices:
        text = choice.message.content.strip()
        tokens = _count_tokens(text, _MODEL_NAME)
        # 規定に満たない場合、文章量を増やすようにリファイン
        if tokens < min_tokens:
            refine_instructions = (
                f"あなたは動画広告のプロコピーライターです。"
                f"以下の台本を添削して、より再生回数が増えるような台本にしてください。"
                f"また、{min_tokens}文字以上になるように、内容を増やしてください。"
                f"出力は、添削後の台本のみを返してください。\n\n"
                f"１分ごとに改行してください。\n\n"
            )
            refine_prompt = f"{refine_instructions}\n\n{text}"  # 元文 + 指示
            refine_resp = _client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=temperature,
                top_p=0.95,
                max_tokens=4096,
            )

    return [choice.message.content.strip() for choice in refine_resp.choices]
