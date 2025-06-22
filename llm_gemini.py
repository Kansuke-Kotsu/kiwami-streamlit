import os
from typing import List, Dict, Any

import google.generativeai as genai
import tiktoken
# "prompt_utils.pyからテンプレートをインポート"
from prompt_utils import build_prompt

# ────────────────────────────────────────────────────────────────────────────
# Google Gemini 1.5 Flash / Pro – 広告台本生成モジュール
# ────────────────────────────────────────────────────────────────────────────

_GEMINI_API_KEY: str | None = os.getenv("GOOGLE_GEMINI_API_KEY")
if not _GEMINI_API_KEY:
    raise EnvironmentError("GOOGLE_GEMINI_API_KEY is not set.")

genai.configure(api_key=_GEMINI_API_KEY)

# 利用モデル（必要に応じて flash / pro を切り替え）
_MODEL_NAME = "gemini-1.5-flash-latest"

def _count_tokens(text: str, model: str) -> int:
    """指定モデルでのトークン数をカウント"""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# ---------------------------------------------------------------------------
# Public API – generate_ad_gemini
# ---------------------------------------------------------------------------

def generate_ad_gemini(req: Dict[str, Any]) -> List[str]:
    """Gemini で広告台本を n_variations 生成し、Markdown 文字列を返す。"""

    model = genai.GenerativeModel(_MODEL_NAME)
    
    duration_sec = int(req.get("duration_sec", 30))
    min_tokens = int(duration_sec * 8)  # 1秒あたり8文字を目安に計算

    # プロンプト組み立て
    prompt_text = build_prompt(req)

    scripts: List[str] = []
    for _ in range(req.get("n_variations", 1)):
        try:
            response = model.generate_content(
                [prompt_text],
                generation_config={
                    "temperature": req.get("temperature", 0.9),
                    "top_p": 0.95,
                    "max_output_tokens": 4096,
                },
            )
            
            text = response.text.strip()
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
                refine_resp = model.generate_content(
                    [refine_prompt],
                    generation_config={
                        "temperature": req.get("temperature", 0.9),
                        "top_p": 0.95,
                        "max_output_tokens": 4096,
                    },
                )
                scripts.append(refine_resp.text.strip())
            else:
                scripts.append(text)

        except Exception as e:
            scripts.append(f"❌ Gemini Error: {e}")

    return scripts
