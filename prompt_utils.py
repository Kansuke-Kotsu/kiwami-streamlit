# app/prompt_utils.py
BASE_PROMPT = """
あなたは動画広告のプロコピーライターです。
以下の条件で、{n_variations} 本の台本を日本語で生成してください。
{char_count}文字程度となるように、構成を考えてください。

【必ず守る書式】
{format_rule}

【出力例】
{example}

【背景情報】
商品名: {product_name}
悩み: {problem}
ベネフィット: {promise}
ターゲット年齢: {audience_age}
尺: {duration_sec} 秒
総文字数: {char_count} 文字
トーン: {tone}

台本を開始してください。
"""

_RULE = {
    "serif_only":            "- 1 行に 1 セリフのみを書く。行頭にタイムコードも演出注釈も書かない。",
    "with_time":             "- 行頭に (0-5s) のような秒数レンジを付け、その後にセリフを書く。",
    "with_time_and_dir":     "- 行頭に (0-5s) を付け、セリフの後に *(演出)* を括弧付きで書く。",
}
_EX = {
    "serif_only":            "こんにちは！寝返りうててますか？",
    "with_time":             "(0-5s) こんにちは！寝返りうててますか？",
    "with_time_and_dir":     "(0-5s) こんにちは！ *(アップで手を振る)*",
}

def build_prompt(req: dict) -> str:
    
    with_timing = "serif_only"
    with_direction = "serif_only"
    
    """req は views で組み立てた dict"""
    key = ("with_time_and_dir" if req["with_timing"] and req["with_direction"]
           else "with_time"    if req["with_timing"]
           else "serif_only")
    # format() に渡す前に重複キーを削除
    safe_req = {k: v for k, v in req.items() if k not in {"n_variations"}}

    return BASE_PROMPT.format(
        n_variations=req["n_variations"],
        format_rule=_RULE[key],
        example=_EX[key],
        char_count=calc_char_count(req["duration_sec"]),
        **safe_req           # ← 重複のない dict
    )

def calc_char_count(duration_sec: int) -> int:
    """動画尺からおおよその文字数を計算"""
    # 1秒あたりの文字数は 10-15 文字程度が目安
    return int(duration_sec * 8)  # 8文字/秒で計算