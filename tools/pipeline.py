
"""
增量同步 pipeline：
- 从 analytics.material_dim 抽取满足 tt_valid=true 且 media_type=2 的素材
- 若 embeddings 中不存在对应 gdt_material_id，则补充插入
- 生成 cover_caption / file_caption / ad_caption（参考 qwen.py）
- 生成 embedding（参考 nvidia_clip.py），存入 embedding 列
"""

from __future__ import annotations

import argparse
import base64
import os
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 环境变量
HF_TOKEN = os.getenv("HF_TOKEN")
MS_TOKEN = os.getenv("MS_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
DSN = os.getenv("DATABASE_CONNECTION_STRING")
OR_API_KEY = os.getenv("OR_API_KEY")


# ---------- LLM & Embedding ----------
def _openai_client() -> OpenAI:
    if not MS_TOKEN:
        raise RuntimeError("MS_TOKEN 未设置，无法调用 Qwen")
    return OpenAI(base_url="https://api-inference.modelscope.cn/v1", api_key=MS_TOKEN)


def describe_image(url: str, prompt: str) -> str:
    client = _openai_client()
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()


def gen_ad_caption(file_caption: str, cover_caption: str) -> str:
    prompt = f"""你是一位资深的广告文案策划师，擅长为游戏和应用创作吸引人的广告文案。

【视频内容描述】
{file_caption}

【封面图片描述】
{cover_caption}

【要求】
1. 文案要简洁有力，突出核心卖点
2. 要有号召力，能激发用户下载/体验的欲望
3. 可以包含：核心玩法亮点、视觉特色、情感共鸣点
4. 长度控制在50字以内
5. 不要使用markdown格式，直接输出文案内容
"""
    client = _openai_client()
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()


def describe_video(url: str, prompt: str) -> str:
    """调用 Qwen3-VL 的 video_url 能力对视频做描述（参考 z-rag/qwen.py）"""
    client = _openai_client()
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": url}},
                ],
            }
        ],
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()



def get_text_embedding(text: str) -> list[float]:
    api_key = NVIDIA_API_KEY
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY 未设置")
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input": [text], "model": "nvidia/nvclip", "encoding_format": "float"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

def get_image_embedding(image_url: str, api_key: str = None) -> list:
    """获取图片的 embedding"""
    api_key = os.environ.get("NVIDIA_API_KEY")

    # 读取图片并转换为 base64
    # with open(image_path, "rb") as f:
    #     image_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.get(image_url, timeout=15)
    resp.raise_for_status()

    image_b64 = base64.b64encode(resp.content).decode("utf-8")

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "input": [f"data:image/jpeg;base64,{image_b64}"],
            "model": "nvidia/nvclip",
            "encoding_format": "float",
        },
        timeout=60,
    )

    return response.json()["data"][0]["embedding"]




# ---------- DB ----------
SELECT_MISSING_SQL = """
SELECT material_id AS gdt_material_id,
       asset_id,
       app_id,
       name,
       file_url,
       cover_url,
       media_type,
       oceanengine_material_id AS tt_material_id,
       tt_valid
FROM analytics.material_dim md
WHERE md.tt_valid = true
  AND md.media_type = 2
  AND NOT EXISTS (
    SELECT 1 FROM material.embeddings eb
    WHERE eb.gdt_material_id = md.material_id
  )
ORDER BY md.material_id DESC
"""

SELECT_MISSING_SQL_WITH_LIMIT = SELECT_MISSING_SQL + "\nLIMIT %(limit)s"


INSERT_SQL = """
INSERT INTO material.embeddings (
  gdt_material_id,
  tt_material_id,
  embedding_model,
  dimensions,
  embedding,
  app_id,
  asset_id,
  name,
  media_type,
  file_url,
  cover_url,
  ad_caption,
  file_caption,
  cover_caption
) VALUES (
  %(gdt_material_id)s,
  %(tt_material_id)s,
  %(embedding_model)s,
  %(dimensions)s,
  %(embedding)s,
  %(app_id)s,
  %(asset_id)s,
  %(name)s,
  %(media_type)s,
  %(file_url)s,
  %(cover_url)s,
  %(ad_caption)s,
  %(file_caption)s,
  %(cover_caption)s
)
ON CONFLICT DO NOTHING;
"""


def fetch_missing(conn, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须大于 0")

    sql = SELECT_MISSING_SQL_WITH_LIMIT if limit else SELECT_MISSING_SQL
    params = {"limit": limit} if limit else None
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET statement_timeout = '60s';")
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        return cur.fetchall()


def insert_rows(conn, rows: list[dict[str, Any]]):
    try:
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
            inserted = cur.rowcount
        conn.commit()
        print(f"insert successful, affected={inserted}")
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        print(f"error: {exc}")



# ---------- Pipeline ----------
def process_row(row: dict[str, Any]) -> dict[str, Any]:
    file_url = row.get("file_url")
    cover_url = row.get("cover_url")
    gdt_material_id = row["gdt_material_id"]
    tt_material_id = row.get("tt_material_id")
    # if not tt_material_id:
    #     raise ValueError(f"tt_material_id 为空，gdt_material_id={gdt_material_id}")
    print(f"  -> gdt_material_id: {gdt_material_id}, tt_material_id: {tt_material_id}")

    cover_prompt = "请用中文详细描述这张图片的内容，包括人物、场景、文字和视觉元素。不要用markdown格式。"
    file_prompt = (
        "请用中文详细描述这个视频的内容，包括主要场景、出现的人物或物体、发生的动作或事件、画面中的文字信息。"
        "用简洁段落描述，不要使用markdown格式。"
    )

    cover_caption = describe_image(cover_url, cover_prompt)
    print(f"  -> cover_caption: {cover_caption}")
    file_caption = describe_video(file_url, file_prompt)
    print(f"  -> file_caption: {file_caption}")
    ad_caption = gen_ad_caption(file_caption, cover_caption)
    print(f"  -> ad_caption: {ad_caption}")


    ad_caption_embedding = get_text_embedding(ad_caption)
    cover_caption_embedding = get_text_embedding(cover_caption)
    file_caption_embedding = get_text_embedding(file_caption)
    cover_embedding = get_image_embedding(cover_url)

    print("  -> generated cover_embeddings:", cover_embedding[:5], "...")

    embedding = (
        np.array(ad_caption_embedding, dtype=np.float32) * 0.25
        + np.array(cover_caption_embedding, dtype=np.float32) * 0.25
        + np.array(file_caption_embedding, dtype=np.float32) * 0.25
        + np.array(cover_embedding, dtype=np.float32) * 0.25
    )

    print("  -> combined embedding:", embedding[:5], "...")

    # psycopg2 不接受 ndarray，存库前转成普通 list
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    return {
        "gdt_material_id": gdt_material_id,
        "file_url": file_url,
        "cover_url": cover_url,
        "tt_material_id": tt_material_id,
        "cover_caption": cover_caption,
        "app_id": row.get("app_id"),
        "name": row.get("name"),
        "file_caption": file_caption,
        "ad_caption": ad_caption,
        "embedding": embedding,
        "embedding_model": "nvidia/nvclip",
        "dimensions": len(embedding),
        "media_type": row.get("media_type"),
        "asset_id": row.get("asset_id"),
    }


def run_pipeline(limit: int | None = None) -> dict[str, Any]:
    if not DSN:
        raise RuntimeError("DATABASE_CONNECTION_STRING not set")

    print("-> connecting db ...")
    summary: dict[str, Any] = {
        "requested_limit": limit,
        "total_candidates": 0,
        "inserted": 0,
        "errors": [],
    }

    with psycopg2.connect(DSN, connect_timeout=100) as conn:
        print("-> fetching missing rows ...")
        rows = fetch_missing(conn, limit=limit)
        summary["total_candidates"] = len(rows)
        print(f"-> need insert: {len(rows)} rows")

        for i, r in enumerate(rows, start=1):
            try:
                print(f"  [{i}/{len(rows)}] gdt_material_id={r['gdt_material_id']}")
                processed_row = process_row(r)
                insert_rows(conn, [processed_row])
                summary["inserted"] += 1
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                summary["errors"].append(
                    {"gdt_material_id": r.get("gdt_material_id"), "error": error_msg}
                )
                print(f"  !! skip {r.get('gdt_material_id')}: {error_msg}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Sync missing materials from analytics.material_dim into material.embeddings"
    )
    parser.add_argument("--limit", type=int, default=None, help="最多处理行数（可选）")
    args = parser.parse_args()

    summary = run_pipeline(limit=args.limit)
    print("-> pipeline finished", summary)


if __name__ == "__main__":
    main()
