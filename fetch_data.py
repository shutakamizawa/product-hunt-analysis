import time
import requests
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from typing import List, Dict

# ----------------------------
# 環境変数の読み込み
# ----------------------------
load_dotenv()  # .env ファイルから環境変数をロード

API_KEY = os.getenv("PRODUCT_HUNT_API_KEY")
if not API_KEY:
    raise ValueError("PRODUCT_HUNT_API_KEY is not set in the environment variables.")

API_URL = "https://api.producthunt.com/v2/api/graphql"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# ----------------------------
# ロギングの設定
# ----------------------------
logger = logging.getLogger("ProductHuntFetcher")
logger.setLevel(logging.DEBUG)  # 全てのレベルのログをキャプチャ

# コンソールハンドラ
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # コンソールにはINFO以上を表示
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# ハンドラをロガーに追加
logger.addHandler(console_handler)

# ----------------------------
# GraphQLクエリの定義
# ----------------------------
QUERY = """
query GetPosts($first: Int, $after: String) {
  posts(
    first: $first
    after: $after
    order: NEWEST  # ORDERをFEATURED_ATからNEWESTに変更
  ) {
    pageInfo {
      endCursor
      hasNextPage
    }
    edges {
      node {
        id
        name
        tagline
        featuredAt
        createdAt
        votesCount
        commentsCount
        media {
          type
          url
        }
        topics(first: 5) {
          edges {
            node {
              id
              name
              slug
            }
          }
        }
      }
    }
  }
}
"""


def get_posts(
    total_posts: int = 1000, batch_size: int = 50, sleep_time: int = 4
) -> List[Dict]:
    """
    Product Huntの投稿データを一括取得する関数

    Parameters
    ----------
    total_posts : int, optional
        取得したい投稿数の上限 (デフォルト: 1000)
    batch_size : int, optional
        1回のAPIリクエストで取得する投稿数 (デフォルト: 50)
    sleep_time : int, optional
        レートリミット回避のため、各バッチ取得の間に入れるスリープ秒数 (デフォルト: 4)

    Returns
    -------
    List[Dict]
        取得した投稿データを格納した辞書のリスト
    """
    all_posts = []
    has_next_page = True
    after_cursor = None

    while has_next_page and len(all_posts) < total_posts:
        variables = {
            "first": batch_size,
            "after": after_cursor,
        }

        try:
            response = requests.post(
                API_URL,
                json={"query": QUERY, "variables": variables},
                headers=HEADERS,
                timeout=30,  # タイムアウト設定
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            logger.info(f"Sleeping for {sleep_time} seconds before retrying...")
            time.sleep(sleep_time)
            continue

        if response.status_code == 200:
            data = response.json()

            # エラーチェック
            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                break

            edges = data["data"]["posts"]["edges"]
            page_info = data["data"]["posts"]["pageInfo"]

            # 取得したPostを追加
            for edge in edges:
                post = edge["node"]

                # topicsの処理
                topics = [
                    topic_edge["node"]
                    for topic_edge in post.get("topics", {}).get("edges", [])
                ]
                category = topics[0]["name"] if len(topics) > 0 else None
                subcategory = topics[1]["name"] if len(topics) > 1 else None

                # mediaのカウント（画像の数）
                media = post.get("media", [])
                image_count = sum(1 for m in media if m["type"] == "image")

                # 新しいフィールドを追加
                post["category"] = category
                post["subcategory"] = subcategory
                post["imageCount"] = image_count

                # featured フィールドを追加
                post["featured"] = post.get("featuredAt") is not None

                # hasVideo フィールドを追加
                post["hasVideo"] = any(m.get("type") == "video" for m in media)

                # 不要なフィールドを削除（必要に応じて）
                post.pop("topics", None)
                post.pop("media", None)
                # post.pop("featuredAt", None)  # オプション: featuredAt を削除する場合

                all_posts.append(post)

                if len(all_posts) >= total_posts:
                    break

            logger.info(f"Fetched {len(edges)} posts. Total fetched: {len(all_posts)}.")

            # ページネーション情報の更新
            after_cursor = page_info["endCursor"]
            has_next_page = page_info["hasNextPage"]

            if not has_next_page:
                logger.info("No more pages to fetch.")
                break

        elif response.status_code == 429:
            # レートリミットに達した場合
            reset_time = int(response.headers.get("X-Rate-Limit-Reset", sleep_time))
            logger.warning(f"Rate limit reached. Sleeping for {reset_time} seconds...")
            time.sleep(reset_time)
            continue

        else:
            logger.error(f"Query failed with status code: {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            break

        # レートリミット残量の確認
        remaining = int(response.headers.get("X-Rate-Limit-Remaining", 0))
        logger.debug(f"Rate limit remaining: {remaining}")

        if remaining < 1:
            reset_time = int(response.headers.get("X-Rate-Limit-Reset", sleep_time))
            logger.warning(f"Rate limit reached. Sleeping for {reset_time} seconds...")
            time.sleep(reset_time)
        else:
            logger.debug(f"Sleeping for {sleep_time} seconds before next request...")
            time.sleep(sleep_time)

    logger.info(f"Finished fetching. Total posts fetched: {len(all_posts)}.")
    return all_posts


if __name__ == "__main__":
    # 取得パラメータの設定
    TOTAL_POSTS = 1000  # 取得したい投稿数の上限
    BATCH_SIZE = 50  # 1回のAPIリクエストで取得する投稿数
    SLEEP_TIME = 4  # 各リクエスト間の待機秒数

    logger.info("Starting to fetch posts from Product Hunt...")
    posts = get_posts(
        total_posts=TOTAL_POSTS, batch_size=BATCH_SIZE, sleep_time=SLEEP_TIME
    )

    # Pandas DataFrame に変換
    if posts:
        df = pd.json_normalize(posts)

        # ログ出力
        logger.info(f"Total posts fetched: {len(df)}")
        logger.debug(f"DataFrame head:\n{df.head()}")

        # 必要な列のみ選択（id, name, tagline, featuredAt, featured, hasVideo, createdAt, votesCount, commentsCount, category, subcategory, imageCount）
        selected_columns = [
            "id",
            "name",
            "tagline",
            "featuredAt",
            "featured",  # 追加
            "hasVideo",  # 追加
            "createdAt",
            "votesCount",
            "commentsCount",
            "category",
            "subcategory",
            "imageCount",
        ]

        df_selected = df[selected_columns]

        # CSVファイルに保存
        csv_filename = "posts.csv"
        df_selected.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        logger.info(f"Saved posts to {csv_filename}.")
    else:
        logger.warning("No posts were fetched.")
