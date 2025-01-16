import time
import requests
import pandas as pd
import logging
from dotenv import load_dotenv
import os
from typing import List, Dict

# ----------------------------
# Load Environment Variables
# 環境変数の読み込み
# ----------------------------
load_dotenv()  # Load environment variables from .env file

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
# Configure Logging
# ロギングの設定
# ----------------------------
logger = logging.getLogger("ProductHuntFetcher")
logger.setLevel(logging.DEBUG)  # Capture all levels of logs

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Display INFO and above on console
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Add handler to logger
logger.addHandler(console_handler)

# ----------------------------
# Define GraphQL Query
# GraphQLクエリの定義
# ----------------------------
QUERY = """
query GetPosts($first: Int, $after: String) {
  posts(
    first: $first
    after: $after
    order: NEWEST  # Changed ORDER from FEATURED_AT to NEWEST
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
    Fetch a bulk of posts from Product Hunt.

    Parameters
    ----------
    total_posts : int, optional
        The maximum number of posts to retrieve (default: 1000)
    batch_size : int, optional
        The number of posts to retrieve per API request (default: 50)
    sleep_time : int, optional
        The number of seconds to sleep between each batch to avoid rate limiting (default: 4)

    Returns
    -------
    List[Dict]
        A list of dictionaries containing the fetched post data.

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
                timeout=30,  # Set timeout
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e} リクエストが失敗しました: {e}")
            logger.info(
                f"Sleeping for {sleep_time} seconds before retrying... {sleep_time}秒待機して再試行します..."
            )
            time.sleep(sleep_time)
            continue

        if response.status_code == 200:
            data = response.json()

            # Check for errors
            if "errors" in data:
                logger.error(
                    f"GraphQL errors: {data['errors']} GraphQLエラー: {data['errors']}"
                )
                break

            edges = data["data"]["posts"]["edges"]
            page_info = data["data"]["posts"]["pageInfo"]

            # Add fetched posts
            for edge in edges:
                post = edge["node"]

                # Process topics
                topics = [
                    topic_edge["node"]
                    for topic_edge in post.get("topics", {}).get("edges", [])
                ]
                category = topics[0]["name"] if len(topics) > 0 else None
                subcategory = topics[1]["name"] if len(topics) > 1 else None

                # Count media (number of images)
                media = post.get("media", [])
                image_count = sum(1 for m in media if m["type"] == "image")

                # Add new fields
                post["category"] = category
                post["subcategory"] = subcategory
                post["imageCount"] = image_count

                # Add 'featured' field
                post["featured"] = post.get("featuredAt") is not None

                # Add 'hasVideo' field
                post["hasVideo"] = any(m.get("type") == "video" for m in media)

                # Remove unnecessary fields (optional)
                post.pop("topics", None)
                post.pop("media", None)
                # post.pop("featuredAt", None)  # Optional: Remove 'featuredAt' if not needed

                all_posts.append(post)

                if len(all_posts) >= total_posts:
                    break

            logger.info(
                f"Fetched {len(edges)} posts. Total fetched: {len(all_posts)} 件取得しました。合計取得数: {len(all_posts)} 件."
            )

            # Update pagination info
            after_cursor = page_info["endCursor"]
            has_next_page = page_info["hasNextPage"]

            if not has_next_page:
                logger.info("No more pages to fetch. これ以上のページはありません。")
                break

        elif response.status_code == 429:
            # Rate limit reached
            reset_time = int(response.headers.get("X-Rate-Limit-Reset", sleep_time))
            logger.warning(
                f"Rate limit reached. Sleeping for {reset_time} seconds... レートリミットに達しました。{reset_time}秒間待機します..."
            )
            time.sleep(reset_time)
            continue

        else:
            logger.error(
                f"Query failed with status code: {response.status_code} クエリがステータスコード {response.status_code} で失敗しました。"
            )
            logger.debug(
                f"Response content: {response.text} レスポンス内容: {response.text}"
            )
            break

        # Check remaining rate limit
        remaining = int(response.headers.get("X-Rate-Limit-Remaining", 0))
        logger.debug(
            f"Rate limit remaining: {remaining} レートリミット残量: {remaining}"
        )

        if remaining < 1:
            reset_time = int(response.headers.get("X-Rate-Limit-Reset", sleep_time))
            logger.warning(
                f"Rate limit reached. Sleeping for {reset_time} seconds... レートリミットに達しました。{reset_time}秒間待機します..."
            )
            time.sleep(reset_time)
        else:
            logger.debug(
                f"Sleeping for {sleep_time} seconds before next request... 次のリクエストまで{sleep_time}秒間待機します..."
            )
            time.sleep(sleep_time)

    logger.info(
        f"Finished fetching. Total posts fetched: {len(all_posts)} 件取得完了。"
    )
    return all_posts


if __name__ == "__main__":
    # Set fetching parameters
    TOTAL_POSTS = 1000  # Maximum number of posts to retrieve
    BATCH_SIZE = 50  # Number of posts to retrieve per API request
    SLEEP_TIME = 4  # Number of seconds to wait between each request

    logger.info(
        "Starting to fetch posts from Product Hunt... Product Huntからの投稿取得を開始します。"
    )
    posts = get_posts(
        total_posts=TOTAL_POSTS, batch_size=BATCH_SIZE, sleep_time=SLEEP_TIME
    )

    # Convert to Pandas DataFrame
    if posts:
        df = pd.json_normalize(posts)

        # Log output
        logger.info(f"Total posts fetched: {len(df)} 件取得した投稿数: {len(df)}")
        logger.debug(f"DataFrame head:\n{df.head()}")

        # Select necessary columns (id, name, tagline, featuredAt, featured, hasVideo, createdAt, votesCount, commentsCount, category, subcategory, imageCount)
        selected_columns = [
            "id",
            "name",
            "tagline",
            "featuredAt",
            "featured",
            "hasVideo",
            "createdAt",
            "votesCount",
            "commentsCount",
            "category",
            "subcategory",
            "imageCount",
        ]

        df_selected = df[selected_columns]

        # Save to CSV file
        csv_filename = "posts.csv"
        df_selected.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        logger.info(
            f"Saved posts to {csv_filename}. '{csv_filename}' に投稿を保存しました。"
        )
    else:
        logger.warning("No posts were fetched. 投稿が取得されませんでした。")
