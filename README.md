# Product Hunt Fetcher

(日本語は英語の後に続きます)

## Overview

Product Hunt Fetcher is a Python script that utilizes the Product Hunt API to retrieve the latest post data and save it as a CSV file. For detailed information on the available data, refer to the [GraphQL Schema](https://github.com/producthunt/producthunt-api/blob/master/schema.graphql).

## Setup

### Prerequisites

- Python 3.7 or higher
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/product-hunt-fetcher.git
   cd product-hunt-fetcher
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a .env file and set your Product Hunt API key**
   ```bash
   PRODUCT_HUNT_API_KEY=your_api_key_here
   ```

## Usage

### Fetching Data

Run the script to fetch data from Product Hunt and save it to `posts.csv`.

```bash
python fetch_data.py
```

### Data Analysis

Use `analyze_data.py` to perform analysis on the retrieved data.

```bash
python analyze_data.py
```

#### Analysis Features

1. **Basic Statistics**

   - Analysis of the overall ratio of featured posts
   - Analysis of featured ratios by category

2. **Media Analysis**

   - Correlation between the number of images and featured status
   - Relationship between the presence of videos and featured status

3. **Category Analysis**

   - Detailed analysis of the top 10 categories
   - Feature analysis by category

4. **Time Series Analysis**

   - Analysis of the number of posts and featured rates by day of the week

#### Statistical Methods

- Chi-Square Test
- t-Test
- Mann-Whitney U Test
- Logistic Regression Analysis
- Odds Ratio Analysis

#### Visualization

- Bar Charts
- Box Plots
- Histograms
- Stacked Bar Charts
- Dual Axis Plots

### Configuring Parameters

Within `fetch_data.py`, you can configure the following parameters:

- `TOTAL_POSTS`: Maximum number of posts to retrieve (default: 1000)
- `BATCH_SIZE`: Number of posts to retrieve per API request (default: 50)
- `SLEEP_TIME`: Waiting time in seconds between each request (default: 4)

The script can fetch up to 1,260 posts per run and will wait for 10 minutes (600 seconds) if the rate limit is reached before continuing. Modify the relevant sections in the script as needed.

## 概要

Product Hunt の API を使用して最新の投稿データを取得し、CSV ファイルとして保存する Python スクリプトです。取得できるデータの詳細については、[GraphQL スキーマ](https://github.com/producthunt/producthunt-api/blob/master/schema.graphql)を参照してください。

## セットアップ

### 必要条件

- Python 3.7 以上
- pip

### インストール

1. リポジトリをクローンします。

   ```bash
   git clone https://github.com/yourusername/product-hunt-fetcher.git
   cd product-hunt-fetcher
   ```

2. 仮想環境を作成し、アクティベートします。

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windowsの場合は `venv\Scripts\activate`
   ```

3. 依存関係をインストールします。

   ```bash
   pip install -r requirements.txt
   ```

4. `.env` ファイルを作成し、Product Hunt API キーを設定します。

   ```env
   PRODUCT_HUNT_API_KEY=your_api_key_here
   ```

## 使い方

### データ取得

スクリプトを実行して、Product Hunt からデータを取得し、`posts.csv` に保存します。

```bash
python fetch_data.py
```

### データ分析

取得したデータの分析は `analyze_data.py` を使用して行います。

```bash
python analyze_data.py
```

#### 分析機能

1. **基本統計**

   - 全体の Featured 記事の比率分析
   - カテゴリー別の Featured 率分析

2. **メディア分析**

   - 画像数と Featured 状態の関連性分析
   - 動画の有無と Featured 状態の関係分析

3. **カテゴリー分析**

   - トップ 10 カテゴリーの詳細分析
   - カテゴリー別の特徴分析

4. **時系列分析**
   - 曜日別の投稿数と Featured 率の分析

#### 統計手法

- カイ二乗検定
- t 検定
- Mann-Whitney U 検定
- ロジスティック回帰分析
- オッズ比分析

#### 可視化

- 棒グラフ
- 箱ひげ図
- ヒストグラム
- 積み上げ棒グラフ
- デュアルアクシスプロット

### パラメータの設定

`fetch_data.py` 内で以下のパラメータを設定できます：

- `TOTAL_POSTS`: 取得したい投稿数の上限（デフォルト: 1000）
- `BATCH_SIZE`: 1 回の API リクエストで取得する投稿数（デフォルト: 50）
- `SLEEP_TIME`: 各リクエスト間の待機秒数（デフォルト: 4）

スクリプトは一回あたり最大 1260 件のデータを取得し、レートリミットに達した場合は 10 分（600 秒）待機してから再開します。必要に応じて、スクリプト内の該当部分を編集してください。

## ロギング

スクリプトは `logging` モジュールを使用してログを出力します。コンソールには INFO 以上のレベルのログが表示され、詳細なデバッグログは DEBUG レベルで記録されます。

## エラーハンドリング

- API リクエストの失敗時やレートリミットに達した場合、スクリプトは指定されたスリープ時間後にリトライします。
- 環境変数 `PRODUCT_HUNT_API_KEY` が設定されていない場合、スクリプトはエラーを投げて終了します。
