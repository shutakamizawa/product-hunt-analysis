# Product Hunt Fetcher

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

```

```
