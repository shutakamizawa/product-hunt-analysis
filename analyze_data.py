import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import Table2x2


def load_data(csv_file_path):
    """
    CSVファイルを読み込み、DataFrameを返す。
    """
    try:
        df = pd.read_csv(csv_file_path)
        print("データの読み込みに成功しました。")
        return df
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_file_path}' が見つかりません。")
        exit(1)
    except pd.errors.ParserError:
        print(f"エラー: ファイル '{csv_file_path}' の解析中にエラーが発生しました。")
        exit(1)


def preprocess_data(df):
    """
    データの前処理:
    - 'featured' と 'hasVideo' をブール型に変換。
    - 'createdAt', 'featuredAt' を日時型に変換(存在する場合)。
    """
    # featured, hasVideo を bool へ
    df["featured"] = df["featured"].astype(bool)
    df["hasVideo"] = df["hasVideo"].astype(bool)

    # createdAt, featuredAt を datetime へ
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    if "featuredAt" in df.columns:
        df["featuredAt"] = pd.to_datetime(df["featuredAt"], errors="coerce")

    return df


def analyze_feature_ratio(df):
    """
    1. 全体の Featured 比率
    """
    total_items = len(df)
    featured_items = df["featured"].sum()
    print("\n--- Overall Featured Ratio ---")
    print(f"Total items: {total_items}")
    print(
        f"Featured items: {featured_items} ({featured_items / total_items * 100:.2f}%)"
    )


def analyze_top10_categories_feature_ratio(df):
    """
    Analyze and visualize the Featured ratio for the top 10 popular categories.
    Categories are sorted by the total number of posts in descending order.
    Additionally, displays the total number of posts per category on the plot
    and sets the y-axis limit to 20%.
    """
    # 記事数が多いトップ10カテゴリを取得
    category_counts = df["category"].value_counts().nlargest(10)
    top10_categories = category_counts.index.tolist()

    # トップ10カテゴリにフィルタリング
    top10_df = df[df["category"].isin(top10_categories)]

    # カテゴリごとのFeatured割合を計算
    category_stats = (
        top10_df.groupby(["category", "featured"]).size().unstack(fill_value=0)
    )
    category_stats["Total"] = category_stats.sum(axis=1)
    if True in category_stats.columns:
        category_stats["Featured %"] = (
            category_stats[True] / category_stats["Total"]
        ) * 100
    else:
        category_stats["Featured %"] = 0

    # Totalで降順にソート
    category_stats = category_stats.sort_values(by="Total", ascending=False)

    print("\n--- Featured Ratio by Top 10 Categories ---")
    print(category_stats[["Total", True, False, "Featured %"]])

    # 可視化: トップ10カテゴリのFeatured割合の棒グラフ
    plt.figure(figsize=(18, 12))  # フィギュアサイズを大きく設定
    sns.barplot(
        x=category_stats.index,
        y="Featured %",
        data=category_stats.reset_index(),
        color="skyblue",
        order=category_stats.index,
    )
    plt.title(
        "Featured Ratio by Top 10 Popular Categories", fontsize=24
    )  # タイトルフォントサイズを大きく
    plt.ylabel("Featured %", fontsize=18)  # y軸ラベルフォントサイズを大きく
    plt.xlabel("Category", fontsize=18)  # x軸ラベルフォントサイズを大きく
    plt.xticks(
        rotation=45, ha="right", fontsize=16
    )  # x軸目盛りラベルフォントサイズを大きく
    plt.yticks(fontsize=16)  # y軸目盛りラベルフォントサイズを大きく
    plt.ylim(0, 20)  # y軸の上限を20%に設定

    # 棒の上にパーセンテージと総数を表示
    for index, (feature_pct, total) in enumerate(
        zip(category_stats["Featured %"], category_stats["Total"])
    ):
        plt.text(
            index,
            feature_pct + 0.5,  # 棒の上に少し離して表示
            f"{feature_pct:.1f}%\n(n={total})",  # パーセンテージと総数を表示
            ha="center",
            va="bottom",
            fontsize=20,  # 注釈のフォントサイズを大きく
            color="black",
        )

    plt.tight_layout()
    plt.show()


def analyze_image_influence(df):
    """
    3. Analyze the relationship between the number of images and the Featured status.
       Visualize and summarize how imageCount differs between Featured and Non-Featured.
    """
    # Summary statistics
    image_stats = df.groupby("featured")["imageCount"].agg(
        ["count", "mean", "median", "std"]
    )
    print("\n--- Image Count Statistics by Featured Status ---")
    print(image_stats)

    # Visualization: Boxplot of imageCount by Featured status
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="featured", y="imageCount", data=df, palette="Set3")
    plt.title("Image Count Distribution by Featured Status")
    plt.xlabel("Featured")
    plt.ylabel("Image Count")
    plt.xticks([0, 1], ["Non-Featured", "Featured"])
    plt.tight_layout()
    plt.show()

    # Visualization: Histogram of imageCount for Featured and Non-Featured
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df[df["featured"] == True]["imageCount"],
        color="green",
        label="Featured",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.6,
    )
    sns.histplot(
        df[df["featured"] == False]["imageCount"],
        color="red",
        label="Non-Featured",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.6,
    )
    plt.title("Image Count Histogram by Featured Status")
    plt.xlabel("Image Count")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_video_influence_improved(df):
    """
    4. Analyze and visualize the relationship between the presence of video and the Featured status.
    """
    # クロス集計: FeaturedとhasVideo
    video_crosstab = pd.crosstab(
        df["featured"], df["hasVideo"], margins=True, margins_name="Total"
    )
    video_crosstab = video_crosstab.rename(
        columns={False: "No Video", True: "Has Video"}
    )
    video_crosstab = video_crosstab.rename(
        index={False: "Non-Featured", True: "Featured", "Total": "Total"}
    )

    print("\n--- Cross Tabulation: Video Presence and Featured Status ---")
    print(video_crosstab)

    # パーセンテージの計算
    video_percentage = (
        pd.crosstab(df["featured"], df["hasVideo"], normalize="index") * 100
    )
    video_percentage = video_percentage.rename(
        columns={False: "No Video (%)", True: "Has Video (%)"}
    )
    video_percentage = video_percentage.rename(
        index={False: "Non-Featured", True: "Featured"}
    )

    print("\n--- Percentage: Video Presence by Featured Status ---")
    print(video_percentage)

    # 可視化: 100%積み上げ棒グラフ
    # -------------------------------
    # データの準備
    video_percentage = video_percentage.reset_index()

    # プロットの準備
    plt.figure(figsize=(8, 6))

    # 各カテゴリーの位置を決定
    bar_width = 0.6
    indices = range(len(video_percentage))

    # 'Has Video (%)' のバーをプロット (下部)
    plt.bar(
        indices,
        video_percentage["Has Video (%)"],
        bar_width,
        label="Has Video (%)",
        color="salmon",
    )

    # 'No Video (%)' のバーを 'Has Video (%)' の上に積み上げてプロット
    plt.bar(
        indices,
        video_percentage["No Video (%)"],
        bar_width,
        bottom=video_percentage["Has Video (%)"],
        label="No Video (%)",
        color="skyblue",
    )

    # 数値ラベルの追加
    for i in indices:
        # 'Has Video (%)' の数値ラベル
        plt.text(
            i,
            video_percentage["Has Video (%)"][i] / 2,
            f"{video_percentage['Has Video (%)'][i]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )

        # 'No Video (%)' の数値ラベル
        plt.text(
            i,
            video_percentage["Has Video (%)"][i]
            + video_percentage["No Video (%)"][i] / 2,
            f"{video_percentage['No Video (%)'][i]:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    # 軸のラベルとタイトルを設定
    plt.xticks(indices, video_percentage["featured"])
    plt.xlabel("Featured Status")
    plt.ylabel("Percentage (%)")
    plt.title("Video Presence Percentage by Featured Status (100% Stacked)")

    # 凡例を追加
    plt.legend(title="Video Presence", loc="upper left")

    # グラフを整える
    plt.tight_layout()
    plt.show()


def analyze_video_feature_association(df):
    """
    Analyze the statistical association between video presence and Featured status.
    """
    # Create cross-tabulation
    contingency_table = pd.crosstab(df["hasVideo"], df["featured"])
    contingency_table.index = ["No Video", "Has Video"]
    contingency_table.columns = ["Non-Featured", "Featured"]

    print("\n--- Cross Tabulation: Video Presence and Featured Status ---")
    print(contingency_table)

    # Chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\n--- Chi-Squared Test Results ---")
    print(f"Chi-squared value: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print(
            "結論: 動画の有無とFeaturedステータスの間には統計的に有意な関連があります。"
        )
    else:
        print(
            "結論: 動画の有無とFeaturedステータスの間に統計的に有意な関連はありません。"
        )

    # Calculate Odds Ratio
    a = contingency_table.loc["No Video", "Featured"]
    b = contingency_table.loc["No Video", "Non-Featured"]
    c = contingency_table.loc["Has Video", "Featured"]
    d = contingency_table.loc["Has Video", "Non-Featured"]

    try:
        table = Table2x2([[a, b], [c, d]])
        odds_ratio = table.oddsratio
        confint = table.oddsratio_confint()
        print(f"\nOdds Ratio: {odds_ratio:.4f}")
        print(f"95% Confidence Interval: ({confint[0]:.4f}, {confint[1]:.4f})")
    except Exception as e:
        print(f"オッズ比の計算中にエラーが発生しました: {e}")

    # Visualization: Bar plot of Featured percentage by Video presence
    featured_percent = df.groupby("hasVideo")["featured"].mean() * 100
    featured_percent = featured_percent.rename({False: "No Video", True: "Has Video"})

    print("\n--- Featured Percentage by Video Presence ---")
    print(featured_percent)

    # Create bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=featured_percent.index, y=featured_percent.values, color="skyblue"
    )  # Use single color
    plt.title("Featured Percentage by Video Presence")
    plt.xlabel("Video Presence")
    plt.ylabel("Featured Percentage (%)")
    for index, value in enumerate(featured_percent.values):
        plt.text(
            index, value + 1, f"{value:.2f}%", ha="center", va="bottom", fontsize=10
        )
    plt.ylim(0, max(featured_percent.values) + 10)
    plt.tight_layout()
    plt.show()


def analyze_image_feature_association(df):
    """
    Analyze the statistical association between the number of images and Featured status.
    """
    # Split data into Featured and Non-Featured
    featured = df[df["featured"] == True]["imageCount"]
    non_featured = df[df["featured"] == False]["imageCount"]

    # Hypothesis Testing
    # Assume normal distribution for t-test
    t_stat, p_value_t = ttest_ind(featured, non_featured, equal_var=False)

    # Non-parametric test
    u_stat, p_value_u = mannwhitneyu(featured, non_featured, alternative="two-sided")

    print("\n--- Image Count Association with Featured Status ---")
    print(
        f"Featured Image Count: Mean={featured.mean():.2f}, Median={featured.median():.2f}"
    )
    print(
        f"Non-Featured Image Count: Mean={non_featured.mean():.2f}, Median={non_featured.median():.2f}"
    )

    print("\n--- T-Test Results ---")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value_t:.4f}")
    if p_value_t < 0.05:
        print("結論: FeaturedとNon-Featured間で画像数に有意な差があります (T-Test)。")
    else:
        print("結論: FeaturedとNon-Featured間で画像数に有意な差はありません (T-Test)。")

    print("\n--- Mann-Whitney U Test Results ---")
    print(f"U-statistic: {u_stat}, P-value: {p_value_u:.4f}")
    if p_value_u < 0.05:
        print(
            "結論: FeaturedとNon-Featured間で画像数に有意な差があります (Mann-Whitney U Test)。"
        )
    else:
        print(
            "結論: FeaturedとNon-Featured間で画像数に有意な差はありません (Mann-Whitney U Test)。"
        )

    # Visualization: Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="featured", y="imageCount", data=df, palette="Set3")
    plt.title("Image Count Distribution by Featured Status")
    plt.xlabel("Featured")
    plt.ylabel("Image Count")
    plt.xticks([0, 1], ["Non-Featured", "Featured"])
    plt.tight_layout()
    plt.show()

    # Visualization: Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(
        featured,
        color="green",
        label="Featured",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.6,
    )
    sns.histplot(
        non_featured,
        color="red",
        label="Non-Featured",
        kde=True,
        stat="density",
        bins=30,
        alpha=0.6,
    )
    plt.title("Image Count Histogram by Featured Status")
    plt.xlabel("Image Count")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_image_feature_association_by_category(df):
    """
    Analyze the statistical association between the number of images and Featured status for each category.
    """
    categories = df["category"].unique()
    results = []

    print("\n--- Image Count Association by Category ---")
    for category in categories:
        category_df = df[df["category"] == category]
        featured = category_df[category_df["featured"] == True]["imageCount"]
        non_featured = category_df[category_df["featured"] == False]["imageCount"]

        # Skip categories with insufficient data
        if len(featured) < 5 or len(non_featured) < 5:
            continue

        # Hypothesis Testing
        t_stat, p_value_t = ttest_ind(featured, non_featured, equal_var=False)
        u_stat, p_value_u = mannwhitneyu(
            featured, non_featured, alternative="two-sided"
        )

        # Save results
        results.append(
            {
                "Category": category,
                "Featured Mean": featured.mean(),
                "Non-Featured Mean": non_featured.mean(),
                "T-Test P-Value": p_value_t,
                "Mann-Whitney U Test P-Value": p_value_u,
            }
        )

        # Display results
        print(f"\nCategory: {category}")
        print(
            f"Featured Image Count: Mean={featured.mean():.2f}, Median={featured.median():.2f}"
        )
        print(
            f"Non-Featured Image Count: Mean={non_featured.mean():.2f}, Median={non_featured.median():.2f}"
        )
        print(f"T-Test P-Value: {p_value_t:.4f} {'*' if p_value_t < 0.05 else ''}")
        print(
            f"Mann-Whitney U Test P-Value: {p_value_u:.4f} {'*' if p_value_u < 0.05 else ''}"
        )

    # Display summary of results
    results_df = pd.DataFrame(results)
    print("\n--- Summary of Image Count Association by Category ---")
    print(results_df)

    # Highlight significant categories
    significant = results_df[
        (results_df["T-Test P-Value"] < 0.05)
        | (results_df["Mann-Whitney U Test P-Value"] < 0.05)
    ]
    if not significant.empty:
        print("\n--- Categories with Significant Image Count Differences ---")
        print(significant)
    else:
        print("\nどのカテゴリでも有意な差は見つかりませんでした。")


def logistic_regression_image_featured(df):
    """
    Perform logistic regression to evaluate the impact of image count on Featured status.
    """
    # Select necessary variables
    data = df[["featured", "imageCount"]].dropna()

    # Define independent variables and dependent variable
    X = data["imageCount"]
    y = data["featured"].astype(int)

    # Add constant term for intercept
    X = sm.add_constant(X)

    # Apply logistic regression
    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False)
        print("\n--- Logistic Regression Results ---")
        print(result.summary())

        # Calculate Odds Ratios
        params = result.params
        conf = result.conf_int()
        conf["OR"] = params
        conf.columns = ["2.5%", "97.5%", "OR"]
        odds_ratio = np.exp(conf)

        print("\n--- Odds Ratios and 95% Confidence Intervals ---")
        print(odds_ratio)
    except Exception as e:
        print(f"ロジスティック回帰中にエラーが発生しました: {e}")


def logistic_regression_image_featured_multivariate(df):
    """
    Perform multivariate logistic regression to evaluate the impact of image count and other factors on Featured status.
    """
    # Select necessary variables and convert categorical variables to dummy variables
    data = df[["featured", "hasVideo", "category", "imageCount"]].dropna()
    data = pd.get_dummies(data, columns=["category"], drop_first=True)

    # Define independent variables and dependent variable
    X = data.drop("featured", axis=1)
    y = data["featured"].astype(int)

    # Add constant term for intercept
    X = sm.add_constant(X)

    # Apply logistic regression
    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False)
        print("\n--- Multivariate Logistic Regression Results ---")
        print(result.summary())

        # Calculate Odds Ratios
        params = result.params
        conf = result.conf_int()
        conf["OR"] = params
        conf.columns = ["2.5%", "97.5%", "OR"]
        odds_ratio = np.exp(conf)

        print("\n--- Odds Ratios and 95% Confidence Intervals ---")
        print(odds_ratio)
    except Exception as e:
        print(f"多変量ロジスティック回帰中にエラーが発生しました: {e}")


def analyze_average_products_and_featured_percentage_per_weekday_dual_axis(df):
    """
    Calculate the average number of products released each day of the week and the percentage of featured products,
    then visualize both metrics using a dual-axis plot with bars for average product count and a line for featured percentage.
    """
    # Ensure 'createdAt' exists in the DataFrame
    if "createdAt" not in df.columns:
        print("Error: 'createdAt' column does not exist in the DataFrame.")
        return

    # Extract the weekday from 'createdAt'
    df["Weekday"] = df["createdAt"].dt.day_name()

    # Determine the date range of the data
    start_date = df["createdAt"].min().date()
    end_date = df["createdAt"].max().date()

    # Generate all dates within the range
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # List of weekdays in order
    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Count the number of each weekday in the date range
    weekday_counts = (
        pd.Series(all_dates.day_name()).value_counts().reindex(weekdays).fillna(0)
    )

    # Count the number of products released each weekday
    product_counts = df["Weekday"].value_counts().reindex(weekdays).fillna(0)

    # Calculate the average number of products per weekday
    average_products = product_counts / weekday_counts

    # Calculate the number of featured products per weekday
    featured_counts = (
        df[df["featured"] == True]["Weekday"].value_counts().reindex(weekdays).fillna(0)
    )

    # Calculate the featured percentage per weekday
    featured_percentage = (featured_counts / product_counts) * 100
    featured_percentage = featured_percentage.replace([np.inf, -np.inf], 0).fillna(0)

    # Create a DataFrame for the results
    average_df = pd.DataFrame(
        {
            "Weekday": weekdays,
            "Average Product Count": average_products.values,
            "Featured Percentage (%)": featured_percentage.values,
        }
    )

    print("\n--- Average Number of Products and Featured Percentage per Weekday ---")
    print(average_df)

    # Set up the matplotlib figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot the average product count as bar graph
    bars = ax1.bar(
        average_df["Weekday"],
        average_df["Average Product Count"],
        color="skyblue",
        label="Average Product Count",
    )
    ax1.set_xlabel("Weekday", fontsize=20)
    ax1.set_ylabel("Average Product Count", fontsize=20, color="skyblue")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    ax1.set_ylim(0, average_df["Average Product Count"].max() * 1.2)

    # Annotate the bars with average product count
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Create a second y-axis for the featured percentage
    ax2 = ax1.twinx()
    ax2.plot(
        average_df["Weekday"],
        average_df["Featured Percentage (%)"],
        color="salmon",
        marker="o",
        label="Featured Percentage (%)",
    )
    ax2.set_ylabel("Featured Percentage (%)", fontsize=20, color="salmon")
    ax2.tick_params(axis="y", labelcolor="salmon")
    ax2.set_ylim(0, average_df["Featured Percentage (%)"].max() * 1.2)

    # Annotate the line points with featured percentage
    for i, value in enumerate(average_df["Featured Percentage (%)"]):
        ax2.annotate(
            f"{value:.2f}%",
            xy=(i, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="salmon",
        )

    # Title and layout
    plt.title(
        "Average Number of Products Released and Featured Percentage per Weekday",
        fontsize=18,
    )

    # Legends
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc="upper left", fontsize=12)

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    # Specify the path to the CSV file
    csv_file = "posts.csv"  # Replace with your actual file name

    # 1. Load data
    df = load_data(csv_file)

    # 2. Preprocess data
    df = preprocess_data(df)

    # ---- Begin Analysis ----
    # # (1) Overall Featured Ratio
    analyze_feature_ratio(df)

    # # (2) Featured Ratio by Top 10 Categories
    analyze_top10_categories_feature_ratio(df)

    # # (3) Image Count Influence
    analyze_image_influence(df)

    # # (4) Video Presence Influence
    analyze_video_influence_improved(df)

    # # (5) Statistical Association between Video Presence and Featured Status
    analyze_video_feature_association(df)

    # # (6) Statistical Association between Image Count and Featured Status
    analyze_image_feature_association(df)

    # # (7) Statistical Association between Image Count and Featured Status by Category
    analyze_image_feature_association_by_category(df)

    # # (8) Logistic Regression: Image Count Impact
    logistic_regression_image_featured(df)

    # # (9) 曜日ごとの平均プロダクト数分析
    analyze_average_products_and_featured_percentage_per_weekday_dual_axis(df)

    # ---- Additional Analyses or Visualizations can be added here ----


if __name__ == "__main__":
    main()
