import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import Table2x2


def load_data(csv_file_path):
    """
    Load a CSV file and return a DataFrame.
    CSVファイルを読み込み、DataFrameを返す。
    """
    try:
        df = pd.read_csv(csv_file_path)
        print("Successfully loaded the data. データの読み込みに成功しました。")
        return df
    except FileNotFoundError:
        print(
            f"Error: File '{csv_file_path}' not found. エラー: ファイル '{csv_file_path}' が見つかりません。"
        )
        exit(1)
    except pd.errors.ParserError:
        print(
            f"Error: Failed to parse the file '{csv_file_path}'. エラー: ファイル '{csv_file_path}' の解析中にエラーが発生しました。"
        )
        exit(1)


def preprocess_data(df):
    """
    Preprocess the data:
    - Convert 'featured' and 'hasVideo' to boolean type.
    - Convert 'createdAt' and 'featuredAt' to datetime type (if they exist).
    データの前処理:
    - 'featured' と 'hasVideo' をブール型に変換。
    - 'createdAt', 'featuredAt' を日時型に変換(存在する場合)。
    """
    # Convert featured and hasVideo to bool
    df["featured"] = df["featured"].astype(bool)
    df["hasVideo"] = df["hasVideo"].astype(bool)

    # Convert createdAt and featuredAt to datetime
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
    if "featuredAt" in df.columns:
        df["featuredAt"] = pd.to_datetime(df["featuredAt"], errors="coerce")

    return df


def analyze_feature_ratio(df):
    """
    1. Overall Featured Ratio
    全体のFeatured比率
    """
    total_items = len(df)
    featured_items = df["featured"].sum()
    print("\n--- Overall Featured Ratio ---")
    print("全体のFeatured比率 ---")
    print(f"Total items: {total_items} 件")
    print(
        f"Featured items: {featured_items} ({featured_items / total_items * 100:.2f}%)"
    )
    print(
        f"Featured items: {featured_items} ({featured_items / total_items * 100:.2f}%)"
    )


def analyze_top10_categories_feature_ratio(df):
    """
    Analyze and visualize the Featured ratio for the top 10 popular categories.
    Categories are sorted by the total number of posts in descending order.
    Additionally, displays the total number of posts per category on the plot
    and sets the y-axis limit to 20%.

    トップ10の人気カテゴリーにおけるFeatured比率を分析・可視化します。
    カテゴリーは投稿数の多い順にソートされています。
    また、プロット上に各カテゴリーの投稿総数を表示し、y軸の上限を20%に設定します。
    """
    # Get the top 10 categories by number of posts
    category_counts = df["category"].value_counts().nlargest(10)
    top10_categories = category_counts.index.tolist()

    # Filter for top 10 categories
    top10_df = df[df["category"].isin(top10_categories)]

    # Calculate featured ratio per category
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

    # Sort by Total in descending order
    category_stats = category_stats.sort_values(by="Total", ascending=False)

    print("\n--- Featured Ratio by Top 10 Categories ---")
    print("トップ10カテゴリーごとのFeatured比率 ---")
    print(category_stats[["Total", True, False, "Featured %"]])

    # Visualization: Bar chart of Featured ratio for top 10 categories
    plt.figure(figsize=(18, 12))  # Increase figure size
    sns.barplot(
        x=category_stats.index,
        y="Featured %",
        data=category_stats.reset_index(),
        color="skyblue",
        order=category_stats.index,
    )
    plt.title(
        "Featured Ratio by Top 10 Popular Categories", fontsize=24
    )  # Increase title font size
    plt.ylabel("Featured %", fontsize=18)  # Increase y-axis label font size
    plt.xlabel("Category", fontsize=18)  # Increase x-axis label font size
    plt.xticks(
        rotation=45, ha="right", fontsize=16
    )  # Increase x-axis tick label font size
    plt.yticks(fontsize=16)  # Increase y-axis tick label font size
    plt.ylim(0, 20)  # Set y-axis upper limit to 20%

    # Display percentage and total number above bars
    for index, (feature_pct, total) in enumerate(
        zip(category_stats["Featured %"], category_stats["Total"])
    ):
        plt.text(
            index,
            feature_pct + 0.5,  # Slightly above the bar
            f"{feature_pct:.1f}%\n(n={total})",  # Display percentage and total
            ha="center",
            va="bottom",
            fontsize=20,  # Increase annotation font size
            color="black",
        )

    plt.tight_layout()
    plt.show()


def analyze_image_influence(df):
    """
    3. Analyze the relationship between the number of images and the Featured status.
       Visualize and summarize how imageCount differs between Featured and Non-Featured.

    画像数とFeaturedステータスの関係を分析します。
    FeaturedとNon-Featured間でimageCountがどのように異なるかを可視化・要約します。
    """
    # Summary statistics
    image_stats = df.groupby("featured")["imageCount"].agg(
        ["count", "mean", "median", "std"]
    )
    print("\n--- Image Count Statistics by Featured Status ---")
    print("Featuredステータス別の画像数統計 ---")
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

    動画の有無とFeaturedステータスの関係を分析・可視化します。
    """
    # Cross-tabulation: Featured and hasVideo
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
    print("動画の有無とFeaturedステータスのクロス集計 ---")
    print(video_crosstab)

    # Calculate percentages
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
    print("Featuredステータス別の動画有無のパーセンテージ ---")
    print(video_percentage)

    # Visualization: 100% Stacked Bar Chart
    # -------------------------------
    # Data preparation
    video_percentage = video_percentage.reset_index()

    # Plot preparation
    plt.figure(figsize=(8, 6))

    # Positions for each category
    bar_width = 0.6
    indices = range(len(video_percentage))

    # Plot 'Has Video (%)' bars (bottom layer)
    plt.bar(
        indices,
        video_percentage["Has Video (%)"],
        bar_width,
        label="Has Video (%)",
        color="salmon",
    )

    # Plot 'No Video (%)' bars stacked on top
    plt.bar(
        indices,
        video_percentage["No Video (%)"],
        bar_width,
        bottom=video_percentage["Has Video (%)"],
        label="No Video (%)",
        color="skyblue",
    )

    # Add numerical labels
    for i in indices:
        # Label for 'Has Video (%)'
        plt.text(
            i,
            video_percentage["Has Video (%)"][i] / 2,
            f"{video_percentage['Has Video (%)'][i]:.1f}%",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
        )

        # Label for 'No Video (%)'
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

    # Axis labels and title
    plt.xticks(indices, video_percentage["featured"])
    plt.xlabel("Featured Status", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.title(
        "Video Presence Percentage by Featured Status (100% Stacked)", fontsize=16
    )

    # Add legend
    plt.legend(title="Video Presence", loc="upper left")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def analyze_video_feature_association(df):
    """
    Analyze the statistical association between video presence and Featured status.

    動画の有無とFeaturedステータス間の統計的関連性を分析します。
    """
    # Create cross-tabulation
    contingency_table = pd.crosstab(df["hasVideo"], df["featured"])
    contingency_table.index = ["No Video", "Has Video"]
    contingency_table.columns = ["Non-Featured", "Featured"]

    print("\n--- Cross Tabulation: Video Presence and Featured Status ---")
    print("動画の有無とFeaturedステータスのクロス集計 ---")
    print(contingency_table)

    # Chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\n--- Chi-Squared Test Results ---")
    print("カイ二乗検定結果 ---")
    print(f"Chi-squared value: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print(
            "Conclusion: There is a statistically significant association between video presence and Featured status. 動画の有無とFeaturedステータスの間には統計的に有意な関連があります。"
        )
    else:
        print(
            "Conclusion: There is no statistically significant association between video presence and Featured status. 動画の有無とFeaturedステータスの間に統計的に有意な関連はありません。"
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
        print("\n--- Odds Ratio and 95% Confidence Intervals ---")
        print("オッズ比と95%信頼区間 ---")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"95% Confidence Interval: ({confint[0]:.4f}, {confint[1]:.4f})")
    except Exception as e:
        print(
            f"Error calculating odds ratio: {e} オッズ比の計算中にエラーが発生しました: {e}"
        )

    # Visualization: Bar plot of Featured percentage by Video presence
    featured_percent = df.groupby("hasVideo")["featured"].mean() * 100
    featured_percent = featured_percent.rename({False: "No Video", True: "Has Video"})

    print("\n--- Featured Percentage by Video Presence ---")
    print("動画有無別のFeatured割合 ---")
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

    画像数とFeaturedステータス間の統計的関連性を分析します。
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
    print("画像数とFeaturedステータスの関連性 ---")
    print(
        f"Featured Image Count: Mean={featured.mean():.2f}, Median={featured.median():.2f}"
    )
    print(
        f"Non-Featured Image Count: Mean={non_featured.mean():.2f}, Median={non_featured.median():.2f}"
    )

    print("\n--- T-Test Results ---")
    print("T検定結果 ---")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value_t:.4f}")
    if p_value_t < 0.05:
        print(
            "Conclusion: There is a significant difference in image count between Featured and Non-Featured (T-Test). FeaturedとNon-Featured間で画像数に有意な差があります (T-Test)。"
        )
    else:
        print(
            "Conclusion: There is no significant difference in image count between Featured and Non-Featured (T-Test). FeaturedとNon-Featured間で画像数に有意な差はありません (T-Test)。"
        )

    print("\n--- Mann-Whitney U Test Results ---")
    print("Mann-Whitney U検定結果 ---")
    print(f"U-statistic: {u_stat}, P-value: {p_value_u:.4f}")
    if p_value_u < 0.05:
        print(
            "Conclusion: There is a significant difference in image count between Featured and Non-Featured (Mann-Whitney U Test). FeaturedとNon-Featured間で画像数に有意な差があります (Mann-Whitney U Test)。"
        )
    else:
        print(
            "Conclusion: There is no significant difference in image count between Featured and Non-Featured (Mann-Whitney U Test). FeaturedとNon-Featured間で画像数に有意な差はありません (Mann-Whitney U Test)。"
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

    各カテゴリーにおける画像数とFeaturedステータス間の統計的関連性を分析します。
    """
    categories = df["category"].unique()
    results = []

    print("\n--- Image Count Association by Category ---")
    print("カテゴリー別の画像数関連性分析 ---")
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
    print("カテゴリー別画像数関連性の要約 ---")
    print(results_df)

    # Highlight significant categories
    significant = results_df[
        (results_df["T-Test P-Value"] < 0.05)
        | (results_df["Mann-Whitney U Test P-Value"] < 0.05)
    ]
    if not significant.empty:
        print("\n--- Categories with Significant Image Count Differences ---")
        print("有意な画像数差異があるカテゴリー ---")
        print(significant)
    else:
        print(
            "\nNo significant differences found in any category. どのカテゴリでも有意な差は見つかりませんでした。"
        )


def logistic_regression_image_featured(df):
    """
    Perform logistic regression to evaluate the impact of image count on Featured status.

    画像数がFeaturedステータスに与える影響を評価するためにロジスティック回帰を実施します。
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
        print("ロジスティック回帰結果 ---")
        print(result.summary())

        # Calculate Odds Ratios
        params = result.params
        conf = result.conf_int()
        conf["OR"] = params
        conf.columns = ["2.5%", "97.5%", "OR"]
        odds_ratio = np.exp(conf)

        print("\n--- Odds Ratios and 95% Confidence Intervals ---")
        print("オッズ比と95%信頼区間 ---")
        print(odds_ratio)
    except Exception as e:
        print(
            f"Error during logistic regression: {e} ロジスティック回帰中にエラーが発生しました: {e}"
        )


def logistic_regression_image_featured_multivariate(df):
    """
    Perform multivariate logistic regression to evaluate the impact of image count and other factors on Featured status.

    画像数や他の要因がFeaturedステータスに与える影響を評価するために多変量ロジスティック回帰を実施します。
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
        print("多変量ロジスティック回帰結果 ---")
        print(result.summary())

        # Calculate Odds Ratios
        params = result.params
        conf = result.conf_int()
        conf["OR"] = params
        conf.columns = ["2.5%", "97.5%", "OR"]
        odds_ratio = np.exp(conf)

        print("\n--- Odds Ratios and 95% Confidence Intervals ---")
        print("オッズ比と95%信頼区間 ---")
        print(odds_ratio)
    except Exception as e:
        print(
            f"Error during multivariate logistic regression: {e} 多変量ロジスティック回帰中にエラーが発生しました: {e}"
        )


def analyze_average_products_and_featured_percentage_per_weekday_dual_axis(df):
    """
    Calculate the average number of products released each day of the week and the percentage of featured products,
    then visualize both metrics using a dual-axis plot with bars for average product count and a line for featured percentage.

    曜日ごとの平均プロダクト数とFeaturedプロダクトの割合を計算し、
    バーで平均プロダクト数、線でFeatured割合を示すデュアルアクシスプロットで可視化します。
    """
    # Ensure 'createdAt' exists in the DataFrame
    if "createdAt" not in df.columns:
        print("Error: 'createdAt' column does not exist in the DataFrame.")
        print("エラー: 'createdAt' 列がDataFrameに存在しません。")
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
    print("曜日ごとの平均プロダクト数とFeatured割合 ---")
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
    plt.title(
        "曜日ごとの平均プロダクト数とFeatured割合",
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
    # 1. Overall Featured Ratio
    analyze_feature_ratio(df)

    # 2. Featured Ratio by Top 10 Categories
    analyze_top10_categories_feature_ratio(df)

    # 3. Image Count Influence
    analyze_image_influence(df)

    # 4. Video Presence Influence
    analyze_video_influence_improved(df)

    # 5. Statistical Association between Video Presence and Featured Status
    analyze_video_feature_association(df)

    # 6. Statistical Association between Image Count and Featured Status
    analyze_image_feature_association(df)

    # 7. Statistical Association between Image Count and Featured Status by Category
    analyze_image_feature_association_by_category(df)

    # 8. Logistic Regression: Image Count Impact
    logistic_regression_image_featured(df)

    # 9. Analyze Average Products and Featured Percentage per Weekday
    analyze_average_products_and_featured_percentage_per_weekday_dual_axis(df)

    # ---- Additional Analyses or Visualizations can be added here ----


if __name__ == "__main__":
    main()
