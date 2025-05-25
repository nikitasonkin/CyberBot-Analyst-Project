# Project Overview
Analysis of data from the [CyberNewsBot](https://github.com/nikitasonkin/CyberNewsBot).
This repository contains SQL schema that creates dedicated tables and Jupyter Notebooks that demonstrate how to process and analyze data efficiently. Below is a summary and explanation of each file in the repository.

## Table of Contents
- [Project Overview](#project-overview)  
- [SQL Schema](#sql-schema)  
- [Analysis Notebook](#analysis-notebook)  
- [Key Sections](#key-sections)  
- [Outputs](#outputs)  
- [JSON Files](#json-files)  
  - [posted_news_ud.json](#posted_news_ud-json)  
  - [skipped_news_ud.json](#skipped_news_ud-json)  
- [Dependencies](#dependencies)  
- [Author](#author)  

---

#### 📄 File: [`create_new_tables.sql`](https://github.com/nikitasonkin/Data-Analyst-Project/blob/main/sql/create_new_tables.sql)

This repository defines five core tables for storing and analyzing news articles, topics and trends. Each table is carefully structured with primary keys, auto-increment IDs, appropriate data types and foreign key constraints to ensure data integrity and optimal query performance.

1. **PostedNews**  
   - **Purpose**: Stores all successfully posted news articles along with full metadata.  
   - **Key columns**:  
     - `ID INT IDENTITY(1,1) PRIMARY KEY` – auto-incrementing unique identifier  
     - `text_hash NVARCHAR(64) UNIQUE` – dedupe key for content  
     - `published_date DATE`, `published_time TIME` – publication timestamp  
     - `rss_source NVARCHAR(100)` – geographic/source channel  
     - `keywords NVARCHAR(MAX)` – raw keyword list (nullable)  

2. **SkippedNews**  
   - **Purpose**: Logs all articles that failed to post, including failure reasons and retry counts.  
   - **Key columns**:  
     - `ID INT IDENTITY(1,1) PRIMARY KEY` – auto-incrementing unique identifier  
     - `text_hash NVARCHAR(64) UNIQUE` – prevents duplicate skip entries  
     - `reason NVARCHAR(300)` – explanation for rejection  
     - `fail_count INT` – number of retry attempts  

3. **Topics**  
   - **Purpose**: Summarizes each clustered topic (connected component) for trend analysis.  
   - **Key columns**:  
     - `topic_id INT PRIMARY KEY` – cluster identifier from DBSCAN  
     - `short_title NVARCHAR(500)` – representative title  
     - `num_articles INT`, `num_days INT` – size and duration of the topic  
     - `first_date`, `last_date` – date range of the topic  
     - `main_country NVARCHAR(100)` – most frequent `rss_source`  
     - `top_keywords NVARCHAR(MAX)` – comma-separated top-5 keywords  
     - `spike_detected BIT DEFAULT 0`, `growth_detected BIT DEFAULT 0` – trend flags  

4. **Articles**  
   - **Purpose**: Stores individual articles linked to topics for detailed, article-level analysis.  
   - **Key columns**:  
     - `article_id INT IDENTITY(1,1) PRIMARY KEY` – auto-incrementing ID  
     - `topic_id INT FOREIGN KEY REFERENCES Topics(topic_id)` – topic association  
     - `similar_count INT` – number of peer articles in the same topic  
     - `cleaned_keywords NVARCHAR(MAX)` – normalized keyword list  

5. **Trends**  
   - **Purpose**: Captures the most important topics for reporting and dashboards.  
   - **Key columns**:  
     - `topic_id INT PRIMARY KEY REFERENCES Topics(topic_id)` – identifies the trend topic  
     - `representative_summary NVARCHAR(MAX)` – text summary of the trend  
     - (inherits all summary fields from `Topics` for easy dashboarding)  


---

#### 📄 File: [`DataAnalystCyberProject.ipynb`](https://github.com/nikitasonkin/Data-Analyst-Project/blob/main/notebooks/DataAnalystCyberProject.ipynb)

A single, end-to-end Python script that drives the CyberNewsBot analytics pipeline—from JSON ingestion and SQL ETL, through exploratory data checks, feature engineering, semantic clustering and topic/trend extraction, to a comprehensive suite of publication-ready visualizations.

##### 🔹 Key Sections

1. **Imports & SQLAlchemy Setup**  
   - Standard, pandas/numpy, SQLAlchemy, matplotlib/seaborn, NLP & ML libraries  
   - Defines `server`, `database`, `driver` and builds the `engine`

2. **ETL Functions**  
   - `load_posted_news(engine, json_path)`  
   - `load_skipped_news(engine, json_path)`  
   - Read JSON → dedupe on `text_hash` → return new entries

3. **Main Entry Point**  
   - `main()` invokes both loaders  
   - Appends new records to `PostedNews` and `SkippedNews`  
   - Prints summary counts and sample titles

4. **Data Retrieval & EDA**  
   - Reads `PostedNews` and `SkippedNews` back into DataFrames  
   - Prints shape, duplicate counts, null counts, `.info()`, `.describe()`, random samples, per-date frequency

5. **Temporal Feature Engineering**  
   - Parses `published_date`/`published_time` → datetime, `.dt.day_name()`, `.dt.hour`  
   - Builds side-by-side `comparison_df` of daily Sent vs Skipped

6. **Keyword Cleaning & Top-N Extraction**  
   - `clean_keywords()` to normalize & filter stopwords  
   - `get_top_keywords()` → DataFrame of top keywords  
   - `plot_top_keywords_pie()` → pie (or donut) chart of keyword share

7. **Publication & Failure Trends**  
   - Daily lineplots of Published vs Failed articles  
   - Growth rate area chart with 7-day MA  
   - Bar-chart of failure reasons (percentage)

8. **Country-Level Breakdown**  
   - Side-by-side barplot of Published vs Skipped by `rss_source`

9. **Heatmap of Failures**  
   - `sns.heatmap()` of failure counts by date × reason

10. **Duplicate Analysis Helpers**  
    - `show_top_dupes()` for title, hash, and URL duplicates  
    - Prints top N with counts and percentages

11. **Hourly Flow Scatter**  
    - `build_hourly_tables()` + `plot_hourly_flow()` for 7-day scatter of published vs duplicates

12. **Semantic Embedding & Clustering**  
    - SentenceTransformer embeddings of `title + summary`  
    - DBSCAN clustering (cosine metric) → `topic_id`, `similar_count`

13. **Topic Filtering & Aggregation**  
    - Identify `large_topics` (≥5 articles)  
    - Build `agg_df` and `topic_summary` for examples

14. **Trend Classification & SQL Write-Back**  
    - Build `topics_df` with counts, date ranges, top keywords, `spike`/`growth` flags  
    - Persist new Topics, Articles, Trends into SQL tables  

15. **Top-N Trending Topics Visualization**  
    - Barplot of top 10 by `num_articles`  
    - Daily line/area chart for top topics  
    - WordCloud of trending keywords  
    - Horizontal bar chart with representative summaries and metadata

##### 📊 Outputs

- **Console**: ETL summaries, EDA statistics, duplicate and topic/trend counts  
- **Charts**: scatterflows, time-series & area plots, pie/donut, bar/heatmap, wordcloud, trend dashboards  

---

##### 📂JSON Files: DATA/`posted_news_ud.json` & DATA/`skipped_news_ud.json`

These files are automatically generated and updated by the **CyberNewsBot** system. They serve as the raw data sources for all analysis and trend detection processes.

##### `posted_news_ud.json`

Contains all cybersecurity news articles that were successfully published and sent by the bot.  
Each article includes the following fields:

- `title` – the article's title  
- `url` – direct link to the article  
- `summary` – short summary of the content  
- `source` – publication source (e.g., website name)  
- `published_date` – date of publication (YYYY-MM-DD)  
- `published_time` – time of publication (HH:MM:SS)  
- `rss_source` – geographic source (country/channel)  
- `keywords` – extracted keywords from the content  
- `text_hash` – unique identifier for duplicate detection  


##### `skipped_news_ud.json`

Stores all articles that were **rejected** during processing due to filters or errors.  
Each skipped entry includes all base fields from `posted_news_ud.json`, plus:

- `reason` – explanation for rejection (e.g., duplicate title, short summary)  
- `fail_count` – how many times the article was skipped  
- `date` – timestamp when the rejection was logged  

---

##### Dependencies

- **Standard Library**  
  - `json`  
  - `warnings`  

- **Data Manipulation**  
  - `pandas`  
  - `numpy`  

- **Database Connectivity**  
  - `sqlalchemy`  
    - `create_engine`  
    - `NVARCHAR`, `Date`, `Time`  

- **Visualization**  
  - `matplotlib`  
    - `pyplot`  
    - `colors` (`mcolors`)  
    - `cm`  
    - `dates` (`DateFormatter`, `mdates`)  
    - `Line2D`  
    - `LogNorm`  
  - `seaborn`  
  - `wordcloud` (`WordCloud`)  

- **Graph Analysis**  
  - `networkx`  

- **Machine Learning / NLP**  
  - `scikit-learn`  
    - `TfidfVectorizer`  
    - `CountVectorizer`  
    - `KMeans`  
    - `DBSCAN`  
  - `sentence-transformers`  
    - `SentenceTransformer`  
    - `util`  

---

### License
-This project is licensed under the [MIT License](LICENSE).

### Author
- **Created by:** Nikita Sonkin  
- **Project repository:** [CyberNewsBot on GitHub](https://github.com/nikitasonkin/CyberNewsBot)
