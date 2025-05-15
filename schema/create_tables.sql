-- Stores all successfully posted news articles with metadata for trend analysis and duplication filtering.Data imported from file datatosql.ipynb
CREATE TABLE PostedNews (
    ID INT IDENTITY(1,1) PRIMARY KEY, 
    title NVARCHAR(500)         NOT NULL,
    url NVARCHAR(1000)          NOT NULL,
    text_hash NVARCHAR(64)      NOT NULL UNIQUE, 
    summary NVARCHAR(MAX)       NOT NULL,
    source NVARCHAR(200)        NOT NULL,
    published_date DATE         NOT NULL,
    published_time TIME         NOT NULL,
    rss_source NVARCHAR(100)    NOT NULL,
    keywords NVARCHAR(MAX)      NULL
);

-- Stores all news articles that failed to post, along with the failure reason and count for quality control and diagnostics. Data imported from file datatosql.ipynb
CREATE TABLE SkippedNews (
    ID INT IDENTITY(1,1) PRIMARY KEY,  
    title NVARCHAR(500)         NOT NULL,
    url NVARCHAR(1000)          NOT NULL,
    text_hash NVARCHAR(64)      NOT NULL UNIQUE, 
    summary NVARCHAR(MAX)       NOT NULL,
    source NVARCHAR(200)        NOT NULL,
    published_date DATE         NOT NULL,
    published_time TIME         NOT NULL,
    rss_source NVARCHAR(100)    NOT NULL,
    reason NVARCHAR(300)        NULL,
    fail_count INT              NOT NULL,
    date DATE                   NOT NULL
);

-- Includes all topics, regardless of whether they are considered trends.
--Built from connected components in a graph of similar articles.
--Serves as the core structure for topic-level insights and trend classification.
CREATE TABLE Topics (
    topic_id INT PRIMARY KEY,
    short_title NVARCHAR(500),
    trend_type NVARCHAR(20),
    num_articles INT,
    num_days INT,
    max_articles_per_day INT,
    first_date DATE,
    dominant_day DATE,
    last_date DATE,
    main_country NVARCHAR(100),
    top_keywords NVARCHAR(MAX),
    spike_detected BIT,
    growth_detected BIT
);

--Only articles that were grouped into a topic (cluster) are included.
--Articles not similar to any others are excluded from this table.
--The table enables trend analysis at the article level
CREATE TABLE Articles (
    article_id INT PRIMARY KEY,
    title NVARCHAR(500),
    summary NVARCHAR(MAX),
    url NVARCHAR(1000),
    published_date DATE,
    published_time TIME,
    rss_source NVARCHAR(100),
    cleaned_keywords NVARCHAR(MAX),
    text_hash NVARCHAR(64) UNIQUE,
    topic_id INT FOREIGN KEY REFERENCES Topics(topic_id),
    similar_count INT
);

--Represents the most important topics for visualization and decision-making.
--Used in dashboards and final visual summaries.
--Ideal for Power BI or summary reports.
CREATE TABLE Trends (
    topic_id INT PRIMARY KEY FOREIGN KEY REFERENCES Topics(topic_id),
    short_title NVARCHAR(500),
    trend_type NVARCHAR(20),
    num_articles INT,
    num_days INT,
    max_articles_per_day INT,
    first_date DATE,
    dominant_day DATE,
    last_date DATE,
    main_country NVARCHAR(100),
    top_keywords NVARCHAR(MAX),
    spike_detected BIT,
    growth_detected BIT,
    representative_summary NVARCHAR(MAX)
);
