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
    fail_count INT              NOT NULL
);

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
    spike_detected BIT DEFAULT 0,
    growth_detected BIT DEFAULT 0
);


CREATE TABLE Articles (
    article_id INT PRIMARY KEY,
    title NVARCHAR(500),
    summary NVARCHAR(MAX),
    url NVARCHAR(1000),
    published_date DATE,
    published_time TIME,
    rss_source NVARCHAR(100),
    cleaned_keywords NVARCHAR(MAX),
    topic_id INT ,
    similar_count INT,
    CONSTRAINT FK_Articles_Topics FOREIGN KEY (topic_id) REFERENCES Topics(topic_id)
);


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
    spike_detected BIT DEFAULT 0,
    growth_detected BIT DEFAULT 0,
    representative_summary NVARCHAR(MAX)
);

