CREATE TABLE PostedNews (
    title NVARCHAR(500)         NOT NULL,
    url NVARCHAR(1000)          NOT NULL,
    text_hash NVARCHAR(64)      NOT NULL PRIMARY KEY,
    summary NVARCHAR(MAX)       NOT NULL,
    source NVARCHAR(200)        NOT NULL,
    published_date DATE         NOT NULL,
    published_time TIME         NOT NULL,
    rss_source NVARCHAR(100)    NOT NULL,
    keywords NVARCHAR(MAX)      NULL  -- רשאי להיות ריק/חסר
);


CREATE TABLE SkippedNews (
    title NVARCHAR(500)         NOT NULL,
    url NVARCHAR(1000)          NOT NULL,
    text_hash NVARCHAR(64)      NOT NULL PRIMARY KEY,
    summary NVARCHAR(MAX)       NOT NULL,
    source NVARCHAR(200)        NOT NULL,
    published_date DATE         NOT NULL,
    published_time TIME         NOT NULL,
    rss_source NVARCHAR(100)    NOT NULL,
    reason NVARCHAR(300)        NULL,      -- לפעמים עשוי להיות ריק
    fail_count INT              NOT NULL,
    date DATE                   NOT NULL
);
