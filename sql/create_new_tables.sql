
CREATE TABLE PostedNews (
  ID               INT IDENTITY(1,1) PRIMARY KEY,
  title            NVARCHAR(500)    NOT NULL,
  url              NVARCHAR(1000)   NOT NULL,
  text_hash        NVARCHAR(64)     NOT NULL UNIQUE,
  summary          NVARCHAR(MAX)    NOT NULL,
  source           NVARCHAR(200)    NOT NULL,
  published_date   DATE             NOT NULL,
  published_time   TIME             NOT NULL,
  rss_source       NVARCHAR(100)    NOT NULL,
  keywords         NVARCHAR(MAX)    NULL
);


CREATE TABLE SkippedNews (
  ID               INT IDENTITY(1,1) PRIMARY KEY,
  title            NVARCHAR(500)    NOT NULL,
  url              NVARCHAR(1000)   NOT NULL,
  text_hash        NVARCHAR(64)     NOT NULL UNIQUE,
  summary          NVARCHAR(MAX)    NOT NULL,
  source           NVARCHAR(200)    NOT NULL,
  published_date   DATE             NOT NULL,
  published_time   TIME             NOT NULL,
  rss_source       NVARCHAR(100)    NOT NULL,
  reason           NVARCHAR(300)    NULL,
  fail_count       INT              NOT NULL
  );


CREATE TABLE Topics (
  topic_id             INT PRIMARY KEY,
  short_title          NVARCHAR(500)    NULL,
  trend_type           NVARCHAR(20)     NULL,
  num_articles         INT              NULL,
  num_days             INT              NULL,
  max_articles_per_day INT              NULL,
  first_date           DATE             NULL,
  dominant_day         DATE             NULL,
  last_date            DATE             NULL,
  main_country         NVARCHAR(100)    NULL,
  top_keywords         NVARCHAR(MAX)    NULL,
  spike_detected       BIT              NULL,
  growth_detected      BIT              NULL
);

CREATE TABLE dbo.Articles (
  article_id       INT           PRIMARY KEY,
  title            NVARCHAR(500) NULL,
  summary          NVARCHAR(MAX) NULL,
  url              NVARCHAR(1000)NULL,
  published_date   DATE          NULL,
  published_time   TIME          NULL,
  rss_source       NVARCHAR(100) NULL,
  cleaned_keywords NVARCHAR(MAX) NULL,
  topic_id         INT           NOT NULL,
  similar_count    INT           NULL,
  CONSTRAINT FK_Articles_PostedNews FOREIGN KEY(article_id)
      REFERENCES PostedNews(ID),
  CONSTRAINT FK_Articles_Topics     FOREIGN KEY(topic_id)
      REFERENCES Topics(topic_id)
);


CREATE TABLE Trends (
  topic_id             INT           PRIMARY KEY,
  short_title          NVARCHAR(500) NULL,
  trend_type           NVARCHAR(20)  NULL,
  num_articles         INT           NULL,
  num_days             INT           NULL,
  max_articles_per_day INT           NULL,
  first_date           DATE          NULL,
  dominant_day         DATE          NULL,
  last_date            DATE          NULL,
  main_country         NVARCHAR(100) NULL,
  top_keywords         NVARCHAR(MAX) NULL,
  spike_detected       BIT           NULL,
  growth_detected      BIT           NULL,
  representative_summary NVARCHAR(MAX) NULL,
  CONSTRAINT FK_Trends_Topics FOREIGN KEY(topic_id)
      REFERENCES Topics(topic_id)
);

--יצירת טבלה DIMDATE
CREATE TABLE DimDate (
  Date DATE PRIMARY KEY,
  Year INT,
  Month INT,
  MonthName NVARCHAR(20),
  MonthShort NVARCHAR(3),
  Quarter NVARCHAR(10),
  Day INT,
  WeekdayName NVARCHAR(20),
  WeekdayNum INT,
  YearMonth NVARCHAR(7)
);

-- הגדרת טווח תאריכים מתוך הנתונים
DECLARE @MinDate DATE = (SELECT MIN(published_date) FROM PostedNews);
DECLARE @MaxDate DATE = (SELECT MAX(published_date) FROM PostedNews);

-- יצירת לולאה עם כל התאריכים בטווח
WITH DateRange AS (
    SELECT @MinDate AS DateValue
    UNION ALL
    SELECT DATEADD(DAY, 1, DateValue)
    FROM DateRange
    WHERE DateValue < @MaxDate
)
INSERT INTO DimDate (Date, Year, Month, MonthName, MonthShort, Quarter, Day, WeekdayName, WeekdayNum, YearMonth)
SELECT
    DateValue,
    YEAR(DateValue),
    MONTH(DateValue),
    DATENAME(MONTH, DateValue),
    LEFT(DATENAME(MONTH, DateValue), 3),
    'Q' + CAST(DATEPART(QUARTER, DateValue) AS NVARCHAR),
    DAY(DateValue),
    DATENAME(WEEKDAY, DateValue),
    DATEPART(WEEKDAY, DateValue),
    FORMAT(DateValue, 'yyyy-MM')
FROM DateRange
OPTION (MAXRECURSION 10000);
