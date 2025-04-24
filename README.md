# Cyber NewsBot – Automated News Pipeline

## Overview
`Cyber NewsBot` is a Python-based ETL system that **retrieves, filters, summarizes, and loads cyber news articles** into an MS SQL Server database, ready for professional data analysis.

The project demonstrates:
- Advanced use of Python (`pandas`, `sqlalchemy`)
- Automated ETL best practices
- Professional handling of news data (JSON → SQL)
- Modular, production-ready code for scaling/automation


## Features
- 🔎 **Automated news retrieval** from RSS/Google Alerts
- 🧹 **Duplicate filtering** (title, URL, `text_hash`)
- ✂️ **Smart summarization** & keyword extraction
- 💾 **Normalized SQL Server storage** (`PostedNews`, `SkippedNews`)
- ♻️ **Incremental load** (no duplicate keys)
- ⏲️ **Ready for scheduled automation (Task Scheduler, Airflow, etc.)**



## ETL Pipeline Steps
The main ETL logic is implemented in [`src/to_sql.py`](src/to_sql.py).

To run the script, execute:
```bash
python src/to_sql.py
```

### 1. **Extract** – Collect news data from multiple JSON files
```python
import pandas as pd
posted_df = pd.read_json('posted_news_ud.json')
skipped_df = pd.read_json('skipped_news_ud.json')
```

### 2.Transform – Filter out already-loaded records, clean and format data
```python
# Convert keywords to comma-separated string
posted_df['keywords'] = posted_df['keywords'].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))

# Remove records with duplicate text_hash
new_posted_df = posted_df[~posted_df['text_hash'].isin(existing_posted_hashes)].drop_duplicates(subset=['text_hash'])
```

### 3. Load – Insert into SQL Server tables using SQLAlchemy
```python
new_posted_df.to_sql(
    "PostedNews",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=1000
)
```
### Database Structure

#### PostedNews

| Column          | Type            | Description                |
|-----------------|-----------------|----------------------------|
| title           | NVARCHAR(500)   | News title                 |
| url             | NVARCHAR(1000)  | Article URL                |
| text_hash       | NVARCHAR(64) PK | Hash of the article summary|
| summary         | NVARCHAR(MAX)   | Summarized text            |
| source          | NVARCHAR(200)   | News source/domain         |
| published_date  | DATE            | Publication date           |
| published_time  | TIME            | Publication time           |
| rss_source      | NVARCHAR(100)   | Country/source of RSS      |
| keywords        | NVARCHAR(MAX)   | Comma-separated keywords   |

#### SkippedNews

| Column          | Type            | Description                        |
|-----------------|-----------------|------------------------------------|
| title           | NVARCHAR(500)   | News title                         |
| url             | NVARCHAR(1000)  | Article URL                        |
| text_hash       | NVARCHAR(64) PK | Hash of the article summary        |
| summary         | NVARCHAR(MAX)   | Summarized text                    |
| source          | NVARCHAR(200)   | News source/domain                 |
| published_date  | DATE            | Publication date                   |
| published_time  | TIME            | Publication time                   |
| rss_source      | NVARCHAR(100)   | Country/source of RSS              |
| reason          | NVARCHAR(300)   | Reason skipped (e.g., duplicate)   |
| fail_count      | INT             | Number of failed attempts          |
| date            | DATE            | Last processed date                |


### Example: Database Validation
## Below is a sample screenshot from SSMS, showing loaded records in PostedNews:<br>
![image](https://github.com/user-attachments/assets/1fbe28fb-39f6-4c42-b626-121832c24b31)

## And a sample screenshot of SkippedNews:<br>
![image](https://github.com/user-attachments/assets/a238d7f5-b847-4638-9f94-3e0dbd2f588d)

## Database Schema (SQL)

Below is the script used to create the tables in SQL Server:

```sql
-- PostedNews Table
CREATE TABLE PostedNews (
    title NVARCHAR(500)         NOT NULL,
    url NVARCHAR(1000)          NOT NULL,
    text_hash NVARCHAR(64)      NOT NULL PRIMARY KEY,
    summary NVARCHAR(MAX)       NOT NULL,
    source NVARCHAR(200)        NOT NULL,
    published_date DATE         NOT NULL,
    published_time TIME         NOT NULL,
    rss_source NVARCHAR(100)    NOT NULL,
    keywords NVARCHAR(MAX)      NULL
);

-- SkippedNews Table
CREATE TABLE SkippedNews (
    title NVARCHAR(500)         NOT NULL,
    url NVARCHAR(1000)          NOT NULL,
    text_hash NVARCHAR(64)      NOT NULL PRIMARY KEY,
    summary NVARCHAR(MAX)       NOT NULL,
    source NVARCHAR(200)        NOT NULL,
    published_date DATE         NOT NULL,
    published_time TIME         NOT NULL,
    rss_source NVARCHAR(100)    NOT NULL,
    reason NVARCHAR(300)        NULL,
    fail_count INT              NOT NULL,
    date DATE                   NOT NULL
);
```


### Future Improvements
`-Automate ETL run using Task Scheduler / Airflow`
` -Add EDA & visualization (matplotlib, seaborn)`
` -Build a simple dashboard/report`
` -Add automatic logging and email summary`


### Author
-Created by Nikita Sonkin.

-Project repository: [CyberNewsBot](https://github.com/nikitasonkin/CyberNewsBot) on GitHub
