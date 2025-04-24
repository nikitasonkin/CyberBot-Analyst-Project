# Cyber NewsBot – Automated News Pipeline

## Overview
`Cyber NewsBot` is a Python-based system that **retrieves, filters, summarizes, and loads cyber news articles** into an MS SQL Server database, fully automated and ready for professional data analysis.

The project demonstrates:
- Advanced use of Python (`pandas`, `sqlalchemy`)
- Working with real-world news data in JSON format
- Loading and managing data in **SQL Server** (SSMS)
- Clean, modular ETL design ready for expansion



## Features
- 🔎 Automated news retrieval from multiple sources (RSS, Google Alerts)
- 🧹 Duplicate filtering by title, URL, and summary hash (`text_hash`)
- ✂️ Smart summarization and keyword extraction
- 💾 Data is loaded into **two normalized SQL tables**: `PostedNews` (published) and `SkippedNews` (filtered/skipped)
- 🕵️‍♂️ Supports incremental loads (no duplicate keys), fully repeatable
- 📅 Designed for weekly/automated scheduled updates



## ETL Pipeline Steps

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
```TABLE
Column | Type | Description
title | NVARCHAR(500) | News title
url | NVARCHAR(1000) | Article URL
text_hash | NVARCHAR(64) PK | Hash of the article summary
summary | NVARCHAR(MAX) | Summarized text
source | NVARCHAR(200) | News source/domain
published_date | DATE | Publication date
published_time | TIME | Publication time
rss_source | NVARCHAR(100) | Country/source of RSS
keywords | NVARCHAR(MAX) | Comma-separated keywords
```

### Example: Database Validation
## Below is a sample screenshot from SSMS, showing loaded records in PostedNews:<br>
![image](https://github.com/user-attachments/assets/1fbe28fb-39f6-4c42-b626-121832c24b31)

## And a sample screenshot of SkippedNews:<br>
![image](https://github.com/user-attachments/assets/a238d7f5-b847-4638-9f94-3e0dbd2f588d)

