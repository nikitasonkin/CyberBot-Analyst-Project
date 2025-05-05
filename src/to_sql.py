import pandas as pd
from sqlalchemy import create_engine
import json
import pyodbc

server = ''
database = ''
driver = ''

connection_string = (
    f"mssql+pyodbc://@{server}/{database}"
    f"?driver={driver.replace(' ', '+')}"
    f"&Trusted_Connection=yes"
)
engine = create_engine(connection_string)

# --- POSTED_NEWS -----------------------------------------------------


posted_df = pd.read_json(r"posted_news_ud.json")
if 'keywords' in posted_df.columns:
    posted_df['keywords'] = posted_df['keywords'].apply(
        lambda x: ','.join(x) if isinstance(x, list) else str(x)
    )
existing_posted = pd.read_sql("SELECT text_hash FROM PostedNews", con=engine)
existing_posted_hashes = set(existing_posted['text_hash'])
new_posted_df = posted_df[~posted_df['text_hash'].isin(existing_posted_hashes)]

if not new_posted_df.empty:
    new_posted_df.to_sql(
        "PostedNews",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000
    )
    print(f"✅ Loaded {len(new_posted_df)} new articles to PostedNews.")
else:
    print("No additional articles for PostedNews.")


with open(r"skipped_news_ud.json", encoding='utf-8') as f:
    skipped_dict = json.load(f)
skipped_values = list(skipped_dict.values())
skipped_df = pd.DataFrame(skipped_values)

existing_skipped = pd.read_sql("SELECT text_hash FROM SkippedNews", con=engine)
existing_skipped_hashes = set(existing_skipped['text_hash'])


new_skipped_df = skipped_df[~skipped_df['text_hash'].isin(existing_skipped_hashes)]
new_skipped_df = new_skipped_df.drop_duplicates(subset=['text_hash'])

if not new_skipped_df.empty:
    new_skipped_df.to_sql(
        "SkippedNews",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000
    )
    print(f"✅ loaded {len(new_skipped_df)} new articles to SkippedNews.")
else:
    print("No additional articles for SkippedNews.")


# בדיקות
print("\n--- Check DataFrame SkippedNews ---")
print(new_skipped_df.dtypes)
print(new_skipped_df.isnull().sum())
print(new_skipped_df.head(2))

# בדיקות
print("\n--- בדיקת DataFrame SkippedNews ---")
print(new_skipped_df.dtypes)
print(new_skipped_df.isnull().sum())
print(new_skipped_df.head(2))
