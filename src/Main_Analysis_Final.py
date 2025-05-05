# ===========================================================================================
# 🔹 1. Imports and Configuration
# ===========================================================================================
#  Data handling and analysis
import pandas as pd                            # ניתוח ועיבוד טבלאות נתונים
import numpy as np                             # חישובים מספריים ומטריצות
#  Visualization
import matplotlib.pyplot as plt                # ציור גרפים
import seaborn as sns                          # ויזואליזציה מתקדמת על בסיס matplotlib
from wordcloud import WordCloud                # יצירת ענני מילים
#  Machine Learning / NLP
from sklearn.feature_extraction.text import TfidfVectorizer  # המרת טקסטים למטריצת TF-IDF
from sklearn.feature_extraction.text import CountVectorizer  # המרת טקסטים למטריצת ספירה
from sklearn.cluster import KMeans                          # אלגוריתם לקיבוץ (Clustering)
from sentence_transformers import SentenceTransformer       # הפקת embeddings למשפטים
from sentence_transformers import util                      # חישוב cosine similarity בין embeddings
#  Database
from sqlalchemy import create_engine          # חיבור ושמירה למסדי נתונים (SQL)
#  Graph & Clustering
import networkx as nx                         # עבודה עם גרפים, מציאת רכיבים מחוברים

# ===========================================================================================
# 🔹 2. loading data
# ===========================================================================================
# הגדרות חיבור ל-SQL
server = 'NIKITA-PC'
database = 'CyberNewsBot'
driver = 'ODBC Driver 17 for SQL Server'

connection_string = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&Trusted_Connection=yes"
engine = create_engine(connection_string)

# שליפת טבלאות
posted_df = pd.read_sql("SELECT * FROM PostedNews", con=engine)
skipped_df = pd.read_sql("SELECT * FROM SkippedNews", con=engine)

# עמודות ID רץ
posted_df['ID'] = posted_df.index
skipped_df['ID'] = skipped_df.index
# ===========================================================================================
# 🔹 3. Initial review
# ===========================================================================================
print("============================================ סקירה ראשונית ===============================================")

print("=== POSTED DF SHAPE ===", posted_df.shape)
print("=== POSTED DUPLICATES ===", posted_df.duplicated().sum())
print("=== POSTED COLUMNS ===", posted_df.columns.tolist())
print("=== POSTED NULLS ===\n", posted_df.isnull().sum())
print("=== POSTED INFO ===")
posted_df.info()
print("=== POSTED DESCRIBE ===\n", posted_df.describe(include='all'))
print("\nמספר ערכים ייחודיים לכל עמודה:")
print(posted_df.nunique())
print("\n דגימה אקראית מ-posted_df:")
print(posted_df.sample(5))
print("\n פילוח כתבות לפי תאריך:")
print(posted_df['published_date'].value_counts().sort_index())
print("-----------------------------------------------------------------------------------")
print("=== SKIPPED DF SHAPE ===", skipped_df.shape)
print("=== SKIPPED DUPLICATES ===", skipped_df.duplicated().sum())
print("=== SKIPPED COLUMNS ===", skipped_df.columns.tolist())
print("=== SKIPPED NULLS ===\n", skipped_df.isnull().sum())
print("=== SKIPPED INFO ===")
skipped_df.info()
print("=== SKIPPED DESCRIBE ===\n", skipped_df.describe(include='all'))
print("\n מספר ערכים ייחודיים לכל עמודה:")
print(skipped_df.nunique())
print("\n דגימה אקראית מ-skipped_df:")
print(skipped_df.sample(5))
print("\n פילוח כתבות לפי תאריך:")
print(skipped_df['published_date'].value_counts().sort_index())

# ===========================================================================================
#  🔹4. Date conversion + auxiliary columns
# ===========================================================================================
print("============================================ המרת תאריכים + עמודות עזר ===============================================")

# המרת עמודות תאריך לזמן תקני (Datetime / Time)
posted_df['published_date'] = pd.to_datetime(posted_df['published_date'], errors='coerce')
posted_df['published_time'] = pd.to_datetime(posted_df['published_time'], format='%H:%M:%S', errors='coerce').dt.time

skipped_df['published_date'] = pd.to_datetime(skipped_df['published_date'], errors='coerce')
skipped_df['published_time'] = pd.to_datetime(skipped_df['published_time'], format='%H:%M:%S', errors='coerce').dt.time

# יצירת עמודות עזר
posted_df['day_of_week'] = posted_df['published_date'].dt.day_name()
posted_df['hour'] = pd.to_datetime(posted_df['published_time'], errors='coerce').apply(lambda x: x.hour if pd.notnull(x) else np.nan)

skipped_df['day_of_week'] = skipped_df['published_date'].dt.day_name()
skipped_df['hour'] = pd.to_datetime(skipped_df['published_time'], errors='coerce').apply(lambda x: x.hour if pd.notnull(x) else np.nan)

print(" המרת תאריכים והוספת עמודות עזר הושלמה.")

# ===========================================================================================
# 🔹 5. Statistical groupings and basic graphs
# ===========================================================================================
print("============================================ קיבוצים סטטיסטיים וגרפים בסיסיים ===============================================")

# סטופ וורדס וניקוי מילות מפתח
stopwords = {"the", "a", "an", "and", "or", "with", "by", "from", "after", "against", "company", "their", "its", "on",
             "for", "in", "of", "to", "is", "are", "was", "as", "this", "that", "these", "those", "about", "into",
             "platform", "portal", "manage", "known", "primarily", "better", "current", "systems"}

def clean_keywords(keyword_str):
    return [
        word.strip().lower()
        for word in str(keyword_str).split(',')
        if word.strip().lower() and word.strip().lower() not in stopwords
    ]
# יצירת עמודת cleaned_keywords
posted_df['cleaned_keywords'] = posted_df['keywords'].apply(clean_keywords)

# -------------------------------------------------------------------------------------------
# 🔹General statistics about the data
# -------------------------------------------------------------------------------------------

# ממוצע מילות מפתח לכתבה
avg_keywords_per_article = posted_df['cleaned_keywords'].apply(len).mean()
print(f" ממוצע מספר מילות מפתח לכתבה: {avg_keywords_per_article:.2f}")

# פירוק keywords לשורות
exploded = posted_df.explode('cleaned_keywords')

# הדפסת מילות מפתח שמופיעות לפחות 10 פעמים
print(" מילות מפתח פופולריות (לפחות 10 מופעים):")
for kw, count in exploded['cleaned_keywords'].value_counts().items():
    if count >= 10:
        print(f"{kw} - {count}")

# -------------------------------------------------------------------------------------------
# 🔹Statistical aggregations
# -------------------------------------------------------------------------------------------

# סך כתבות שפורסמו לפי תאריך
articles_per_day = posted_df.groupby('published_date').size()
# סך כתבות שנכשלו לפי תאריך
failures_per_day = skipped_df.groupby('published_date').size()
# סך כתבות פורסמות לפי מדינה
articles_per_country = posted_df['rss_source'].value_counts()
# סיבות כישלון לפי תאריך
failures_by_reason_date = skipped_df.groupby(['published_date', 'reason']).size().unstack(fill_value=0)

# -------------------------------------------------------------------------------------------
# 🔹Graph 1: Number of articles published by date
# -------------------------------------------------------------------------------------------

plt.figure(figsize=(14,7))
articles_per_day.plot(kind='line', marker='o')
plt.title('Number of Published Articles Per Day')
plt.xlabel('Published Date')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------
# 🔹Graph 2: Daily number of failures
# -------------------------------------------------------------------------------------------

plt.figure(figsize=(14,7))
failures_per_day.plot(kind='line', color='red', marker='o')
plt.title('Number of Failed Articles Per Day')
plt.xlabel('Published Date')
plt.ylabel('Number of Failed Articles')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------
# 🔹Graph 3: Segmentation of failure reasons
# -------------------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
skipped_df['reason'].value_counts().plot(kind='bar', color='tomato')
plt.title('Failure Reasons Distribution')
plt.xlabel('Reason')
plt.ylabel('Number of Failures')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------
# 🔹Graph 4: Segmentation of articles by country
# -------------------------------------------------------------------------------------------

plt.figure(figsize=(12,6))
articles_per_country.plot(kind='bar')
plt.title('Number of Published Articles by Country')
plt.xlabel('Country')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------
#  🔹Graph 5: Heatmap – Failure Reason by Date
# -------------------------------------------------------------------------------------------

failures_by_reason_date.index = failures_by_reason_date.index.strftime('%Y-%m-%d')
plt.figure(figsize=(14,7))
sns.heatmap(failures_by_reason_date, annot=True, fmt='d', cmap='YlOrBr')
plt.title('Failures by Reason and Date')
plt.xlabel('Failure Reason')
plt.ylabel('Published Date')
plt.tight_layout()
plt.show()

# 🟢 סיום שלב גרפים סטטיסטיים
print(" גרפים סטטיסטיים בסיסיים נוצרו בהצלחה.")

# ===========================================================================================
# ️🔹 Step 6: Identify and compare duplicates between published and rejected articles (title + summary hash)
# ===========================================================================================
print("============================================ זיהוי והשוואת כפילויות בין כתבות שפורסמו ונדחו ===============================================")

#  סופרים כמה פעמים הופיעו כותרים שהובילו לדחייה (Duplicate by title)
title_counts = skipped_df[skipped_df['reason'] == 'Duplicate by title'].groupby('title').size()

#  סופרים כמה פעמים הופיע תקציר זהה (Duplicate by summary hash)
hash_counts = skipped_df[skipped_df['reason'] == 'Duplicate by summary hash'].groupby('text_hash').size()

#  הוספת עמודות ל-posted_df: כמה כפילויות כל כתבה גרמה
posted_df['duplicated_titles_in_skipped'] = posted_df['title'].map(title_counts).fillna(0).astype(int)
posted_df['duplicated_hashes_in_skipped'] = posted_df['text_hash'].map(hash_counts).fillna(0).astype(int)

# חישוב כולל של כפילויות שכתבה אחת גרמה להן
posted_df['total_duplicates_caused'] = posted_df['duplicated_titles_in_skipped'] + posted_df['duplicated_hashes_in_skipped']

# -------------------------------------------------------------------------------------------
#  🔹Printing the most "noisy" articles that caused the most duplications
# -------------------------------------------------------------------------------------------

print("\n=== דוגמאות של כתבות עם הכי הרבה כפילויות בכותרת (Top 10) ===")
print(posted_df[['title', 'duplicated_titles_in_skipped']].sort_values('duplicated_titles_in_skipped', ascending=False).head(10))

print("\n=== דוגמאות של כתבות עם הכי הרבה כפילויות בתקציר (Top 10) ===")
print(posted_df[['title', 'duplicated_hashes_in_skipped']].sort_values('duplicated_hashes_in_skipped', ascending=False).head(10))

# -------------------------------------------------------------------------------------------
#  🔹Graph: General distribution of the number of duplicates by article
# -------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.hist(
    posted_df['total_duplicates_caused'],
    bins=range(0, posted_df['total_duplicates_caused'].max() + 2),
    edgecolor='black'
)
plt.title('Distribution of Total Duplicates Caused by Published Articles')
plt.xlabel('Number of Duplicated Articles Caused')
plt.ylabel('Number of Published Articles')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print(" חישוב כפילויות ויצירת גרף התפלגות בוצעו בהצלחה.")

# ===========================================================================================
# 🔹 Step 7: Identify similar articles using cosine similarity on title + summary
# ===========================================================================================
print("============================================ זיהוי כתבות דומות (cosine similarity) ===============================================")

# שילוב כותרת + תקציר לטקסט מלא לכל כתבה
posted_df['super_text'] = posted_df['title'].astype(str) + " " + posted_df['summary'].astype(str)

# המרת טקסט ל־embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2') #טעינת המודל זהו מודל מאומן מראש שיודע להמיר משפטים לווקטורים (vectors) שמייצגים את המשמעות של המשפט.
embeddings = model.encode(posted_df['super_text'].tolist(), show_progress_bar=True) # המרת טקסט ל־embeddings משתנה embeddings מכיל מערך בגודל (מספר כתבות, גודל embedding)

print("\n=== זיהוי כתבות דומות לפי cosine similarity ===")

cosine_sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy() #חישוב מטריצת דמיון קוסיני

# ===========================================================================================
#  🔹Threshold for filtering and mapping similar articles by similarity level > 0.8
# ===========================================================================================

threshold = 0.8 # זיהוי כתבות עם דמיון גבוה מ־0.8 (ללא זהות עצמית)
similar_articles_dict = {} #יוצרים מילון שיאחסן עבור כל כתבה את הרשימה של כתבות שדומות לה ברמת סף שנקבע.

for i in range(len(posted_df)):
    similar_indices = np.where((cosine_sim_matrix[i] >= threshold) & (cosine_sim_matrix[i] < 0.999))[0]
    if len(similar_indices) > 0:
        similar_articles_dict[i] = list(similar_indices)

# עמודת כתבות דומות
posted_df['similar_articles'] = posted_df.index.map(similar_articles_dict)
# עמודת מונה כתבות דומות
posted_df['similar_count'] = posted_df['similar_articles'].apply(lambda x: len(x) if isinstance(x, list) else 0)#אם x הוא רשימה → מחזירה את האורך שלה (כלומר: כמה כתבות דומות יש).

# ===========================================================================================
#  🔹View the top 10 most "repeated" articles by ideal similarity
# ===========================================================================================

print("\n כתבות עם הכי הרבה דמיון (cosine > 0.8):")
print(posted_df[['ID', 'title', 'similar_count']].sort_values('similar_count', ascending=False).head(10))

# גרף התפלגות: כמה כתבות דומות יש לכל כתבה
plt.figure(figsize=(10, 6))
plt.hist(posted_df['similar_count'], bins=range(0, posted_df['similar_count'].max() + 2), edgecolor='black')
plt.title('Distribution of Similar Articles per Article (cosine > 0.8)')
plt.xlabel('Number of Similar Articles')
plt.ylabel('Number of Articles')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print(" זיהוי כתבות דומות הושלם בהצלחה.")


# ===========================================================================================
# 🔹 Step 8: Building topic clusters (Topics) based on conceptual similarities between articles
# ===========================================================================================
print("============================================ בניית אשכולות נושא (Topics) על בסיס דמיון רעיוני בין כתבות ===============================================")

print(" בניית קבוצות נושא לפי cosine similarity...")

# -------------------------------------------------------------------------------------------
#  🔹1. Building a graph of connections between similar articles (undirected graph)
# -------------------------------------------------------------------------------------------

# גרף קשרים
G = nx.Graph() #יוצר גרף בלתי מכוון
for i, sims in posted_df['similar_articles'].dropna().items():#זוגות (i, sims), כאשר i הוא אינדקס וה־sims הם רשימת אינדקסים דומים. מסנן כתבות שאין להן דומות.
    G.add_node(i) # מוסיף את הכתבה כצומת בגרף
    for j in sims: # עבור כל כתבה דומה
        G.add_edge(i, j) # מוסיף קשת בין הכתבה הנוכחית לכתבה הדומה לה

# -------------------------------------------------------------------------------------------
#  🔹2. Identifying connected components (article clusters)
# -------------------------------------------------------------------------------------------

connected_groups = list(nx.connected_components(G)) # מחזיר רשימה של קבוצות מחוברות בגרף. כל קבוצה היא אוסף של צמתים (כתבות) שמחוברות זו לזו.

# מיפוי ל־topic_id
topic_mapping = {} #  כל כתבה → מספר אשכול
topic_article_ids = {}  # כל topic_id → רשימת כתבות באשכול

for idx, group in enumerate(connected_groups): #enumerate נותן מספר מזהה (idx) לכל קבוצת כתבות.
    for article_id in group: #group זה אוסף האינדקסים של הכתבות באותו אשכול.
        topic_mapping[article_id] = idx # מיפוי הכתבה למספר האשכול שלה
    topic_article_ids[idx] = list(group) # מיפוי מספר האשכול לכתבות שלו

posted_df['topic_id'] = posted_df['ID'].map(topic_mapping)# מיפוי לכל כתבה את מספר topic_id

# -------------------------------------------------------------------------------------------
#  🔹3. Create a topic table with at least 5 articles
# -------------------------------------------------------------------------------------------

# טבלת נושאים
topic_df = posted_df[posted_df['topic_id'].notnull()].copy()
topic_df['topic_id'] = topic_df['topic_id'].astype(int)

# נושאים עם לפחות 5 כתבות
topic_sizes = topic_df['topic_id'].value_counts()
large_topics = topic_sizes[topic_sizes >= 5].index.tolist()
print(f" נמצאו {len(large_topics)} נושאים עם לפחות 5 כתבות")

# -------------------------------------------------------------------------------------------
#  🔹4. Building a summary table for topics
# -------------------------------------------------------------------------------------------

topic_summary = pd.DataFrame({
    'topic_id': large_topics ,# רשימת נושאים עם לפחות 5 כתבות,
    'num_articles': [topic_sizes[t] for t in large_topics], # מספר כתבות בכל נושא
    'first_date': [topic_df[topic_df['topic_id'] == t]['published_date'].min().date() for t in large_topics], # תאריך ראשון מחפש את התאריך הכי מוקדם
    'last_date': [topic_df[topic_df['topic_id'] == t]['published_date'].max().date() for t in large_topics],    # תאריך אחרון מחפש את התאריך הכי מאוחר
    'main_country': [topic_df[topic_df['topic_id'] == t]['rss_source'].mode()[0] for t in large_topics], # מדינה עיקרית mode()[0] מחזיר את המדינה הכי שכיחה
    'article_ids': [topic_article_ids[t] for t in large_topics], # רשימת כתבות בכל נושא
})

# -------------------------------------------------------------------------------------------
#  🔹5. Printing examples from subject groups (up to 5)
# -------------------------------------------------------------------------------------------

print("\n=== דוגמאות לנושאים עם לפחות 5 כתבות ===")
for topic in large_topics:
    subset = topic_df[topic_df['topic_id'] == topic].sort_values('published_date') #subset = טבלה קטנה עם כל הכתבות של הנושא הזה.  מוציא את כל הכתבות ששייכות לנושא הזה (topic_id = topic) וממיין אותן לפי תאריך פרסום.
    print(f"\n🟨 Topic ID: {topic} - {len(subset)} כתבות") # מדפיס איזה נושא אתה מציג, וכמה כתבות יש בו.
    for _, row in subset.iterrows(): # עובר על כל שורה (כתבה) בתת-הטבלה subset.
        print(f"  - {row['published_date'].date()} | {row['rss_source']} | {row['title'][:70]}...")

# -------------------------------------------------------------------------------------------
#  🔹6. Graphs - Trend development by number of articles per day (including moving average)
# -------------------------------------------------------------------------------------------

for topic in large_topics:#עובר על כל נושא שמכיל לפחות 5 כתבות
    subset = topic_df[topic_df['topic_id'] == topic] # subset = טבלה קטנה עם כל הכתבות של הנושא הזה.
    topic_dates = subset['published_date'].value_counts().sort_index() # מונה את מספר הכתבות בכל תאריך וממיין לפי תאריך

    # תנאי סינון – רק אם יש לפחות 3 ימים שונים ו-5 כתבות לפחות
    if len(topic_dates) >= 3 and len(subset) >= 5: #האם יש לפחות שני תאריכים שונים
        ma = topic_dates.rolling(window=2).mean() # חישוב ממוצע נע של 2 ימים
        topic_dates.index = topic_dates.index.strftime('%Y-%m-%d')
        ma.index = ma.index.strftime('%Y-%m-%d')

        plt.figure(figsize=(10, 5))
        plt.plot(topic_dates.index, topic_dates.values, marker='o', label='Actual')
        plt.plot(ma.index, ma.values, linestyle='--', label='Moving Average (2 days)')
        plt.title(f"Trend for Topic ID {topic} - {len(topic_dates)} Days")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

print(" זיהוי אשכולות וניתוח התפתחות טרנדים הושלמו.")


# ===========================================================================================
# 🔹Step 9: Create summary tables – Topics, Articles, Trends
# ===========================================================================================
print("============================================ יצירת טבלאות מסכמות – Topics, Articles, Trends ===============================================")

# סיכום נושאים עם שדות תיאוריים
topics_df = topic_df.groupby('topic_id').agg(
    num_articles=('ID', 'count'), # מספר כתבות בכל נושא
    first_date=('published_date', 'min'), # תאריך ראשון
    last_date=('published_date', 'max'), # תאריך אחרון
    main_country=('rss_source', lambda x: x.mode()[0] if not x.mode().empty else None), # מדינה עיקרית
    top_keywords=('cleaned_keywords', lambda x: ', '.join(pd.Series([kw for kws in x for kw in kws]).value_counts().head(5).index)) , # 5 מילות מפתח פופולריות
).reset_index()

# short_title: הכותרת של הכתבה הכי מייצגת (הכי דומה לאחרות)
representative_titles = (
    topic_df.loc[topic_df.groupby('topic_id')['similar_count'].idxmax()] # מוצא לכל topic_id את הכתבה עם הכי הרבה כתבות דומות אליה
    .set_index('topic_id')['title']  #בונה סדרה: לכל topic_id הכותרת הכי מייצגת.
)
topics_df['short_title'] = topics_df['topic_id'].map(representative_titles) #מוסיף את הכותרת המייצגת לטבלת הנושאים.

# כמה ימים שונים הופיעו כתבות לכל נושא
topic_day_counts = topic_df.groupby('topic_id')['published_date'].nunique()
max_articles_per_day = (# מקסימום כתבות ביום אחד לכל נושא
    topic_df.groupby(['topic_id', 'published_date']).size()  # מונה את מספר הכתבות בכל נושא לפי תאריך
    .groupby('topic_id').max() # מקסימום כתבות ביום אחד לכל נושא
)

# תנאים לזיהוי טרנדים
spike_condition = max_articles_per_day >= 4 #נושא שהיה לו לפחות יום אחד עם 4 כתבות ומעלה
growth_condition = topic_day_counts >= 3 #נושא שהיה לו לפחות 3 ימים שונים עם כתבות

# מיפוי העמודות המחושבות
topics_df['num_days'] = topics_df['topic_id'].map(topic_day_counts) #בכמה ימים שונים הוא הופיע
topics_df['max_articles_per_day'] = topics_df['topic_id'].map(max_articles_per_day) #מקסימום כתבות ביום אחד לכל נושא
topics_df['spike_detected'] = topics_df['topic_id'].map(spike_condition) #נושא שהיה לו לפחות יום אחד עם 4 כתבות ומעלה
topics_df['growth_detected'] = topics_df['topic_id'].map(growth_condition) #נושא שהיה לו לפחות 3 ימים שונים עם כתבות

#  סיווג סוג הטרנד (Both / Spike / Growth / None)
def classify_trend(spike, growth):
    if spike and growth:
        return 'Both'
    elif spike:
        return 'Spike'
    elif growth:
        return 'Growth'
    else:
        return 'None'

topics_df['trend_type'] = topics_df.apply(
    lambda row: classify_trend(row['spike_detected'], row['growth_detected']),
    axis=1
)

#  dominant_day: היום שבו התרחש השיא עבור כל topic
dominant_day = (
    topic_df.groupby(['topic_id', 'published_date']).size() #סופר כמה כתבות היו בכל יום לכל נושא.
    .groupby('topic_id').idxmax() #מוצא את התאריך שבו היה הכי הרבה כתבות לכל נושא
    .apply(lambda x: x[1]) #לוקח רק את התאריך (x[1] כי זה טופל כ־tuple), וממיר לפורמט יפה YYYY-MM-DD.
)
topics_df['dominant_day'] = topics_df['topic_id'].map(dominant_day)

# סדר עמודות סופי
topics_df = topics_df[[
    'topic_id', 'short_title', 'trend_type',
    'num_articles', 'num_days', 'max_articles_per_day','first_date','dominant_day', 'last_date', 'main_country', 'top_keywords',
    'spike_detected', 'growth_detected'
]]

print("\n דוגמה לטבלת Topics:")
print(topics_df.head())

# -------------------------------------------------------------------------------------------
#  🔹2. Building an Articles table – articles with association to topics
# -------------------------------------------------------------------------------------------

# רק כתבות שמשויכות לנושאים
articles_df = topic_df[[
    'ID', 'title', 'summary', 'url',
    'published_date', 'published_time', 'rss_source',
    'cleaned_keywords', 'text_hash', 'topic_id', 'similar_count'
]].copy()

articles_df.rename(columns={'ID': 'article_id'}, inplace=True)# שינוי שם העמודה ל-article_id

# ניקוי cleaned_keywords לרשימה מופרדת בפסיקים
articles_df['cleaned_keywords'] = articles_df['cleaned_keywords'].apply(
    lambda kws: ', '.join(kws) if isinstance(kws, list) else ''
)

print("\n דוגמה לטבלת Articles:")
print(articles_df.head())

# -------------------------------------------------------------------------------------------
#  🔹3. Build a Trends table – only topics that are Spike / Growth / Both
# -------------------------------------------------------------------------------------------

trends_df = topics_df[topics_df['trend_type'] != 'None'].copy()
print("\n דוגמה לטבלת Trends:")
print(trends_df.head())

# -------------------------------------------------------------------------------------------
#  🔹4. Saving the tables to the database
# -------------------------------------------------------------------------------------------

topics_df.to_sql('Topics', con=engine, index=False, if_exists='replace')
articles_df.to_sql('Articles', con=engine, index=False, if_exists='replace')
trends_df.to_sql('Trends', con=engine, index=False, if_exists='replace')

print(f"\n✅ Topics saved: {len(topics_df)}")
print(f"✅ Articles saved: {len(articles_df)}")
print(f"✅ Trends saved: {len(trends_df)}")


# ===========================================================================================
# 🔹 10. Business visualization of trends
# ===========================================================================================
print("============================================ ויזואליזציה עסקית של טרנדים ===============================================")

# -------------------------------------------------------------------------------------------
# 🔹Graph 1: Columns – Top 10 trends by number of articles
# -------------------------------------------------------------------------------------------

top_trends = trends_df.sort_values(['num_articles', 'num_days'], ascending=False).head(10)
top_topic_ids = top_trends['topic_id'].tolist() #רשימת מזהי topic (topic_id) עבור הגרפים
id_to_title = dict(zip(top_trends['topic_id'], top_trends['short_title'])) #מילון שממפה מזהה לנושא קצר להצגה (short_title) שימושי בתוויות של הגרפים.

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_trends,
    x='short_title',
    y='num_articles',
    hue='short_title',
    palette='Blues_d',
    legend=False
)
plt.title('Top 10 Trending Topics by Number of Articles')
plt.xlabel('Topic')
plt.ylabel('Number of Articles')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------
# 🔹 Graph 2: Lines – daily development of 10 leading trends
# -------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 7))
for topic_id in top_topic_ids:
    short_title = id_to_title[topic_id]
    subset = topic_df[topic_df['topic_id'] == topic_id]
    counts = subset['published_date'].value_counts().sort_index()
    plt.plot(counts.index, counts.values, marker='o', label=short_title)

plt.title('Daily Article Count for Top Trending Topics')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.legend(title='Topics')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------
# 🔹Printing 10 sample articles from each leading trend
# -------------------------------------------------------------------------------------------
for topic_id in top_topic_ids: #עובר על כל topic_id שנמצא ברשימת top_topic_ids, שנבחרו קודם כ־Top 10 טרנדים.
    short_title = id_to_title[topic_id] #מוצא את הכותרת הקצרה של הנושא הזה (short_title) מתוך המילון שהכנו קודם.
    print(f"\n🔹 {short_title} (Topic ID: {topic_id})\n") #מדפיס כותרת ברורה למסך, כדי להבחין בין נושאים — כולל השם המקוצר של ה־topic וה־ID שלו.

    print(
        topic_df[topic_df['topic_id'] == topic_id][[ #מסנן רק את הכתבות ששייכות לנושא הנוכחי.
             'title','published_date','cleaned_keywords', 'rss_source' #בוחר רק את העמודות הרלוונטיות לתצוגה.
        ]].sort_values('published_date').head(10) #ממיין לפי תאריך.
    )

# -------------------------------------------------------------------------------------------
# 🔹Graph 3: WordCloud of keywords from trends
# -------------------------------------------------------------------------------------------
# חיבור כל מילות המפתח של טרנדים ל־String אחד
trend_keywords_series = trends_df['top_keywords'].dropna().astype(str)
text_blob = ' '.join(trend_keywords_series)

# יצירת WordCloud
wordcloud = WordCloud(width=1000, height=600, background_color='white', colormap='Blues').generate(text_blob)

# תצוגה
plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Keywords in Trending Topics (WordCloud)', fontsize=16)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------
# 🔹Graph 4: Horizontal bar graph – Top 5 trends with representative summary
# -------------------------------------------------------------------------------------------
top_topics = trends_df.sort_values('num_articles', ascending=False).head(5)['topic_id'] # 5 נושאים עם הכי הרבה כתבות
top_articles = articles_df[articles_df['topic_id'].isin(top_topics)] # בוחר את כל הכתבות ששייכות לאותם 5 טרנדים.

# מאחד מילות מפתח לפי נושא
topic_keywords = (
    top_articles.groupby('topic_id')['cleaned_keywords']
    .apply(lambda x: ' '.join(x))
)

# הופך את הטקסטים (לפי טרנד) ל־וקטורים של ספירת מילים (Bag of Words).
vec = CountVectorizer()
X = vec.fit_transform(topic_keywords)

# מייצר מטריצה של מילות מפתח לפי נושא
keyword_matrix = pd.DataFrame(X.toarray(), index=topic_keywords.index, columns=vec.get_feature_names_out())


# --- פונקציית קיצור תקציר לעד 7 שורות של כ-100 תווים ---
def wrap_summary(text, max_len=100, max_lines=7):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line + " " + word) <= max_len:
            line += " " + word
        else:
            lines.append(line.strip())
            line = word
            if len(lines) == max_lines:
                break
    if line and len(lines) < max_lines:
        lines.append(line.strip())
    return '\n'.join(lines)

# --- שלב 1: מיפוי התקציר הכי מייצג לכל טרנד לפי הכי הרבה similar_count ---
topic_summary_map = (
    articles_df
    .sort_values(['topic_id', 'similar_count'], ascending=[True, False])
    .drop_duplicates('topic_id')
    .set_index('topic_id')['summary']
)

# --- שלב 2: הוספת התקציר ל־trends_df ---
trends_df['representative_summary'] = trends_df['topic_id'].map(topic_summary_map)

# --- שלב 3: בחירת 5 הטרנדים הכי חזקים ---
top5 = trends_df.sort_values(by=['num_articles', 'num_days'], ascending=False).head(5).copy()

# --- שלב 4: קיצור התקציר לתצוגה בגרף ---
top5['wrapped_title'] = top5['representative_summary'].apply(wrap_summary)

# --- שלב 5: יצירת תווית טקסט עם כל המידע ---
top5['label'] = top5.apply(lambda row: (
    f"Trend Type: {row['trend_type']}\n"
    f"Articles: {row['num_articles']} | Days: {row['num_days']}\n"
    f"Max per Day: {row['max_articles_per_day']}\n"
    f"Dominant Day: {row['dominant_day']}\n"
    f"Country: {row['main_country']}\n"
    f"Keywords: {row['top_keywords']}"
), axis=1)

# --- שלב 6: ציור גרף אופקי מסודר ---
plt.figure(figsize=(14, 10))
bars = plt.barh(top5['wrapped_title'], top5['num_articles'], color='skyblue')

for bar, label in zip(bars, top5['label']):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             label, va='center', fontsize=9)

plt.title("Top 5 Cyber Trends – Based on Most Representative Article Summary", fontsize=16)
plt.xlabel("Number of Articles")
plt.ylabel("Representative Summary (trimmed)")
plt.grid(axis='x')
plt.tight_layout(rect=[0, 0, 0.95, 1])  # הרחבת השוליים ימינה
plt.show()

