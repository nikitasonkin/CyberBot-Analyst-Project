# Project Overview
Analysis of data from the Cyber News Bot.
This repository contains Jupyter Notebooks that demonstrate how to process and analyze data efficiently. Below is a summary and explanation of each file in the repository.

### 📄 File: Notebook: 01-datatosql.ipynb
This Jupyter Notebook demonstrates how to load JSON data into a SQL Server database using Python libraries like pandas and SQLAlchemy. It is designed to process data from the Cyber News Bot and perform the following tasks:

🔑 Key Features
1️⃣ Database Connection Setup
Configures a connection to SQL Server using SQLAlchemy with ODBC Driver 17.
Ensures secure and efficient data transfer to the database.
2️⃣ Loading PostedNews JSON Data
📂 Reads JSON files containing posted news articles (posted_news_ud.json).
🔄 Converts the keywords column into a string format.
🔍 Checks for duplicates in the database using the text_hash column.
✅ Loads only new articles into the PostedNews table.
3️⃣ Loading SkippedNews JSON Data
📂 Reads skipped articles from skipped_news_ud.json.
🔍 Identifies and skips duplicate articles using the text_hash column.
✅ Populates the SkippedNews table with unique entries.
4️⃣ Data Quality Checks
👁️ Displays examples of skipped articles for review.
🧐 Performs a missing value check to ensure data integrity in the SkippedNews table.
📊 Outputs
Successfully loaded articles are printed with summary messages:
✅ Number of new articles loaded into PostedNews.
✅ Number of new articles added to SkippedNews.
🗒️ Logs any skipped or duplicate entries detected during the process.
📦 Dependencies
🐍 Python Libraries: pandas, SQLAlchemy, pyodbc, json.
🛠️ A running SQL Server instance with tables PostedNews and SkippedNews predefined.
⚙️ Usage
To run the notebook, ensure that:

📁 The JSON files (posted_news_ud.json, skipped_news_ud.json) are placed in the working directory.
⚙️ The SQL Server connection settings (e.g., server, database, driver) are correctly configured.
✅ All required Python packages are installed.


### 📄 File: `notebooks/Main_Analysis_Fina.ipynb`

This Jupyter Notebook represents the core analytical and AI-driven operations for processing and deriving insights from the data collected by the Cyber News Bot. The notebook is designed to perform comprehensive data cleaning, analysis, and visualization while also leveraging machine learning and natural language processing (NLP) techniques to extract deeper insights.

---

#### 🔹 Key Objectives

1. **Data Retrieval and Processing**:
   - Establish a connection to the SQL database to fetch two main datasets:
     - `PostedNews`: Articles successfully published by the bot.
     - `SkippedNews`: Articles that failed and were not published.
   - Conduct initial exploratory data analysis (EDA) to cleanse and understand the structure of both datasets.

2. **Data Cleaning and Preparation**:
   - Standardize date and time fields for temporal analysis.
   - Remove duplicates and handle null values.
   - Generate auxiliary columns (e.g., day of the week, publication hour) to support further analysis.

3. **Embedding and Similarity Analysis**:
   - Use **Sentence Transformers** to create embeddings for article summaries:
     - Embeddings are numerical vector representations of text, enabling semantic understanding.
   - Perform **cosine similarity** calculations between embeddings:
     - Identify relationships between articles based on their content similarity.
     - Useful for clustering articles with similar topics or detecting duplicate or near-duplicate content.

4. **Keyword Analysis**:
   - Extract and clean keywords from articles, removing stopwords for clarity.
   - Aggregate and visualize the most frequently occurring keywords.
   - Generate a cleaned version of the keywords for improved relevance.

5. **Statistical Insights and Visualizations**:
   - Analyze trends in article publication and failures:
     - Daily trends for both published and skipped articles.
     - Breakdown of failure reasons (e.g., duplicates, invalid content).
   - Identify popular RSS sources contributing to the dataset.
   - Visualize the dataset using line graphs, pie charts, and comparison plots.

6. **Clustering and Machine Learning**:
   - Use **KMeans Clustering** to group articles based on their embeddings:
     - Helps categorize articles into thematic clusters.
   - Leverage TF-IDF (Term Frequency-Inverse Document Frequency) and Count Vectorization techniques to analyze textual patterns and relationships.

7. **SQL Integration and Data Management**:
   - Seamless integration with a SQL Server database using the `sqlalchemy` library.
   - Retrieve structured datasets directly into Pandas DataFrames.
   - Save processed or cleaned data back to the SQL database by creating new tables:
     - Example: Tables summarizing daily trends or keyword analyses can be written back to SQL for persistence.

---

#### 🔹 Key Features and Analytical Steps

1. **Data Retrieval**:
   - The notebook connects to a SQL Server database using the `sqlalchemy` library.
   - Retrieves structured datasets (`PostedNews` and `SkippedNews`) for analysis.

2. **Exploratory Data Analysis (EDA)**:
   - Perform an initial review of both datasets:
     - Summarize key statistics (e.g., shape, null counts, duplicates).
     - Examine column types and unique values.
   - Generate random samples to inspect the quality and structure of the data.

3. **Embeddings and Similarity**:
   - **Embedding Creation**:
     - Use the `SentenceTransformer` model to compute embeddings for article summaries.
     - These embeddings encode the semantic meaning of text into high-dimensional vectors.
   - **Cosine Similarity**:
     - Measure the similarity between articles by calculating the cosine angle between their embeddings.
     - Applications include identifying duplicate articles, grouping similar topics, and detecting content overlap.

4. **Keyword Analysis**:
   - Analyze the `keywords` field for each article:
     - Clean and normalize keywords by removing common stopwords.
     - Explode keywords into individual rows for frequency analysis.
   - Generate a ranked list of the most frequently occurring keywords and visualize their distribution.

5. **Data Visualization**:
   - Create detailed graphs and charts:
     - Line graph comparing published versus failed articles over time.
     - Pie chart highlighting the share of top keywords.
     - Daily breakdown of article submissions and failures.

6. **Clustering and Text Analysis**:
   - Use clustering techniques to group articles into thematic clusters:
     - Apply KMeans clustering on embedding vectors to identify content categories.
   - Employ TF-IDF and Count Vectorization to identify significant terms and patterns in article text.

7. **Failure Analysis**:
   - Analyze the reasons for article failures:
     - Categorize failures (e.g., duplicate titles, invalid content).
     - Visualize failure trends over time and by RSS source.

---

#### 🔹 Outputs and Example Results

1. **Top Keywords**:
   ```plaintext
      keyword       count
        cyber         941
     security         354
  cybersecurity         289
       attack         115
        marks          84
  ```

2. **Daily Article Breakdown**:
   ```plaintext
   Daily breakdown of articles sent vs. failed:
                  Sent  Skipped
   2025-04-23       40        0
   2025-04-24      132        0
   2025-04-25      118        0
   ```

3. **Similarity Insights**:
   - Articles with high cosine similarity are flagged as potential duplicates or related content.
   - Example:
     ```plaintext
     Similar Articles:
     Article 1: "Cybersecurity Trends in 2025"
     Article 2: "Emerging Cyber Threats This Year"
     Similarity: 0.92
     ```

4. **Graphs**:
   - **Published vs. Failed Articles Per Day**:
     - Line graph showing the trends of successful and failed submissions.
   - **Top 8 Keywords by Frequency Share**:
     - Pie chart visualizing the most frequently occurring keywords.

---

#### 🔹 How to Use

1. **Setup**:
   - Ensure all dependencies are installed:
     - Key libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `sqlalchemy`, `networkx`, `wordcloud`, `sentence-transformers`.
   - Configure SQL Server connection settings:
     - Update `server`, `database`, and `driver` variables to match your environment.

2. **Execution**:
   - Open the notebook in Jupyter or JupyterLab.
   - Execute cells sequentially to reproduce the analysis and visualizations.

3. **Customization**:
   - Modify clustering parameters (e.g., number of clusters) to suit your analysis needs.
   - Update stopword lists or keyword cleaning logic to refine keyword analysis.

---

This notebook combines robust data processing, advanced machine learning, and intuitive visualizations to deliver actionable insights into the performance and content of articles processed by the Cyber News Bot. It leverages state-of-the-art embedding techniques, cosine similarity analysis, and clustering to uncover hidden patterns and relationships within the data while maintaining seamless integration with SQL for efficient data management.

### Author
-Created by Nikita Sonkin.

-Project repository: [CyberNewsBot](https://github.com/nikitasonkin/CyberNewsBot) on GitHub
