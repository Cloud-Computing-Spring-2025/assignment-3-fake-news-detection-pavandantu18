# Assignment-5: Fake News Detection with Spark MLlib

## 🚀 Project Overview
This project demonstrates a complete Spark MLlib pipeline to detect “FAKE” vs. “REAL” news articles. You will:

1. Ingest and explore a CSV dataset of news articles.
2. Preprocess text (lowercasing, tokenization, stop-word removal).
3. Extract TF–IDF features.
4. Train and evaluate a logistic regression classifier.
5. Save intermediate outputs and final metrics.

## 🎯 Goals
- Build a scalable text-classification pipeline in PySpark.  
- Understand common NLP preprocessing steps.  
- Learn how to extract TF–IDF features at scale.  
- Practice training and evaluating MLlib models.

---

## 📂 Repository Structure
├── fake_news_sample.csv # Raw dataset
├── fakenews.py # Main PySpark script
├── requirements.txt # Python dependencies
├── README.md # This file
└── outputs/ # Generated CSVs and logs after running
├── task1_output.csv
├── task2_output.csv
├── task3_output.csv
├── task4_output.csv
└── task5_output.csv


---

## 📋 Dataset Details
- **Filename**: `fake_news_sample.csv`  
- **Columns**:
  - `id` (integer) — Unique article identifier  
  - `title` (string) — Headline of the article  
  - `text` (string) — Full content  
  - `label` (string) — Ground truth (`FAKE` or `REAL`)

---

## ⚙️ Prerequisites & Setup

1. **Install Python 3.7+**  

2.
```bash
pip install pyspark 
pip install faker 
spark submit --version 

Run the script by 
spark-submit fakenews.py
```

## Project Tasks

### Task 1: Load & Basic Exploration
1. **Read the CSV into a Spark DataFrame**  
   - Use `spark.read.csv(...)` with `header=True` and `inferSchema=True` to automatically detect column types.  
   - This gives you a distributed DataFrame (`news_df`) containing all articles.

2. **Create a Temporary View for SQL Queries**  
   - Call `news_df.createOrReplaceTempView("news_data")`.  
   - Enables ad-hoc exploration using Spark SQL (`spark.sql("SELECT ... FROM news_data")`).

3. **Inspect the Data**  
   - **Show first 5 rows** (`news_df.show(5)`) to verify schema and sample content.  
   - **Count total articles** (`news_df.count()`) to confirm dataset size.  
   - **List distinct labels** (`news_df.select("label").distinct().show()`) to see how many “FAKE” vs. “REAL” entries exist.

4. **Save a Small Sample**  
   - Limit the DataFrame to 5 rows (`news_df.limit(5)`) and convert to Pandas.  
   - Write to `task1_output.csv` for quick sanity-checks or sharing with others.

---

### Task 2: Text Preprocessing
1. **Normalize Text**  
   - Convert the entire `text` column to lowercase (`.withColumn("text", lower(col("text")))`) to eliminate case variations.

2. **Tokenization**  
   - Instantiate `Tokenizer(inputCol="text", outputCol="words")`.  
   - Splits each article’s body into an array of word tokens, stored in `words`.

3. **Stop-Word Removal**  
   - Use `StopWordsRemover(inputCol="words", outputCol="filtered_words")`.  
   - Filters out common English stop-words (“the”, “and”, etc.) that carry little semantic meaning.

4. **Select Relevant Columns**  
   - Keep only `id`, `title`, `filtered_words`, and `label` for downstream tasks.  
   - Write this cleaned, tokenized view to `task2_output.csv`.

---

### Task 3: Feature Extraction
1. **Term Frequency via HashingTF**  
   - Apply `HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)`.  
   - Maps each word into a fixed-length feature vector (using the hashing trick) representing raw term frequencies.

2. **Inverse Document Frequency (IDF)**  
   - Fit an `IDF` model on the raw features to down-weight very common terms and up-weight rare but informative terms.  
   - Transform `rawFeatures` into final `features` vectors (TF–IDF).

3. **Label Indexing**  
   - Use `StringIndexer(inputCol="label", outputCol="label_index")` to convert “FAKE”→0.0 and “REAL”→1.0.  
   - Ensures the label is in numeric form for the classifier.

4. **Prepare Output**  
   - Select `id`, `filtered_words`, `features`, and `label_index`.  
   - Save to `task3_output.csv` for potential reuse or inspection.

---

### Task 4: Model Training
1. **Train/Test Split**  
   - Randomly split the feature DataFrame into **80%** training and **20%** test sets using a fixed seed for reproducibility.

2. **Train Logistic Regression**  
   - Initialize `LogisticRegression(featuresCol="features", labelCol="label_index")`.  
   - Fit the model on the training set.  

3. **Make Predictions**  
   - Apply the trained model to the test set to produce a `prediction` column alongside `label_index`.

4. **Attach Article Titles**  
   - Join predictions back with the `title` DataFrame (on `id`) so you can see human-readable article titles in the output.

5. **Save Predictions**  
   - Export `id`, `title`, `label_index` (true), and `prediction` to `task4_output.csv`.

---

### Task 5: Model Evaluation
1. **Select Evaluation Metrics**  
   - **Accuracy**: overall fraction of correct predictions.  
   - **F1 Score**: harmonic mean of precision and recall, balances false positives vs. false negatives.

2. **Compute Metrics**  
   - Use two `MulticlassClassificationEvaluator` instances (one for `"accuracy"`, one for `"f1"`).  
   - Evaluate against the `prediction` and `label_index` columns in the predictions DataFrame.

3. **Save & Display Results**  
   - Create a small Pandas DataFrame with columns `Metric` & `Value`.  
   - Write it out to `task5_output.csv` and print to console for a quick summary:

