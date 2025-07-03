from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample job descriptions
job_descriptions = [
"We are looking for a data analyst with experience in Python, Excel, and SQL.",
"Frontend developer needed with React, JavaScript, and CSS skills.",
"Backend developer with experience in Django, REST APIs, and PostgreSQL.",
"QA engineer with knowledge of test automation using Selenium and Python."
]

# Extract top keywords using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
X = vectorizer.fit_transform(job_descriptions)

# Convert to DataFrame
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df.index = [f"Job {i+1}" for i in range(len(job_descriptions))]

# Show results
print("Top keywords for each job:")
print(df)
