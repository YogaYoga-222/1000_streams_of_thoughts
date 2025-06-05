import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean the CSV
df = pd.read_csv("/home/stemland/Downloads/stream_of_thoughts.csv")

# Combine important columns into one text column
columns = ['Raw', 'Text', 'Text.1', 'Unnamed: 4', 'Unnamed: 5']
df['Raw Text'] = df[columns].astype(str).agg(' '.join, axis=1).str.strip()

# Keep only the 'Raw Text' column and clean rows
df = df[['Raw Text']].dropna()
df = df[df['Raw Text'].str.lower() != 'nan']
df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows

# Split into training and testing
train_data = df[:100]
test_data = df[100:].sample(n=10, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['Raw Text'])
X_test = vectorizer.transform(test_data['Raw Text'])

# Compare test thoughts with training thoughts
similarities = cosine_similarity(X_test, X_train)

known = 0
for i, row in enumerate(similarities):
    test_thought = test_data['Raw Text'].iloc[i]
    most_similar = train_data['Raw Text'].iloc[row.argmax()]
    score = row.max()

    print(f"\nTest Thought {i+1}: {test_thought}")
    print(f"Most similar training thought: {most_similar}")
    print(f"Similarity Score: {score:.2f}")

    if score > 0.7:
        print("This is a known or similar thought.")
        known += 1
    else:
        print("This might be a new or different thought.")

# Summary
print(f"\n Known thoughts: {known}")
print(f" New or different thoughts: {len(test_data) - known}")
