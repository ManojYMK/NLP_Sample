# nlp_toxicity_baseline.py
# Minimal baseline for toxicity classification using scikit-learn
# Author: Manoj Kumar Yasangi

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Tiny synthetic dataset
X = [
    "I hope you have a great day",
    "You are terrible and no one likes you",
    "Thanks for sharing your thoughts",
    "This is hateful and disgusting",
    "I appreciate your perspective",
    "Go away, nobody wants you here",
]
y = [0, 1, 0, 1, 0, 1]  # 1 = toxic, 0 = non-toxic

clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
clf.fit(X, y)

test_examples = [
    "I disagree with you",
    "You are awful",
    "Thank you for the info",
    "Your comment is disgusting"
]
pred = clf.predict(test_examples)

for text, label in zip(test_examples, pred):
    tag = "Toxic" if label == 1 else "Non-toxic"
    print(f"{tag} → {text}")

