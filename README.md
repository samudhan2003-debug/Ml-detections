import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# load dataset
data = pd.read_csv("dataset/news.csv")

# features and labels
X = data["text"]
y = data["label"]

# convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_vectorized = vectorizer.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model trained and saved!")
