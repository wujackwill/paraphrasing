from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random

original_text = "我是一名大学生，我喜欢学习，我喜欢读书，我喜欢运动，我喜欢旅游。"

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([original_text])

tfidf_array = tfidf_matrix.toarray()[0]

words = np.array(tfidf_vectorizer.get_feature_names_out())

sorted_indices = np.argsort(tfidf_array)[::-1]

top_n = int(len(sorted_indices) * 0.5)
selected_features = words[sorted_indices][:top_n]

punctuations = [char for char in original_text if char in "，。！？；"]

random_punctuation = random.choice(punctuations)

new_text = ' '.join(selected_features)

new_text += random_punctuation

print("original text:", original_text)
print("new text:", new_text)

