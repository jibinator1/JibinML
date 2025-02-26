import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gemini import gemini_answer

vectorizer = TfidfVectorizer()
X  = pd.DataFrame("x")
fact_vectors = vectorizer.fit_transform(X)

def get_best_matches(query, n):
    query_vector = vectorizer.transform([query])#convert the question asked to a vector
    similarities = cosine_similarity(query_vector, fact_vectors).flatten()#compare the question asks to the data about Jibin using cosine similarity model
    top_indices = np.argsort(similarities)[-n:][::-1]# get the  indexes for the top n best matches in order
    return [(X[i], similarities[i]) for i in top_indices]#return the data where the index is the in the top indices

user_question = input("What question do you want to ask Jibin?")

best_responses = get_best_matches(user_question, 5)

best_response = ""
for i in best_responses:
    best_response += f"{i[0]}, {i[1]:.4f}, " 
print(gemini_answer(best_responses, user_question)) #print whatever gemini could summarize from the top 5 best matches