import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gemini import gemini_answer
import re

vectorizer = TfidfVectorizer()
df  = pd.read_csv("jibin.csv")
X = df['Value']

fact_vectors = vectorizer.fit_transform(X)

def get_best_matches(query, n):
    query_vector = vectorizer.transform([query])#convert the question asked to a vector
    similarities = cosine_similarity(query_vector, fact_vectors).flatten()#compare the question asks to the data about Jibin using cosine similarity model
    top_indices = np.argsort(similarities)[-n:][::-1]# get the  indexes for the top n best matches in order
    return [(df.iloc[i], similarities[i]) for i in top_indices]#return the data where the index is the in the top indices

user_question = input("What question do you want to ask Jibin?: ")
user_question = re.sub(r'\b(Jibin Im|Jibin|Jibins|jibin\'s)\b', 'Anonymous', user_question, flags=re.IGNORECASE)



best_responses = get_best_matches(user_question, 5)

best_response_str = ""
for i in best_responses:
    best_response_str += f"Category: {i[0]['Category']}, Detail: {i[0]['Detail']}, Value: {i[0]['Value']}\n"
    print(f"Category: {i[0]['Category']}, Detail: {i[0]['Detail']}, Value: {i[0]['Value']}")
print(gemini_answer(best_responses, user_question)) #print whatever gemini could summarize from the top 5 best matches
