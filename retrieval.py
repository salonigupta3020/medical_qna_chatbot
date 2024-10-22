import pandas as pd
from sentence_transformers import SentenceTransformer, util

class Retrieval:
    def __init__(self, df):
        self.df = df
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(df['question'].tolist(), convert_to_tensor=True)

    def get_answer(self, user_question):
        user_embedding = self.model.encode(user_question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(user_embedding, self.embeddings)[0]
        top_result = cos_scores.argmax().item()
        return self.df.iloc[top_result]['answer']
