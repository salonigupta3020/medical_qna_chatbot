import pandas as pd
from sentence_transformers import SentenceTransformer, util
from src.retrieval import Retrieval

class MedicalChatbot:
    def __init__(self, data_path):
        # Load and preprocess the dataset
        try:
            self.df = pd.read_json(data_path)
            print(f"Dataset loaded with columns: {self.df.columns}")

            # Ensure 'question' and 'answer' columns exist
            for column in ['question', 'answer']:
                if column not in self.df.columns:
                    raise KeyError(f"'{column}' column not found in the dataset")

            # Initialize the SentenceTransformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            # Encode questions into embeddings
            self.embeddings = self.model.encode(self.df['question'].tolist(), convert_to_tensor=True)

            # Initialize Retrieval with the loaded dataframe
            self.retrieval_model = Retrieval(self.df)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except KeyError as e:
            print(f"Error: {e}")
            raise
        except ValueError as e:
            print(f"Error loading JSON: {e}")
            raise

    def get_answer(self, query):
        # Encode the user query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calculate cosine similarities
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]

        # Check if there are any scores and find the best match
        if scores.size(0) == 0:
            return "Sorry, no answers could be found."

        # Get the index of the highest score
        best_match_idx = scores.argmax().item()  # Ensure it's an integer

        # Check if the index is valid
        if best_match_idx < 0 or best_match_idx >= len(self.df):
            return "Sorry, I couldn't find an answer."

        # Return the answer for the best matching question
        return self.df.iloc[best_match_idx]['answer']

    def ask(self, query):
        try:
            return self.get_answer(query)
        except Exception as e:
            return f"Sorry, I couldn't find an answer. Error: {e}"
