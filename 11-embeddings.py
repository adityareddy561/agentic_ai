from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings


load_dotenv()  # load environment variables from .env file

model =OpenAIEmbeddings(
    model="text-embedding-3-small")

# Example text to embed

text =["This is a test sentence for embedding.",
       "Another sentence to be embedded.",
       "Embedding is a way to represent text in a vector space."
       "This is a different sentence for embedding."
       "Yet another example of text to embed."
       "Embedding can capture semantic meaning of text."
       "This is a short sentence.",
       "A longer sentence that contains more information and context for embedding purposes.",
       "Embedding is useful for various NLP tasks.",
       "This sentence is specifically designed to test the embedding capabilities of the model."]

embeddings = model.embed_documents(text)

print("Embeddings:")
for i, emb in enumerate(embeddings):
    print(f"Total dimensions: {len(emb)}",emb[:10])  # Print first 10 dimensions for brevity
