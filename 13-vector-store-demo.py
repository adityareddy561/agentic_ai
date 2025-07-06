from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_postgres import PGVector 
load_dotenv()  # load environment variables from .env file

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"


model =OpenAIEmbeddings(
    model="text-embedding-3-small")

# Create blob from file
blob = Blob.from_path('senior_resume_gen_ai_fixed.pdf')

# Parse PDF into Documents
parser = PyPDFParser()
documents = list(parser.lazy_parse(blob))   # materialize generator!

# Inspect parsed docs
for doc in documents:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")

# Initialize splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# Split while preserving metadata
chunks = text_splitter.split_documents(documents)

# Inspect splits
for chunk in chunks:
    print(chunk.page_content)
    print("-----")
    print(chunk.metadata)
    print("-----")


vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=model,
    connection=connection,
    collection_name="resume_chunks"
)

print("vestor store")
print(vector_store)


# Query the vector store
query = "What is the candidate's experience with AI?"
results = vector_store.similarity_search(query, k=10)

print("Query results:")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(result.page_content)
    print("Metadata:", result.metadata)
    print("-----")