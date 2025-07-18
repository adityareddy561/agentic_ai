from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
split_docs = text_splitter.split_documents(documents)

# Inspect splits
for chunk in split_docs:
    print(chunk.page_content)
    print("-----")
    print(chunk.metadata)
    print("-----")
