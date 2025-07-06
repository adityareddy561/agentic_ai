from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob


blob = Blob.from_path('senior_resume_gen_ai_fixed.pdf')

parser = PyPDFParser()

documents =parser.lazy_parse(blob)

for doc in documents:
    print(doc.page_content)
    print(doc.metadata)
    print("-----")
