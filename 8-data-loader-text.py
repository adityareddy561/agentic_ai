from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
load_dotenv()  # load environment variables from .env file
# initialize the OpenAI chat model
