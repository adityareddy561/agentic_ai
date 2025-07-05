from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI

load_dotenv() # load environment variables from .env file

# initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-4.1-nano")

# response
response = model.invoke("What is the capital of India?")
print(response)  # print the response from the model
print("--------------------------------")
print(response.content)  # print the content of the response
print("--------------------------------")
response = model.invoke("Who won f1 race in June 2025?")
print(response)  # print the response from the model
print("--------------------------------")
print(response.content)  # print the content of the response
print("--------------------------------")