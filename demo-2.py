#2 - demo-2.py
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv() # load environment variables from .env file

# initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-4.1-nano")

sys_message = SystemMessage(
    content="""You are a helpful assistant that answers questions about current events and general knowledge for the children in the age group 8 to 12 years.
    Ensure that your responses are simple, clear, and age-appropriate.
    If you don't know the answer, say 'I don't know' instead of making up an answer.
    Always provide accurate and factual information.""")

human_text = HumanMessage("Tell me about large language models and how they work.")
# response
response = model.invoke([sys_message, human_text])
print(response)  # print the response from the model
print("--------------------------------")
print(response.content)  # print the content of the response