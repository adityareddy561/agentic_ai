from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

model = ChatOpenAI(model="gpt-4.1-nano")

# Define a prompt template

usr_input = ["Jhon  has won the best student in the class","Who is the best student in the class?"]

response = model.batch(usr_input)

print("Batch Response:")

for i, res in enumerate(response):
    print(f"Response {i+1}: {res.content}")  # print the content of each response
    print("--------------------------------")
    