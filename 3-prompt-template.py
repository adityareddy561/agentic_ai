from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv() # load environment variables from .env file

# initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-4.1-nano")

# Define a prompt template
template = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on the provided context.if you cant find the answer in the context, say 'I don't know'.
    Context: {context}
    Question: {question}
    Answer:"""
)

# generate a prompt using the template

prompt = template.invoke({
    "context": "The capital of India is New Delhi. It is the seat of the government of India and is known for its rich history and cultural heritage.",
    "question": "What is the capital of France?"
})

# generate response using the model
response = model.invoke(prompt)
print(response)  # print the response from the model
print("--------------------------------")
print(response.content)  # print the content of the response