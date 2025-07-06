from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


load_dotenv()  # load environment variables from .env file

class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")

# initialize the OpenAI chat model with structured output
# # Note: The model "gpt-4.1-nano" is used for structured output, which is a smaller version of GPT-4.1.
# If you want to use a different model, you can change the model name accordingly.
model = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(Movie)
# Define a prompt template
template = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions based on the provided context.
    Context: {context}
    Question: {question}
    Answer:"""
)

user_input = {
    "context": "The Bahubali movie series is a two-part Indian epic action film directed by S. S. Rajamouli. The first part, Baahubali: The Beginning, was released in 2015, and the second part, Baahubali: The Conclusion, was released in 2017. The films are known for their grand scale, visual effects, and compelling storytelling.",
    "question": "Who directed the movie The Shawshank Redemption?"
}


chain = template | model

response = chain.invoke(user_input)

print(response)  # print the response from the model


print("--------------------------------")

#convert the response to a Movie object
response_movie = response.model_dump_json()
print(response_movie)  # print the content of the response as a Movie object
print("--------------------------------")
