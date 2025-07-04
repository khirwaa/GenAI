#A starter file for the HomeMatch application if you want to build your solution in a Python program instead of a notebook. 

import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter # Importing text splitter from Langchain
from langchain_community.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain_community.vectorstores import Chroma # Importing Chroma vector store from Langchain
from langchain.chat_models import ChatOpenAI
#from langchain.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator, OPENAI_TEMPLATE
from pydantic import BaseModel
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
import shutil # Importing shutil module for high-level file operations

os.environ["OPENAI_API_KEY"] = "voc-20679776171266773807325679e50ea654ed1.55149321"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"


file_path = "./listings.csv"
CHROMA_PATH = "./chroma_db"
model_name = "gpt-3.5-turbo"
temperature = 0.5

llm = ChatOpenAI(
           model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None, 
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://openai.vocareum.com/v1")

# Criteria:1 Synthetic data generation
class Listing(BaseModel):
  Neighborhood: str
  Price: str
  Bedrooms: int
  Bathrooms: int
  House_size: str
  House_features: str 
  Neighborhood_description: str

# Use the below example to build a dataset 
examples = [
    {"example": """
    Neighborhood: Green Oaks, Price: $800000, Bedrooms: 3, Bathrooms: 2, House_size: 2000 sqft, House_features: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces; highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden which is perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem, Neighborhood_description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
    """}
]

def data_generator(Listing):
    OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

    datagen_prompt_template = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["subject", "extra"],
        example_prompt=OPENAI_TEMPLATE,
    )

    listing_data_generator = create_openai_data_generator(
        output_schema=Listing,
        llm=ChatOpenAI(
           model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None, 
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://openai.vocareum.com/v1"),
        prompt=datagen_prompt_template,
    )
    synthetic_data = listing_data_generator.generate(
        subject="Listing",
        extra="Include details typical for a property listing, such as amenities, schools around, commute options, renovations and unique selling features",
        runs=50,
    )
    return synthetic_data

# data = data_generator(Listing=Listing)

def generate_listings_file(Listing, file_path):
    """Generate a CSV file with synthetic property listings data.
    """
    data = data_generator(Listing=Listing)
    synthetic_data = []
    for item in data:
        synthetic_data.append({
            'Neighborhood': item.Neighborhood,
            'Price': item.Price,
            'Bedrooms': item.Bedrooms,
            'Bathrooms': item.Bathrooms,
            'House_size': item.House_size,
            'House_features': item.House_features,
            'Neighborhood_description': item.Neighborhood_description
        })

    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv(file_path, index=False)

# Check if listings.csv already exists
if not os.path.exists(file_path):
   print(f"Generating synthetic data and saving to {file_path}...")
   generate_listings_file(Listing=Listing, file_path=file_path)
    
# Criteria:2 Creating a Vector Database and Storing Listings
# Loading the source data
loader = CSVLoader(file_path=file_path)
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PATH)

# Persist the database to disk
db.persist()
print(f"Saved {len(documents)} chunks to {CHROMA_PATH}.")


PROMPT_TEMPLATE = """
    Based on the listings in the context data {context}, please tell me which listing would be a suitable match 
    for the user based on the user preference in the query {question}.
    Make sure you present the listing highlighting the features important to the user without falsifying the data"
    """

# Criteria:3 Semantic Search of Listings Based on Buyer Preferences

def query_rag(query_text):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_text (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  # YOU MUST - Use same embedding function as before
  embedding_function = OpenAIEmbeddings()

  # Prepare the database
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  
  # Retrieving the context from the DB using similarity search
  similar_docs = db.similarity_search(query, k=5)
  # results = db.similarity_search_with_relevance_scores(query_text, k=3)
  # Check if there are any matching results or if the relevance score is too low
#   if len(similar_docs) == 0 or similar_docs[0][1] < 0.7:
#     print(f"Unable to find matching results.")

  # Combine context from matching documents
  context_text = "\n\n - -\n\n".join([doc.page_content for doc in similar_docs])
 
  # Create prompt template using context and query text
  prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  
  # Initialize OpenAI chat model
  # model = ChatOpenAI()

  # Generate response text based on the prompt
  response_text = llm.predict(prompt)
 
   # Get sources of the matching documents
  sources = [doc.metadata.get("source", None) for doc in similar_docs]
 
  # Format and return response including generated text and sources
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response, response_text


query = "I am looking for a 2 bedroom apartment in the city center which is pet-friendly and has a balcony. My budget is $500,000. I prefer a neighborhood with good schools and parks nearby."
# Let's call our function we have defined
formatted_response, response_text = query_rag(query)
# and finally, inspect our final response!
print(response_text)

# Criteria:4 Logic for Searching and Augmenting Listing Descriptions
rephrase_prompt_template = """
Generate a rephrased summary of the listing {response_text} and present it to user highlighting the important features as desired in the query {query} without changing any facts. 
"""
from langchain_core.prompts import ChatPromptTemplate
rewriter_prompt = ChatPromptTemplate.from_template(rephrase_prompt_template)

# rewriter_prompt.format(response_text=response_text, query=query)
# Now, construct the chain to execute the rewriting process:
rewriter_chain = rewriter_prompt | llm | StrOutputParser()
search_query = rewriter_chain.invoke({"response_text":response_text, "query":query})
print(search_query)
