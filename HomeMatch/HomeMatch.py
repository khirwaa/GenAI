#A starter file for the HomeMatch application if you want to build your solution in a Python program instead of a notebook. 

import os
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter # Importing text splitter from Langchain
from langchain.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
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
import shutil # Importing shutil module for high-level file operations

os.environ["OPENAI_API_KEY"] = "voc-20679776171266773807325679e50ea654ed1.55149321"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"


file_path = "./real_estate_listings.csv"

model_name = "gpt-3.5-turbo"
temperature = 0.0
llm = OpenAI(model_name=model_name, temperature=temperature, max_tokens = 1000)

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
# OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")
# OPENAI_TEMPLATE
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
            api_key="voc-20679776171266773807325679e50ea654ed1.55149321",
            base_url="https://openai.vocareum.com/v1"),
        prompt=datagen_prompt_template,
    )
    synthetic_data = listing_data_generator.generate(
        subject="Listing",
        extra="Include details typical for a property listing, such as amenities, schools around, commute options, renovations and unique selling features",
        runs=15,
    )
    return synthetic_data

data = data_generator(Listing=Listing)
# print(data)


synthetic_data = []
for item in synthetic_results:
    synthetic_data.append({
        'patient_id': item.patient_id,
        'patient_name': item.patient_name,
        'diagnosis_code': item.diagnosis_code,
        'procedure_code': item.procedure_code,
        'total_charge': item.total_charge,
        'insurance_claim_amount': item.insurance_claim_amount
    })

synthetic_df = pd.DataFrame(synthetic_data)
synthetic_df
# Loading the source data
# loader = CSVLoader(file_path=file_path)
# data = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(data)

# embeddings = OpenAIEmbeddings()
# CHROMA_PATH = "./chroma_db"
# db = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PATH)

# # Persist the database to disk
# db.persist()
# print(f"Saved {len(documents)} chunks to {CHROMA_PATH}.")


# PROMPT_TEMPLATE = """
#     Based on the listings in the context data {context}, please tell me which listing would be a suitable match 
#     for the user based on the user preference in the query {question}.
#     Make sure you present the listing highlighting the features important to the user without falsifying the data"
#     """





# def query_rag(query_text):
#   """
#   Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
#   Args:
#     - query_text (str): The text to query the RAG system with.
#   Returns:
#     - formatted_response (str): Formatted response including the generated text and sources.
#     - response_text (str): The generated response text.
#   """
#   # YOU MUST - Use same embedding function as before
#   embedding_function = OpenAIEmbeddings()

#   # Prepare the database
#   db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  
#   # Retrieving the context from the DB using similarity search
#   similar_docs = db.similarity_search(query, k=5)
#   # results = db.similarity_search_with_relevance_scores(query_text, k=3)
#   # Check if there are any matching results or if the relevance score is too low
# #   if len(similar_docs) == 0 or similar_docs[0][1] < 0.7:
# #     print(f"Unable to find matching results.")

#   # Combine context from matching documents
#   context_text = "\n\n - -\n\n".join([doc.page_content for doc in similar_docs])
 
#   # Create prompt template using context and query text
#   prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
#   prompt = prompt_template.format(context=context_text, question=query_text)
  
#   # Initialize OpenAI chat model
#   # model = ChatOpenAI()

#   # Generate response text based on the prompt
#   response_text = llm.predict(prompt)
 
#    # Get sources of the matching documents
#   sources = [doc.metadata.get("source", None) for doc in similar_docs]
 
#   # Format and return response including generated text and sources
#   formatted_response = f"Response: {response_text}\nSources: {sources}"
#   return formatted_response, response_text


# query = "I am looking for a big house with an open concept kitchen and a large yard in country side"
# # Let's call our function we have defined
# formatted_response, response_text = query_rag(query)
# # and finally, inspect our final response!
# print(response_text)

