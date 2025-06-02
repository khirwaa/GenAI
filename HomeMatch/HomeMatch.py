#A starter file for the HomeMatch application if you want to build your solution in a Python program instead of a notebook. 

import os

os.environ["OPENAI_API_KEY"] = "voc-20679776171266773807325679e50ea654ed1.55149321"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

from langchain.llms import OpenAI

model_name = "gpt-3.5-turbo"
temperature = 0.0
llm = OpenAI(model_name=model_name, temperature=temperature, max_tokens = 1000)


# Use the below example to build a dataset 
example = [
    """
    Neighborhood: Green Oaks
    Price: $800,000
    Bedrooms: 3
    Bathrooms: 2
    House Size: 2,000 sqft

    Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

    Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
"""
]