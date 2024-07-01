import os

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL_NAME = 'gpt-3.5-turbo-0125'

TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
FROM = os.getenv('FROM')

CONNECTION_STRING = os.getenv('CONNECTION_STRING')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

QDRANT_URL = os.getenv('QDRANT_URL')
# In case using the Cloud Instance you need this
# QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = 'my_collection'

cwd = os.getcwd()

OUTPUT_DIR = os.path.join(
    cwd,
    'output'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model='text-embedding-3-large'
)

chat_model = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model_name=OPENAI_MODEL_NAME
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

qdrant_client = QdrantClient(
    url=QDRANT_URL, https=True,
    # api_key=QDRANT_API_KEY
)