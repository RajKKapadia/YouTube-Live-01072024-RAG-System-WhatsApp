from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Qdrant
from pypdf import PdfReader

import config


def create_index(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        with open(f'{config.OUTPUT_DIR}/output.txt', 'w') as file:
            file.write(text)
        loader = DirectoryLoader(
            config.OUTPUT_DIR,
            glob='**/*.txt',
            loader_cls=TextLoader
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=64
        )
        texts = text_splitter.split_documents(documents)
        Qdrant.from_documents(
            texts,
            config.embeddings,
            collection_name=config.COLLECTION_NAME,
            url=config.QDRANT_URL,
            # In case using the Cloud Instance you need this
            # api_key=config.QDRANT_API_KEY
        )
        return 'Documents uploaded and index created successfully. You can chat now.'
    except Exception as e:
        return e