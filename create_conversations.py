from langchain_community.vectorstores import Qdrant
from langchain.chains.llm import LLMChain
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate


import config


def format_context(documents: list[Document]) -> str:
    formated_context = ''
    for doc in documents:
        formated_context += f'\n{doc.page_content.strip()}\n'
    return formated_context


def format_chat_history(chat_history: list[list[str, str]]) -> str:
    formated_chat_history = ''
    for ch in chat_history:
        formated_chat_history += f"HUMAN: {ch['query']}\nAI: {ch['response']}\n"
    return formated_chat_history


def get_system_template() -> str:
    system_prompt = '''You are a helpful assistant. \
# Parameters:
- CONTEXT
- CHAT HISTORY
- QUESTION

# Instructions:
- Use the CONTEXT and CHAT HISTORY to answer the QUESTION. \
- If you don't know the answer and the CONTEXT doesn't contain the answer truthfully say I don't know. \
- Keep an informative tone.

# Answer:'''
    instruction = "CONTEXT: {context}\n\nCHAT HISTORY:\n\n{formated_chat_history}\n\nHUMAN: {question}\n\nAI:"
    template = f'{system_prompt}\n{instruction}'
    return template


def condense_user_query(query: str, formated_chat_history: list[list[str, str]]) -> str:
    system_prompt = f'''Given the following CHAT HISTORY and a FOLLOW UP QUESTION, \
rephrase the FOLLOW UP QUESTION to be a STANDALONE QUESTION in its original language. \

# Parameters \
- QUERY: {query} \
- CHAT_HISTORY: {formated_chat_history} \

# Instructions \
- Keep the context of the CHAT HISTORY in the standalone question.

# STANDALONE QUESTION: '''
    instruction = "CHAT HISTORY:\n\n{formated_chat_history}\n\nFOLLOW UP QUESTION: {question}\n\nSTANDALONE QUESTION:"
    template = f'{system_prompt}\n{instruction}'
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(template)
        ]
    )
    if len(formated_chat_history) <= 1:
        return query
    llm_chain = LLMChain(
        llm=config.chat_model,
        prompt=prompt,
        verbose=True
    )
    response = llm_chain.predict(
        question=query, formated_chat_history=formated_chat_history)
    response = response.strip()
    return response


def handle_create_conversation(formated_chat_history: list[list[str, str]], query: str) -> str:
    vector_db = Qdrant(client=config.qdrant_client, embeddings=config.embeddings,
                                   collection_name=config.COLLECTION_NAME)
    template = get_system_template()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(template)
        ]
    )
    llm_chain = LLMChain(
        llm=config.chat_model,
        prompt=prompt,
        verbose=True
    )
    condense_query = condense_user_query(query, formated_chat_history)
    searched_docs = vector_db.similarity_search(
        condense_query, k=5)
    formated_context = format_context(searched_docs)
    response = llm_chain.predict(
        question=query, context=formated_context, formated_chat_history=formated_chat_history)
    response = response.strip()
    return response
