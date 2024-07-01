def get_chat_history_from_mongodb(data: list) -> str:
    chat_history = []
    for d in data:
        chat_history.append(
            (
                d['query'],
                d['response']
            )
        )
    return chat_history

def create_string_chunks(string, length):
    words = string.split()
    sentences = []
    temp_string= ''
    for w in words:
        if len(temp_string) > length:
            sentences.append(f'{temp_string}...')
            temp_string = ''
        temp_string += f'{w} '
    sentences.append(temp_string)
    return sentences
