from datetime import datetime
from typing import Any

from pymongo import MongoClient

import config

client = MongoClient(config.CONNECTION_STRING)
db = client[config.DATABASE_NAME]
user_collection = db[config.COLLECTION_NAME]


def update_messages(sender_id: int, query: str, response: str, message_count: int) -> bool:
    '''
    Update messages for the user and reduce the messages_count by one

    Parameters:
        - sender_id(int): user telegram id
        - response(str): response of the bot
        - query(str): query of the user

    Returns:
        - bool, 0 for failure and 1 for success
    '''
    message = {
        'query': query,
        'response': response,
        'createdAt': datetime.now().strftime('%d/%m/%Y, %H:%M')
    }

    result = user_collection.find_one_and_update(
        {
            'senderId': sender_id
        },
        {
            '$push': {
                'messages': message
            },
            '$set': {
                'messageCount': message_count + 1
            }
        }
    )
    if not result:
        return False
    else:
        return True


def update_user(sender_id: int, update: dict) -> bool:
    '''
    Update messages for the user and reduce the messages_count by one

    Parameters:
        - sender_id(int): user telegram id
        - update(dict): update to push to the record

    Returns:
        - bool, 0 for failure and 1 for success
    '''
    result = user_collection.find_one_and_update(
        {
            'senderId': sender_id
        },
        {
            '$set': update
        }
    )
    if not result:
        return False
    else:
        return True


def create_user(user: dict) -> bool:
    '''
    Process the whole body and update the db

    Parameters:
        - data(dict): the incoming request from Telegram

    Returns:
        - bool, 0 for failure and 1 for success
    '''
    result = user_collection.insert_one(user)
    return result.acknowledged


def get_user(sender_id: str) -> Any:
    '''
    Get user

    Parameters:
        - sender_id(str): sender id of the user

    Returns:
        - bool, 0 for failure and 1 for success
    '''
    result = user_collection.find_one(
        {
            'senderId': sender_id
        }
    )
    if not result:
        None
    return result