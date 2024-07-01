from datetime import datetime
import threading

from flask import Flask, request

from create_conversations import format_chat_history, handle_create_conversation
from database_functions import create_user, get_user, update_messages
from twilio_functions import send_message

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return 'OK', 200


def handle_request(data: dict) -> None:
    sender_id = data['From']
    query = data['Body']
    user_name = data['ProfileName']
    user = get_user(sender_id)
    print(user)
    # create chat_history from the previous conversations
    if user:
        messages = format_chat_history(user['messages'][-3:])
    else:
        messages = format_chat_history([])
    print(query)
    print(sender_id)
    print(messages)
    response = handle_create_conversation(messages, query)
    print(response)
    send_message(sender_id, response)
    if user:
        update_messages(sender_id, query, response,
                        user['messageCount'])
    else:
        # if not create
        message = {
            'query': query,
            'response': response,
            'createdAt': datetime.now().strftime('%d/%m/%Y, %H:%M')
        }
        user = {
            'userName': user_name,
            'senderId': sender_id,
            'messages': [message],
            'messageCount': 1,
            'mobile': sender_id.split(':')[-1],
            'channel': 'WhatsApp',
            'is_paid': False,
            'created_at': datetime.now().strftime('%d/%m/%Y, %H:%M')
        }
        create_user(user)


@app.route('/twilio', methods=['POST'])
def handle_twilio_webhook():
    try:
        print('A new twilio request...')
        data = request.form.to_dict()
        print(data)
        # Create a new thread to handle the time consuming request
        threading.Thread(
            target=handle_request,
            args=[data]
        ).start()
        print('Request success.')
    except:
        print('Request failed.')
    finally:
        return 'OK', 200
