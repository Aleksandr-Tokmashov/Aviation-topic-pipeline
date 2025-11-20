# Парсинг постов из Telegram

from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel
from dotenv import load_dotenv
import os
import csv

load_dotenv()

api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")
phone = os.getenv("PHONE")




client = TelegramClient(phone, api_id, api_hash)

def parse_messages(client, channel_name, total_count_limit, limit = 100, offset_id = 0):
    # для хранения спарсенных сообщений
    all_messages = [] 

    total_messages = 0
    
    client.start()

    channel = client.get_entity(channel_name)  

    # Получаем по 100 сообщений на каждой итерации
    while True:
        history = client(GetHistoryRequest(
        peer=channel,
        offset_id=offset_id,
        offset_date=None,
        add_offset=0,
        limit=limit,
        max_id=0,
        min_id=0,
        hash=0
        ))
        if not history.messages:
            break
        messages = history.messages
        for message in messages:
            all_messages.append(message.message) #Добавим параметр message к методу message.
        offset_id = messages[len(messages) - 1].id
        total_messages += limit
        print("Спарсили " + str(total_messages) + f" сообщений из {channel_name}.")
        if total_count_limit != 0 and total_messages >= total_count_limit:
            break
    
    print("Сохраняем данные в файл...") 
    
    with open(f"./data/raw/{channel_name}_posts.csv", "w", encoding="UTF-8") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        writer.writerow(["message"])
        for message in all_messages:
            writer.writerow([message])    

        print(f"Парсинг сообщений {channel_name} успешно выполнен.") 


channels = {'favt_ru': 3000, 'aviatorshina': 1000, 's7newsroom': 500, 'aeroflot_official': 500}

for channel_name, total_count_limit in channels.items():
    parse_messages(client, channel_name, total_count_limit)




