import discord
from discord.ext import commands
import torch
from ollama import chat
import traceback

# Настройки бота
TOKEN = 'MTE2MTM5NjA0ODkxMzI0MDA4Ng.GGgerJ.a6C1ShrN_NAhmAV8JR28NxvKTKaHEXTyOo-RfI'
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

chat_history = []

# Функция для генерации ответа
def generate(message, model="gemma3:1b"):
    response = chat(model=model,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return response["message"]["content"]

@bot.event
async def on_message(message):
    # Игнорируем сообщения от самого бота
    if message.author == bot.user:
        return

    # Список разрешённых каналов для отправки сообщений
    allowed_channel_ids = {
        1355493651974717540,  # Второй канал
        1355606855144968202  # Четвёртый канал
    }

    try:
        # Проверяем, находится ли сообщение в разрешённом канале
        if message.channel.id not in allowed_channel_ids:
            return  # Если канал не разрешён, игнорируем сообщение

        # Добавляем сообщение с указанием автора
        chat_history.append({
            "author": message.author.name,
            "content": message.content
        })
        if len(chat_history) > 1000:
            chat_history.pop(0)

        # Генерация ответа
        context = "\n".join(
            [msg["content"] for msg in chat_history[-1:]])  # Используем последние 5 сообщений как контекст
        response = generate(context)

        # Отправляем ответ в тот же канал
        await message.channel.send(response)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()


# Запуск бота
bot.run(TOKEN)
