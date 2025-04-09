import discord
from discord.ext import commands
import torch
from ollama import chat
import traceback

# Настройки бота
with open("../token.txt", "r") as file: TOKEN = file.read()
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Функция для генерации ответа
def generate(message, author, model="gemma3:1b"):
    response = chat(model=model,
        messages=[
            {"role": "system", "content": "Старайся не писать больше 2000 тысяч символов и отвечай на сообщения на русском."},
            {"role": "user", "content": f"{author}: {message}"}
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

        #if "@romangobb’s_bot#7030" not in message.content:
        #    return

        # Генерация ответа
        #context = "\n".join(
        #    [msg["content"] for msg in chat_history[-1:]])  # Используем последние 1 сообщение как контекст
        response = generate(message.content, message.author.name)

        # Отправляем ответ в тот же канал
        if len(response) > 2000:
            await message.reply("Мужик, ну ты это, извини, но бот настрочил слишком много и дискорд зажевал сообщение. \
            Переспроси вопрос и попроси её написать ответ покороче", mention_author=True)

        await message.reply(response, mention_author=True)


    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()


# Запуск бота
bot.run(TOKEN)
