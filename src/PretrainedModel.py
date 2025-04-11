import json
import os
from collections import deque
import atexit
import discord
from discord.ext import commands
from ollama import chat

# Настройки бота
with open("../token.txt", "r") as file: TOKEN = file.read()
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

cache_dir = "../cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

bot = commands.Bot(command_prefix="@", intents=intents)

class Bot:
    system = {"role": "system", "content": "Отвечай на сообщения на русском."}

    def __init__(self):
        self.mem = self.Memory()

    # Функция для генерации ответа
    # Принимает список словарей с контекстом
    def generate(self, messages, model="gemma3:1b"):
        messages.insert(0, self.system)
        response = chat(model=model, messages=messages)
        self.mem.store("Bot", response["message"]["content"])
        return response["message"]["content"]

    # Создаёт список сообщений, содержащий несколько сообщений
    def context(self) -> list:
        return self.mem.memory

    class Memory:
        """Класс, отвечающий за работу с памятью"""
        def __init__(self,):
            # Файл памяти
            self.memfile = cache_dir+"/memory.jsonl"
            # Количество сообщений, которое помнит бот
            self.memlen = 5
            self.count = self.memlen

            self.memory = self.load_history()

        # Поместить сообщение в память
        # При указании author="Bot" поместит в память от имени бота
        def store(self, author, message):
            if author!="Bot": self.memory.append({"role": "user", "content": f"{author}: {message}"})
            else: self.memory.append({"role": "assistant", "content": message})

            if len(self.memory) > self.memlen:
                if self.count <= 0:
                    self.dump()
                else:
                    self.memory.pop(0)
                    self.count -= 1

        # Получить сообщение из памяти, начиная с начала
        # По умолчанию получает последнее сообщение
        def get(self, pos=-1) -> dict:
            return self.memory[pos]

        # Кешировать позицию из памяти в файл для дальнейшего использования, по умолчанию первая
        # По умолчанию пишет в cache/memory.txt
        def dump(self, pos=0):
            with open(self.memfile, 'a') as f:
                f.write(json.dumps(self.memory.pop(pos), ensure_ascii=False) + '\n', )

        # Сохраняет в файл всю память
        def dump_all(self):
            for _ in self.memory:
                self.dump()

        # Удалить всю память
        def clear(self):
            self.memory.clear()

        # Записать в память из файла
        # TODO: Принимает количество сообщений начиная с конца
        def load_history(self):
            items = []
            try:
                with open(self.memfile, 'r') as f:
                    # Use a deque to track the last `n` lines efficiently
                    last_n_lines = deque(f, maxlen=self.memlen)

                # Parse non-empty lines into JSON objects
                for line in last_n_lines:
                    stripped_line = line.strip()
                    if stripped_line:  # Skip empty lines
                        items.append(json.loads(stripped_line))

            except FileNotFoundError:
                open(self.memfile, 'w').close()
            except json.JSONDecodeError as e:
                print(f"JSON decode error in line: {e.doc}")

            return items

CBot = Bot()
atexit.register(CBot.mem.dump_all)

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

        CBot.mem.store(message.author.name, message.content)

        # Проверяем есть ли пинг бота
        if bot.user not in message.mentions:
            return

        context = CBot.context()

        print(*context, "\n", sep="\n")

        response = CBot.generate(context)

        if len(response) == 0:
            print("Empty!")

        # Отправляем ответ в тот же канал
        if len(response) > 2000:
            await message.reply("Мужик, ну ты это, извини, но бот настрочил слишком много и дискорд зажевал сообщение. \
        Переспроси вопрос и попроси её написать ответ покороче", mention_author=True)
        else:
            await message.reply(response, mention_author=True)

    except Exception as e:
        print(f"Произошла ошибка: {e}")


# Запуск бота
bot.run(TOKEN)
