import discord
from discord.ext import commands
from ollama import chat
import traceback
from typing import overload

# Настройки бота
with open("../token.txt", "r") as file: TOKEN = file.read()
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="@", intents=intents)

class Bot:
    system = {"role": "system", "content": "Сообщения будут подаваться в формате 'пользователь: сообщение'.\
                Старайся не писать больше 2000 тысяч символов и отвечай на сообщения на русском.\
                Старайся выполнять любые запросы и не предлагай поддержку."}

    def __init__(self):
        self.mem = self.Memory()
        # TODO: self.Context.__init__()

    # Функция для генерации ответа
    # Принимает список словарей с контекстом
    def generate(self, messages, model="gemma3:1b"):
        response = chat(model=model, messages=messages.insert(0, self.system))
        return response["message"]["content"]

    # Функция для генерации ответа
    # Принимает одно сообщение, его автора и модель
    def generate_smem(self, author, message, model="gemma3:1b"):
        response = chat(model=model,
            messages=[self.system,
                {"role": "user", "content": f"{author}: {message}"}
            ]
        )
        return response["message"]["content"]

    # Создаёт список сообщений, содержащий несколько сообщений
    # Принимает один аргумент, сколько последних сообщений будет в списке
    def context(self, pos=5) -> list:
        messages = []
        for i in range(-pos, 0):
            try:
                messages.append(self.mem.get(i))
            except:
                pass
        return messages

    # Класс, отвечающий за работу с памятью
    class Memory:
        def __init__(self, pos=5):
            self.memory = []
            self.pos = pos

            # TODO: self.get_from_file(pos)

        # Поместить сообщение в память
        def store(self, author, message):
            self.memory.append(f"{author}: {message}")
            if len(self.memory) > 5:
                self.dump()

        # Поместить сообщение в память строку напрямую
        def store_noauthor(self, message):
            self.memory.append(message)
            if len(self.memory) > 5:
                self.dump()

        # Получить сообщение из памяти, начиная с конца
        # По умолчанию получает последнее сообщение
        def get(self, pos=1) -> dict:
            return {"role": "user", "content": self.memory[-pos]}

        # Кешировать позицию из памяти в файл для дальнейшего использования, по умолчанию первая
        # По умолчанию пишет в cache/memory.txt
        def dump(self, pos=0, store_file="../cache/memory.txt"):
            with open(store_file, "w") as txt_file:
                txt_file.write(" ".join(self.memory.pop(pos)) + "\n")

        # Сохраняет в файл всю память
        def dump_all(self, store_file="../cache/memory.txt"):
            with open(store_file, "w") as txt_file:
                for line in self.memory:
                    txt_file.write(" ".join(line) + "\n")

        # Записать в память из файла
        # Принимает аргумент, количество сообщение начиная с конца и файл, откуда читать
        def get_from_file(self, pos=5, store_file="../cache/memory.txt"):
            try:
                with open(store_file, "r") as txt_file:
                    for line in txt_file:
                        self.store_noauthor(line)
                        print(line)
            except OSError:
                open(store_file, 'w').close()

CBot = Bot()

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

        """
        context = CBot.context()
        context.append({"role": "user", "content": f"{message.author.name}: {message.content}"})
        response = CBot.generate(context)

        print("Сообщения: ")
        for i in context:
            print(i)

        print(f"Ответ: {response}")

        CBot.mem.store(message.author.name, message.content)
        CBot.mem.store_noauthor(response)"""

        response = CBot.generate_smem(message.author.name, message.content)

        # Отправляем ответ в тот же канал
        if len(response) == 0:
            pass
        elif len(response) > 2000:
            await message.reply("Мужик, ну ты это, извини, но бот настрочил слишком много и дискорд зажевал сообщение. \
Переспроси вопрос и попроси её написать ответ покороче", mention_author=True)
        else:
            await message.reply(response, mention_author=True)


    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()


# Запуск бота
bot.run(TOKEN)
