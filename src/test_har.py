import discord
from discord.ext import commands
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from torch.optim import Adam
import random
import os
import traceback

# Настройки бота
TOKEN = 'MTE2MTM5NjA0ODkxMzI0MDA4Ng.GGgerJ.a6C1ShrN_NAhmAV8JR28NxvKTKaHEXTyOo-RfI'
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Путь для сохранения модели
MODEL_SAVE_PATH = "./saved_model"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Переменные для управления обучением
retrain = False  # Если True, обучение с нуля; если False, загрузка сохранённой модели
is_train = False # Если True, обучение при старте; если False, сразу начинает отвечать

# Инициализация токенизатора и модели T5
model_name = "cointegrated/rut5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
if retrain:
    print("Создание новой модели с нуля...")
    config = T5Config.from_pretrained(model_name)  # Загружаем конфигурацию модели
    model = T5ForConditionalGeneration(config)  # Создаем модель с нуля без предобученных весов
else:
    print("Загрузка предварительно сохранённой модели...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH)

# Перенос модели на GPU
model.to(device)

# Список для хранения истории чата (для обучения)
chat_history = []

# Функция для генерации ответа
def generate_response(context, max_new_tokens=50):
    # Токенизация входного текста
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Перенос данных на GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Генерация ответа
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Максимальное количество новых токенов
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        #repetition_penalty=2.0,
        top_k=50,  # топ-50 
        top_p=0.95,  
        do_sample=True,  # Вsampling
        #temperature=0.2
    )
    
    # Декодирование ответа
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Обучение модели на основе истории чата
def train_model_on_chat():
    if len(chat_history) < 2:
        print("Недостаточно данных для обучения.")
        return
    
    optimizer = Adam(model.parameters(), lr=1e-4)  # Используем Adam
    model.train()

    # Создаем пары "контекст-ответ"
    contexts = []
    targets = []
    for i in range(1, len(chat_history)):
        prev_message = chat_history[i - 1]
        current_message = chat_history[i]

        # Проверяем, что авторы сообщений разные
        if prev_message["author"] != current_message["author"]:
            contexts.append(prev_message["content"])
            targets.append(current_message["content"])

    if not contexts or not targets:
        print("Нет подходящих пар для обучения.")
        return

    # Токенизация всех данных
    inputs = tokenizer(contexts, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(targets, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    # Выравнивание длин input_ids и labels
    min_length = min(inputs["input_ids"].shape[1], labels["input_ids"].shape[1])
    inputs["input_ids"] = inputs["input_ids"][:, :min_length]
    inputs["attention_mask"] = inputs["attention_mask"][:, :min_length]
    labels["input_ids"] = labels["input_ids"][:, :min_length]

    # Перенос данных на GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels["input_ids"].to(device)

    # Разделяем данные на батчи
    batch_size = 4
    total_batches = (len(inputs["input_ids"]) + batch_size - 1) // batch_size  # Количество батчей

    # Добавляем 5 эпох обучения
    epochs = 20
    for epoch in range(epochs):
        print(f"\nЭпоха {epoch + 1}/{epochs}")
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(inputs["input_ids"]))

            batch_inputs = {
                "input_ids": inputs["input_ids"][start_idx:end_idx],
                "attention_mask": inputs["attention_mask"][start_idx:end_idx]
            }
            batch_labels = labels[start_idx:end_idx]

            # Простой цикл обучения
            optimizer.zero_grad()
            outputs = model(input_ids=batch_inputs["input_ids"], attention_mask=batch_inputs["attention_mask"], labels=batch_labels)
            loss = outputs.loss

            # Добавляем штраф за повторение контекста
            penalty_loss = add_penalty_for_repetition(batch_inputs["input_ids"], batch_labels)
            loss += penalty_loss

            # Добавляем награду за количество слов в ответе
            #reward = add_reward_for_word_count(batch_inputs["input_ids"], batch_labels)
            #loss -= reward

            loss.backward()
            optimizer.step()

            print(f"Обучение батча {batch_idx + 1}/{total_batches}, Loss: {loss.item()}")
        save_model()
    model.eval()

# Функция для добавления штрафа за повторение контекста
def add_penalty_for_repetition(input_ids, labels):
    penalty = 0.0
    batch_size, seq_len = labels.shape
    for i in range(batch_size):
        label_text = tokenizer.decode(labels[i], skip_special_tokens=True)
        words = label_text.split()  # Разбиваем текст на слова
        
        # Подсчёт частоты повторений для каждого слова
        word_counts = {}
        for word in words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        
        # Накладываем штраф за повторения
        for count in word_counts.values():
            if count > 1:  # Если слово повторяется
                penalty += (count - 1) * 0.2  # Штраф пропорционален количеству повторений

    return torch.tensor(penalty, device=device)

# Функция для добавления награды за количество слов в ответе
def add_reward_for_word_count(input_ids, labels):
    reward = 0.0
    batch_size, seq_len = labels.shape
    for i in range(batch_size):
        input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        label_text = tokenizer.decode(labels[i], skip_special_tokens=True)
        
        # Подсчёт количества слов в контексте и ответе
        input_word_count = len(input_text.split())
        label_word_count = len(label_text.split())
        
        # Награда даётся только если ответ длиннее контекста
        #if label_word_count > input_word_count:
        #    reward += (label_word_count - input_word_count) * 0.1  # Небольшая награда за каждое дополнительное слово
    return torch.tensor(reward, device=device)

# Функция для сохранения модели
def save_model():
    print("Сохранение модели...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Модель успешно сохранена.")

# Получение последних 1000 сообщений из канала
async def fetch_initial_messages(channel):
    print("Загрузка последних 1000 сообщений для обучения...")
    async for message in channel.history(limit=100):
        chat_history.append({
            "author": message.author.name,  # Сохраняем имя автора
            "content": message.content     # Сохраняем текст сообщения
        })
    print(f"Загружено {len(chat_history)} сообщений.")
    if is_train:
        try:
            train_model_on_chat()
        except Exception as e:
            print(f"Ошибка при обучении на сообщениях из канала {channel.name}: {e}")
            traceback.print_exc()

# Обработка сообщений
@bot.event
async def on_ready():
    print(f"Бот {bot.user} подключен к Discord!")
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.name == "🌐-oбщий-чат":
                try:
                    await fetch_initial_messages(channel)
                    if not is_train:
                        break
                except Exception as e:
                    print(f"Ошибка при загрузке сообщений из канала {channel.name}: {e}")

@bot.event
async def on_message(message):
    # Игнорируем сообщения от самого бота
    if message.author == bot.user:
        return

    # Список разрешённых каналов для отправки сообщений
    allowed_channel_ids = {
        1355493651974717540,  # Второй канал
        1355606855144968202   # Четвёртый канал
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
        context = "\n".join([msg["content"] for msg in chat_history[-5:]])  # Используем последние 5 сообщений как контекст
        response = generate_response(context, max_new_tokens=50)

        # Отправляем ответ в тот же канал
        await message.channel.send(response)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()

# Запуск бота
bot.run(TOKEN)
