import telebot
import numpy as np
import librosa
import subprocess
import os
from tensorflow.keras.models import load_model

# 🔐 Токен Telegram-бота
TOKEN = "7424010381:AAF1_4x5XJpUj7V_d0KgmbZynggT7bJqxvg"  # 
bot = telebot.TeleBot(TOKEN)

# 🧠 Загрузка обученной модели
model = load_model("speaker_classifier.keras")

# 🏷️ Словарь меток
labels = {0: "speaker1", 1: "speaker2", 2: "speaker3"}

# 💬 Ответ на текстовые сообщения
@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.reply_to(
        message,
        "👋 Привет! Пожалуйста, отправьте аудиофайл (WAV, MP3, OGG и т.д.), чтобы определить говорящего."
    )

# 🎧 Обработка аудиофайлов
@bot.message_handler(content_types=['audio', 'document'])
def handle_audio(message):
    try:
        # Получаем файл от пользователя
        file_info = bot.get_file(message.audio.file_id if message.audio else message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Сохраняем файл с оригинальным расширением
        original_extension = os.path.splitext(file_info.file_path)[1]
        input_filename = f"input{original_extension}"
        with open(input_filename, 'wb') as f:
            f.write(downloaded_file)

        # Конвертируем в WAV, если нужно
        if original_extension.lower() != '.wav':
            subprocess.call(['ffmpeg', '-y', '-i', input_filename, 'converted.wav'])
            input_path = 'converted.wav'
        else:
            input_path = input_filename

        # Извлекаем MFCC-признаки
        y, sr = librosa.load(input_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        mfcc = mfcc[:348] if mfcc.shape[0] > 348 else np.pad(mfcc, ((0, 348 - mfcc.shape[0]), (0, 0)))
        mfcc = np.expand_dims(mfcc, axis=0)

        # Предсказание
        pred = model.predict(mfcc)
        speaker = labels[np.argmax(pred)]
        bot.reply_to(message, f"🔊 Говорящий: {speaker}")

        # Удаление временных файлов
        os.remove(input_filename)
        if os.path.exists("converted.wav"):
            os.remove("converted.wav")

    except Exception as e:
        bot.reply_to(message, f"❌ Произошла ошибка: {e}")

# ▶️ Запуск бота
bot.polling()
