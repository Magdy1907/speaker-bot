
import telebot
import numpy as np
import librosa
import subprocess
from tensorflow.keras.models import load_model

# 🔐 Токен Telegram-бота
TOKEN = "7424010381:AAF1_4x5XJpUj7V_d0KgmbZynggT7bJqxvg"
bot = telebot.TeleBot(TOKEN)

# 🧠 Загрузка модели
model = load_model("speaker_classifier.keras")

# 🏷️ Классы
labels = {0: "speaker1", 1: "speaker2", 2: "speaker3"}

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Сохранение ogg-файла
        with open("input.ogg", 'wb') as f:
            f.write(downloaded_file)

        # Конвертация в wav
        subprocess.call(['ffmpeg', '-y', '-i', 'input.ogg', 'input.wav'])

        # Извлечение MFCC
        y, sr = librosa.load("input.wav", sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        mfcc = mfcc[:348] if mfcc.shape[0] > 348 else np.pad(mfcc, ((0, 348 - mfcc.shape[0]), (0, 0)))
        mfcc = np.expand_dims(mfcc, axis=0)

        # Предсказание
        pred = model.predict(mfcc)
        speaker = labels[np.argmax(pred)]

        bot.reply_to(message, f"🔊 Предсказанный говорящий: {speaker}")

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {e}")

# Запуск
bot.polling()
