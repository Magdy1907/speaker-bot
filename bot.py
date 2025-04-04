import telebot
import numpy as np
import librosa
import subprocess
from tensorflow.keras.models import load_model
import threading

# 🔐 Токен Telegram-бота
TOKEN = "7424010381:AAF1_4x5XJpUj7V_d0KgmbZynggT7bJqxvg"
bot = telebot.TeleBot(TOKEN)

# 🧠 Загрузка модели
model = load_model("speaker_classifier.keras")

# 🏷️ Классы
labels = {0: "speaker1", 1: "speaker2", 2: "speaker3"}

def process_audio(file_info, message):
    try:
        downloaded_file = bot.download_file(file_info.file_path)
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

        # Проверка уверенности модели (если менее 60% точности)
        confidence = np.max(pred)
        if confidence < 0.6:
            bot.reply_to(message, "❌ Извините, я не смог распознать голос.")
        else:
            bot.reply_to(message, f"🔊 Говорящий: {speaker}")

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {e}")

@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id)

    # Создаем новый поток для обработки файла
    thread = threading.Thread(target=process_audio, args=(file_info, message))
    thread.start()

# Запуск
bot.polling()
