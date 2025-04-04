import telebot
import numpy as np
import librosa
import os
import subprocess
from tensorflow.keras.models import load_model

# 🔐 Токен Telegram-бота
TOKEN = "7424010381:AAF1_4x5XJpUj7V_d0KgmbZynggT7bJqxvg"
bot = telebot.TeleBot(TOKEN)

# 🧠 Загрузка модели
model = load_model("speaker_classifier.keras")

# 🌿 Классы
labels = {0: "speaker1", 1: "speaker2", 2: "speaker3"}

@bot.message_handler(content_types=['text'])
def greet_user(message):
    if message.text.lower() in ["привет", "здравствуйте", "начать", "/start"]:
        bot.reply_to(message, "👋 Привет! Пожалуйста, отправьте голосовое сообщение или аудиофайл для определения говорящего.")

@bot.message_handler(content_types=['voice', 'audio'])
def handle_voice(message):
    try:
        file_info = bot.get_file(message.voice.file_id if message.content_type == 'voice' else message.audio.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Сохранение файла
        input_path = "input.ogg" if message.content_type == 'voice' else "input.mp3"
        with open(input_path, 'wb') as f:
            f.write(downloaded_file)

        # Конвертация в WAV только если не .wav
        wav_path = "input.wav"
        if not input_path.endswith(".wav"):
            subprocess.call(['ffmpeg', '-y', '-i', input_path, wav_path])
        else:
            wav_path = input_path

        # MFCC
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
        mfcc = mfcc[:348] if mfcc.shape[0] > 348 else np.pad(mfcc, ((0, 348 - mfcc.shape[0]), (0, 0)))
        mfcc = np.expand_dims(mfcc, axis=0)

        # Предсказание
        pred = model.predict(mfcc)
        print(pred)  # Для отладки

        confidence = np.max(pred)
        if confidence < 0.6:
            bot.reply_to(message, "🔍 Говорящий не распознан или слишком слабый сигнал.")
        else:
            speaker = labels[np.argmax(pred)]
            bot.reply_to(message, f"🎤 Говорящий: {speaker}")

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {e}")

# Запуск
bot.polling()
