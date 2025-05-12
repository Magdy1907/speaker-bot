import telebot
import numpy as np
import librosa
import subprocess
import os
from tensorflow.keras.models import load_model

# 🔐 Токен бота
TOKEN = "7424010381:AAFhJOwnBKclkx4WVs6cG1btN_vnSK1tLVk"
bot = telebot.TeleBot(TOKEN)

# 📦 Загрузка модели
model = load_model("speaker_classifier.keras")

# 🏷️ Метки классов
labels = {
    0: "Анна",
    1: "Бабушка",
    2: "Влад",
    3: "Дедушка",
    4: "Никита"
}

# 💬 Ответ на текст
@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.reply_to(message, "👋 Привет! Отправь голосовое сообщение или аудиофайл (WAV, MP3, OGG), и я скажу, кто говорит.")

# 🎧 Обработка аудио и voice
@bot.message_handler(content_types=['audio', 'document', 'voice'])
def handle_audio(message):
    try:
        # 📥 Получение файла
        if message.voice:
            file_info = bot.get_file(message.voice.file_id)
            ext = ".ogg"
        elif message.audio:
            file_info = bot.get_file(message.audio.file_id)
            ext = os.path.splitext(file_info.file_path)[1]
        elif message.document:
            file_info = bot.get_file(message.document.file_id)
            ext = os.path.splitext(file_info.file_path)[1]
        else:
            bot.reply_to(message, "⚠️ Неподдерживаемый формат файла.")
            return

        file_data = bot.download_file(file_info.file_path)
        input_file = f"input{ext}"
        with open(input_file, "wb") as f:
            f.write(file_data)

        # 🔄 Конвертация в WAV
        wav_file = "converted.wav"
        if ext.lower() != ".wav":
            subprocess.run(['ffmpeg', '-y', '-i', input_file, wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            wav_file = input_file

        # 🧠 Извлечение MFCC
        y, sr = librosa.load(wav_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T

        max_len = model.input_shape[1]
        if mfcc.shape[0] > max_len:
            mfcc = mfcc[:max_len]
        else:
            mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))

        mfcc = np.expand_dims(mfcc, axis=0)

        # 🤖 Предсказание
        pred = model.predict(mfcc)
        confidence = np.max(pred)
        predicted = np.argmax(pred)

        if confidence < 0.3:
            bot.reply_to(message, "❌ Не удалось распознать голос.")
        else:
            bot.reply_to(message, f"🗣️ Говорящий: {labels[predicted]}")

        # 🧹 Очистка
        for f in [input_file, "converted.wav"]:
            if os.path.exists(f):
                os.remove(f)

    except Exception as e:
        bot.reply_to(message, f"⚠️ Ошибка: {e}")

# ▶️ Запуск бота
bot.remove_webhook()
bot.polling()
