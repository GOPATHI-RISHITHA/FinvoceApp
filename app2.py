import streamlit as st
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from googletrans import Translator
import speech_recognition as sr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
from sklearn.preprocessing import MinMaxScaler
from gtts import gTTS
import pygame
import tempfile
import time

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

translator = Translator()

# Language mappings
languages = {
    "English": ("en-IN", "en"),
    "Hindi": ("hi-IN", "hi"),
    "Telugu": ("te-IN", "te"),
    "Tamil": ("ta-IN", "ta"),
    "Kannada": ("kn-IN", "kn"),
    "Bengali": ("bn-IN", "bn")
}

def translate_text(text, lang_code):
    translated = translator.translate(text, dest=lang_code)
    return translated.text

def speak_text(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            temp_path = fp.name

        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.3)

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        os.remove(temp_path)

    except Exception as e:
        st.warning(f"Voice output failed: {e}")

def get_voice_input(selected_lang_display):
    lang_code, trans_code = languages[selected_lang_display]
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(f"Speak now in {selected_lang_display}...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=10, phrase_time_limit=8)
    try:
        query = r.recognize_google(audio, language=lang_code)
        st.success(f"You said: {query}")
        if trans_code != 'en':
            translated = translator.translate(query, src=trans_code, dest='en')
            st.info(f"Translated: {translated.text}")
            return translated.text.lower()
        return query.lower()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return ""

def match_company_files(query):
    mapping = {
        "hdfc": "HDFC_Bank.csv",
        "tcs": "TCS.csv",
        "reliance": "RELIANCE.csv",
        "infosys": "INFOSYS.csv",
        "hindustan": "HINDUSTAN.csv",
        "itc": "ITC.csv"
    }
    matched = [(name.upper(), fname) for name, fname in mapping.items() if name in query]
    return matched

def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    df['close'] = df['close'].astype(str).str.replace(',', '').astype(float)
    df = df.sort_values('date')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']].values)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler, df

def build_model():
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def suggest_investment(last, predicted):
    pct = ((predicted - last) / last) * 100
    msgs = []
    if pct > 2:
        msgs = [
            f"🔼 Strong Buy: Expected rise of {pct:.2f}%",
            "💰 Suggested Investment: ₹10,000",
            f"📈 Estimated Profit: ₹{(predicted - last) * (10000 / last):.2f}",
            "🟢 Risk: Low"
        ]
    elif 0.5 < pct <= 2:
        msgs = [
            f"🟡 Moderate Buy: Expected rise of {pct:.2f}%",
            "💰 Suggested Investment: ₹5,000",
            f"📈 Estimated Profit: ₹{(predicted - last) * (5000 / last):.2f}",
            "🟡 Risk: Medium"
        ]
    elif -0.5 <= pct <= 0.5:
        msgs = [f"⚖️ Hold: Small change ({pct:.2f}%)", "🕒 Action: Wait"]
    else:
        msgs = [f"🔻 Avoid: Expected drop of {pct:.2f}%", "❌ Action: Do not invest"]
    return msgs, pct

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="FinVoice", layout="centered")
st.markdown("<style>body, .stApp { background-color: #ADD8E6; color: black; }</style>", unsafe_allow_html=True)

st.title("📊 FinVoice: Talk to AI for Smarter Investment")
st.markdown("Speak or type your company name(s) in any supported language and get predictions instantly.")

selected_lang = st.selectbox("Choose language:", list(languages.keys()))
lang_code = languages[selected_lang][1]

input_mode = st.radio("Input Mode", ["🎤 Voice", "⌨️ Text"], horizontal=True)

query = ""
if input_mode == "🎤 Voice":
    if st.button("Start Voice Input"):
        query = get_voice_input(selected_lang)
else:
    query = st.text_input("Enter company name(s), e.g., HDFC or TCS and RELIANCE").lower()

enable_voice = st.checkbox("🔊 Voice output for suggestions", value=True)

if query:
    matched = match_company_files(query)
    if matched:
        predictions = {}
        for company, file in matched:
            st.subheader(f"📈 {company} Prediction")
            X, y, scaler, df = load_and_prepare_data(file)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = build_model()
            model.fit(X, y, epochs=20, batch_size=32, verbose=0)

            last_close = df['close'].values[-1]
            last_60 = df['close'].values[-60:]
            last_scaled = scaler.transform(last_60.reshape(-1, 1))
            X_test = np.array([last_scaled]).reshape(1, 60, 1)

            prediction = model(X_test, training=False).numpy()
            predicted_price = scaler.inverse_transform(prediction)[0][0]
            predictions[company] = (last_close, predicted_price)

            st.write(f"Last Close: ₹{last_close:.2f}")
            st.write(f"Predicted: ₹{predicted_price:.2f}")

            # Plot
            plt.figure(figsize=(8, 3))
            plt.plot(df['close'].values[-50:], label="Recent")
            plt.plot([None]*49 + [last_close, predicted_price], label="Predicted", linestyle='--', marker='o')
            plt.legend()
            plt.title(f"{company} - Close Price & Forecast")
            st.pyplot(plt)

            # Suggestion
            original_msgs, _ = suggest_investment(last_close, predicted_price)
            translated_msgs = [translate_text(msg, lang_code) for msg in original_msgs]
            for msg in translated_msgs:
                st.info(msg)

            if enable_voice:
                full_text = " ".join(translated_msgs)
                speak_text(full_text, lang=lang_code)

        # Best investment message
        if len(predictions) > 1:
            best = max(predictions.items(), key=lambda x: ((x[1][1] - x[1][0]) / x[1][0]))
            best_company = best[0]
            msg = f" Best Investment Opportunity: {best_company}\nBased on the predictions, {best_company} has the highest expected rise!"
            translated = translate_text(msg, lang_code)
            st.success(translated)
            if enable_voice:
                speak_text(translated, lang=lang_code)

    else:
        st.warning("No matching company found. Try names like HDFC, TCS, RELIANCE, etc.")
