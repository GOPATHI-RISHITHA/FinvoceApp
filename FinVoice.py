# Translated version of FinVoice with multilingual output

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
import base64

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
translator = Translator()

# Language code map
languages = {
    "English": ("en-IN", "en"),
    "Hindi": ("hi-IN", "hi"),
    "Telugu": ("te-IN", "te"),
    "Tamil": ("ta-IN", "ta"),
    "Kannada": ("kn-IN", "kn"),
    "Bengali": ("bn-IN", "bn")
}

company_mapping = {
    "hdfc": "HDFC_Bank.csv",
    "tcs": "TCS.csv",
    "reliance": "RELIANCE.csv",
    "infosys": "INFOSYS.csv",
    "hindustan": "HINDUSTAN.csv",
    "itc": "ITC.csv"
}

def set_background(png_file_path):
    with open(png_file_path, "rb") as f:
        data_url = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def translate_text(text, lang_code):
    if lang_code == "en":
        return text
    try:
        return translator.translate(text, dest=lang_code).text
    except:
        return text

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
        pygame.mixer.quit()
        os.remove(temp_path)
    except Exception as e:
        st.warning(f"Voice output failed: {e}")

def get_voice_input(selected_lang_display):
    lang_code, trans_code = languages[selected_lang_display]
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(f"🎤 Speak now in {selected_lang_display}...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=10, phrase_time_limit=8)
    try:
        query = r.recognize_google(audio, language=lang_code)
        if trans_code != 'en':
            query = translator.translate(query, src=trans_code, dest='en').text
        return query.lower()
    except Exception as e:
        st.error(f"Voice Error: {str(e)}")
        return ""

def match_company_files(query):
    matched_companies = []
    if "which company" in query or not any(company in query for company in company_mapping.keys()):
        matched_companies = [(name.upper(), fname) for name, fname in company_mapping.items()]
    else:
        matched_companies = [(name.upper(), fname) for name, fname in company_mapping.items() if name in query]
    return matched_companies

def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
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
        msgs = [f" Strong Buy: Expected rise of {pct:.2f}%",
                " Suggested Investment: ₹10,000",
                f" Estimated Profit: ₹{(predicted - last) * (10000 / last):.2f}",
                " Risk: Low"]
    elif 0.5 < pct <= 2:
        msgs = [f" Moderate Buy: Expected rise of {pct:.2f}%",
                " Suggested Investment: ₹5,000",
                f" Estimated Profit: ₹{(predicted - last) * (5000 / last):.2f}",
                " Risk: Medium"]
    elif -0.5 <= pct <= 0.5:
        msgs = [f" Hold: Small change ({pct:.2f}%)", "🕒 Action: Wait"]
    else:
        msgs = [f" Avoid: Expected drop of {pct:.2f}%", " Action: Do not invest"]
    return msgs, pct

# UI setup
st.set_page_config(page_title="FinVoice", layout="centered")
set_background("D:/PROJECT/image2.png")
st.markdown("""
    <style>
    /* Remove dark overlay */
    .stApp::before {
        background: none !important;
    }

    /* Global text color */
    html, body, [class*="css"]  {
        color: #000000 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    /* Inputs and widgets */
    .stTextInput input, .stSelectbox div, .stRadio div, .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        border: 1px solid #00000033 !important;
        border-radius: 10px;
    }

    ::placeholder {
        color: #555555 !important;
    }

    /* Alert boxes */
    .stAlert, .stInfo, .stSuccess, .stError, .stWarning {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        border: 1px solid #00000033 !important;
    }

    /* White content block with padding and subtle shadow */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1> FinVoice: Talk to AI for Smarter Investment</h1>", unsafe_allow_html=True)

selected_lang = st.selectbox("🌐 Choose language:", list(languages.keys()))
lang_code = languages[selected_lang][1]
input_mode = st.radio("🗣️ Input Mode", ["🎤 Voice", "⌨️ Text"], horizontal=True)

query = ""
if input_mode == "🎤 Voice":
    if st.button("🎙️ Start Voice Input"):
        query = get_voice_input(selected_lang)
else:
    query = st.text_input("Enter company name(s), or ask 'Which company to invest in?'", key="input").lower()

enable_voice = st.checkbox("🔊 Enable voice output", value=True)



if query:
    translated_input = translate_text(query, lang_code)

    st.markdown("### 🗨️ Input Statement:")
    st.markdown(f"- **English:** `{query}`")
    st.markdown(f"- **{selected_lang}:** `{translated_input}`")

    if enable_voice:
        speak_text(f"You said: {translated_input}", lang=lang_code)
    matched = match_company_files(query)
    if not matched:
        st.warning(translate_text("No matching company found. Try HDFC, TCS, RELIANCE, etc.", lang_code))
    else:
        predictions = {}
        suggestion_msgs = {}

        # Always evaluate all companies for best recommendation
        all_companies = [(name.upper(), fname) for name, fname in company_mapping.items()]

        for company, file in all_companies:
            try:
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

                msgs, _ = suggest_investment(last_close, predicted_price)
                translated_msgs = [translate_text(msg, lang_code) for msg in msgs]
                suggestion_msgs[company] = translated_msgs

            except Exception as e:
                st.error(f"Failed for {company}: {e}")

        # Show prediction details only for matched companies
        for company, file in matched:
            st.markdown(f"<h2>📈 {company} Prediction</h2>", unsafe_allow_html=True)
            last_close, predicted_price = predictions[company]
            time_estimate = "in next 3–5 days" if abs(predicted_price - last_close)/last_close > 0.02 else "in next 7–10 days"
            st.write(translate_text("Last Close", lang_code) + f": ₹{last_close:.2f}")
            st.write(translate_text("Predicted", lang_code) + f": ₹{predicted_price:.2f} ({translate_text(time_estimate, lang_code)})")

            df = pd.read_csv(company_mapping[company.lower()])
            df.columns = [col.strip().lower() for col in df.columns]
            if 'close' not in df.columns:
                st.error("CSV file missing 'close' column.")
                st.stop()
            df['close'] = df['close'].astype(str).str.replace(',', '').astype(float)

            plt.figure(figsize=(8, 3))
            plt.plot(df['close'].values[-50:], label="Recent")
            plt.plot([None]*49 + [last_close, predicted_price], label="Prediction", linestyle='--', marker='o')
            plt.title(f"{company} - Close Price & Forecast")
            plt.legend()
            st.pyplot(plt)
            plt.clf()

            for msg in suggestion_msgs[company]:
                st.info(msg)

            if enable_voice:
                speak_text(" ".join(suggestion_msgs[company]), lang=lang_code)

        if predictions:
            best = max(predictions.items(), key=lambda x: ((x[1][1] - x[1][0]) / x[1][0]))
            best_company = best[0]
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header(translate_text("Best Investment Opportunity", lang_code))
            st.success(translate_text(f"Best company to invest is {best_company}.", lang_code))
            for msg in suggestion_msgs[best_company]:
                st.info(msg)
            if enable_voice:
                speak_text(f"{translate_text('Best company to invest is', lang_code)} {best_company}. " + " ".join(suggestion_msgs[best_company]), lang=lang_code)
