import os
import streamlit as st
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoModelForSeq2SeqLM

# Atur direktori cache khusus Hugging Face
HF_CACHE_DIR = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# Inisialisasi session_state untuk menyimpan entri
if "entries" not in st.session_state:
    st.session_state.entries = []

# Load pipelines dan cache ke disk
@st.cache_resource
def load_pipelines():
    cache_dir = "./hf_cache"

    # Summarizer
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("farizkuy/bart-xsum-finetuned-fariz", cache_dir=cache_dir)
    summarizer_tokenizer = AutoTokenizer.from_pretrained("farizkuy/bart-xsum-finetuned-fariz", cache_dir=cache_dir)
    summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

    # NER
    ner_model = AutoModelForTokenClassification.from_pretrained("farizkuy/bert-laskar-ner", cache_dir=cache_dir)
    ner_tokenizer = AutoTokenizer.from_pretrained("farizkuy/bert-laskar-ner", cache_dir=cache_dir)
    ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

    # Emotion classifier (TensorFlow model)
    emo_model = TFAutoModelForSequenceClassification.from_pretrained(
        "farizkuy/emotion_tf",
        cache_dir=cache_dir
    )
    emo_tokenizer = AutoTokenizer.from_pretrained("farizkuy/emotion_tf", cache_dir=cache_dir)
    emo_pipeline = pipeline(
        "text-classification",
        model=emo_model,
        tokenizer=emo_tokenizer,
        top_k=None,
        framework="tf"
    )

    return summarizer, ner_pipeline, emo_pipeline

summarizer, ner_model, emotion_classifier = load_pipelines()

st.title("📓 Aplikasi Jurnal Harian dengan Analisis Emosi & Entitas")

# --- FORM INPUT ENTRI ---
with st.form("entry_form"):
    st.subheader("✍️ Tambahkan Entri Baru")
    entry_date = st.date_input("Tanggal Entri", datetime.today())
    entry_text = st.text_area("Isi Jurnal")
    submitted = st.form_submit_button("Simpan Entri")

    if submitted and entry_text.strip():
        st.session_state.entries.append({
            "date": entry_date,
            "text": entry_text.strip()
        })
        st.success("✅ Entri disimpan!")

# --- TAMPILKAN ENTRI YANG SUDAH ADA ---
if st.session_state.entries:
    with st.expander("📚 Lihat Semua Entri"):
        for e in st.session_state.entries:
            st.markdown(f"**{e['date']}**\n\n> {e['text']}")

# --- ANALISIS ---
st.subheader("📊 Analisis Entri Berdasarkan Rentang Tanggal")

if st.session_state.entries:
    dates = [e["date"] for e in st.session_state.entries]
    min_date, max_date = min(dates), max(dates)

    date_range = st.date_input("Pilih rentang tanggal", value=(min_date, max_date))

    if len(date_range) == 2:
        start_date, end_date = date_range

        selected_entries = [e for e in st.session_state.entries if start_date <= e["date"] <= end_date]

        if selected_entries:
            texts = [e["text"] for e in selected_entries]
            combined_text = " ".join(texts)

            st.markdown("### ☁️ WordCloud")
            wc = WordCloud(width=800, height=400, background_color="white").generate(combined_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            st.markdown("### 😊 Klasifikasi Emosi")
            emotion_results = []
            for text in texts:
                preds = emotion_classifier(text)
                for p in preds[0]:  # ambil top-k
                    emotion_results.append(p['label'])

            if emotion_results:
                emotion_df = pd.DataFrame(emotion_results, columns=["emotion"])
                emotion_counts = emotion_df["emotion"].value_counts().reset_index()
                emotion_counts.columns = ["Emotion", "Count"]
                fig = px.bar(emotion_counts, x="Emotion", y="Count", title="Distribusi Emosi")
                st.plotly_chart(fig)

            st.markdown("### 🧠 Ringkasan Entri")
            if len(combined_text.split()) >= 30:
                summary = summarizer(combined_text, max_length=130, min_length=30, do_sample=False)
                st.success(summary[0]['summary_text'])
            else:
                st.info("Teks terlalu pendek untuk diringkas.")

            st.markdown("### 🧬 Named Entity Recognition (NER)")
            ner_results = ner_model(combined_text)
            entity_counter = {}

            for ent in ner_results:
                label = ent['entity_group'].upper()
                entity_counter[label] = entity_counter.get(label, 0) + 1

            if entity_counter:
                entity_df = pd.DataFrame(entity_counter.items(), columns=["Entity", "Count"])
                fig = px.pie(entity_df, values='Count', names='Entity', title='Distribusi Entitas Terdeteksi')
                st.plotly_chart(fig)

        else:
            st.warning("Tidak ada entri pada rentang tanggal yang dipilih.")
else:
    st.info("Belum ada entri. Silakan tambahkan entri terlebih dahulu.")
