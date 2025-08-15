import streamlit as st
import subprocess
import pandas as pd
import os
import re
import glob
import pickle
import torch
import tempfile
import gdown
from transformers import AutoTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
/* Background keseluruhan */
.stApp {
    background: linear-gradient(135deg, #ffecd2, #fcb69f); /* Peach pastel */
    font-family: 'Segoe UI', sans-serif;
    color: #222;
}

/* Judul utama */
h1 {
    text-align: center;
    color: #2c3e50;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f9f9f9);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

/* Tombol */
button[kind="primary"] {
    background: linear-gradient(90deg, #4facfe, #00f2fe) !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    transition: all 0.3s ease;
}
button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* Dataframe */
.dataframe {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Box Panduan */
.panduan {
    font-family: 'Segoe UI', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    background: linear-gradient(135deg, #ffffff, #f8f9fa);
    padding: 18px 20px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

/* Teks dalam Box Panduan */
.panduan p {
    margin: 0;
    font-size: 0.95rem;
}

/* Animasi hover untuk card */
div[data-testid="stHorizontalBlock"] > div {
    transition: all 0.3s ease;
}
div[data-testid="stHorizontalBlock"] > div:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.keyword-btn {
    display: inline-block;
    padding: 8px 16px;
    margin: 6px 6px 0 0;
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: white;
    border-radius: 25px;
    font-size: 14px;
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
    border: none;
    transition: all 0.3s ease;
}
.keyword-btn:hover {
    background: linear-gradient(135deg, #43e97b, #38f9d7);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)


# ==============================
# Judul
# ==============================
st.title("📊 Twitter Keyword Scraper & Sentiment Analysis")

# ==============================
# Panduan Pengguna
# ==============================
# Variabel untuk menyimpan keyword yang dipilih
if "selected_keyword" not in st.session_state:
    st.session_state.selected_keyword = ""

# Fungsi untuk memilih keyword
def set_keyword(kw):
    st.session_state.selected_keyword = kw

# Panduan
with st.expander("📘 Panduan Penggunaan", expanded=False):
    st.markdown("""
    <div class="panduan">
    <b>Langkah-langkah penggunaan:</b><br>
    1. Masukkan <b>keyword pencarian</b> tweet.<br>
    2. Atur <b>limit</b> jumlah tweet yang ingin diambil.<br>
    3. Klik tombol <b>"Ambil Data dari Twitter"</b> untuk mulai scraping.<br>
    4. Pilih kolom teks yang akan dianalisis.<br>
    5. Klik <b>"Analisis Data"</b> untuk memproses dan melihat hasil analisis sentimen.<br>
    6. Unduh hasil dalam bentuk CSV dengan tombol <b>Download</b>.<br><br>
    
    <b>Catatan:</b>
    <ul>
        <li>Pastikan koneksi internet stabil.</li>
        <li>Gunakan keyword spesifik agar hasil lebih relevan.</li>
        <li>Klik salah satu keyword di bawah untuk otomatis terisi.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Tombol keyword
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Barak Militer", use_container_width=True):
            set_keyword("Barak Militer")
    with col2:
        if st.button("Barak Anak Nakal", use_container_width=True):
            set_keyword("Barak Anak Nakal")
    with col3:
        if st.button("Anak Nakal Barak Militer", use_container_width=True):
            set_keyword("Anak Nakal Barak Militer")

# ==============================
# Token Twitter
# ==============================
twitter_auth_token = "a2f1685ce71dc7af5dc850b9f8b2534a785cba61"

keyword = st.text_input("🔍 Masukkan Keyword Pencarian" , value=st.session_state.selected_keyword, key="keyword")
limit = st.number_input("📌 Limit jumlah tweet", min_value=10, max_value=1000, value=100, step=10)

output_folder = "tweets-data"
os.makedirs(output_folder, exist_ok=True)
filename = None

# Tools & Stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan', 'pada', 'juga', 'karena',
    'ada', 'tidak', 'sudah', 'saja', 'lebih', 'akan', 'bagi', 'para', 'sebagai', 'oleh',
    'tentang', 'maka', 'atau', 'jadi', 'namun'
])

# ==============================
# Preprocessing
# ==============================
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|pic.twitter\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.lower().strip()

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def tokenize_text(text):
    return text.split()

# Kata/frasa kunci yang diizinkan
allowed_keywords = [
    "barak militer",
    "anak nakal barak militer",
    "barak anak nakal",
    ""
]

def filter_by_keywords(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in allowed_keywords)

# ==============================
# Load Model & Tokenizer
# ==============================
@st.cache_resource
def load_model_and_tokenizer():
    try:
        file_id = "1xiww2ex8Mnaa-WrVQYZpd4MEg04aWRKs"
        drive_url = f"https://drive.google.com/uc?id={file_id}"
        temp_path = os.path.join(tempfile.gettempdir(), "model_temp.pkl")

        if not os.path.exists(temp_path):
            with st.spinner("📥 Mengunduh model dari Google Drive..."):
                gdown.download(drive_url, temp_path, quiet=False)

        with open(temp_path, "rb") as f:
            model = pickle.load(f)

        # Patch config untuk kompatibilitas versi baru
        if not hasattr(model.config, "output_attentions"):
            model.config.output_attentions = False
        if not hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = False

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer

    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

# ==============================
# Prediksi Sentimen
# ==============================
def predict_sentiment_local(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        label_map = {0: "Negatif", 1: "Positif", 2: "Netral"}
        return [label_map[pred.item()] for pred in predictions]

# ==============================
# Ambil Data
# ==============================
if st.button("📥 Ambil Data dari Twitter"):
    if not keyword:
        st.warning("⚠️ Mohon masukkan keyword terlebih dahulu.")
    else:
        command = [
            "tweet-harvest",
            "-o", output_folder,
            "-s", keyword,
            "--tab", "LATEST",
            "-l", str(limit),
            "--token", twitter_auth_token
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                csv_files = glob.glob(os.path.join(output_folder, "*.csv"))
                if csv_files:
                    filename = max(csv_files, key=os.path.getctime)
                else:
                    st.error("❌ CSV tidak ditemukan.")
            else:
                st.error(f"❌ Error saat mengambil data:\n{result.stderr}")
        except Exception as e:
            st.error(f"❌ Terjadi error:\n{str(e)}")

# ==============================
# Cek file terbaru
# ==============================
if filename is None:
    csv_files = glob.glob(os.path.join(output_folder, "*.csv"))
    if csv_files:
        filename = max(csv_files, key=os.path.getctime)

# ==============================
# Analisis Data
# ==============================
if filename and os.path.exists(filename):
    df = pd.read_csv(filename)
    df = df[df.apply(lambda row: filter_by_keywords(" ".join(map(str, row.values))), axis=1)]

    if df.empty:
        st.warning("⚠️ Tidak ditemukan tweet yang sesuai kata/frasa kunci.")
    else:
        st.subheader("📋 Data Mentah (Contoh 5 baris)")
        st.dataframe(df.head())

        text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 10]

        if text_columns:
            if "full_text" in text_columns:
                selected_col = "full_text"
            else:
                selected_col = text_columns[0]
            st.markdown(
                f"""
                <div style="
                    padding: 14px;
                    border-radius: 10px;
                    background: linear-gradient(135deg, #56CCF2, #2F80ED);
                    color: white;
                    margin-bottom: 12px;
                    font-size: 16px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                    ">
                    📂 <b>Kolom teks yang digunakan untuk analisis:</b> {selected_col}
                </div>
                """,
                unsafe_allow_html=True
            )


            if st.button("📊 Analisis Data"):
                with st.spinner("⏳ Sedang memproses..."):
                    df["clean_text"] = df[selected_col].apply(clean_text)
                    df["no_stopwords"] = df["clean_text"].apply(remove_stopwords)
                    df["tokenized"] = df["no_stopwords"].apply(tokenize_text)
                    df["stemmed"] = df["tokenized"].apply(lambda tokens: " ".join([stemmer.stem(word) for word in tokens]))
                    df["sentiment"] = predict_sentiment_local(df["stemmed"].tolist())
                    
                    st.subheader("📄 Hasil Preprocessing & Sentimen")
                    st.dataframe(df[[selected_col, "clean_text", "no_stopwords", "tokenized", "stemmed", "sentiment"]])

                    st.subheader("📈 Visualisasi Sentimen")
                    sentiment_counts = df["sentiment"].value_counts().reset_index()
                    sentiment_counts.columns = ["sentiment", "count"]
                    fig = px.pie(
                        sentiment_counts,
                        values="count",
                        names="sentiment",
                        title="Distribusi Sentimen",
                        hole=0.3
                    )
                    fig.update_traces(
                        textinfo="percent+label",
                        pull=[0.05] * len(sentiment_counts),  
                        marker=dict(line=dict(color="#FFFFFF", width=2))  
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',   
                        paper_bgcolor='rgba(0,0,0,0)',  
                        title_font=dict(size=20, color="white"),
                        legend_title=dict(font=dict(color="white")),
                        legend=dict(font=dict(color="white")),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("☁️ WordCloud per Sentimen")
                    for label in df["sentiment"].unique():
                            label_text = " ".join(df[df["sentiment"] == label]["stemmed"])
                            if label_text.strip():
                                wordcloud = WordCloud(
                                    width=800, height=400, background_color="white",
                                    max_words=100, colormap="viridis"
                                ).generate(label_text)
                                fig_wc, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation="bilinear")
                                ax.axis("off")

                                # Tambahkan judul sesuai label sentimen
                                ax.set_title(f"WordCloud - Sentimen: {label}", fontsize=16, fontweight="bold", pad=20)

                                # Tampilkan di Streamlit
                                st.pyplot(fig_wc)

                    st.download_button("⬇️ Download Hasil Sentimen", df.to_csv(index=False), "hasil_sentimen.csv", "text/csv")
        else:
            st.warning("⚠️ Tidak ditemukan kolom teks yang valid.")
else:
    st.info("ℹ️ Belum ada data yang tersedia.")
