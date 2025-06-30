import os
import random
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from openai import OpenAI
from langchain.prompts import PromptTemplate
from operator import itemgetter

# --- 1. Muat variabel lingkungan dari file .env ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --- 2. Ambil API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY tidak ditemukan.")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY tidak ditemukan.")

# --- 3. Konfigurasi LLM (DeepSeek via OpenRouter) ---
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

LLM_MODEL_ID = "deepseek/deepseek-r1-0528:free"

# --- 4. Konfigurasi Embedding ---
embedding_function = MistralAIEmbeddings(
    api_key=MISTRAL_API_KEY,
    model="mistral-embed"
)

# --- 5. Fungsi untuk ingestion ke Vector Store ---
def ingest_data_to_vector_db(csv_file_path="data/data_toba_guide.csv", persist_directory="chroma_db"):
    if not os.path.exists(csv_file_path):
        print(f"File tidak ditemukan: {csv_file_path}")
        return None

    try:
        df = pd.read_csv(csv_file_path)
        print(f"CSV dimuat. Jumlah baris: {len(df)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# --- Global vector store ---
try:
    global_vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_function
    )
    print("ChromaDB dimuat.")
except Exception as e:
    print(f"ChromaDB gagal dimuat: {e}")
    global_vector_store = None

def get_relevant_context(retrieved_docs, question, top_k=5):
    scored = []
    q_lower = question.lower()
    for doc in retrieved_docs:
        score = 0
        metadata = doc.metadata
        if metadata.get("title", "").lower() in q_lower:
            score += 3
        if metadata.get("kategori", "").lower() in q_lower:
            score += 2
        if metadata.get("aktivitas", "").lower() in q_lower:
            score += 2
        if metadata.get("kecamatan", "").lower() in q_lower:
            score += 1
        scored.append((score, doc))
        rating = metadata.get("rating")
        if rating is not None:
            try:
                rating_float = float(rating)
                if rating_float >= 4.0:
                    score += 2
                elif rating_float >= 3.0:
                    score += 1
            except ValueError:
                pass    

    sorted_docs = sorted(scored, key=itemgetter(0), reverse=True)
    return [doc.page_content for _, doc in sorted_docs[:top_k]]

# --- Chatbot utama dengan RAG ---
def get_chatbot_response_with_rag(user_message: str, chat_history: list = None):
    if not chat_history:
        chat_history = [{
            "role": "system",
            "content": (
                "Anda adalah asisten AI khusus untuk pariwisata Danau Toba dan sekitarnya. "
                "PENTING: Anda HANYA boleh menjawab pertanyaan yang berkaitan dengan:"
                "\n• Destinasi wisata di kawasan Danau Toba dan Pulau Samosir"
                "\n• Aktivitas wisata (berenang, hiking, fotografi, kuliner lokal)"
                "\n• Informasi praktis (lokasi, jam buka, biaya masuk, akomodasi)"
                "\n• Rekomendasi perjalanan dan tips wisata"
                "\n• Budaya dan sejarah lokal Batak"
                "\n• Transportasi menuju dan di sekitar Danau Toba"
                "\n\nJika user bertanya tentang topik di luar pariwisata Danau Toba "
                "(seperti politik, berita umum, teknologi, dll), jawab dengan:"
                "\n'Maaf, saya adalah asisten khusus untuk pariwisata Danau Toba. "
                "Saya hanya dapat membantu dengan informasi seputar destinasi wisata, "
                "aktivitas, dan tips perjalanan di kawasan Danau Toba dan sekitarnya. "
                "Apakah ada yang ingin Anda ketahui tentang wisata Danau Toba?'"
                "\n\nGunakan informasi dari konteks yang diberikan sebagai referensi utama. "
                "Jawab dengan gaya ramah dan informatif seperti pemandu wisata lokal."
            )
        }]

    if global_vector_store is None:
        # Kalau vector DB gak ada, langsung ke LLM tanpa konteks
        messages_for_llm = chat_history + [{"role": "user", "content": user_message}]
    else:
        # 1. Cari dokumen relevan
        retrieved_docs = global_vector_store.similarity_search(user_message, k=20)

        # 2. Hitung skor dan pilih top-k terbaik berdasarkan metadata dan kecocokan
        context_texts = get_relevant_context(retrieved_docs, user_message, top_k=5)

        # Gabungkan konteks dari dokumen terpilih
        context_combined = "\n\n".join(context_texts)

        # 3. Buat prompt khusus dengan konteks + pertanyaan
        rag_prompt_template = """
Anda adalah asisten wisata khusus untuk kawasan Danau Toba.

IMPORTANT: HANYA jawab pertanyaan tentang pariwisata Danau Toba dan sekitarnya.
Jika pertanyaan di luar topik wisata Danau Toba, gunakan respons penolakan yang telah ditentukan.

KONTEKS:
{context}

PEDOMAN MENJAWAB:
• Periksa apakah pertanyaan berkaitan dengan pariwisata Danau Toba
• Jika TIDAK berkaitan dengan wisata Danau Toba, jawab: "Maaf, saya adalah asisten khusus untuk pariwisata Danau Toba. Saya hanya dapat membantu dengan informasi seputar destinasi wisata, aktivitas, dan tips perjalanan di kawasan Danau Toba dan sekitarnya. Apakah ada yang ingin Anda ketahui tentang wisata Danau Toba?"
• Jika berkaitan dengan wisata Danau Toba, berikan jawaban lengkap berdasarkan konteks
• Untuk informasi yang tidak tersedia dalam konteks, sampaikan dengan jujur

PERTANYAAN USER:
{question}

JAWABAN:
        """

        rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=rag_prompt_template
        )
        formatted_prompt = rag_prompt.format(context=context_combined, question=user_message)

        # 4. Kirim ke LLM API
        messages_for_llm = [{"role": "system", "content": formatted_prompt}] + chat_history[1:] + [{"role": "user", "content": user_message}]

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=messages_for_llm,
            stream=False,
            temperature=0,
            top_p=0.9,
            max_tokens=0
        )
        assistant_message = response.choices[0].message.content
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message, chat_history
    except Exception as e:
        print(f"LLM API error: {e}")
        chat_history.append({"role": "user", "content": user_message})
        return "Maaf, saya sedang tidak bisa menjawab saat ini.", chat_history

# --- CLI ---
if __name__ == "__main__":
    print("\n--- Memeriksa Database Vektor ---")
    need_ingest = False
    if not os.path.exists("chroma_db"):
        need_ingest = True
    else:
        try:
            test_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
            if test_store._collection.count() == 0:
                need_ingest = True
        except Exception as e:
            print(f"Gagal memuat ChromaDB: {e}")
            need_ingest = True

    if need_ingest:
        print("ChromaDB kosong atau rusak. Menjalankan ingestion...")
        ingest_data_to_vector_db()
        global_vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
    else:
        global_vector_store = test_store
        print("ChromaDB sudah siap.")

    print("\n--- Chatbot Toba Guide Siap ---")
    print("Ketik 'keluar' untuk keluar.\n")

    conversation_history = [{
        "role": "system",
        "content": (
            "Anda adalah asisten AI untuk Toba Guide. Tugas Anda adalah memberikan informasi "
            "pariwisata Danau Toba dan sekitarnya (termasuk Pulau Samosir). Gunakan informasi dari "
            "'Konteks:' yang akan diberikan. Jika 'Konteks:' tidak mencukupi, katakan terus terang bahwa "
            "Anda tidak memiliki informasi spesifik tersebut. Jawaban Anda harus terdengar alami, seperti "
            "seorang pemandu wisata yang sedang menjelaskan dengan ramah dan informatif."
        )
    }]

    while True:
        user_input = input("Anda: ")
        if user_input.lower() == 'keluar':
            print("Chatbot: Sampai jumpa!")
            break
        response, conversation_history = get_chatbot_response_with_rag(user_input, conversation_history)
        print(f"Chatbot: {response}")