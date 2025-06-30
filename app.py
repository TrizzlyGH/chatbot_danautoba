import os
import random
from difflib import SequenceMatcher

import pandas as pd
from flask import Flask, jsonify, request

from llm_service import (
    get_chatbot_response_with_rag,
    ingest_data_to_vector_db
)

if not os.path.exists("chroma_db/chroma.sqlite3"):
    print("ChromaDB belum ada, melakukan ingest data...")
    vectordb = ingest_data_to_vector_db("data/data_toba_guide.csv", "chroma_db")
    if vectordb is not None:
        print(f"Jumlah dokumen di vector storage: {vectordb._collection.count()}")
        try:
            df_check = pd.read_csv("data/data_toba_guide.csv")
            print(f"Jumlah baris di CSV: {len(df_check)}")
        except Exception as e:
            print(f"Gagal membaca CSV untuk validasi: {e}")
else:
    print("ChromaDB sudah ada, skip ingest data.")

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# filepath: [app.py](http://_vscodecontentref_/7)
def search_csv_for_answer(user_message, df):
    search_cols = [
        'title', 'link', 'rating', 'reviews', 'address', 'latitude', 'longitude',
        'kategori', 'aktivitas', 'deskripsi', 'kecamatan'
    ]
    user_message_lower = user_message.lower()
    exclude_titles = []

    # Deteksi permintaan "selain ..."
    for title in df['title']:
        if f"selain {title.lower()}" in user_message_lower or f"kecuali {title.lower()}" in user_message_lower:
            exclude_titles.append(title.lower())

    # Deteksi pertanyaan spesifik
    for _, row in df.iterrows():
        title_lower = row['title'].lower()
        if title_lower in user_message_lower:
            if any(k in user_message_lower for k in ['lokasi', 'link', 'dimana', 'di mana', 'letak', 'alamat']):
                return [{'type': 'lokasi', 'data': row}]
            if any(k in user_message_lower for k in ['rating', 'bintang', 'nilai']):
                return [{'type': 'rating', 'data': row}]

    # Pencarian umum berbasis kemiripan
    results = []
    for _, row in df.iterrows():
        if row['title'].lower() in exclude_titles:
            continue
        for col in search_cols:
            val = str(row.get(col, '')).lower()
            if val and (val in user_message_lower or similar(val, user_message_lower) > 0.8):
                results.append({'type': 'umum', 'data': row})
                break
    return results

def format_response_towhere(where, user_message=None):
    templates = [
        f"{where['title']} terletak di {where['kecamatan']}. Kamu dapat menggunakan koordinat GPS ({where['latitude']}, {where['longitude']}) untuk menemukannya. Info lengkap di {where['link']}.",
        f"Lokasi {where['title']} berada di kecamatan {where['kecamatan']}. Alamat: {where['address']}. Cek koordinat: ({where['latitude']}, {where['longitude']}) dan kunjungi {where['link']}.",
        f"Kamu bisa menemukan {where['title']} di {where['kecamatan']} pada alamat {where['address']}. Koordinatnya adalah ({where['latitude']}, {where['longitude']}). Klik {where['link']} untuk detail lebih lanjut.",
        f"{where['title']} berlokasi di {where['address']}, {where['kecamatan']}. Pastikan untuk menggunakan koordinat ({where['latitude']}, {where['longitude']}). Kunjungi {where['link']} untuk info peta.",
        f"Untuk mencapai {where['title']}, kamu bisa pergi ke {where['address']} di {where['kecamatan']}. Lokasi GPS: ({where['latitude']}, {where['longitude']}). Info selengkapnya: {where['link']}.",
        f"{where['title']} berada di kawasan {where['kecamatan']}, dengan alamat lengkap: {where['address']}, dan terletak pada koordinat ({where['latitude']}, {where['longitude']}). Cek link: {where['link']}.",
        f"Tempat bernama {where['title']} bisa ditemukan di daerah {where['kecamatan']} (alamat: {where['address']}). Dengan koordinat ({where['latitude']}, {where['longitude']}), kamu bisa cek detailnya di {where['link']}.",
        f"{where['title']} dapat kamu kunjungi di {where['address']}, wilayah {where['kecamatan']}. Gunakan koordinat ({where['latitude']}, {where['longitude']}) dan informasi lebih lanjut di {where['link']}.",
        f"Jika kamu mencari {where['title']}, tempat ini berada di kecamatan {where['kecamatan']} dengan titik koordinat ({where['latitude']}, {where['longitude']}). Kunjungi {where['link']} untuk melihat lokasinya di peta.",
        f"{where['title']} berada di {where['kecamatan']} dan merupakan salah satu destinasi menarik di kawasan tersebut. Alamat: {where['address']}, koordinat: ({where['latitude']}, {where['longitude']}). Info peta: {where['link']}."
    ]
    return random.choice(templates)


def format_response_rating(rat):
    templates = [
        f"{rat['title']} memiliki rating sebesar {rat['rating']}.",
        f"Tempat ini, yaitu {rat['title']}, mendapatkan nilai {rat['rating']} dari pengunjung.",
        f"Rating dari {rat['title']} adalah {rat['rating']}, cukup menarik untuk dikunjungi!",
        f"Dengan skor {rat['rating']}, {rat['title']} menjadi salah satu tempat yang direkomendasikan.",
        f"{rat['title']} terletak di {rat['address']} dan memiliki rating {rat['rating']}.",
        f"Apakah kamu tahu? {rat['title']} punya rating {rat['rating']} menurut ulasan para wisatawan.",
        f"Jika kamu mencari tempat dengan rating bagus, {rat['title']} punya skor {rat['rating']}!",
        f"{rat['title']} mendapatkan penilaian {rat['rating']} dari para pengunjungnya.",
        f"Rating {rat['title']} adalah {rat['rating']} — nilai yang cukup baik menurut standar wisatawan.",
        f"Menurut data, {rat['title']} memiliki rating {rat['rating']} yang bisa jadi pertimbanganmu.",
        f"{rat['title']} menerima nilai {rat['rating']} dari pengunjung yang pernah berkunjung.",
        f"Dari skala rating yang ada, {rat['title']} memperoleh nilai {rat['rating']}.",
        f"Nilai rating {rat['title']} adalah {rat['rating']} yang menunjukkan kualitas yang cukup baik.",
        f"{rat['title']} dikenal dengan rating sebesar {rat['rating']} dari berbagai sumber review.",
        f"Rating {rat['title']} cukup tinggi, yakni {rat['rating']}.",
        f"Skor rating {rat['title']} menurut pengunjung adalah {rat['rating']}, layak untuk dikunjungi."
    ]
    return random.choice(templates)

def format_response_from_row(row):
    templates = [
        f"{row['title']} adalah tempat dengan kategori {row['kategori']} yang bisa kamu kunjungi di kecamatan {row['kecamatan']}. Tempat ini menawarkan aktivitas seperti {row['aktivitas']}. Deskripsinya: {row['deskripsi']}",
        f"Kamu dapat mengunjungi {row['title']}, yang berlokasi di kecamatan {row['kecamatan']}. Tempat ini terkenal dengan kategori {row['kategori']} dan aktivitas {row['aktivitas']}.",
        f"Jika kamu mencari tempat untuk {row['aktivitas']}, {row['title']} di kecamatan {row['kecamatan']} adalah pilihan yang tepat. Berikut deskripsinya: {row['deskripsi']}",
        f"{row['title']} termasuk dalam kategori {row['kategori']} dan berada di kecamatan {row['kecamatan']}. Tempat ini menawarkan berbagai aktivitas seperti {row['aktivitas']}.",
        f"Tempat bernama {row['title']} di kecamatan {row['kecamatan']} ini populer untuk aktivitas {row['aktivitas']}. Detail: {row['deskripsi']}",
        f"{row['title']} adalah salah satu tempat menarik di kecamatan {row['kecamatan']}. Dengan kategori {row['kategori']}, tempat ini menawarkan beragam aktivitas seperti {row['aktivitas']}. Jangan lewatkan pengalaman serunya!",
        f"Jika kamu mencari tempat untuk {row['aktivitas']}, maka {row['title']} bisa jadi pilihan menarik. Terletak di {row['kecamatan']}, tempat ini menawarkan pengalaman yang unik.",
        f"{row['title']} dikenal sebagai lokasi yang cocok untuk {row['aktivitas']}. Terletak di {row['kecamatan']}, tempat ini menawarkan suasana yang berbeda.",
        f"{row['title']}, yang berlokasi di kecamatan {row['kecamatan']}, menghadirkan berbagai aktivitas seru seperti {row['aktivitas']}. Tempat ini punya daya tarik tersendiri—berikut sedikit deskripsi: {row['deskripsi']}.",
        f"Tempat {row['title']} di kecamatan {row['kecamatan']} terkenal dengan kategori {row['kategori']} dan aktivitas yang ditawarkan seperti {row['aktivitas']}. Deskripsi singkat: {row['deskripsi']}.",
        f"Kamu bisa menemukan {row['title']} di kecamatan {row['kecamatan']}. Tempat ini cocok untuk aktivitas seperti {row['aktivitas']}, dengan kategori {row['kategori']}.",
        f"{row['title']} di kecamatan {row['kecamatan']} merupakan destinasi favorit bagi yang suka {row['aktivitas']}. Tempat ini memiliki deskripsi sebagai berikut: {row['deskripsi']}.",
        f"Destinasi {row['title']} berada di kecamatan {row['kecamatan']}. Tempat ini menawarkan kategori {row['kategori']} dan aktivitas seru seperti {row['aktivitas']}.",
        f"Jika kamu ingin beraktivitas {row['aktivitas']}, {row['title']} di kecamatan {row['kecamatan']} adalah tempat yang tepat. Deskripsi: {row['deskripsi']}.",
        f"{row['title']} adalah lokasi populer di kecamatan {row['kecamatan']} dengan kategori {row['kategori']} dan aktivitas yang bisa dilakukan antara lain {row['aktivitas']}."
    ]
    return random.choice(templates)

# ...fungsi format_response_towhere, format_response_rating, dst...

def get_top_destinations(df, n=5):
    df_sorted = df.sort_values(by=['rating', 'reviews'], ascending=[False, False])
    tops = df_sorted.head(n)
    lines = []
    for i, row in tops.iterrows():
        lines.append(f"{row['title']} (Deskripsi: {row['deskripsi']}, Kecamatan: {row['kecamatan']})")
    return "Berikut beberapa destinasi wisata paling terkenal di sekitar Danau Toba:\n" + "\n".join(lines)

def get_info_from_csv(user_message, df):
    return "Fungsi pencarian CSV belum diimplementasikan."

app = Flask(__name__)

# --- Muat Data CSV Saat Aplikasi Dimulai ---
# Pastikan path ke file CSV sudah benar
try:
    df_toba_info = pd.read_csv("data/data_toba_guide.csv")
    print("Data CSV berhasil dimuat di app.py.")
except FileNotFoundError:
    print("ERROR: File data_toba_guide.csv tidak ditemukan di folder 'data/'. Pastikan file ada!")
    df_toba_info = None # Pastikan df_toba_info tetap None jika file tidak ada

# filepath: [app.py](http://_vscodecontentref_/8)
def format_detail_row(row):
    return (
        f"Nama: {row['title']}\n"
        f"Latitude: {row['latitude']}\n"
        f"Longitude: {row['longitude']}\n"
        f"Kategori: {row['kategori']}\n"
        f"Aktivitas: {row['aktivitas']}\n"
        f"Kecamatan: {row['kecamatan']}\n"
        f"Deskripsi: {row['deskripsi']}\n"
    )

def find_exact_title(user_message, df):
    user_message_lower = user_message.lower()
    best_match = None
    best_score = 0
    
    print(f"DEBUG: Searching for '{user_message_lower}'")  # Debug
    
    for _, row in df.iterrows():
        title_lower = str(row['title']).lower()
        
        # Prioritaskan kecocokan kata kunci spesifik dulu
        if 'holbung' in user_message_lower and 'holbung' in title_lower:
            print(f"DEBUG: Found HOLBUNG match: {title_lower}")
            return row  # Langsung return jika ada match spesifik
        elif 'burung' in user_message_lower and 'burung' in title_lower:
            print(f"DEBUG: Found BURUNG match: {title_lower}")
            return row
        elif 'gibeon' in user_message_lower and 'gibeon' in title_lower:
            print(f"DEBUG: Found GIBEON match: {title_lower}")
            return row
        
        # Fallback: Hitung skor berdasarkan jumlah kata yang cocok
        user_words = set(user_message_lower.split())
        title_words = set(title_lower.split())
        common_words = user_words & title_words
        
        if len(common_words) > 0:
            # Prioritaskan yang memiliki lebih banyak kata cocok
            score = len(common_words) * 100 + len(title_lower)
            print(f"DEBUG: {title_lower} -> {common_words} -> score: {score}")
            
            if score > best_score:
                best_score = score
                best_match = row
    
    print(f"DEBUG: Best match: {best_match['title'] if best_match is not None else 'None'}")
    return best_match

def parse_multiple_destinations(user_message, df):
    """
    Parse pertanyaan user untuk mendeteksi multiple destinasi dan intent secara dinamis dari CSV.
    """
    try:
        print(f"DEBUG: Starting parse_multiple_destinations with message: '{user_message}'")
        user_message_lower = user_message.lower()
        mentioned_destinations = []
        # Ambil semua title dari CSV sebagai keyword dinamis
        for _, row in df.iterrows():
            title = str(row['title'])
            title_lower = title.lower()
            idx = user_message_lower.find(title_lower)
            if idx != -1:
                mentioned_destinations.append({
                    'keyword': title_lower,
                    'position': idx,
                    'row': row
                })
        print(f"DEBUG: Found {len(mentioned_destinations)} mentioned destinations (dynamic)")
        # Sort berdasarkan posisi kemunculan dalam kalimat user
        mentioned_destinations.sort(key=lambda x: x['position'])
        # Tentukan destinasi primer dan additional
        primary_destination = None
        additional_destinations = []
        if mentioned_destinations:
            primary_destination = mentioned_destinations[0]['row']
            additional_destinations = [dest['row'] for dest in mentioned_destinations[1:]]
        # Deteksi intent rekomendasi
        has_recommendation_request = any(phrase in user_message_lower for phrase in [
            'selain itu', 'apa lagi', 'rekomendasi', 'wisata lain', 'tempat lain'
        ])
        result = {
            'primary': primary_destination,
            'additional': additional_destinations,
            'has_recommendation_request': has_recommendation_request,
            'mentioned_count': len(mentioned_destinations)
        }
        print(f"DEBUG: Returning result: {result}")
        return result
    except Exception as e:
        print(f"ERROR in parse_multiple_destinations: {e}")
        import traceback
        traceback.print_exc()
        return {
            'primary': None,
            'additional': [],
            'has_recommendation_request': False,
            'mentioned_count': 0
        }

def format_comprehensive_response(parsed_data, df_toba_info):
    """
    Format response yang komprehensif berdasarkan parsed data
    """
    try:
        print("DEBUG: Starting format_comprehensive_response")
        response_parts = []
        
        # 1. Jawab destinasi primary
        if parsed_data['primary'] is not None:
            primary_row = parsed_data['primary']
            print(f"DEBUG: Processing primary destination: {primary_row['title']}")
            response_parts.append("=== DESTINASI UTAMA ===")
            response_parts.append(format_detail_row(primary_row))
        
        # 2. Jawab destinasi additional yang disebutkan eksplisit
        if parsed_data['additional']:
            print(f"DEBUG: Processing {len(parsed_data['additional'])} additional destinations")
            response_parts.append("\n=== DESTINASI LAIN YANG DISEBUTKAN ===")
            for additional_row in parsed_data['additional']:
                print(f"DEBUG: Processing additional destination: {additional_row['title']}")
                response_parts.append(format_detail_row(additional_row))
                response_parts.append("---")
        
        # 3. Berikan rekomendasi jika diminta
        if parsed_data['has_recommendation_request']:
            print("DEBUG: Processing recommendation request")
            response_parts.append("\n=== REKOMENDASI WISATA SERUPA ===")
            
            # Ambil destinasi serupa (kategori sama atau aktivitas serupa)
            if parsed_data['primary'] is not None:
                primary_kategori = parsed_data['primary']['kategori']
                mentioned_titles = [parsed_data['primary']['title'].lower()]
                if parsed_data['additional']:
                    mentioned_titles.extend([row['title'].lower() for row in parsed_data['additional']])
                
                print(f"DEBUG: Looking for recommendations in category: {primary_kategori}")
                print(f"DEBUG: Excluding titles: {mentioned_titles}")
                
                recommendations = []
                for _, row in df_toba_info.iterrows():
                    if (row['title'].lower() not in mentioned_titles and 
                        row['kategori'] == primary_kategori):
                        recommendations.append(
                            f"- {row['title']} (Kategori: {row['kategori']}, Kecamatan: {row['kecamatan']})"
                        )
                        print(f"DEBUG: Added recommendation: {row['title']}")
                    if len(recommendations) >= 3:
                        break
                
                if recommendations:
                    response_parts.extend(recommendations)
                else:
                    response_parts.append("- Tidak ada rekomendasi serupa ditemukan dalam kategori yang sama.")
        
        result = "\n".join(response_parts)
        print(f"DEBUG: Generated response length: {len(result)}")
        return result
        
    except Exception as e:
        print(f"ERROR in format_comprehensive_response: {e}")
        import traceback
        traceback.print_exc()
        return "Maaf, terjadi kesalahan dalam memformat response."

def detect_intent_and_entities(user_message, df):
    """
    Deteksi intent dan entity (destinasi, kategori, aktivitas, dsb) secara dinamis dari CSV.
    Return: dict {intent, entities, kategori, aktivitas, is_greeting, is_unknown}
    """
    user_message_lower = user_message.lower()
    # 1. Deteksi salam/basa-basi
    greetings = ["halo", "hai", "selamat pagi", "selamat siang", "selamat sore", "selamat malam", "assalamualaikum"]
    if any(greet in user_message_lower for greet in greetings):
        return {"intent": "greeting", "entities": [], "kategori": None, "aktivitas": None, "is_greeting": True, "is_unknown": False}
    # 2. Deteksi intent opini/umum
    opini_phrases = ["menurutmu", "bagi kamu", "kalau kamu", "apa yang menarik", "apa yang berkesan", "apa yang paling", "jika ya", "jika belum", "kamu pernah", "bagimu", "kenapa", "mengapa"]
    if any(phrase in user_message_lower for phrase in opini_phrases):
        return {"intent": "opini", "entities": [], "kategori": None, "aktivitas": None, "is_greeting": False, "is_unknown": False}
    # 3. Deteksi intent rekomendasi
    rekom_phrases = ["rekomendasi", "wisata lain", "tempat lain", "apa lagi", "selain itu", "selain ", "kecuali "]
    if any(phrase in user_message_lower for phrase in rekom_phrases):
        # Cek entity destinasi yang ingin dikecualikan
        excluded = []
        for title in df['title']:
            if f"selain {title.lower()}" in user_message_lower or f"kecuali {title.lower()}" in user_message_lower:
                excluded.append(title)
        # Cek kategori/aktivitas
        kategori = None
        aktivitas = None
        for kat in df['kategori'].unique():
            if kat and str(kat).lower() in user_message_lower:
                kategori = kat
        for akt in df['aktivitas'].unique():
            if akt and str(akt).lower() in user_message_lower:
                aktivitas = akt
        return {"intent": "recommendation", "entities": excluded, "kategori": kategori, "aktivitas": aktivitas, "is_greeting": False, "is_unknown": False}
    # 4. Deteksi intent detail destinasi
    mentioned = []
    for title in df['title']:
        if title.lower() in user_message_lower:
            mentioned.append(title)
    if mentioned:
        return {"intent": "detail", "entities": mentioned, "kategori": None, "aktivitas": None, "is_greeting": False, "is_unknown": False}
    # 5. Deteksi intent berdasarkan kategori/aktivitas
    for kat in df['kategori'].unique():
        if kat and str(kat).lower() in user_message_lower:
            return {"intent": "category", "entities": [], "kategori": kat, "aktivitas": None, "is_greeting": False, "is_unknown": False}
    for akt in df['aktivitas'].unique():
        if akt and str(akt).lower() in user_message_lower:
            return {"intent": "activity", "entities": [], "kategori": None, "aktivitas": akt, "is_greeting": False, "is_unknown": False}
    # 6. Jika tidak terdeteksi
    return {"intent": "unknown", "entities": [], "kategori": None, "aktivitas": None, "is_greeting": False, "is_unknown": True}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("DEBUG: Starting chat endpoint")
        data = request.json
        user_message = data.get('message')
        chat_history = data.get('history', [])

        if not user_message:
            return jsonify({"error": "Pesan tidak boleh kosong"}), 400

        print(f"DEBUG: Processing message: '{user_message}'")

        # --- Jawab langsung jika pertanyaan destinasi populer/terkenal ---
        if any(k in user_message.lower() for k in ['terkenal', 'populer', 'terbaik', 'favorit']):
            response = get_top_destinations(df_toba_info)
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"response": response, "history": chat_history})

        # --- INTENT DETECTION FIRST ---
        intent_data = detect_intent_and_entities(user_message, df_toba_info)
        print(f"DEBUG: Detected intent: {intent_data['intent']}")
        if intent_data.get('is_greeting'):
            return jsonify({'response': 'Halo! Ada yang bisa saya bantu seputar wisata Danau Toba? 😊', 'history': chat_history})
        if intent_data['intent'] == 'opini':
            rag_response, updated_history = get_chatbot_response_with_rag(user_message, chat_history)
            return jsonify({'response': rag_response, 'history': updated_history})
        # --- END INTENT DETECTION ---

        if df_toba_info is not None:
            print("DEBUG: DataFrame loaded, parsing multiple destinations")
            # Parse multiple destinations dan intent
            parsed_data = parse_multiple_destinations(user_message, df_toba_info)
            print(f"DEBUG: Parsed data: {parsed_data}")
            # Jika ada destinasi yang terdeteksi, proses semuanya
            if (parsed_data['primary'] is not None or 
                parsed_data['additional'] or 
                parsed_data['mentioned_count'] > 0):
                print("DEBUG: Found destinations, checking for special question type")
                row = parsed_data['primary']
                if row is not None:
                    msg = user_message.lower()
                    if any(k in msg for k in ['lokasi', 'dimana', 'di mana', 'letak', 'alamat']):
                        response = format_response_towhere(row, user_message)
                    elif any(k in msg for k in ['rating', 'bintang', 'nilai']):
                        response = format_response_rating(row)
                    else:
                        response = format_response_from_row(row)
                else:
                    response = format_comprehensive_response(parsed_data, df_toba_info)
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response})
                return jsonify({"response": response, "history": chat_history})
            print("DEBUG: No explicit destinations found, trying fuzzy search")
            rows = search_csv_for_answer(user_message, df_toba_info)
            if rows:
                # Integrasi formatter baru
                if isinstance(rows[0], dict) and 'type' in rows[0]:
                    if rows[0]['type'] == 'lokasi':
                        response = format_response_towhere(rows[0]['data'], user_message)
                    elif rows[0]['type'] == 'rating':
                        response = format_response_rating(rows[0]['data'])
                    else:
                        response = format_response_from_row(rows[0]['data'])
                else:
                    # fallback lama jika rows berupa DataFrame row
                    row = rows[0]
                    response = format_detail_row(row)
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response})
                return jsonify({"response": response, "history": chat_history})
        # Fallback ke LLM/RAG jika tidak ada jawaban dari CSV
        print("DEBUG: Falling back to LLM/RAG")
        response, updated_history = get_chatbot_response_with_rag(user_message, chat_history)
        return jsonify({"response": response, "history": updated_history})
    except Exception as e:
        print(f"ERROR in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Terjadi kesalahan internal", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)