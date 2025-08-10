import os
import numpy as np
import pandas as pd
import faiss
import google.generativeai as genai
import psycopg2
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure PostgreSQL connection from environment variables
db_params = {
    'dbname': os.getenv('POSTGRES_DB', 'kkuai'),
    'user': os.getenv('POSTGRES_USER', 'r4ll0d'),
    'password': os.getenv('POSTGRES_PASSWORD', '020345Jk'),
    'host': os.getenv('POSTGRES_HOST', 'postgres_container'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

# Connect to PostgreSQL
def get_db_connection():
    try:
        return psycopg2.connect(**db_params)
    except psycopg2.Error as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        raise

# -- ขั้นตอนที่ 1: ตั้งค่าและโหลดข้อมูล Q&A จาก PostgreSQL --
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Fetch data from PostgreSQL
query = "SELECT category, question, answer FROM qa_data"
try:
    conn = get_db_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        print("Error: No data found in qa_data table")
        raise ValueError("No data found in qa_data table")
    print("DataFrame ที่โหลดจาก PostgreSQL เสร็จเรียบร้อย:")
    print(df)
except Exception as e:
    print(f"Error loading data from PostgreSQL: {e}")
    df = pd.DataFrame()

# -- ขั้นตอนที่ 2: สร้าง Vector DBs แยกตามหมวดหมู่ --
embedding_model = 'models/text-embedding-004'
vector_stores = {}
if not df.empty:
    all_categories = df['category'].unique()
    print("กำลังสร้าง Vector DBs แยกตามหมวดหมู่...")
    for category in all_categories:
        print(f"  - กำลังประมวลผลหมวดหมู่: {category}")
        
        category_df = df[df['category'] == category].reset_index(drop=True)
        questions_for_embedding = category_df['question'].tolist()
        question_embeddings = genai.embed_content(
            model=embedding_model,
            content=questions_for_embedding,
            task_type="retrieval_document"
        )["embedding"]
        
        d = np.array(question_embeddings).shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(question_embeddings))
        
        vector_stores[category] = {
            'index': index,
            'dataframe': category_df
        }
    print("\nสร้าง Vector DBs ทั้งหมดเสร็จสิ้น!")
else:
    print("Skipping Vector DB creation due to empty DataFrame")

# -- ขั้นตอนที่ 3: ฟังก์ชันถามคำถาม --
def ask_chatbot(user_question):
    print(f"\n[?] คำถาม: {user_question}")
    
    if not vector_stores:
        return "ขออภัย ระบบไม่สามารถโหลดข้อมูลจากฐานข้อมูลได้ กรุณาตรวจสอบการเชื่อมต่อฐานข้อมูลหรือข้อมูลในตาราง qa_data"
    
    classifier_model = genai.GenerativeModel('gemini-1.5-flash')
    category_list_str = "\n".join([f"- {cat}" for cat in vector_stores.keys()])
    classifier_prompt = f"""
    จากคำถามของผู้ใช้ โปรดวิเคราะห์และเลือกหมวดหมู่ที่เกี่ยวข้องที่สุดเพียง 1 หมวดหมู่จากรายการต่อไปนี้
    
    รายการหมวดหมู่:
    {category_list_str}

    คำถามของผู้ใช้: "{user_question}"

    โปรดตอบเฉพาะชื่อหมวดหมู่ที่เลือกเท่านั้น ห้ามมีคำอธิบายอื่น
    """
    
    response = classifier_model.generate_content(classifier_prompt)
    predicted_category = response.text.strip().replace("-","").strip()
    
    print(f"[🤖 ขั้นตอนที่ 1] วิเคราะห์หมวดหมู่ได้ว่า: {predicted_category}")

    if predicted_category in vector_stores:
        store = vector_stores[predicted_category]
        category_index = store['index']
        category_df = store['dataframe']

        question_embedding = genai.embed_content(
            model=embedding_model,
            content=user_question,
            task_type="retrieval_query"
        )["embedding"]
        
        distances, indices = category_index.search(np.array([question_embedding]), k=1)
        retrieved_answer = category_df.iloc[indices[0][0]]['answer']
        
        generator_prompt = f"""
        คุณคือ Chatbot ผู้ช่วยของมหาวิทยาลัยขอนแก่น
        โปรดตอบคำถามของผู้ใช้โดยอ้างอิงจาก "ข้อมูลสำหรับตอบคำถาม" ที่ให้มาเท่านั้นเท่านั้น
        และอย่าพยายามสร้างข้อมูลขึ้้นมาเอง
        
        ข้อมูลสำหรับตอบคำถาม: "{retrieved_answer}"
        คำถามของผู้ใช้: "{user_question}"
        
        คำตอบ (เรียบเรียงให้เป็นธรรมชาติ):
        """
        generator_model = genai.GenerativeModel('gemini-1.5-flash')
        final_response = generator_model.generate_content(generator_prompt)
        
        print(f"[🤖 ขั้นตอนที่ 2] คำตอบ: {final_response.text}")
        return final_response.text
    else:
        print("[🤖 ขั้นตอนที่ 2] ไม่สามารถระบุหมวดหมู่ที่ชัดเจนได้ ขอตอบแบบกว้างๆ")
        final_response = "ขออภัย ฉันไม่สามารถหาข้อมูลที่ตรงกับคำถามของคุณได้ในขณะนี้"
        print(f"[🤖] คำตอบ: {final_response}")
        return final_response

# -- ขั้นตอนที่ 4: สร้าง API ด้วย Flask --
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        user_question = data['question']
        response = ask_chatbot(user_question)
        return jsonify({'question': user_question, 'answer': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)