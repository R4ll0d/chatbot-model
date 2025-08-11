# KKU IntelliSphere Chatbot

A React-based AI chatbot interface for Khon Kaen University, inspired by Grok, with Docker deployment.

## Folder Structure
```
chatbot-model/
├── .env
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── init.sql
├── chatbot.py

```

## Prerequisites
- Docker installed on your machine

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd chatbot-model
   ```

2. **Build the Docker Image**:
   ```bash
   docker-compose up -d --build
   ```

3. **Run the Docker Container**:
   ```bash
   docker exec -it postgres_container psql -U r4ll0d -d kkuai
   
    CREATE TABLE qa_data (
    id SERIAL PRIMARY KEY,
    category VARCHAR(255) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL
    );
   
   ```
&copy; 2025 Khon Kaen University. Developed by the Office of Digital Technology.
