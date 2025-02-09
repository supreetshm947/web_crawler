# 📚 Dynamic RAG Chatbot with Web Crawling

An AI assistant that dynamically learns from **web pages** using **retrieval-augmented generation (RAG)**.  

---

## 🛠️ Technologies Used
- **PipeCat** 🐈 → Building Pipeline
- **Daily** 💬 → Chatroom
- **LangChain** 🦜🔗 → Prompt chaining, retriever, and memory  
- **FAISS** 🏪 → In-memory vector database for storing embeddings  
- **all-MiniLM-L6-v2  Embeddings** 🤖 → Text vectorization  
- **ChatCohere** 🧠 → Language model for generating responses  
- **BeautifulSoup** 🍜 → Web scraping for learning new knowledge  
- **Python** 🐍 → The foundation of our AI chatbot  

---
## Workflow

<img src="workflow.png" alt="grid"><br>

---

## 📥 Setup

### 1️⃣ Set up environment variables
Add config in .env (.env_example for reference)

### 2️⃣ Setup Poetry Virtual Environment
```bash
poetry install
```

### 3️⃣ Start the Embedding Server
```
docker compose up
```

### 4️⃣ Run the rag_chatbot_server.py
```bash
py rag_chatbot_server.py
```
