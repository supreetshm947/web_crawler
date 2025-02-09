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
## Architecture

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

---

## Pipeline

- ```DailyTransport``` initializes the Daily Chat room (```transport```).
- Pipecat binds together the agentic Flow as pipeline
```
 pipeline = Pipeline([
        transport.input(),
        tma_in,
        lc,
        tma_out
    ])
```
- ```tma_in``` mantains User context and ```tma_out``` mantains LLM Response/Assistant context.
- ```lc``` encapsulates a Langchain Retrieval Chain as an object of ```LLMUserResponseAggregator``` customized as ```LangchainRAGProcessor``` to implement fuzzy logic matching for URL and initiating web crawling.
- Pipeline gets initiates with ```on_chat_message``` listener on ```transport``` and initiates the Pipecat pipeline.

