import faiss
import numpy as np
import os
from langchain_core.tools import tool
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_cohere import ChatCohere
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere embeddings & LLM
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)
llm = ChatCohere()

# Sample documents (Initial Knowledge Base)
documents = [
    Document(page_content="George Costanza is the master of deception."),
    Document(page_content="Arthur Vandaley wrote Venetian Blinds."),
    Document(page_content="Penny Pecker published Venetian Blinds."),
    Document(page_content="The chinese medicine claimed it would have made George's head like Stalin."),
]

# Convert documents to embeddings
texts = [doc.page_content for doc in documents]
doc_embeddings = embeddings.embed_documents(texts)

# Convert to NumPy array
doc_embeddings_np = np.array(doc_embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
index.add(doc_embeddings_np)

# Store texts alongside FAISS
index_to_docstore_id = {i: str(i) for i in range(len(documents))}  # Map index to doc IDs
docstore = {str(i): doc for i, doc in enumerate(documents)}  # Create a docstore

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)


# 🔹 Tool: Fake Web Scraper to Add Documents to Vector DB
@tool
def add_website_content(url: str) -> str:
    """
    Simulates fetching content from a webpage and adding it to the vector DB.
    """
    fake_content = f"Extracted useful data from {url}. It's about AI advancements."
    new_doc = Document(page_content=fake_content)

    # Embed new document
    new_embedding = embeddings.embed_documents([fake_content])
    new_embedding_np = np.array(new_embedding).astype("float32")

    # Add to FAISS index
    # index.add(new_embedding_np)
    # vector_store.add_documents([new_doc])

    return f"New content from {url} has been added!"


# 🔹 Create a RAG Retriever (FAISS-Based)
class FAISSRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        """
        Searches vector DB and retrieves relevant documents.
        """
        query_embedding = embeddings.embed_query(query)
        results = vector_store.similarity_search_by_vector(query_embedding, k=2)

        return results

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)


faiss_retriever = FAISSRetriever()

# 🔹 RAG Chain (Retriever + LLM)
rag_chain = create_retrieval_chain(llm, faiss_retriever)

# 🔹 Define the RAG Agent (LLM + Tools + RAG Retrieval + Memory)
tools = [
    Tool(name="add_content", func=add_website_content, description="If user supplies a url, use this to fetch content from a url."),
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    retriever=faiss_retriever  # 🔥 Now the agent can retrieve from RAG dynamically!
)

# 🔹 Test the Agent
while True:
    user_query = input("\nYou: ")

    if user_query.lower() == "exit":
        break

    # Let the LLM decide whether to call the tool or query RAG
    response = agent.run(user_query)

    print(f"Assistant: {response}")
