from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.transports.services.daily import DailyParams, DailyTransport
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from pipecat.pipeline.runner import PipelineRunner
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from rag_processor import LangchainRAGProcessor
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator
from embedding_model import MyEmbeddingModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
import asyncio

load_dotenv()

message_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]

async def main(room_url: str):
    transport = DailyTransport(
        room_url,
        None,
        "Chatbot",
        DailyParams(
            api_key=os.getenv("DAILY_API_KEY"),
            chat_enabled=True
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant. For specific queries by user, TRY to answer questions using the provided context. "
             "If the question is too specific and you do not know the answer based on the given context, say:\n\n"
             "'I don't have enough knowledge on this. Please provide a URL so I can learn."
             ),
            ("user", "Context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

    embedding_dim = os.getenv("EMBEDDING_DIM")
    embedding_model = MyEmbeddingModel()
    index = faiss.IndexFlatL2(int(embedding_dim))

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    combine_docs_chain = create_stuff_documents_chain(
        ChatCohere(), prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever, combine_docs_chain
    )

    history_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        history_messages_key="chat_history",
        input_messages_key="input"
    )

    lc = LangchainRAGProcessor(transport, history_chain, vector_store)

    tma_in = LLMUserResponseAggregator()
    tma_out = LLMAssistantResponseAggregator()

    tma_in.messages.append({
        "role": "model",
        "content": "Hello! I'm your helpful AI assistant. How can I assist you today?"
    })

    pipeline = Pipeline([
        transport.input(),
        tma_in,
        lc,
        tma_out
    ])

    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True
        ),
    )

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        lc.set_participant_id(participant["id"])
        system_message = tma_in.messages[-1]
        await transport.send_prebuilt_chat_message(system_message["content"], "Chatbot")

    @transport.event_handler("on_app_message")
    async def on_chat_message(transport, data, sender):
        if "message" in data:
            tma_in.messages.append({
                "role": "user",
                "content": data["message"]
            })
            await task.queue_frame(LLMMessagesFrame(tma_in.messages))

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main(os.getenv("YOUR_ROOM_URL")))
