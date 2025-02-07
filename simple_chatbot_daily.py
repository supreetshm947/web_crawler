from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.services.daily import DailyParams, DailyTransport
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import ChatCohere
from pipecat.pipeline.runner import PipelineRunner
from google.ai.generativelanguage_v1beta.types.content import Content, Part
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from rag_processor import LangchainRAGProcessor
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator
from dotenv import load_dotenv
import os
import asyncio

from typing import Optional
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics

load_dotenv()

message_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]

async def main(room_url: str):
    transport = DailyTransport(
        room_url,
        None,  # No token needed for public rooms
        "Chatbot",
        DailyParams(
            api_key=os.getenv("DAILY_API_KEY"),
            chat_enabled=True
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Be nice and helpful. Answer very briefly"
             ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

    llm = prompt | ChatCohere()

    history_chain = RunnableWithMessageHistory(
        llm,
        get_session_history,
        history_messages_key="chat_history",
        input_messages_key="input")
    lc = LangchainRAGProcessor(transport, history_chain)

    tma_in = LLMUserResponseAggregator()
    tma_out = LLMAssistantResponseAggregator()

    tma_in.messages.append({
        "role": "model",
        "content": "Hello! I'm your helpful AI assistant. How can I assist you today?"
    })

    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.frames.frames import Frame

    class URLProcessor(FrameProcessor):
        def __init__(
                self,
                *,
                name: Optional[str] = None,
                metrics: Optional[FrameProcessorMetrics] = None,
                **kwargs,
        ):
            super().__init__(name=name, metrics=metrics, **kwargs)
        # async def process_frame(self, frame: Frame, direction: FrameDirection) -> Frame:
        #     # await super().process_frame(frame, direction)
        #     # query = frame.messages[-1].parts[0].text  # Extract user query
        #     # import re
        #     # # Check if query contains a URL
        #     # url_match = re.search(r'https?://\S+', query)
        #     # if url_match:
        #     #     url = url_match.group(0)
        #     #     print(f"🔍 Found URL: {url}. Crawling...")
        #     #
        #     # return frame
        #     await super().process_frame(frame, direction)
            # pass

    pipeline = Pipeline([
        transport.input(),
        # tma_in,
        # URLProcessor(),
        lc,
        # tma_out
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
        lc.set_participant_id("123")
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
