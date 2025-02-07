from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.google import GoogleLLMService
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor
from google.ai.generativelanguage_v1beta.types.content import Content, Part
from dotenv import load_dotenv
import os
import asyncio
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import ChatCohere

from langchain_core.runnables import RunnableWithMessageHistory
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from rag_processor import LangchainRAGProcessor
import aiohttp
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent

load_dotenv()


prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Be nice and helpful. Answer very briefly"
                 ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])


@tool
def magic_function(input: int)->int:
    """Applies a magic function to an input"""
    with open("some_file.txt", "w") as file:
        file.write(f"{input+2}")
    return input+2

message_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]

tools = [magic_function]

llm = ChatCohere()
# llm=llm.bind_tools(tools)
#
# chain = prompt | llm
# chain = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

agent=create_tool_calling_agent(llm, tools, prompt)
chain = AgentExecutor(agent=agent, tools=tools, verbose=True)

history_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input")
lc = LangchainRAGProcessor(history_chain)
lc.set_participant_id("123")

llm = GoogleLLMService(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-pro",
    params=GoogleLLMService.InputParams(
        temperature=0.7,
        max_tokens=1000
    )
)


def get_user_input():
    return input("You: ")


def print_bot_response(response):
    print(f"Bot: {response}")


# context = OpenAILLMContext()
# # messages=[
# #     {"role": "system", "content": "You are a helpful assistant."}
# # ])
# context_aggregator = llm.create_context_aggregator(context)

tma_in = LLMUserResponseAggregator()
tma_out = LLMAssistantResponseAggregator()


from pipecat.processors.frame_processor import FrameProcessor
# from pipecat.frames.frames import Framef
#
# class URLProcessor(FrameProcessor):
#     async def process_frame(self, frame: Frame, direction: FrameDirection) -> Frame:
#         query = frame.messages[-1].parts[0].text  # Extract user query
#         import re
#         # Check if query contains a URL
#         url_match = re.search(r'https?://\S+', query)
#         if url_match:
#             url = url_match.group(0)
#             print(f"🔍 Found URL: {url}. Crawling...")
#
#         return frame




pipeline = Pipeline([
    # context_aggregator.user(),
    tma_in,
    # URLProcessor(),
    lc,
    # context_aggregator.assistant()
    tma_out
])





async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            task = PipelineTask(
                pipeline,
                PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True,
                    enable_usage_metrics=True,
                )
            )
            runner = PipelineRunner()
            user_input = input("You: ")

            # frame = LLMMessagesFrame([Content(role="user", parts=[Part(text=user_input)])])
            #
            # await task.queue_frames([frame])

            # context.messages.append(
            #     Content(role="user", parts=[Part(text=user_input)])
            # )
            tma_in.messages.append({
                "role": "user",
                "content": user_input
            })
            await task.queue_frame(LLMMessagesFrame(tma_in.messages))

            try:
                await asyncio.wait_for(runner.run(task), timeout=10)
            except asyncio.TimeoutError:
                print("Runner took too long to execute!")

            # Get the assistant's response from the context
            assistant_message = tma_out.messages[-1]
            assistant_response = assistant_message["content"]
            print_bot_response(assistant_response)


if __name__ == "__main__":
    asyncio.run(main())