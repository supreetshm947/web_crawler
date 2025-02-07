from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyOutputTransport
from pipecat.services.google import GoogleLLMService
from pipecat.frames.frames import Frame, LLMFullResponseEndFrame
from pipecat.services.google.google import GoogleLLMContext
from google.ai.generativelanguage_v1beta.types.content import Content, Part
from dotenv import load_dotenv
import os
import asyncio
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frame_processor import FrameProcessor
from typing import List, Optional, Dict, Any
from googlellmsvcwrapper import GoogleLLMServiceWrapper

load_dotenv()

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



    llm = GoogleLLMServiceWrapper(
        transport=transport,
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-pro",
        params=GoogleLLMService.InputParams(
        temperature=0.7,
        max_tokens=1000
    ))

    context = GoogleLLMContext(
        messages=[
            Content(
                role="user",
                parts=[Part(
                    text="You are a helpful assistant. Your goal is to demonstrate your capabilities in a succinct way. Keep your responses brief and creative.")]
            ),
            Content(
                role="model",
                parts=[Part(text="Hello! I'm your helpful AI assistant. How can I assist you today?")]
            )
        ]
    )

    context_aggregator = llm.create_context_aggregator(context)

    class SendMessage(DailyOutputTransport):
        def __init__(self, transport: DailyTransport, **kwargs):
            super().__init__(transport._client, transport._params, **kwargs)
            self.transport: transport

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            if isinstance(frame, LLMFullResponseEndFrame):
                print("ok we can send mesage here")
            await self.push_frame(frame, direction)




    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        llm,
        # transport.output(),
        context_aggregator.assistant(),
        #
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
        system_message = context.messages[-1]
        await transport.send_prebuilt_chat_message( system_message.parts[-1].text, "Chatbot")

    @transport.event_handler("on_app_message")
    async def on_chat_message(transport, data, sender):
        if "message" in data:
            context.messages.append(
                Content(role="user", parts=[Part(text=data["message"])])
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main(os.getenv("YOUR_ROOM_URL")))