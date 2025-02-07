from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google import GoogleLLMService
from pipecat.services.google.google import GoogleLLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.runner import PipelineRunner
from google.ai.generativelanguage_v1beta.types.content import Content, Part
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

async def main():
    llm_service = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-1.5-flash-latest"
    )

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

    context_aggregator = llm_service.create_context_aggregator(context)

    pipeline = Pipeline([
        context_aggregator.user(),
        llm_service,
        context_aggregator.assistant()
    ])





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

        if user_input.lower() == 'exit':
            break

        context.messages.append(
            Content(role="user", parts=[Part(text=user_input)])
        )
        await task.queue_frame(LLMMessagesFrame(context.messages))

        try:
            await asyncio.wait_for(runner.run(task), timeout=5)
        except asyncio.TimeoutError:
            print("Runner took too long to execute!")
        # await runner.run(task)
        assistant_message = context.messages[-1]
        assistant_response = assistant_message.parts[0].text
        print(f"Assistant: {assistant_response}")


if __name__ == "__main__":
    asyncio.run(main())
