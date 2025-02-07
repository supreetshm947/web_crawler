from pipecat.services.google import GoogleLLMService
from typing import Any, Dict, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)

from pipecat.services.google.frames import LLMSearchResponseFrame

try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
    from google import genai
    from google.cloud import texttospeech_v1
    from google.genai import types
    from google.generativeai.types import GenerationConfig
    from google.oauth2 import service_account
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set the environment variable GOOGLE_API_KEY for the GoogleLLMService and GOOGLE_APPLICATION_CREDENTIALS for the GoogleTTSService`."
    )
    raise Exception(f"Missing module: {e}")

from pipecat.transports.services.daily import DailyTransport


class GoogleLLMServiceWrapper(GoogleLLMService):
    def __init__(
            self,
            transport: DailyTransport,
            api_key: str,
            model: str = "gemini-1.5-flash-latest",
            params: GoogleLLMService.InputParams = GoogleLLMService.InputParams(),
            system_instruction: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_config: Optional[Dict[str, Any]] = None,
            **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            params=params,
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
            **kwargs,
        )
        self._transport = transport

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        grounding_metadata = None
        search_result = ""

        try:
            logger.debug(
                # f"Generating chat: {self._system_instruction} | {context.get_messages_for_logging()}"
                f"Generating chat: {context.get_messages_for_logging()}"
            )

            messages = context.messages
            if context.system_message and self._system_instruction != context.system_message:
                logger.debug(f"System instruction changed: {context.system_message}")
                self._system_instruction = context.system_message
                self._create_client()

            # Filter out None values and create GenerationConfig
            generation_params = {
                k: v
                for k, v in {
                    "temperature": self._settings["temperature"],
                    "top_p": self._settings["top_p"],
                    "top_k": self._settings["top_k"],
                    "max_output_tokens": self._settings["max_tokens"],
                }.items()
                if v is not None
            }

            generation_config = GenerationConfig(**generation_params) if generation_params else None

            await self.start_ttfb_metrics()
            tools = []
            if context.tools:
                tools = context.tools
            elif self._tools:
                tools = self._tools
            tool_config = None
            if self._tool_config:
                tool_config = self._tool_config
            response = await self._client.generate_content_async(
                contents=messages,
                tools=tools,
                stream=True,
                generation_config=generation_config,
                tool_config=tool_config,
            )
            await self.stop_ttfb_metrics()

            if response.usage_metadata:
                # Use only the prompt token count from the response object
                prompt_tokens = response.usage_metadata.prompt_token_count
                total_tokens = prompt_tokens

            async for chunk in response:
                if chunk.usage_metadata:
                    # Use only the completion_tokens from the chunks. Prompt tokens are already counted and
                    # are repeated here.
                    completion_tokens += chunk.usage_metadata.candidates_token_count
                    total_tokens += chunk.usage_metadata.candidates_token_count
                try:
                    for c in chunk.parts:
                        if c.text:
                            search_result += c.text
                            await self.push_frame(LLMTextFrame(c.text))
                        elif c.function_call:
                            logger.debug(f"Function call: {c.function_call}")
                            args = type(c.function_call).to_dict(c.function_call).get("args", {})
                            await self.call_function(
                                context=context,
                                tool_call_id="what_should_this_be",
                                function_name=c.function_call.name,
                                arguments=args,
                            )
                    # Handle grounding metadata
                    # It seems only the last chunk that we receive may contain this information
                    # If the response doesn't include groundingMetadata, this means the response wasn't grounded.
                    if chunk.candidates:
                        for candidate in chunk.candidates:
                            # logger.debug(f"candidate received: {candidate}")
                            # Extract grounding metadata
                            grounding_metadata = (
                                {
                                    "rendered_content": getattr(
                                        getattr(candidate, "grounding_metadata", None),
                                        "search_entry_point",
                                        None,
                                    ).rendered_content
                                    if hasattr(
                                        getattr(candidate, "grounding_metadata", None),
                                        "search_entry_point",
                                    )
                                    else None,
                                    "origins": [
                                        {
                                            "site_uri": getattr(grounding_chunk.web, "uri", None),
                                            "site_title": getattr(
                                                grounding_chunk.web, "title", None
                                            ),
                                            "results": [
                                                {
                                                    "text": getattr(
                                                        grounding_support.segment, "text", ""
                                                    ),
                                                    "confidence": getattr(
                                                        grounding_support, "confidence_scores", None
                                                    ),
                                                }
                                                for grounding_support in getattr(
                                                    getattr(candidate, "grounding_metadata", None),
                                                    "grounding_supports",
                                                    [],
                                                )
                                                if index
                                                   in getattr(
                                                    grounding_support, "grounding_chunk_indices", []
                                                )
                                            ],
                                        }
                                        for index, grounding_chunk in enumerate(
                                            getattr(
                                                getattr(candidate, "grounding_metadata", None),
                                                "grounding_chunks",
                                                [],
                                            )
                                        )
                                    ],
                                }
                                if getattr(candidate, "grounding_metadata", None)
                                else None
                            )
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logger.debug(
                            f"LLM refused to generate content for safety reasons - {messages}."
                        )
                    else:
                        logger.exception(f"{self} error: {e}")

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            if grounding_metadata is not None and isinstance(grounding_metadata, dict):
                llm_search_frame = LLMSearchResponseFrame(
                    search_result=search_result,
                    origins=grounding_metadata["origins"],
                    rendered_content=grounding_metadata["rendered_content"],
                )
                await self.push_frame(llm_search_frame)

            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())
            if search_result:
                await self._transport.send_prebuilt_chat_message( search_result, "Chatbot")