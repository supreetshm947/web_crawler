from typing import Union

from pipecat.processors.frameworks.langchain import LangchainProcessor

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable


from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
)

from pipecat.transports.services.daily import DailyTransport
import re
from fuzzywuzzy import fuzz
from loguru import logger


class LangchainRAGProcessor(LangchainProcessor):
    def __init__(self, transport: DailyTransport, chain: Runnable, transcript_key: str = "input"):
        super().__init__(chain, transcript_key)
        self._transport = transport
        self._chain = chain
        self._transcript_key = transcript_key

    @staticmethod
    def __get_token_value(text: Union[str, AIMessageChunk]) -> str:
        match text:
            case str():
                return text
            case AIMessageChunk():
                return text.content
            case dict() as d if 'answer' in d:
                return d['answer']
            case _:
                return ""

    @staticmethod
    def __detect_url_masking(query):
        # URL pattern (basic regex for detecting links)
        url_pattern = re.compile(r'https?://\S+|www\.\S+')

        # Common query patterns (can be expanded)
        retrieval_phrases = [
            "summarize the content of",
            "extract key points from",
            "get details from",
            "retrieve information from",
            "scrape and summarize",
            "what does"
        ]

        detected_url = url_pattern.search(query)
        if detected_url:
            url = detected_url.group()  # Extract URL

            # Check for fuzzy match with common retrieval phrases
            for phrase in retrieval_phrases:
                if fuzz.partial_ratio(phrase, query.lower()) > 80:
                    logger.debug(f"Detected masked query for URL: {url}")
                    return url

        return None

    async def _ainvoke(self, text: str):
        search_result = ""
        logger.debug(f"Invoking chain with {text}")
        if url:=self.__detect_url_masking(text):
            await self._transport.send_prebuilt_chat_message(f"Crawling {url}, it could take a while", "Chatbot")
            return


        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for token in self._chain.astream(
                    {self._transcript_key: text},
                    config={"configurable": {"session_id": self._participant_id}},
            ):
                c = self.__get_token_value(token)
                search_result += c
                await self.push_frame(TextFrame(c))
            await self._transport.send_prebuilt_chat_message(search_result, "Chatbot")
        except GeneratorExit:
            print(f"{self} generator was closed prematurely")
        except Exception as e:
            print(f"{self} an unknown error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())