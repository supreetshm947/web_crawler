import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

url_cache = {}

def scrape_and_store(url, vector_store):

    if url in url_cache:
        logger.info(f"🔹 Using cached embeddings for {url}")
        return url_cache[url]

    logger.info(f"🌐 Scraping {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch page: {url}")

    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])  # Extract paragraph text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)


    vector_store.add_texts(text_chunks)

    url_cache[url] = text_chunks
    logger.info(f"✅ {len(text_chunks)} chunks stored for {url}")

