from langchain_core.embeddings import Embeddings
import requests

class MyEmbeddingModel(Embeddings):
    def __init__(self, service_url='http://127.0.0.1:8080/predictions/my_model'):
        self.service_url = service_url

    def embed_query(self, text):
        payload = {"input": [text]}
        response = requests.post(self.service_url, json=payload)

        if response.status_code == 200:
            return response.json()[0]
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")

    def embed_documents(self, texts):
        payload = {"input": texts}
        response = requests.post(self.service_url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")