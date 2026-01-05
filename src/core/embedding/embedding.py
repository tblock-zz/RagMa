import os
import torch
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoModel, AutoTokenizer
from ...setting import RAGSettings
from dotenv import load_dotenv

load_dotenv()

class LocalEmbedding:
    @staticmethod
    def set(setting: RAGSettings | None = None, **kwargs):
        setting = setting or RAGSettings()
        model_name = setting.ingestion.embed_llm
        
        if model_name == "text-embedding-ada-002":
            return OpenAIEmbedding()
        elif "/" not in model_name:
            # Assume local/Ollama model if no slash (e.g. "nomic-embed-text")
            # Add :latest tag if not present
            if ":" not in model_name:
                model_name = f"{model_name}:latest"
            
            ollama_url = f"http://localhost:11434"
            print(f"[DEBUG] Connecting to Ollama at: {ollama_url} with model: {model_name}")
            
            return OllamaEmbedding(
                model_name=model_name,
                base_url=ollama_url,
                ollama_additional_kwargs={"mirostat": 0}
            )
        else:
            return HuggingFaceEmbedding(
                model=AutoModel.from_pretrained(
                    model_name, torch_dtype=torch.float16, trust_remote_code=True
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    model_name, torch_dtype=torch.float16
                ),
                cache_folder=os.path.join(os.getcwd(), setting.ingestion.cache_folder),
                trust_remote_code=True,
                embed_batch_size=setting.ingestion.embed_batch_size,
            )

    @staticmethod
    def pull(**kwargs):
        setting = RAGSettings()
        payload = {"name": setting.ingestion.embed_llm}
        return requests.post(f"http://localhost:11434/api/pull", json=payload, stream=True)

    @staticmethod
    def check_model_exist(**kwargs) -> bool:
        setting = RAGSettings()
        data = requests.get(f"http://localhost:11434/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        
        # Check exact match first
        if setting.ingestion.embed_llm in list_model:
            return True
        
        # Check if model name matches without tag (e.g., "nomic-embed-text" matches "nomic-embed-text:latest")
        for model in list_model:
            if model.startswith(setting.ingestion.embed_llm + ":"):
                return True
        
        return False
