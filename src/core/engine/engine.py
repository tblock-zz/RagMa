from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode
from typing import List
from .retriever import LocalRetriever
from ...setting import RAGSettings


class LocalChatEngine:
    def __init__(
        self, setting: RAGSettings | None = None
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._retriever = LocalRetriever(self._setting)

    def set_engine(
        self,
        llm: LLM,
        nodes: List[BaseNode],
        language: str = "eng",
        vector_index=None,
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:
        # Normal chat engine
        # Only use simple chat if no nodes provided AND (no vector index OR empty vector index)
        has_docs = len(nodes) > 0
        if not has_docs and vector_index is not None:
             try:
                 # Check if index has nodes in docstore
                 has_docs = len(vector_index.docstore.docs) > 0
             except:
                 has_docs = False

        if not has_docs:
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.ollama.chat_token_limit
                ),
            )

        # Chat engine with documents
        retriever = self._retriever.get_retrievers(
            llm=llm, 
            language=language, 
            nodes=nodes,
            vector_index=vector_index
        )
        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=ChatMemoryBuffer(token_limit=self._setting.ollama.chat_token_limit),
        )
