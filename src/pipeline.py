from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    LocalVectorStore,
    get_system_prompt,
)
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole
#------------------------------------------------------------------------------
class LocalRAGPipeline:
    def __init__(self) -> None:
        self._language = "eng"
        self._model_name = ""
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        self._engine = LocalChatEngine()
        self._default_model = LocalRAGModel.set(self._model_name)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStore()
        Settings.llm = LocalRAGModel.set()
        Settings.embed_model = LocalEmbedding.set()        
        # Initialize persistent index
        self._vector_index = self._vector_store.get_index()        
        # Initialize query engine with existing index if available
        self._query_engine = self._engine.set_engine(
            llm=self._default_model, 
            nodes=[], 
            language=self._language,
            vector_index=self._vector_index
        )
    #----
    def get_model_name(self):
        return self._model_name
    #----
    def set_model_name(self, model_name: str):
        self._model_name = model_name
    #----
    def get_language(self):
        return self._language
    #----
    def set_language(self, language: str):
        self._language = language
    #----
    def get_system_prompt(self):
        return self._system_prompt
    #----
    def set_system_prompt(self, system_prompt: str | None = None):
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language, is_rag_prompt=self._ingestion.check_nodes_exist()
        )
    #----
    def set_model(self):
        Settings.llm = LocalRAGModel.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
        )
        self._default_model = Settings.llm
    #----
    def reset_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model, 
            nodes=[], 
            language=self._language,
            vector_index=self._vector_index
        )
    #----
    def reset_documents(self):
        self._ingestion.reset()
    #----
    def clear_conversation(self):
        self._query_engine.reset()
    #----
    def reset_conversation(self):
        self.reset_engine()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
        )
    #----
    def delete_database(self, entire_db: bool = False):
        """Clear the vector store (optionally entire DB) and reset the pipeline state."""
        if entire_db:
            self._vector_store.clear_all_database()
        else:
            self._vector_store.clear_database()            
        # Re-initialize state based on whatever topic the vector store is now on (fallback or default)
        self._vector_index = self._vector_store.get_index()
        self.reset_documents()
        self.reset_conversation()
    #----
    def get_topics(self) -> list[str]:
        """Get available topics."""
        return self._vector_store.get_topics()
    #----
    def get_current_topic(self) -> str:
        """Get the name of the current topic."""
        return self._vector_store._current_topic
    #----
    def switch_topic(self, topic_name: str):
        """Switch to a different topic and refresh the index."""
        self._vector_store.change_topic(topic_name)
        self._vector_index = self._vector_store.get_index()
        self.reset_documents()
        self.reset_conversation()
    #----
    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name)
    #----
    def pull_model(self, model_name: str):
        return LocalRAGModel.pull(model_name)
    #----
    def pull_embed_model(self, model_name: str):
        return LocalEmbedding.pull(model_name)
    #----
    def get_installed_models(self) -> list[str]:
        return LocalRAGModel.get_installed_models()
    #----
    def check_exist(self, model_name: str) -> bool:
        return LocalRAGModel.check_model_exist(model_name)
    #----
    def check_exist_embed(self, model_name: str) -> bool:
        return LocalEmbedding.check_model_exist(model_name)
    #----
    def store_nodes(self, input_files: list[str] = None) -> None:
        nodes = self._ingestion.store_nodes(input_files=input_files)
        if nodes:
            # Get current index and insert new nodes
            self._vector_index.insert_nodes(nodes)
            
            # Persist the storage context (docstore, index_store, etc.)
            self._vector_index.storage_context.persist(
                persist_dir=self._vector_store.get_persist_dir()
            )
            
            # Update query engine with refreshed index
            self.set_engine()
    #----
    def set_chat_mode(self, system_prompt: str | None = None):
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()
    #----
    def set_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._ingestion.get_ingested_nodes(),
            language=self._language,
            vector_index=self._vector_index
        )
    #----
    def get_history(self, chatbot: list[dict[str, str]]):
        history = []
        for chat in chatbot:
            if chat["role"] == "user":
                history.append(ChatMessage(role=MessageRole.USER, content=chat["content"]))
            elif chat["role"] == "assistant":
                history.append(ChatMessage(role=MessageRole.ASSISTANT, content=chat["content"]))
        return history
    #----
    def query(
        self, mode: str, message: str, chatbot: list[dict[str, str]]
    ) -> StreamingAgentChatResponse:
        if mode == "chat":
            history = self.get_history(chatbot)
            return self._query_engine.stream_chat(message, history)
        else:
            self._query_engine.reset()
            return self._query_engine.stream_chat(message)
