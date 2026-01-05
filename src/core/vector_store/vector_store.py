from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from ...setting import RAGSettings

load_dotenv()


# ------------------------------------------------------------------------------
class LocalVectorStore:
  # ----------------------------------------------------------------------------
  def __init__(
    self,
    setting: RAGSettings | None = None,
  ) -> None:
    self._setting = setting or RAGSettings()
    
    # Initialize persistent local client
    import chromadb
    from chromadb.config import Settings
    
    print(f"Using local persistent storage at {self._setting.storage.persist_dir_chroma}")
    self._client = chromadb.PersistentClient(
      path=self._setting.storage.persist_dir_chroma,
      settings=Settings(anonymized_telemetry=False)
    )

    self._current_topic = self._setting.storage.collection_name
    self._collection = self._client.get_or_create_collection(
      name=self._current_topic
    )

  # ----------------------------------------------------------------------------
  def get_topics(self) -> list[str]:
    """List all available collections (topics) in ChromaDB."""
    return [c.name for c in self._client.list_collections()]

  # ----------------------------------------------------------------------------
  def change_topic(self, topicName: str):
    """Switch to a different collection."""
    if not topicName:
      return
    self._current_topic = topicName
    self._collection = self._client.get_or_create_collection(name=topicName)
    print(f"Switched to topic: {topicName}")

  # ----------------------------------------------------------------------------
  def get_persist_dir(self) -> str:
    """Get the storage directory for the current topic."""
    import os
    baseDir = self._setting.storage.persist_dir_storage
    if self._current_topic == self._setting.storage.collection_name:
      return baseDir
    return os.path.join(baseDir, self._current_topic)

  # ----------------------------------------------------------------------------
  def get_index(self, nodes=None):
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext, load_index_from_storage
    import os

    vectorStore = ChromaVectorStore(chroma_collection=self._collection)
    persistDir = self.get_persist_dir()
    
    if os.path.exists(persistDir) and os.path.exists(os.path.join(persistDir, "docstore.json")):
      print(f"Loading existing index from {persistDir}")
      storageContext = StorageContext.from_defaults(
        vector_store=vectorStore,
        persist_dir=persistDir
      )
      return load_index_from_storage(storageContext)

    storageContext = StorageContext.from_defaults(vector_store=vectorStore)
    if nodes and len(nodes) > 0:
      index = VectorStoreIndex(
        nodes=nodes, 
        storage_context=storageContext
      )
    else:
      index = VectorStoreIndex.from_vector_store(
        vectorStore,
        storage_context=storageContext,
      )
    
    # Ensure the directory exists before returning, though from_vector_store doesn't persist yet
    os.makedirs(persistDir, exist_ok=True)
    return index

  # ----------------------------------------------------------------------------
  def clear_database(self):
    """Delete all data for the current topic and NOT recreate the collection."""
    import shutil
    import os
    import gc
    import time
    
    # 1. Release handles and clear Chroma collection
    collectionToDelete = self._current_topic
    print(f"Deleting collection: {collectionToDelete}")
    
    # To avoid file locks on Windows/SQLite, we clear our references
    self._collection = None
    gc.collect()
    time.sleep(0.5)
    
    try:
      self._client.delete_collection(name=collectionToDelete)
    except Exception as e:
      print(f"Warning: Could not delete collection {collectionToDelete}: {e}")
      
    # 2. Clear Storage Context (LlamaIndex files)
    persistDir = self.get_persist_dir()
    if os.path.exists(persistDir):
      try:
        shutil.rmtree(persistDir)
        print(f"Cleared LlamaIndex storage at {persistDir}")
      except Exception as e:
        print(f"Warning: Could not delete storage directory {persistDir}: {e}")
      
    # 3. Fallback to default topic so we are not in a 'zombie' state
    fallbackTopic = self._setting.storage.collection_name
    # If we just deleted the default topic, we still need to ensure at least one exists
    self.change_topic(fallbackTopic)
      
    print(f"Topic '{collectionToDelete}' removed.")

  # ----------------------------------------------------------------------------
  def clear_all_database(self):
    """Delete all collections (topics) and the entire storage directory."""
    import shutil
    import os
    import gc
    import time
    
    # 1. Clear Storage Context (LlamaIndex)
    baseDirStorage = self._setting.storage.persist_dir_storage
    if os.path.exists(baseDirStorage):
      try:
        shutil.rmtree(baseDirStorage)
        print(f"Cleared LlamaIndex storage at {baseDirStorage}")
      except Exception as e:
        print(f"Warning: Could not fully clear storage directory: {e}")

    # 2. Clear ChromaDB
    # To avoid file locks on Windows, we try to 'close' the client by dereferencing it
    # and forcing garbage collection before deleting the directory.
    chromaPath = self._setting.storage.persist_dir_chroma
    
    print("Closing ChromaDB client to release file locks...")
    self._collection = None
    self._client = None
    gc.collect() # Force collection to close sqlite handles
    time.sleep(1) # Give OS a moment to release handles
    
    if os.path.exists(chromaPath):
      try:
        shutil.rmtree(chromaPath)
        print(f"Cleared ChromaDB directory at {chromaPath}")
      except Exception as e:
        print(f"Warning: Could not delete ChromaDB directory {chromaPath}: {e}")
        # Fallback: if we can't delete the directory, we'll try to delete collections later
    
    # 3. Re-initialize empty state
    import chromadb
    from chromadb.config import Settings
    self._client = chromadb.PersistentClient(
      path=chromaPath,
      settings=Settings(anonymized_telemetry=False)
    )
    self._current_topic = self._setting.storage.collection_name
    self._collection = self._client.get_or_create_collection(name=self._current_topic)
      
    print("Entire vector store and storage context cleared.")
