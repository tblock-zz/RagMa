import sys
import os
import shutil
# Add project root to path
sys.path.append(os.getcwd())

from src.core.vector_store import LocalVectorStore
from src.setting import RAGSettings
from llama_index.core.schema import TextNode

from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding

def test_persistence():
    print("----------------------------------------------------------------")
    print("Testing ChromaDB Persistence...")
    
    # Set mock embedding to avoid external API calls/errors during test
    Settings.embed_model = MockEmbedding(embed_dim=384) # 384 is common for BGE
    print("Configured MockEmebedding for testing.")

    # 1. Setup - Clean previous local data if exists for clean test
    settings = RAGSettings()
    persist_dir = settings.storage.persist_dir_chroma
    
    # Only clean if we are definitely using local fallback (checked via connection attempt)
    # But for safety in this test script, let's use a custom path or just append a test suffix
    # However, to test the ACTUAL code, we should probably just rely on the existing logic.
    # Let's try to initialize and see what happens.
    
    print(f"Initializing Vector Store...")
    try:
        vs = LocalVectorStore()
    except Exception as e:
        print(f"Failed to initialize LocalVectorStore: {e}")
        return

    # 2. Add Data
    print("\nCreating new index with test data...")
    node = TextNode(text="This is a persistent test node.", id_="test_node_1")
    try:
        index = vs.get_index(nodes=[node])
        print("Index created and node added.")
    except Exception as e:
        print(f"Error creating index: {e}")
        return

    # 3. Simulate Restart (Re-initialize)
    print("\nSimulating application restart...")
    del vs
    del index
    
    print("Re-initializing Vector Store...")
    vs_new = LocalVectorStore()
    
    # 4. Verify Data Exists
    print("Loading existing index...")
    try:
        index_new = vs_new.get_index() # No nodes passed
        # Simple verification: check if we can retrieve the node or if doc count > 0
        # Accessing the underlying vector store to check count
        count = index_new.vector_store._collection.count()
        print(f"Documents in collection: {count}")
        
        if count >= 1:
            print("SUCCESS: Data persisted and loaded!")
        else:
            print("FAILURE: Collection is empty.")
            
    except Exception as e:
        print(f"Error loading existing index: {e}")

if __name__ == "__main__":
    test_persistence()
