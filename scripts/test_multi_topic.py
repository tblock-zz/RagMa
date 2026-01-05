import os
import sys
# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import LocalRAGPipeline
from llama_index.core import Document
import shutil

def test_multi_topic():
    pipeline = LocalRAGPipeline()
    
    # Topic 1: Fruits
    topic1 = "Fruits"
    pipeline.switch_topic(topic1)
    print(f"Switched to topic: {topic1}")
    
    # Manually create nodes for testing
    doc1 = Document(text="An apple is a sweet, edible fruit produced by an apple tree.", metadata={"file_name": "apple.txt"})
    nodes1 = pipeline._ingestion.store_nodes(input_files=[]) # Clear ingestion
    pipeline._vector_index.insert_nodes([doc1])
    pipeline._vector_index.storage_context.persist(persist_dir=pipeline._vector_store.get_persist_dir())
    print("Stored fruit info in Topic 1")
    
    # Topic 2: Planets
    topic2 = "Planets"
    pipeline.switch_topic(topic2)
    print(f"Switched to topic: {topic2}")
    
    doc2 = Document(text="Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System.", metadata={"file_name": "mars.txt"})
    pipeline._vector_index.insert_nodes([doc2])
    pipeline._vector_index.storage_context.persist(persist_dir=pipeline._vector_store.get_persist_dir())
    print("Stored planet info in Topic 2")
    
    # Verify Topics
    topics = pipeline.get_topics()
    print(f"Available topics: {topics}")
    assert topic1 in topics
    assert topic2 in topics
    
    # Query Topic 2 (Planets)
    pipeline.switch_topic(topic2)
    response = pipeline.query("QA", "What is Mars?", [])
    print(f"Querying {topic2}: {response.response}")
    assert "planet" in response.response.lower()
    assert "apple" not in response.response.lower()
    
    # Switch back to Topic 1 (Fruits)
    pipeline.switch_topic(topic1)
    response = pipeline.query("QA", "What is an apple?", [])
    print(f"Querying {topic1}: {response.response}")
    assert "fruit" in response.response.lower()
    assert "mars" not in response.response.lower()
    
    print("Multi-topic verification successful!")

if __name__ == "__main__":
    try:
        test_multi_topic()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
