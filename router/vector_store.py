
import os
from dotenv import load_dotenv
from pathlib import Path
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType


env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)


MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "semantic_memory"
DIMENSION = 384

class VectorStore:
    def __init__(self):
        self.collection_name = COLLECTION_NAME
        self._connect()
        self._create_collection_if_not_exists()
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _connect(self):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            print(f"[VectorStore] Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            print(f"[VectorStore] Connection Failed: {e}")

    def _create_collection_if_not_exists(self):
        if utility.has_collection(self.collection_name):
            return
        
        print(f"[VectorStore] Creating collection '{self.collection_name}'...")
        
        # Define Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), 
            FieldSchema(name="metadata", dtype=DataType.JSON) 
        ]
        
        schema = CollectionSchema(fields, "Semantic Memory Storage")
        
        # Create
        collection = Collection(self.collection_name, schema)
        
        # Build Index for fast search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"[VectorStore] Collection created and indexed.")

    def search_similar(self, query_embedding: list, top_k: int = 3, threshold: float = 0.7):
        if not query_embedding:
            return []
            
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            matches = []
            for hit in results[0]:
                if hit.score < threshold: # Cosine similarity: 1.0 is exact match
                    continue
                    
                matches.append({
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.score,
                    "id": hit.id
                })
            
            return matches
            
        except Exception as e:
            print(f"[VectorStore] Search Error: {e}")
            return []

    def insert_memory(self, text: str, embedding: list, metadata: dict):
        try:
            
            
            data = [
                [embedding],
                [text],
                [metadata]
            ]
            
            self.collection.insert(data)
            self.collection.flush() 
            print(f"[VectorStore] Memory inserted.")
            
        except Exception as e:
            print(f"[VectorStore] Insert Error: {e}")

# Global instance
vector_store = VectorStore()
