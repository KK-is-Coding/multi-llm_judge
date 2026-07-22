import json
import os
from pathlib import Path
from datetime import datetime

class MemoryStore:
    def __init__(self, filepath="memory_store.json"):
        self.filepath = filepath
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                self.data = []
        else:
            self.data = []

    def _save_memory(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def retrieve_memory(self, query: str):
        # validation: simple keyword matching for now, can be upgraded to vector search later
        # Returning all memory for now as the dataset is small
        return self.data

    def update_memory(self, memory_type: str, content: str):
        entry = {
            "type": memory_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.data.append(entry)
        self._save_memory()
        return entry
