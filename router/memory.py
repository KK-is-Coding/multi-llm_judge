import json
import os
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path(__file__).resolve().parent / "memory_store.json"

class MemoryStore:
    def __init__(self):
        self.MEMORY_FILE = MEMORY_FILE
        self._load_memory()

    def _load_memory(self):
        if not MEMORY_FILE.exists():
            self.memory = {}
            self._save_to_disk()
        else:
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
            except:
                self.memory = {}

    def _save_to_disk(self):
        with open(self.MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2)

    def get_intent_answer(self, intent_signature: str):
        """
        Returns the accepted answer if it exists for this intent.
        """
        record = self.memory.get(intent_signature)
        if record:
            # Update last_used_at if it exists (schema support)
            if "last_used_at" in record:
                record["last_used_at"] = datetime.now().isoformat()
                self._save_to_disk()
        return record

    def save_intent_answer(self, intent_data: dict, answer: str, generated_by_models: list, confidence: float, auto_saved: bool = False):
        """
        Saves the intent and answer to the JSON store with Rich Schema.
        intent_data must contain: 'intent_signature', 'domain', 'task', 'object'
        """
        signature = intent_data.get("intent_signature")
        if not signature:
            print("[Memory] Error: No intent_signature provided.")
            return

        timestamp = datetime.now().isoformat()
        
        # Check for existing record to handle history/versioning
        existing_record = self.memory.get(signature)
        history_log = existing_record.get("history_log", []) if existing_record else []
        
        # If updating an *existing* record, archive the OLD answer
        if existing_record:
            archive_entry = {
                "archived_at": timestamp,
                "previous_answer": existing_record.get("approved_answer") or existing_record.get("answer"),
                "previous_confidence": existing_record.get("confidence")
            }
            history_log.append(archive_entry)
            
        new_record = {
            "intent": signature, # Composite key
            "domain": intent_data.get("domain", "general"),
            "task": intent_data.get("task", "unknown"),
            "object": intent_data.get("object", "unknown"),
            "approved_answer": answer,
            "source": {
                "generated_by": generated_by_models,
                "judge": "Gemini_Judge",
                "human_verified": not auto_saved, # If auto-saved, it is NOT verified
                "auto_saved": auto_saved
            },
            "confidence": confidence,
            "created_at": existing_record.get("created_at", timestamp) if existing_record else timestamp,
            "last_used_at": timestamp,
            "history_log": history_log
        }
        
        self.memory[signature] = new_record
        self._save_to_disk()

    def list_intents(self):
        return list(self.memory.keys())
        
    def get_intents_by_domain(self, domain: str) -> list:
        """
        Returns a list of intent signatures that belong to the specified domain.
        """
        return [k for k, v in self.memory.items() if v.get("domain") == domain]

# Global instance
memory = MemoryStore()
