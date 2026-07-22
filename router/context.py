
class ContextManager:
    def __init__(self, max_turns=10):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2: # 2 messages per turn
            self.history = self.history[-self.max_turns*2:]

    def get_context_formatted(self) -> str:
        """Returns history formatted for the LLM prompt."""
        formatted = ""
        for msg in self.history:
            role = "AI" if msg["role"] == "assistant" else "USER"
            formatted += f"{role}: {msg['content']}\n"
        return formatted

    def get_history(self) -> list:
        return self.history

    def clear(self):
        self.history = []

class EntityTraceMemory:
    """
    Medium-term memory for tracking recently mentioned entities.
    Survives topic switches to allow 'skip-back' references.
    """
    def __init__(self, max_size=5):
        self.entities = [] # List of dicts: {name, type, domain}
        self.max_size = max_size

    def add_entity(self, name: str, type: str, domain: str):
        # Avoid duplicates, move to top if exists
        self.entities = [e for e in self.entities if e['name'].lower() != name.lower()]
        self.entities.insert(0, {"name": name, "type": type, "domain": domain})
        if len(self.entities) > self.max_size:
            self.entities = self.entities[:self.max_size]

    def get_recent_entities(self):
        return self.entities

    def resolve_reference(self, domain_filter=None):
        """Returns the most recent entity matching the filter."""
        for e in self.entities:
            if domain_filter and e['domain'] != domain_filter:
                continue
            return e
        return None
