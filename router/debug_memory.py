
from memory import memory
import json
from datetime import datetime

print("--- Memory Debugger ---")
# print(f"Memory File Path: {memory.MEMORY_FILE}")

# 1. Print current state
print(f"Current Keys in Memory: {list(memory.memory.keys())}")

# 2. Try simple save
test_intent = {
    "intent_signature": "debug_test_intent",
    "domain": "debug"
}
test_answer = "This is a debug answer."
test_models = ["DebugModel"]

print(f"\n[Action] Attempting to save intent: {test_intent['intent_signature']}")
try:
    memory.save_intent_answer(test_intent, test_answer, test_models)
    print("[Success] save_intent_answer called without error.")
except Exception as e:
    print(f"[Error] save_intent_answer failed: {e}")

# 3. Verify file on disk
print("\n[Action] Reading file from disk...")
try:
    with open("memory_store.json", 'r') as f:
        data = json.load(f)
        if "debug_test_intent" in data:
            print("[Success] 'debug_test_intent' FOUND in JSON file.")
            print(json.dumps(data["debug_test_intent"], indent=2))
        else:
            print("[FAILURE] 'debug_test_intent' NOT FOUND in JSON file.")
except Exception as e:
    print(f"[Error] Could not read file: {e}")
