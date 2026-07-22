
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from router.orchestrator import process_query

if __name__ == "__main__":
    print("Multi-LLM Orchestrator Started.")
    print("-------------------------------")
    
    while True:
        query = input("\nUser Query (Ctrl+C to quit): ").strip()
        if not query: continue
        
        asyncio.run(process_query(query))
