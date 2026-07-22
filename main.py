import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "router"))

from router.orchestrator import process_query

async def main():
    print("="*60)
    print("🤖 Multi-LLM Chatbot System (4-Layer Memory Architecture)")
    print("="*60)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            await process_query(user_input)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    try:
        # Quick health check of imports
        from router.judge import JUDGE_SYSTEM_PROMPT
        from router.intent import INTENT_SYSTEM_PROMPT
        print("[System] Core modules (Judge, Intent, Router) imported successfully.")
        asyncio.run(main())
    except ImportError as e:
        print(f"\n[CRITICAL ERROR] Failed to import required modules: {e}")
        print("Please ensure you are running from the correct directory and 'router' package exists.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] System startup failed: {e}")
        sys.exit(1)