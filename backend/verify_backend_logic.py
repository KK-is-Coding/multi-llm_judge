import sys
# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from orchestrator import Orchestrator

async def main():
    print("--- Initializing Orchestrator ---")
    orchestrator = Orchestrator()
    
    prompt = "best all time best playing 11 for team india??"
    print(f"\n[Step 1] Starting Workflow with prompt: '{prompt}'")
    
    # 1. Start Workflow
    result = await orchestrator.start_workflow(prompt)
    
    while True:
        workflow_id = result.get("workflow_id")
        status = result.get("status")
        draft = result.get("draft_answer")
        msg = result.get("message", "")
        
        print(f"\nStatus: {status}")
        if msg:
            print(f"System Message: {msg}")
        
        if status == "completed":
            print(f"Final Answer: {result.get('final_answer')}")
            print(f"Memory Saved: {result.get('memory_saved')}")
            print("\nWorkflow Completed Successfully.")
            break
            
        print(f"Draft Answer Preview: {draft[:150]}...")
        
        # Human Feedback Step
        print("\n[Human Feedback Required]")
        print("Draft Answer is above.\n")
        
        print("Enter feedback/corrections for the next cycle (or 'exit' to stop): ", end="")
        feedback = input().strip()
        
        if feedback.lower() == 'exit':
            print("Exiting...")
            break
            
        feedback_payload = {
            "workflow_id": workflow_id,
            "feedback": feedback
        }
        
        print("\nProcessing feedback... (this triggers re-generation)")
        result = await orchestrator.process_human_feedback(workflow_id, feedback_payload)

if __name__ == "__main__":
    asyncio.run(main())
