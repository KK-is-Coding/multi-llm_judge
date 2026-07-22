
import sys
# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import json
import sys
from pathlib import Path

# Add parent dir to sys.path to allow imports if run directly
sys.path.append(str(Path(__file__).parent.parent))

from router import intent, memory, llm_generators, judge
from router.context import ContextManager, EntityTraceMemory

# Initialize Context & Entity Trace
context_manager = ContextManager()
entity_trace = EntityTraceMemory()
IMPLICIT_TRIGGERS = {"jo", "us", "usi", "that", "it", "him", "her", "that movie", "that film", "woh"}

async def main_loop(prompt: str = None):
    print("--- Initializing Router System ---")
    
    if not prompt:
        # You can change this prompt to test different queries
        print("Enter your query: ", end="")
        prompt = input().strip()
    print(f"\n[Step 1] Starting Workflow with prompt: '{prompt}'")
    
    # --- STEP 0: RESOLVE IMPLICIT REFERENCES (Entity Trace) ---
    if any(w in prompt.lower().split() for w in IMPLICIT_TRIGGERS):
        print(f"[Router] Implicit trigger found in '{prompt}'")
        
        # DOMAIN INFERENCE
        domain_filter = None
        if any(w in prompt.lower() for w in ["movie", "film", "cinema", "actor", "show"]):
            domain_filter = "entertainment"
            print("[Router] Context implies domain: 'entertainment'")
            
        resolved_entity = entity_trace.resolve_reference(domain_filter=domain_filter)
        if resolved_entity:
            print(f"[Router] Resolved implicit reference -> {resolved_entity['name']} ({resolved_entity['type']})")
            prompt = f"{prompt} ({resolved_entity['name']})"
            print(f"[Router] Updated Prompt: '{prompt}'")
    
    # 1. Intent Extraction
    print(f"\n[Router] Extracting intent and domain...")
    intent_data = await intent.extract_intent_signature(prompt)
    intent_sig = intent_data["intent_signature"]
    current_domain = intent_data["domain"]
    print(f"Intent Signature: {intent_sig} | Domain: {current_domain}")
    
    # NEW: Update Entity Trace
    if intent_data.get("object") and intent_data.get("object") != "unknown_intent":
        entity_trace.add_entity(intent_data["object"], "entity", intent_data["domain"])
        print(f"[Router] Added '{intent_data['object']}' to Entity Trace Memory.")
    
    # 2. Check Memory (Exact + Semantic)
    cached = memory.memory.get_intent_answer(intent_sig)
    
    if not cached:
        # SEMANTIC FALLBACK
        print(f"[Router] Exact match failed. Checking semantic similarity in domain '{current_domain}'...")
        candidates = memory.memory.get_intents_by_domain(current_domain)
        if candidates:
            match = await judge.find_matching_intent(intent_sig, candidates)
            if match:
                print(f"[Router] Semantic Match Found! Mapping '{intent_sig}' -> '{match}'")
                intent_sig = match # Redirect to existing intent
                cached = memory.memory.get_intent_answer(intent_sig)
    
    if cached:
        print("\n[Status] Existing Intent found in memory.")
        answer_text = cached.get("approved_answer") or cached.get("answer")
        print(f"Memorized Answer: {answer_text}")
        # Display History if available
        history = cached.get("history_log", [])
        if history:
            print(f"\n[History] Found {len(history)} previous versions:")
            for idx, entry in enumerate(history, 1):
                print(f"   {idx}. [{entry['archived_at']}] {entry['previous_answer'][:50]}...")

        print("\n--- Running Judge (Adaptation) ---")
        # Judge adapts the stored answer to current phrasing
        final_response = await judge.judge_from_memory(prompt, answer_text)
        print(f"Final Adapted Answer: {final_response}")
        
        # Set variables for feedback loop
        draft = final_response
        generator_models = cached.get("source", {}).get("generated_by", ["Memory"])
        print("\n[Info] Proceeding to Feedback Loop (opportunity to refine memory)...")

    else:
        # 3. New Intent Flow
        print("\n[Status] New Intent. Calling Generator LLMs...")
        
        print("\n--- Model Responses ---")
        # Prepend history so Generators have context
        history_context = context_manager.get_context_formatted()
        augmented_prompt = f"""POST HISTORY:
{history_context}

CURRENT QUERY:
{prompt}"""

        # This calls all LLMs (Gemini, ChatGPT, Groq, Ollama)
        responses = await llm_generators.generate_all(augmented_prompt)
        generator_models = list(responses.keys())
        
        for model_name, res in responses.items():
            print(f"[{model_name}]: {res}")
            
        print("\n--- Running Judge ---")
        judge_result = await judge.judge_responses(prompt, responses)
        # Handle new schema
        draft = judge_result.get("corrected_answer") or judge_result.get("final_answer")

    print("\n" + "="*50)
    print("DRAFT FINAL ANSWER")
    print("="*50)
    print(f"{draft}")
    print("="*50)
    
    # 4. Human Feedback Step
    print("\n[Human Feedback Required]")
    print("Draft Answer is above.\n")
    
    print("Press [ENTER] to approve, or type your correction/feedback below:")
    
    final_answer = draft
    status = "pending"
    
    while True:
        feedback = input(">>> ").strip()
        
        if not feedback:
            print("\n[Feedback Received] Approved.")
            status = "approved"
            break
        else:
            # Simple check for demo (could call LLM, but let's match orchestrator logic)
            fb_class = await intent.classify_query(feedback, final_answer, intent_sig)
            
            if fb_class['query_type'] == 'follow_up':
                print(f"\n[Feedback Received] Routing to Judge for refinement...")
                # We treat the *latest* final_answer as the draft for the next round
                final_answer = await judge.review_correction(prompt, final_answer, feedback)
                print(f"\nRefined Answer:\n{final_answer}")
                print("\n" + "="*50)
                print("Press [ENTER] to approve this new version, or type further feedback:")
                status = "corrected"
            else:
                print(f"\n[Feedback Received] Classified as NEW QUESTION ({fb_class['reasoning']}).")
                
                # AUTO-SAVE LOGIC (The "Remember Previous" Feature)
                print(f"[Auto-Save] Saving current acceptable draft for '{intent_sig}' before switching topics...")
                try:
                    memory.memory.save_intent_answer(
                        intent_data=intent_data, 
                        answer=final_answer, 
                        generated_by_models=generator_models,
                        confidence=0.85, # UPGRADE 4: Lower confidence for auto-save
                        auto_saved=True # UPGRADE 4: Mark as unverified
                    )
                    print(f"[Success] Previous intent '{intent_sig}' auto-saved (Unverified).")
                except Exception as e:
                    print(f"[Error] Could not auto-save: {e}")

                print("Restarting Workflow for new query...")
                return await main_loop(feedback) # Calling self for the new prompt
        
    # 5. Save to Memory
    print(f"\n[Debug] Status: {status}")
    if status in ["approved", "corrected"]:
        file_path = memory.memory.MEMORY_FILE
        print(f"[Debug] Memory File Path: {file_path}")
        
        try:
            print("[Debug] Executing save_intent_answer...")
            memory.memory.save_intent_answer(
                intent_data=intent_data, 
                answer=final_answer, 
                generated_by_models=generator_models,
                confidence=0.95
            )
            print("[Debug] Save function returned.")
            
            # Immediate verification
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    if intent_sig in saved_data:
                        print(f"\n[SUCCESS] Confirmed '{intent_sig}' is on disk.")
                        # print(json.dumps(saved_data[intent_sig], indent=2))
                    else:
                        print(f"\n[FAILURE] Key '{intent_sig}' NOT found in file after save!")
                        print(f"Keys found: {list(saved_data.keys())}")
            else:
                print(f"[FAILURE] File {file_path} does not exist after save!")

            print(f"Memory Saved: True (Intent: {intent_sig})")
            
            # Update Context History (RAM)
            context_manager.add_turn("user", prompt)
            context_manager.add_turn("assistant", final_answer)
            print("[Debug] Added turn to Context Manager.")

            print("\nWorkflow Completed Successfully.")
            
        except Exception as e:
            print(f"[ERROR] Failed to save memory: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Debug] Workflow Aborted (Status not approved/corrected).")

if __name__ == "__main__":
    asyncio.run(main_loop())
