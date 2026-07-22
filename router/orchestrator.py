import asyncio
# State tracking for follow-ups (in memory variable for this session)
last_system_response = ""
last_intent_sig = ""
last_intent_data = {}
from . import intent
from .memory import memory
from .llm_generators import generate_all
from . import judge
from .feedback import get_user_feedback
from .context import ContextManager, EntityTraceMemory

# Initialize Context Manager
# Initialize Context Manager
context_manager = ContextManager()
entity_trace = EntityTraceMemory() # NEW: Trace Memory (Corrected Instantiation)

IMPLICIT_TRIGGERS = {"jo", "us", "usi", "that", "it", "him", "her", "that movie", "that film", "woh"}

async def process_query(user_query: str):
    global last_system_response, last_intent_sig, last_intent_data
    
    # --- STEP 0: RESOLVE IMPLICIT REFERENCES ---
    if any(w in user_query.lower().split() for w in IMPLICIT_TRIGGERS):
        print(f"[Router] distinct implicit trigger found in '{user_query}'")
        
        # DOMAIN INFERENCE
        domain_filter = None
        if any(w in user_query.lower() for w in ["movie", "film", "cinema", "actor", "show"]):
            domain_filter = "entertainment"
            print("[Router] Context implies domain: 'entertainment'")

        # Try to resolve from trace
        resolved_entity = entity_trace.resolve_reference(domain_filter=domain_filter) 
        if resolved_entity:
            print(f"[Router] Resolved implicit reference -> {resolved_entity['name']} ({resolved_entity['type']})")
            # Append context to query to help classifier
            user_query = f"{user_query} ({resolved_entity['name']})"
    
    print(f"\n[System] Processing: '{user_query}'")
    
    # 1. Classification (Follow-up vs New)
    history_str = context_manager.get_context_formatted()
    classification = await intent.classify_query(user_query, history_str, last_intent_sig)
    print(f"[Router] Classification: {classification['query_type']} ({classification['reasoning']})")
    
    # --- ROUTE 1: FOLLOW-UP (Contextual Refinement) ---
    if classification['query_type'] == 'follow_up':
        print("[Router] Routing to Judge for Refinement (Contextual Follow-up)...")
        # We treat this as a "correction" or enhancement of the previous answer
        # If we have no previous response, we must treat it as new, but the classifier handles that.
        
        refined_answer = await judge.review_correction(user_query, last_system_response, user_query)
        print(f"\n[Result] (Refined by Judge): {refined_answer}")
        
        last_system_response = refined_answer
        return refined_answer

    # --- ROUTE 2: NEW QUESTION (Standard Flow) ---
    
    # 2. Intent Extraction
    print("[Router] Extracting intent signature and domain...")
    intent_data = await intent.extract_intent_signature(user_query)
    current_intent_sig = intent_data["intent_signature"]
    current_domain = intent_data["domain"]
    print(f"[Router] Intent Signature: {current_intent_sig} | Domain: {current_domain}")
    
    # Update global intent state for NEXT turn
    last_intent_sig = current_intent_sig
    # Update global intent state for NEXT turn
    last_intent_sig = current_intent_sig
    last_intent_data = intent_data
    
    # NEW: Update Entity Trace
    if intent_data.get("object") and intent_data.get("object") != "unknown_intent":
        entity_trace.add_entity(intent_data["object"], "entity", intent_data["domain"])
        print(f"[Router] Added '{intent_data['object']}' to Entity Trace Memory.")
    
    # 3. Check Memory (Only if same domain context if we wanted to be strict, but intent key implies uniqueness)
    cached_record = memory.get_intent_answer(current_intent_sig)
    
    if cached_record:
        print("[Router] Intent found in memory! Routing to Judge for final delivery.")
        # Handle new vs legacy schema key
        answer_text = cached_record.get("approved_answer") or cached_record.get("answer")
        
        final_response = await judge.judge_from_memory(user_query, answer_text)
        print(f"\n[Result] (From Memory): {final_response}")
        last_system_response = final_response
        return final_response
    
    else:
        print("[Router] New Intent. Calling Generator LLMs...")
        
        # 4. Multi-LLM Generation
        # Prepend history to prompt so Generators have context
        history_context = context_manager.get_context_formatted()
        augmented_prompt = f"""POST HISTORY:
{history_context}

CURRENT QUERY:
{user_query}"""

        responses = await generate_all(augmented_prompt)
        generator_models = list(responses.keys()) # ["Gemini", "ChatGPT", "Groq", "Ollama"]
        
        # 5. Judge Evaluation
        print("[Router] Generators finished. Judging...")
        judge_result = await judge.judge_responses(user_query, responses)
        
        # 6. Iterative Human Feedback
        final_answer = judge_result.get("corrected_answer") or judge_result.get("final_answer")
        
        while True:
            print(f"\n[Proposed Answer]: {final_answer}")
            print(f"[Domain]: {current_domain}")
            
            print("Press [ENTER] to approve, or type your correction/feedback below:")
            user_feedback = input(">>> ").strip()
            
            if not user_feedback:
                print("[Router] Feedback approved.")
                # Store in Memory (New Schema)
                memory.save_intent_answer(
                    intent_data=intent_data,
                    answer=final_answer,
                    generated_by_models=generator_models,
                    confidence=0.95 # Validated by human
                )
                print("[Router] Answer saved to memory.")
                
                # Update Context History
                context_manager.add_turn("user", user_query)
                context_manager.add_turn("assistant", final_answer)
                
                last_system_response = final_answer
                return final_answer
            
            else:
                # SMART FEEDBACK CHECK: Is this a correction or a new topic?
                fb_classification = await intent.classify_query(user_feedback, final_answer, current_intent_sig)
                
                if fb_classification['query_type'] == 'follow_up':
                    print(f"[Router] Feedback is a Follow-up ({fb_classification['reasoning']}). Refinement cycle...")
                    # Update Memory timestamp for "last used" if we were editing a memory item (though here it's new)
                    # For follow-ups effectively we are refining the intent.
                    
                    refined = await judge.review_correction(user_query, final_answer, user_feedback)
                    final_answer = refined
                    # Loop continues...
                else:
                # Intent Change
                    print(f"[Router] Feedback is a NEW QUESTION ({fb_classification['reasoning']}). Switching context...")
                    
                    try:
                        # Auto-save logic here if we were implementing it fully
                        pass
                    except Exception as e:
                        print(f"[Auto-Save] Error saving draft: {e}")

                    # We abandon the current iterative learning and start a fresh query
                    return await process_query(user_feedback)

                # We abandon the current iterative learning and start a fresh query
                return await process_query(user_feedback)

if __name__ == "__main__":
    # Test specific flow
    while True:
        try:
            q = input("\nEnter query (or 'exit'): ")
            if q.lower() == 'exit': break
            asyncio.run(process_query(q))
        except KeyboardInterrupt:
            break
