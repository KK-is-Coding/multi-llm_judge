import asyncio
from typing import Dict, Any, List
from memory_store import MemoryStore
import llm_clients
import json

class Orchestrator:
    def __init__(self):
        self.memory = MemoryStore()
        # Simple in-memory state tracking for demo purposes. 
        # In production, use a proper DB (Redis/Postgres).
        self.current_state = {} 

    async def _run_cycle(self, prompt: str, memory_context: List[Dict]):
        """
        Runs Stage 1 (Generation) and Stage 2 (Judging).
        Returns the result dict.
        """
        # Stage 1: Multi-LLM Generation
        # We append memory findings to the prompt for the LLMs so they are aware
        augmented_prompt = f"USER PROMPT: {prompt}\n\nSTRICT MEMORY CONSTRAINTS:\n{json.dumps(memory_context)}"
        
        raw_responses = await llm_clients.generate_all(augmented_prompt)
        
        # Stage 2: AI Judge & Debate
        judge_result = await llm_clients.judge_responses(prompt, raw_responses)
        
        return {
            "raw_responses": raw_responses,
            "judge_result": judge_result
        }

    async def start_workflow(self, prompt: str):
        """
        Starts the workflow from Stage 0 to Stage 2.
        Pauses for Stage 3 (Human Verification).
        """
        # Stage 0: Memory Retrieval
        retrieved_memory = self.memory.retrieve_memory(prompt)
        
        cycle_result = await self._run_cycle(prompt, retrieved_memory)
        
        # Prepare state for human review
        workflow_id = "default_session" # Simplified for single user
        self.current_state[workflow_id] = {
            "prompt": prompt,
            "raw_responses": cycle_result["raw_responses"],
            "judge_result": cycle_result["judge_result"],
            "stage": "waiting_for_human",
            "memory_context": retrieved_memory
        }
        
        return {
            "status": "waiting_for_human_verification",
            "workflow_id": workflow_id,
            "draft_answer": cycle_result["judge_result"].get("corrected_answer"),
            "critique": cycle_result["judge_result"].get("rationale"),
            "model_scores": cycle_result["judge_result"].get("scores"),
            "raw_responses": cycle_result["raw_responses"],
            "full_judge_result": cycle_result["judge_result"]
        }

    async def process_human_feedback(self, workflow_id: str, human_input: Dict[str, Any]):
        """
        Resumes from Stage 3 with human input.
        Every input triggers a re-generation cycle for refinement.
        """
        state = self.current_state.get(workflow_id)
        if not state:
            return {"error": "Workflow session not found"}

        # Extract feedback from input
        feedback = human_input.get("feedback") or human_input.get("corrections")
        if not feedback:
            return {"error": "No feedback or corrections provided"}

        # Store error correction immediately
        self.memory.update_memory("error_correction", feedback)
        
        # Re-retrieve memory including the new error correction
        prompt = state["prompt"]
        updated_memory = self.memory.retrieve_memory(prompt)
        
        # Construct refined prompt with explicit user feedback
        refined_prompt = f"{prompt}\n\nUSER FEEDBACK / CORRECTION: {feedback}"
        
        # Loop back to Stage 1 (Re-run Cycle)
        cycle_result = await self._run_cycle(refined_prompt, updated_memory)
        
        # Update state with new results
        self.current_state[workflow_id] = {
            "prompt": refined_prompt,
            "raw_responses": cycle_result["raw_responses"],
            "judge_result": cycle_result["judge_result"],
            "stage": "waiting_for_human",
            "memory_context": updated_memory
        }
        
        # Return new draft to user
        return {
            "status": "waiting_for_human_verification",
            "workflow_id": workflow_id,
            "draft_answer": cycle_result["judge_result"].get("corrected_answer"),
            "critique": cycle_result["judge_result"].get("rationale"),
            "model_scores": cycle_result["judge_result"].get("scores"),
            "raw_responses": cycle_result["raw_responses"],
            "full_judge_result": cycle_result["judge_result"],
            "message": "Re-generated based on feedback."
        }
