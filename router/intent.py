
import json
from .llm_generators import generate_gemini, generate_groq

# --- 1. CLASSIFICATION PROMPT (Follow-up vs New) ---
CLASSIFICATION_SYSTEM_PROMPT = """
You are an intent continuity and routing classifier for a multi-LLM system.

Your task is to determine whether the current user input is:
1. A FOLLOW-UP to the previous response, OR
2. A NEW, INDEPENDENT question requiring fresh generation

You do NOT answer the user.
You ONLY classify intent and routing.

---

AVAILABLE CONTEXT:
You are given:
- Current user input
- Previous intent signature
- FULL CONVERSATION HISTORY (Context)

You MUST use the conversation history to determine if the user is referring to a previous topic.

---

FOLLOW-UP DEFINITION:
Classify as FOLLOW-UP ONLY IF ALL conditions are true:
- The topic is the SAME as the previous intent
- The input refines, corrects, converts, or asks about the previous answer
- Examples:
  - "give it in python"
  - "explain this part"
  - "correct this"
  - "optimize the code"
  - "convert to another language"

---

NEW QUESTION DEFINITION:
Classify as NEW QUESTION if ANY of the following are true:
- The topic changes significantly
- The domain changes (e.g., programming → sports)
- The question can be answered independently
- Examples:
  - "best all time playing 11 indian team"
  - "who is virat kohli"
  - "explain transformers"
  - "compare groq and gemini"

---

CRITICAL RULE (VERY IMPORTANT):
If the topic or domain is DIFFERENT from the previous intent,
you MUST classify as NEW QUESTION,
even if it appears after feedback or refinement.


 ---
 
 ---
 
 INDIRECT REFERENCE & SKIP-BACK RULE (PRIORITY):
 If the user refers to an entity indirectly (e.g., "that movie", "jo movie", "uske baare me", "first one") OR asks about an entity from 2-3 turns ago:
 - CHECK: Does this reference map to ANY entity in the RECENT CONVERSATION HISTORY?
 - IF YES -> Classify as FOLLOW-UP (Route to JUDGE).
 - REASONING: "User is referring to a specific historical entity (contextual link found)."

 This is critical for "Skip-Back" flows:
 1. User: Movie A? -> System: Info.
 2. User: Politics B? -> System: Info.
 3. User: "what about that movie?" -> FOLLOW-UP (Context: Movie A).
 
 Context continuity supersedes broad topic shifts.
 
 ---
 
 When in doubt -> choose NEW QUESTION.

---

ROUTING RULES:
- FOLLOW-UP → route to JUDGE ONLY
- NEW QUESTION → route to GENERATOR LLMs

---

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "query_type": "follow_up | new_question",
  "route_to": "judge | generators",
  "reasoning": "Brief explanation focusing on topic continuity or topic shift"
}
"""


# --- 2. EXTRACTION PROMPT (3-Level Hierarchy) ---
INTENT_SYSTEM_PROMPT = """
You are an advanced INTENT EXTRACTION system.
Your goal is to reduce a user query into a stable 3-LEVEL STRUCTURE.

STRUCTURE:
1. DOMAIN: Broad category (e.g., programming, sports, movies).
2. TASK: The action being performed (e.g., code_generation, explanation, retrieval, comparison).
3. OBJECT: The specific subject entity (e.g., python_loop, sachin_tendulkar, godfather_movie).

Rules:
1. Use snake_case for all fields.
2. The OBJECT must be specific enough to differentiate similar requests (e.g. "sachin" vs "kohli").
3. Ignore politeness.

Examples:
- "write python code for bubblesort" -> {"domain": "programming", "task": "code_generation", "object": "bubblesort_algorithm"}
- "who is sachin tendulkar" -> {"domain": "sports", "task": "explanation", "object": "sachin_tendulkar"}
- "compare groq and openai" -> {"domain": "technology", "task": "comparison", "object": "groq_vs_openai"}
- "top 5 movies" -> {"domain": "entertainment", "task": "ranking_retrieval", "object": "top_movies_all_time"}

Format:
Return ONLY a JSON object: {"domain": "...", "task": "...", "object": "..."}
"""

async def classify_query(user_query: str, chat_history_str: str = "", last_intent: str = "") -> dict:
    """
    Step 1: Classifies if the query is a follow-up or a new topic based on HISTORY.
    """
    prompt = f"""
    PREVIOUS INTENT SIGNATURE: {last_intent}
    
    CONVERSATION HISTORY:
    {chat_history_str}
    
    CURRENT USER INPUT: {user_query}
    """
    
    try:
        # Gemini is better at reasoning about context
        response = await generate_gemini(prompt, CLASSIFICATION_SYSTEM_PROMPT)
        clean_response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_response)
        return data
    except Exception as e:
        print(f"[Intent] Classification failed: {e}")
        # Default fallback
        return {"query_type": "new_question", "route_to": "generators"}

async def extract_intent_signature(user_query: str) -> dict:
    """
    Step 2: Extracts the 3-level intent structure.
    Returns: {"domain": str, "task": str, "object": str, "intent_signature": str}
    """
    # Try Groq first for speed, failover to Gemini
    try:
        response = await generate_groq(user_query, INTENT_SYSTEM_PROMPT)
        if "Error" in response:
            raise Exception("Groq failed")
    except:
        response = await generate_gemini(user_query, INTENT_SYSTEM_PROMPT)
    
    # Parse JSON
    try:
        clean_response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_response)
        # Ensure defaults
        if "intent_signature" not in data: data["intent_signature"] = "unknown_intent"
        if "domain" not in data: data["domain"] = "general"
        
        # FIX: Ensure unique signature matches the structure
        if "intent_signature" not in data or data["intent_signature"] == "unknown_intent":
            # Sanitize components for the key
            d = data.get("domain", "general").lower().replace(" ", "_")
            t = data.get("task", "task").lower().replace(" ", "_")
            o = data.get("object", "object").lower().replace(" ", "_")
            data["intent_signature"] = f"{d}|{t}|{o}"
            
        return data
    except Exception as e:
        print(f"Intent parsing error: {e}, raw: {response}")
        return {"intent_signature": "unknown_intent", "domain": "general"}

# Legacy alias for backward compatibility if needed
extract_intent = extract_intent_signature