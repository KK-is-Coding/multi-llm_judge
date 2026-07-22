
import json
from .llm_generators import generate_gemini

JUDGE_SYSTEM_PROMPT = """
You are an expert AI Judge.
Your task is to evaluate responses to a user query and select the best one, or synthesize a better answer.
"""

async def judge_responses(query: str, responses: dict) -> dict:
    """
    Evaluates a dictionary of {model: response} and returns the best answer + metadata.
    """
    
    judge_prompt =f"""
You are an expert AI JUDGE and RESPONSE SYNTHESIZER in a multi-LLM system.

Your role is NOT to reinterpret the task.
Your role is to evaluate responses strictly against the USER QUERY as given.

DO NOT expand scope.
DO NOT introduce new perspectives unless required by the query.
DO NOT add historical or era-wise analysis unless the query explicitly asks for it.

────────────────────────
USER QUERY
────────────────────────
"{query}"

────────────────────────
MODEL RESPONSES
────────────────────────
1. Gemini: {responses.get('Gemini')}
2. ChatGPT: {responses.get('ChatGPT')}
3. Groq: {responses.get('Groq')}
4. Ollama: {responses.get('Ollama')}

────────────────────────
STEP 1: INVALID RESPONSE FILTER
────────────────────────
Mark a response INVALID if it:
- starts with "Error"
- mentions missing API keys
- is empty or irrelevant to the query

Invalid responses:
- receive scores of 0
- are excluded from synthesis

────────────────────────
STEP 2: IDEAL ANSWER CRITERIA
────────────────────────
Define what a correct answer requires BASED ONLY on the user query.

Examples:
- If the query is technical → correctness, clarity, relevance
- If the query is comparative → fair comparison
- If the query is factual → accuracy
- If the query is historical → contextual accuracy

Do NOT add requirements the query did not ask for.

────────────────────────
STEP 3: RESPONSE EVALUATION
────────────────────────
For each VALID response:
- Identify strengths
- Identify errors or omissions
- Note verbosity or irrelevance penalties

────────────────────────
STEP 4: CROSS-RESPONSE ANALYSIS
────────────────────────
- Identify which response best matches the IDEAL CRITERIA
- Note any useful insights present in weaker responses

────────────────────────
STEP 5: SYNTHESIS
────────────────────────
Produce a corrected_answer by:
- Using ONLY information present in the responses
- Removing irrelevant or over-expanded content
- Correcting mistakes
- Keeping the answer aligned to the original query

DO NOT introduce new facts.

────────────────────────
STEP 6: SCORING
────────────────────────
Score each model (0–10):
- accuracy
- clarity
- completeness

Scores must reflect usefulness for THIS query.

────────────────────────
STEP 7: FINAL DECISION
────────────────────────
Select best_model based on:
- correctness
- relevance
- minimal critical errors

If all responses fail → best_model = "None"

────────────────────────
OUTPUT FORMAT (STRICT JSON ONLY)
────────────────────────
{{
  "best_model": "Gemini | ChatGPT | Groq | Ollama | None",
  "rationale": "Why this model best satisfies the query",
  "scores": {{
    "Gemini": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "" }},
    "ChatGPT": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "" }},
    "Groq": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "" }},
    "Ollama": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "" }}
  }},
  "corrected_answer": "Synthesized, corrected response based only on the best parts of the LLM outputs."
}}
"""

    try:
        raw_result = await generate_gemini(judge_prompt, JUDGE_SYSTEM_PROMPT)
        # Clean markdown
        clean_result = raw_result.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_result)
        return data
    except Exception as e:
        return {
            "best_model": "Error",
            "rationale": f"Judge failed: {str(e)}",
            "final_answer": "Error during evaluation.",
            "corrected_answer": "Error during evaluation."
        }

async def judge_from_memory(query: str, stored_answer: str) -> str:
    """
    Called when intent is found in memory. The Judge takes the stored answer and the current query
    to ensure the response is contextually perfect (e.g. slight phrasing differences), 
    or simply returns it if it's identical.
    """
    prompt = f"""
    The user has asked: "{query}"
    
    We have a TRUSTED, MEMORIZED answer for this intent:
    "{stored_answer}"
    
    Task:
    Adapt the memorized answer to fit the user's current query if needed (e.g. tone, minor formatting).
    If the memorized answer is perfect, return it as is.
    Do NOT change the core meaning or facts.
    Do NOT add new information.
    Do NOT expand beyond the original intent.
    Only adapt wording or formatting.   
    """
    
    return await generate_gemini(prompt, "You are a helpful assistant delivering a verified answer.")


async def review_correction(query: str, original_draft: str, feedback: str) -> str:
    """
    Called when the user REJECTS or CORRECTS the judge's draft.
    The Judge must now synthesize the FINAL answer by respecting the user's authortity
    but maintaining the professional structure.
    """
    prompt = f"""
    User Query: "{query}"
    
    Previous Draft Answer:
    "{original_draft}"
    
    USER FEEDBACK / CORRECTION:
    "{feedback}"
    
    Task:
    1. If the user is correcting valid facts, accept them as absolute truth.
    2. If the user is asking "Why?" or for more detail, ENRICH the answer to provide that reasoning.
    3. If the user is asking to change style (shorter/longer), adapt accordingly.
    4. Rewrite the FINAL answer to be the perfect response to the original query, now improved by this feedback.
    5. Return ONLY the new final text.
    6. If user feedback introduces a NEW topic, STOP and signal routing.
    """
    
    return await generate_gemini(prompt, "You are an expert editor incorporating user feedback.")

async def find_matching_intent(new_intent: str, candidates: list) -> str:
    """
    Checks if 'new_intent' is semantically equivalent to any in 'candidates'.
    Returns the matching intent signature from candidates, or None.
    """
    if not candidates:
        return None
        
    prompt = f"""
    NEW INTENT SIGNATURE: "{new_intent}"
    
    EXISTING CANDIDATES:
    {json.dumps(candidates, indent=1)}
    
    Task:
    Identify if ANY of the candidates represents the EXACT SAME TASK as the new intent, just named differently.
    
    Rules:
    1. "parity_check" == "check_odd_even" (SAME)
    2. "python_code" != "cpp_code" (DIFFERENT)
    3. Return the exact candidate string if match found.
    4. Return "None" if no strong match.
    5. Only match intents if they belong to the SAME domain.
    
    Format:
    Return ONLY the candidate string or "None".
    """
    
    try:
        match = await generate_gemini(prompt, "You are a semantic equivalency classifier.")
        match = match.strip().replace('"', '')
        if match in candidates:
            return match
        return None
    except Exception as e:
        print(f"[Judge] Similarity check failed: {e}")
        return None
