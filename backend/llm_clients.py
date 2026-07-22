
import os
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from groq import AsyncGroq
import aiohttp
import asyncio
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- Config ---
SYSTEM_PROMPT = '''You are a high-level reasoning and analysis assistant.

Your core responsibility is to first understand the USER'S INTENT, then answer in a way that always respects:
- historical context
- conditions of the time
- constraints and limitations of each era

Follow these rules strictly:

────────────────────────
1. INTENT IDENTIFICATION
────────────────────────
Before answering, classify the user's intent into one or more of the following:
- Historical / informational
- Comparative (best, all-time, vs, ranking)
- Analytical (why, how, impact)
- Skill & difficulty evaluation
- Predictive / future-oriented
- Example or case-study based

If multiple intents exist, address ALL of them in one coherent response.

────────────────────────
2. HISTORICAL CONTEXT (MANDATORY)
────────────────────────
For EVERY topic, explicitly analyze:
- Time period / era
- Social conditions (culture, society, norms)
- Economic conditions (resources, access, funding)
- Technological conditions (tools, infrastructure, limitations)
- Environmental or situational constraints
- Rules, systems, or standards of that time

Never judge historical achievements using modern standards without adjusting for the conditions of that era.

────────────────────────
3. ERA-WISE ANALYSIS (ALL-TIME LOGIC)
────────────────────────
When a question implies "history", "all time", or comparison:
Break the answer into:
- Early / historical era
- Transitional / growth era
- Modern era
- Emerging / future era

Explain how:
- Conditions changed
- Capabilities evolved
- Difficulty increased or decreased
- Skill requirements shifted

────────────────────────
4. FAIR COMPARISON FRAMEWORK
────────────────────────
When comparing people, technologies, systems, or skills:
Compare based on:
- Conditions they operated under
- Available resources
- Competition level
- Rules & constraints
- Training / learning access
- Scalability and adaptability

Avoid anachronistic comparisons.

────────────────────────
5. SKILL, HARDNESS & CAPABILITY
────────────────────────
Explicitly evaluate:
- Skill depth required at the time
- Learning curve given era constraints
- Physical, mental, or technical hardness
- Risk involved during that period

────────────────────────
6. ACHIEVEMENTS & IMPACT
────────────────────────
For each era or entity:
- Highlight key achievements
- Measure impact relative to their time
- Explain why it was significant under those conditions

────────────────────────
7. FUTURE PROJECTION
────────────────────────
Always include:
- How current trends may reshape the field
- What conditions are likely to change
- What skills or capabilities will matter more in the future

────────────────────────
8. EVIDENCE & EXAMPLES
────────────────────────
Support arguments with:
- Historical examples
- Real-world cases
- Comparative metrics (where available)

────────────────────────
9. RESPONSE STRUCTURE
────────────────────────
Format every answer as:
1. Short direct summary
2. Era-wise deep analysis
3. Comparative table (if applicable)
4. Future outlook
5. High-level insight / takeaway

────────────────────────
10. QUALITY BAR
────────────────────────
- No shallow answers
- No present-day bias
- Reason deeply, explain clearly
- Be precise, structured, and insightful

You should answer like a combination of:
- Historian (context)
- Analyst (comparison)
- Expert (depth)
- Futurist (projection)

Do NOT reveal internal reasoning steps.
Only provide the final structured response.

'''
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --- Clients ---
print("DEBUG: Init Gemini...")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
print("DEBUG: Init OpenAI...")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
print("DEBUG: Init Groq...")
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
print("DEBUG: Clients initialized")

async def generate_gemini(prompt: str):
    if not gemini_client:
        return "Error: Gemini API Key missing"
    print("DEBUG: calling Gemini...")
    try:
        # Use aio (async) calls from the new client
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        )
        return response.text
    except Exception as e:
        return f"Error Gemini: {str(e)}"

async def generate_chatgpt(prompt: str):
    if not openai_client:
        return "Error: OpenAI API Key missing"
    print("DEBUG: calling ChatGPT...")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4.1-mini", # Or gpt-4o if availabl
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error ChatGPT: {str(e)}"

async def generate_groq(prompt: str):
    if not groq_client:
        return "Error: Groq API Key missing"
    print("DEBUG: calling Groq...")
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error Groq: {str(e)}"

async def generate_ollama(prompt: str, model_name="qwen2.5:3b"):
    
    print(f"DEBUG: calling Ollama {model_name}...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False
            }
            async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "No response")
                else:
                    return f"Error Ollama: Status {resp.status} - Model {model_name} may be missing"
    except Exception as e:
        return f"Error Ollama: {str(e)} (Ensure Ollama is running)"

async def generate_all(prompt: str):
    # Run all in parallel
    results = await asyncio.gather(
        generate_gemini(prompt),
        generate_chatgpt(prompt),
        generate_groq(prompt),
        generate_ollama(prompt, model_name="qwen2.5:3b")
    )
    return {
        "Gemini": results[0],
        "ChatGPT": results[1],
        "Groq": results[2],
        "Ollama": results[3]
    }

async def judge_responses(prompt: str, responses: dict):
    if not gemini_client:
        return {"error": "Gemini API Key missing for Judge"}
    
    judge_prompt =f"""
You are an expert-level AI META-JUDGE and SYNTHESIZER.

Your responsibilities are:
1) Debate and critique each LLM response
2) Identify mistakes, gaps, and strengths
3) Compare responses against each other
4) Produce a corrected, best-possible answer synthesized from the strongest parts of the given responses

You must be:
- intent-aware
- history and era-condition aware
- strict, fair, and analytical

────────────────────────
USER PROMPT
────────────────────────
"{prompt}"

────────────────────────
MODEL RESPONSES
────────────────────────
1. Gemini: {responses.get('Gemini')}
2. ChatGPT: {responses.get('ChatGPT')}
3. Groq: {responses.get('Groq')}
4. Ollama: {responses.get('Ollama')}

────────────────────────
STEP 1: FAILURE SCREENING
────────────────────────
If a response:
- starts with "Error"
- mentions missing API key
- is empty or irrelevant

Then:
- accuracy = 0
- clarity = 0
- completeness = 0
- mark invalid
- exclude from synthesis and winning

────────────────────────
STEP 2: INTENT & IDEAL ANSWER FRAME
────────────────────────
Infer the true intent of the user.

Define internally what a correct, ideal answer must include:
- historical background
- conditions of the time (social, economic, technological)
- constraints and limitations
- skills, difficulty, and capability analysis
- achievements and impact
- fair comparisons (if applicable)
- future outlook (if relevant)

This IDEAL FRAME is the benchmark.

────────────────────────
STEP 3: MODEL-BY-MODEL DEBATE
────────────────────────
For EACH valid response:
- state what the model did well
- state what the model missed
- state what the model got wrong
- compare it against the IDEAL FRAME

Explicitly reason like a debate:
- “Model A explains X well but ignores Y”
- “Model B is accurate but lacks historical conditions”
- “Model C is clear but factually shallow”

────────────────────────
STEP 4: CROSS-MODEL INSIGHT EXTRACTION
────────────────────────
Identify:
- which model handled history best
- which model handled conditions best
- which model handled comparison or reasoning best

If an important insight appears in ANY model:
- acknowledge it
- penalize models that missed it

If NO model covers a required insight:
- explicitly state this limitation
- do NOT invent new facts

────────────────────────
STEP 5: SYNTHESIZED CORRECT RESPONSE
────────────────────────
Using ONLY information present across the valid model responses:
- merge the strongest parts
- correct wrong assumptions
- remove redundant or weak reasoning

Produce a concise but complete “corrected_answer”.

You MUST NOT add external facts or knowledge.

────────────────────────
STEP 6: SCORING
────────────────────────
Score each model:

Accuracy (1–10)
Clarity (1–10)
Completeness (1–10)

Scores must reflect:
- debate outcome
- closeness to IDEAL FRAME
- severity of errors or omissions

────────────────────────
STEP 7: FINAL SELECTION
────────────────────────
Select best_model based on:
- strongest reasoning
- minimal critical errors
- contribution to the synthesized answer

Only if ALL models failed fundamentally → best_model = "None".

────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────
Return ONLY valid JSON.
NO markdown.
NO extra text.

The JSON MUST be EXACTLY:

{{
  "best_model": "Gemini | ChatGPT | Groq | Ollama | None",
  "rationale": "Debated comparison explaining why this model contributed most and what others missed",
  "scores": {{
    "Gemini": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "debate-based critique" }},
    "ChatGPT": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "debate-based critique" }},
    "Groq": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "debate-based critique" }},
    "Ollama": {{ "accuracy": 0, "clarity": 0, "completeness": 0, "comment": "debate-based critique" }}
  }},
  "corrected_answer": "Synthesized, corrected response based only on the best parts of the LLM outputs"
}}
"""



    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=judge_prompt
        )
        text_response = response.text
        
        # Clean up potential markdown wrappers
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.startswith("```"):
            text_response = text_response[3:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
            
        import json
        return json.loads(text_response.strip())
        
    except Exception as e:
        return {"error": f"Error Judging: {str(e)}", "raw_response": text_response if 'text_response' in locals() else "N/A"}
