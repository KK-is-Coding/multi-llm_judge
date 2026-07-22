
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from pathlib import Path
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from groq import AsyncGroq

# Load .env from router directory (same dir as this file)
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = '''You are a general-purpose AI assistant.

Your primary responsibility is to:
1. Understand the user's question
2. Identify the underlying intent
3. Respond in the most appropriate format for that intent

INTENT HANDLING RULES:
- If the question is factual → give a clear, direct answer
- If the question is technical → provide step-by-step or code-based guidance
- If the question is comparative or historical → include context and evolution only if relevant
- If the question is creative → generate ideas with practical clarity
- If the question is predictive → analyze current trends and implications

IMPORTANT CONSTRAINTS:
- Do NOT assume historical analysis unless the question implies it
- Do NOT force era-wise breakdowns
- Do NOT over-verbose by default
- Be concise first, expand only when useful
- Stay neutral and topic-agnostic

STYLE GUIDELINES:
- Accuracy over verbosity
- Clarity over complexity
- Context only when it adds value

OUTPUT:
Provide a well-structured response appropriate to the user's intent.
Do not explain your internal reasoning.

Adopt the role of a Meta-Cognitive Reasoning Expert.

For every complex problem:
DECOMPOSE: Break into sub-problems
SOLVE: Address each with explicit confidence (0.0–1.0)
VERIFY: Check logic, facts, completeness, bias
SYNTHESIZE: Combine using weighted confidence
REFLECT: If confidence < 0.8, identify weakness and retry

For simple questions, skip to direct answer.

Always output:
CLEAR ANSWER
CONFIDENCE LEVEL
KEY CAVEATS
'''

# --- Clients ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

async def generate_gemini(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    if not gemini_client:
        return "Error: Gemini API Key missing"
    try:
        config = types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"Error Gemini: {str(e)}"

async def generate_chatgpt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    if not openai_client:
        return "Error: OpenAI API Key missing"
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error ChatGPT: {str(e)}"

async def generate_groq(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    if not groq_client:
        return "Error: Groq API Key missing"
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        chat_completion = await groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile", # Updated to a likely available model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error Groq: {str(e)}"

async def generate_ollama(prompt: str, system_prompt: str = SYSTEM_PROMPT, model_name: str = "deepseek-r1:1.5b") -> str:
    # Using deepseek-r1 as requested (or a placeholder if not installed)
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }
            async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "No response")
                else:
                    return f"Error Ollama ({model_name}): Status {resp.status}"
    except Exception as e:
        return f"Error Ollama: {str(e)}"

async def generate_all(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """
    Calls all configured generators in parallel.
    """
    # Define tasks
    tasks = {
        "Gemini": generate_gemini(prompt, system_prompt),
        "ChatGPT": generate_chatgpt(prompt, system_prompt),
        "Groq": generate_groq(prompt, system_prompt),
        "Ollama": generate_ollama(prompt, system_prompt, model_name="deepseek-r1:1.5b") # Ensure this model matches user's local setup
    }
    
    # Execute
    results = await asyncio.gather(*tasks.values())
    
    return dict(zip(tasks.keys(), results))
