import sys
sys.stdout.reconfigure(encoding='utf-8')
print("Python is working")

try:
    print("Importing os...")
    import os
    print("Importing google.genai...")
    from google import genai
    print("Importing google.genai.types...")
    from google.genai import types
    print("Importing openai...")
    from openai import AsyncOpenAI
    print("Importing groq...")
    from groq import AsyncGroq
    print("Imports successful")
except Exception as e:
    print(f"Import failed: {e}")
