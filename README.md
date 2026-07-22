# 🤖 Multi-LLM Judge System

An intelligent orchestration system that queries multiple LLMs in parallel, uses a judge to select the best response, and maintains semantic memory for context-aware conversations.

## 🌟 Features

- **Multi-LLM Orchestration**: Parallel queries to Gemini, ChatGPT, Groq, and Ollama
- **AI Judge System**: Intelligent evaluation and selection of best responses
- **Semantic Memory**: Vector-based memory storage using Milvus for context retention
- **Intent Recognition**: Smart routing based on query intent and conversation history
- **Context-Aware Routing**: Distinguishes between follow-ups and new topics
- **Human Feedback Loop**: Approve, modify, or reject AI responses

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Milvus**: Vector database (optional, for memory features)
- **Ollama**: Local LLM runtime (optional, for Ollama support)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/KK-is-Coding/multi-llm_judge.git
cd Multi-LLM_JUDGE
```

### 2. Install Dependencies

```bash
pip install -r router/requirements.txt
```

**Required packages:**
- `aiohttp` - Async HTTP client
- `google-genai` - Google Gemini API
- `openai` - OpenAI API
- `groq` - Groq API
- `python-dotenv` - Environment variable management
- `pymilvus` - Milvus vector database client

### 3. Set Up Environment Variables

Create a `.env` file in the `router/` directory:

```bash
# router/.env

# Required API Keys (get at least one)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
```

**Where to get API keys:**
- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Groq**: [Groq Console](https://console.groq.com/keys)

### 4. (Optional) Set Up Milvus

For full memory capabilities, install and run Milvus:

**Using Docker:**
```bash
# Download Milvus standalone
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
docker-compose up -d
```

**Or use Milvus Lite (simpler):**
```bash
pip install milvus
```

### 5. (Optional) Set Up Ollama

For local LLM support:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the DeepSeek model:
```bash
ollama pull deepseek-r1:1.5b
```

### 6. Run the System

```bash
python main.py
```

## 💡 Usage

### Basic Interaction

```
User: What is machine learning?
```

The system will:
1. Extract intent from your query
2. Send to all configured LLMs in parallel
3. Judge evaluates all responses
4. Returns the best answer

### Follow-up Questions

```
User: What is machine learning?
Assistant: [Provides answer]

User: Can you explain it more simply?
```

The system recognizes follow-ups and routes appropriately.

### Human Feedback

After receiving a response, you can:
- **Approve**: Accept the answer (saved to memory)
- **Modify**: Edit and improve the response
- **Reject**: Request regeneration with feedback

## 📁 Project Structure

```
Multi-LLM_JUDGE/
├── main.py                 # Entry point
├── router/
│   ├── orchestrator.py     # Main orchestration logic
│   ├── llm_generators.py   # LLM API integrations
│   ├── judge.py            # Response evaluation
│   ├── intent.py           # Intent extraction
│   ├── context.py          # Context management
│   ├── memory.py           # Memory operations
│   ├── vector_store.py     # Milvus integration
│   ├── feedback.py         # Human feedback handling
│   └── requirements.txt    # Dependencies
├── backend/                # Alternative backend implementation
└── .env                    # Environment variables (create this)
```

## 🔧 Configuration

### Customize LLM Models

Edit `router/llm_generators.py` to change models to the their latest versions:

```python
# Gemini
model="gemini-flash-latest"

# ChatGPT
model="gpt-4o-mini"

# Groq
model="llama-3.3-70b-versatile"

# Ollama
model_name="deepseek-r1:1.5b"
```

### Adjust System Prompts

Modify prompts in:
- `llm_generators.py` - Generator behavior
- `judge.py` - Judge evaluation criteria
- `intent.py` - Intent extraction logic

## 🧪 Testing

Run the verification script:

```bash
python router/verify_router.py
```

## 🐛 Troubleshooting

### "API Key missing" errors
- Ensure your `.env` file is in the `router/` directory
- Check that API keys are valid and have proper permissions

### Milvus connection errors
- Verify Milvus is running: `docker ps` (if using Docker)
- Check connection settings in `vector_store.py`
- System will work without Milvus (limited memory)

### Ollama errors
- Ensure Ollama is running: `ollama list`
- Verify model is installed: `ollama pull deepseek-r1:1.5b`
- Check `OLLAMA_BASE_URL` in `.env`

### Import errors
- Reinstall dependencies: `pip install -r router/requirements.txt`
- Ensure you're running from the project root directory

## 📊 System Architecture

```
User Query
    ↓
Intent Extraction
    ↓
Context Analysis ──→ Memory Check
    ↓
Routing Decision
    ├─→ New Query: All LLMs in parallel
    └─→ Follow-up: Judge only
    ↓
Judge Evaluation
    ↓
Best Response
    ↓
Human Feedback ──→ Save to Memory
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Repository**: [https://github.com/KK-is-Coding/multi-llm_judge](https://github.com/KK-is-Coding/multi-llm_judge.git)

## 📧 Support

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ using multiple LLMs**