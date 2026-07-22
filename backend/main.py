from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm_clients import generate_all, judge_responses
from orchestrator import Orchestrator
from typing import Optional, Dict, Any
import uvicorn
import os

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In dev, allow all. In prod, lock this down.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

class JudgeRequest(BaseModel):
    prompt: str
    responses: dict

class OrchestratorStartRequest(BaseModel):
    prompt: str

class OrchestratorFeedbackRequest(BaseModel):
    workflow_id: str
    feedback: str

orchestrator = Orchestrator()

@app.post("/api/arena")
async def run_arena(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")
    
    results = await generate_all(request.prompt)
    return results

@app.post("/api/judge")
async def run_judge(request: JudgeRequest):
    evaluation = await judge_responses(request.prompt, request.responses)
    return {"evaluation": evaluation}

@app.post("/api/orchestrator/start")
async def start_orchestrator(request: OrchestratorStartRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")
    return await orchestrator.start_workflow(request.prompt)

@app.post("/api/orchestrator/human-feedback")
async def process_feedback(request: OrchestratorFeedbackRequest):
    return await orchestrator.process_human_feedback(
        request.workflow_id,
        request.model_dump()
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
