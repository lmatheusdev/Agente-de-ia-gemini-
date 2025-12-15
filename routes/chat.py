from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.ai_agent import run_agent

router = APIRouter()

class UserMessage(BaseModel):
    message: str

class AgentResponse(BaseModel):
    reply: str

@router.post("/chat/send")
async def chat(payload: UserMessage):
    try:
        reply = run_agent(payload.message)
        return {"reply": reply}

    except Exception as e:
        print("ERRO NO AGENTE:", e)
        raise HTTPException(500, "Erro no agente de IA")
