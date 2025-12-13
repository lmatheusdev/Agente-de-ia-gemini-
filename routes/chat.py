from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class UserMessage(BaseModel):
    message: str

@router.post("/chat/send")
async def receive_message(payload: UserMessage):
    user_text = payload.message
    # por agora, só confirmamos o recebimento
    return {"received": user_text}