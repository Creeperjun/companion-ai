import os, json, asyncio
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from agent_core import AgentCore

load_dotenv()
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = AgentCore(memory_persist_dir=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    yield
    agent = None

app = FastAPI(title="CompanionAI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel): message: str

class ChatResponse(BaseModel): response: str; emotion: str; memory_count: int; conversation_round: int

class StatusResponse(BaseModel): emotion: str; memory_count: int; conversation_round: int; history_size: int; memories: List[dict] = []

@app.get("/")  
async def index(): return FileResponse("static/index.html")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    st = agent.get_status()
    all_m = agent.memory.get_all_memories()
    return StatusResponse(emotion=st["emotion"], memory_count=st["memory_count"], conversation_round=st["conversation_round"], history_size=st["history_size"], memories=all_m[-10:] if all_m else [])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip(): raise HTTPException(status_code=400, detail="消息不能为空")
    response = agent.chat(request.message)
    st = agent.get_status()
    return ChatResponse(response=response, emotion=st["emotion"], memory_count=st["memory_count"], conversation_round=st["conversation_round"])

@app.post("/reset")
async def reset(): agent.reset(); return {"status": "ok", "message": "对话已重置"}

@app.get("/history")
async def get_history():
    history = agent.get_history()
    messages = [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in history]
    return {"messages": messages}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg_data = json.loads(data)
            user_input = msg_data.get("message", "").strip()
            if not user_input: await websocket.send_json({"type": "error", "content": "消息不能为空"}); continue
            emotion_info = agent.emotion.get_emotion_info()
            await websocket.send_json({"type": "emotion", "emotion": emotion_info["current"], "style": emotion_info["style_prompt"]})
            full_response = ""
            for chunk in agent.stream_chat(user_input):
                full_response += chunk; await websocket.send_json({"type": "chunk", "content": chunk})
            st = agent.get_status()
            await websocket.send_json({"type": "done", "emotion": st["emotion"], "memory_count": st["memory_count"], "conversation_round": st["conversation_round"]})
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await websocket.send_json({"type": "error", "content": str(e)})
        except: pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
