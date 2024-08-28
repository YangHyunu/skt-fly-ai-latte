import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from api.routes import chat_router, refine_router, elevenlabs_router, sam_router

from api.user import user_router, login_router
from api.persona import persona_router
from api.recallbook import recallbook_router

from dotenv import load_dotenv
from config import settings
import asyncio

load_dotenv()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await settings.initialize_database()
    print("[INFO] Starting up application...")
    settings.initialize_clients()
    print("[INFO] Application startup completed.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 특정 도메인만 허용할 수 있습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router) # /chat
app.include_router(refine_router)
app.include_router(elevenlabs_router)
app.include_router(sam_router)
app.include_router(user_router)
app.include_router(login_router)
app.include_router(persona_router)
app.include_router(recallbook_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1",port=8000, reload=True)
    
    