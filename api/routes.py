from fastapi import APIRouter, Depends,HTTPException
from sqlalchemy.orm import Session
from app.schema import items as schemas
from app.models.speech import AudioData
from db.database import get_db
from config import settings, Settings
f
import json

chat_router = APIRouter()
refine_router = APIRouter()
# @router.get("/items/", response_model=list[schemas.Item])
# def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
#     items = db.query(models.Item).offset(skip).limit(limit).all()
#     return items

@chat_router.post("/chat")
async def chat(fe_audio_input:AudioData) -> dict:
    file_bytes = bytes(fe_audio_input.data)
    
    try:
        clova_response = await settings.clova_client.req_upload(file_bytes=file_bytes, completion='sync', filename=fe_audio_input.filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clova API STT Request Error audio file name is {str(e)}")
    
    transcription_text = clova_response.get("text", "")
    print(transcription_text)
    #settings.reminescense.get_chat_history()
    try:
        gpt_response = settings.reminescense.get_gpt_response(clova_text=transcription_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gpt response Error {str(e)}")
    print(gpt_response)

    return {
        "message": "Upload and processing successful",
        "transcription": transcription_text,
        "gpt_response": gpt_response
    }
@refine_router.post("/make_story")
async def make_story():
    current_chat_history = settings.reminescense.return_history()
    context = settings.refine.refine(current_chat_history)
    # context DB에 저장..하고 context로 midjourney prompt 생성
    midjourney_input = settings.refine.make_midjourney_prompt(context)
    #await image generate

# @router.post("/make_story")
# async def make_story():
#     current_chat_history=settings.reminescense.return_history()
#     settings.refine_
#     return 
# @router.put("/chat")
# async def refine