from fastapi import APIRouter, Depends,HTTPException
from sqlalchemy.orm import Session
from app.schema import items as schemas
from app.models.speech import AudioData
from db.database import get_db
from config import settings, Settings
import json

router = APIRouter()

# @router.get("/items/", response_model=list[schemas.Item])
# def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
#     items = db.query(models.Item).offset(skip).limit(limit).all()
#     return items

@router.post("/chat")
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

@router.post("/make_history")
async def make_story():
    settings.reminescense.return_history
    return 
# @router.put("/chat")
# async def refine