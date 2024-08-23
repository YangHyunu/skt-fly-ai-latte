from fastapi import APIRouter, Depends,HTTPException
from sqlalchemy.orm import Session
from app.models.speech import AudioData
from app.models.image import ImageBase64Data
from db.database import get_db
from config import settings, Settings

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse


chat_router = APIRouter()
refine_router = APIRouter()
elevenlabs_router = APIRouter()
replicate_router = APIRouter()
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


# 목소리 추가 엔드포인트
# 현재는 voice_id를 return하지만, 나중에는 user의 고유 id를 이용해 DB에 해당 값을 올릴 것이다.
@elevenlabs_router.post("/voice/clone")
def clone_voice(voice_request: AudioData):

    # Array 형식의 데이터를 bytes로 변환
    voice_bytes = bytes(voice_request.data)

    try:
        response = settings.elevenlabs.clone_voice(name=voice_request.filename, voice_bytes=voice_bytes)

        return {
            "message": f"Voice '{voice_request.filename}' has been cloned successfully.",
            "voice_id": response.voice_id,
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while cloning the voice, {e}")

# 복제 목소리 이용 TTS 엔드포인트
# 현재는 voice_id를 직접적으로 받지만, 나중에는 user의 고유 id를 이용해 DB에서 해당 값을 가져올 것이다.
@elevenlabs_router.post("/voice/text_to_voice")
def text_to_cloned_voice(text: str, voice_id: str):

    try:
        response = settings.elevenlabs.text_to_cloned_voice(text=text, voice_id=voice_id)

        # with open(response, "rb") as audio_file:
        #     audio_data = audio_file.read()

        audio_name = "test_audio.mp3"

        headers = {
            "Content-Disposition": f"attachment; filename={audio_name}"
        }

        # FastAPI Response로 음성 파일 반환
        return StreamingResponse(content=response, media_type="audio/mpeg", headers=headers)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with text-to-speech, {e}")


# 얼굴 이미지 나이 변경
@replicate_router.post("/face/transform")
def transform_face_age(input_face: ImageBase64Data, target_age: str):

    try:
        container_name = "selfie"

        # Base64 형식으로 받아 이미지로 변환 후 Azure blob storage에 업로드하여 이미지 url을 얻음
        image_url = settings.azure_client.upload_image_to_storage(input_face=input_face, container_name=container_name)

        # 사진 이미지 url으로 target_age의 얼굴 사진 url을 얻음
        output_url = settings.replicate.transform_face_age(image=image_url, target_age=target_age)
        
        # 위의 url로 이미지를 받아 Azure bloc storage에 업로드하고 해당 url을 제공할 예정
        '''추가 예정'''

        return output_url
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while transforming the face age, {e}")