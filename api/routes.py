from fastapi import APIRouter, Depends,HTTPException
from fastapi.responses import StreamingResponse
from app.models.speech import AudioData, PersonaAudioData, PersonaBase64AudioData
from app.models.image import ImageBase64Data
from db.database import get_db
from config import settings, Settings
from langchain.prompts import ChatPromptTemplate
from app.schema.db_schema import RecallBook, User, Persona, Recallbook_header
from beanie import PydanticObjectId
import base64

import json

chat_router = APIRouter()
refine_router = APIRouter()
elevenlabs_router = APIRouter()
replicate_router = APIRouter()
user_router = APIRouter()
recallbook_router = APIRouter()
login_router = APIRouter()

@chat_router.post("/chat")
async def chat(persona_audio_input: PersonaBase64AudioData) -> dict:
    
    # voice cloning
    persona = await Persona.find_one(Persona.persona_id == persona_audio_input.persona_id)
    if not persona.voice_id:
        # Base64로 인코딩된 데이터를 bytes로 변환
        voice_bytes = base64.b64decode(persona_audio_input.base64_audio)

        try:
            response = settings.elevenlabs.clone_voice(name=persona_audio_input.filename, voice_bytes=voice_bytes)

            # voice_id를 해당 persona에 저장
            updated_persona = await Persona.find_one(Persona.persona_id == persona_audio_input.persona_id).update({"$set": {Persona.voice_id: response.voice_id}})

            # 업데이트가 완료되었는지 확인하기 위해 문서를 다시 읽음
            updated_persona = await Persona.find_one(Persona.persona_id == persona_audio_input.persona_id)

            # 업데이트된 voice_id가 있는지 확인
            if not updated_persona.voice_id:
                raise HTTPException(status_code=500, detail="Failed to update voice_id in Persona")
                
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while cloning the voice, {e}")
        
    # 음성 디코딩
    file_bytes = base64.b64decode(persona_audio_input.base64_audio)
    
    # Speech-To-Text
    try:
        clova_response = await settings.clova_client.req_upload(file_bytes=file_bytes, completion='sync', filename=persona_audio_input.filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clova API STT Request Error audio file name is {str(e)}")
    
    transcription_text = clova_response.get("text", "")
    print(transcription_text)
    #settings.reminescense.get_chat_history()

    # GPT Response 얻기
    try:
        gpt_response = settings.reminescense.get_gpt_response(clova_text=transcription_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gpt response Error {str(e)}")
    print(gpt_response)

    
    # Text-To-Speech
    try:
        # voice_id 받아오기
        persona = await Persona.find_one(Persona.persona_id == persona_audio_input.persona_id)
        voice_id = persona.voice_id
        if not voice_id:
            raise HTTPException(status_code=404, detail="Voice ID not found after cloning.")
        
        # text to cloned voice
        response = settings.elevenlabs.text_to_cloned_voice(text=gpt_response, voice_id=voice_id)

        # 제너레이터로부터 바이너리 데이터 수집
        audio_bytes = bytearray()
        for chunk in response:
            if chunk:
                audio_bytes.extend(chunk)
        
        # 바이너리 데이터를 base64로 인코딩
        audio_base64 = base64.b64encode(audio_bytes)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with text-to-speech, {e}")
    
    return {
        "message": "Upload and processing successful",
        "transcription": transcription_text,
        "gpt_response": gpt_response,
        "response_voice": audio_base64
    }


@refine_router.post("/chat/make_story")
async def make_story(uid:Recallbook_header):
    current_chat_history = settings.reminescense.return_history()
    if len(current_chat_history) == 0:
        return {"message": "no chatting history"}
    
    context = settings.refine.refine(current_chat_history)
    # context DB에 저장..하고 context로 midjourney prompt 생성
    title, midjourney_input = settings.refine.make_midjourney_prompt(context)

    #await image generate
    try:
        image_id = await settings.image_generate.create_image(midjourney_input)
        if 'data' not in image_id or 'id' not in image_id['data']:
            raise HTTPException(status_code=501, detail=f"image create Error")
    
    except Exception as e:
        print(f"image create Error{e}")

    image_info = image_id['data']['id']

    try:
        image_urls = await settings.image_generate.check_image_status(image_info)
    
    except Exception as e:
        print(f"Error detected in image_checking process")
    
    recallbook_data = RecallBook( # 삽입할 데이터 생성
        title=title,
        context=context,
        paint_url=image_urls[1],
        user_id=uid.user_id
    ) 
    try:
        await recallbook_data.insert()
    except Exception as e:
        print(f"Error detected in recallbook data insertion")
    
    user = await User.find_one({"user_id":recallbook_data.user_id})
    if recallbook_data.recallbook_id not in user.recallbooks:
        user.recallbooks.append(recallbook_data.recallbook_id)
        await user.save() # 유저 정보 저장
        # 현재 chatting History 초기화
        settings.reminescense.chat_history = []
    else:
        return {"message":"samebook exist"}

    return {"message": "Recallbook added to user's recallbook list", 
            "user_id": str(recallbook_data.user_id), 
            "recallbook_id": str(recallbook_data.recallbook_id)}


# 얼굴 이미지 나이 변경
@replicate_router.post("/face/aging")
def transform_face_age(input_face: ImageBase64Data, target_age: str):

    try:
        modifiec_image_container_name = "persona-image"

        # Base64 형식으로 받아 이미지 변환 후 Base64 형식으로 결과를 얻음
        result_face = settings.sam_client.transform_face_age(input_image=input_face, target_age=target_age)

        # Base64 형식으로 받아 이미지로 변환 후 Azure blob storage에 업로드하여 이미지 url을 얻음
        result_url = settings.azure_client.upload_image_to_storage(input_face=result_face, container_name=modifiec_image_container_name)

        return result_url
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while aging the face, {e}")



# # 목소리 추가 엔드포인트
# @elevenlabs_router.post("/voice/clone")
# async def clone_voice(persona_audio_input: PersonaAudioData):

#     # 해당 persona에 voice_id가 존재한다면 에러
#     persona = await Persona.find_one(Persona.persona_id == persona_audio_input.persona_id)
#     if persona.voice_id:
#         raise HTTPException(status_code=400, detail="Persona already have cloned voice")

#     # Array 형식의 데이터를 bytes로 변환
#     voice_bytes = bytes(persona_audio_input.data)

#     try:
#         response = settings.elevenlabs.clone_voice(name=persona_audio_input.filename, voice_bytes=voice_bytes)

#         # voice_id를 해당 persona에 저장
#         await Persona.find_one(Persona.persona_id == persona_audio_input.persona_id).update({"$set": {Persona.voice_id: response.voice_id}})

#         return {
#             "message": f"Voice '{persona_audio_input.filename}' has been cloned successfully.",
#             "voice_id": response.voice_id
#         }
    
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred while cloning the voice, {e}")


# # 복제 목소리 이용 TTS 엔드포인트
# # 현재는 voice_id를 직접적으로 받지만, 나중에는 user의 고유 id를 이용해 DB에서 해당 값을 가져올 것이다.
# @elevenlabs_router.post("/voice/text_to_voice")
# def text_to_cloned_voice(text: str, voice_id: str):

#     try:
#         response = settings.elevenlabs.text_to_cloned_voice(text=text, voice_id=voice_id)

#         # with open(response, "rb") as audio_file:
#         #     audio_data = audio_file.read()

#         audio_name = "test_audio.mp3"

#         headers = {
#             "Content-Disposition": f"attachment; filename={audio_name}"
#         }

#         # FastAPI Response로 음성 파일 반환
#         return StreamingResponse(content=response, media_type="audio/mpeg", headers=headers)
    
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred with text-to-speech, {e}")


# # 얼굴 이미지 나이 변경
# @replicate_router.post("/face/transform")
# def transform_face_age(input_face: ImageBase64Data, target_age: str):

#     try:
#         original_image_container_name = "selfie"
#         modifiec_image_container_name = "persona-image"

#         # Base64 형식으로 받아 이미지로 변환 후 Azure blob storage에 업로드하여 이미지 url을 얻음
#         image_url = settings.azure_client.upload_image_to_storage(input_face=input_face, container_name=original_image_container_name)

#         # 사진 이미지 url으로 target_age의 얼굴 사진 url을 얻음
#         replicate_url = settings.replicate.transform_face_age(image=image_url, target_age=target_age)
        
#         # 위의 url로 이미지를 받아 Azure bloc storage에 업로드하고 해당 url을 제공
#         output_url = settings.azure_client.upload_image_url_to_storage(image_url=replicate_url, container_name=modifiec_image_container_name)

#         return output_url
    
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred while transforming the face age, {e}")
