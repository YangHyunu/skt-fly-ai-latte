from app.schema.db_schema import Persona, PersonaCreate, User
from fastapi import APIRouter, HTTPException
from beanie import PydanticObjectId
from config import settings, Settings

persona_router = APIRouter()

# id_example = PydanticObjectId('66ca1b0ea4c6cbb2eb856b92')

# fe_input_example = {
#     "image_url": "https://cl.imagineapi.dev/assets/249f1fe7-55c9-430d-8cb3-0b53a4fc30e2/249f1fe7-55c9-430d-8cb3-0b53a4fc30e2.png",
#     "voice_id" : "voice_id_input",
#     "user_id": id_example
# }

@persona_router.post("/persona/create")
async def create_persona(fe_input: PersonaCreate):
    # feinput 변환 Persona 객체로
    persona_data = Persona(
        image_url=fe_input.image_url,
        voice_id=None,
        user_id=fe_input.user_id
    )
    
    # DB에 삽입
    await persona_data.insert()

    # 유저 정보에 persona_id 등록 -> 후에 유저가 가지고 있는 페르소나 조회 가능
    user = await User.find_one(User.user_id == fe_input.user_id)
    # user = await User.find_one({"user_id": user_id})
    user.persona.append(persona_data.persona_id)
    await user.save() # 유저 정보 저장

    return {"message": "Persona added to user's persona list",
            "user_id": str(persona_data.user_id),
            "persona_id": str(persona_data.persona_id),
            "image_url": persona_data.image_url}


@persona_router.delete("/persona/delete/{persona_id}")
async def delete_persona(persona_id: PydanticObjectId, user_id: PydanticObjectId):
    
    # DB에서 persona_id의 페르소나 찾기
    persona = await Persona.find_one(Persona.persona_id == persona_id)

    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    # voice cloning이 되어 있다면 해당 목소리도 제거
    if persona.voice_id:
        voice_id = persona.voice_id
        response = settings.elevenlabs.delete_voice(voice_id=voice_id)

    persona = await Persona.find_one(Persona.persona_id == persona_id)
    
    # DB에서 persona 삭제
    await persona.delete()

    # persona를 가지고 있는 user 확인
    user = await User.find_one(User.user_id == user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # 리스트에서 특정 persona_id 삭제 (순서 유지)
        user.persona.remove(persona_id)
    except ValueError:
        # 만약 리스트에 persona_id가 없으면 예외 발생
        raise HTTPException(status_code=404, detail="Persona ID not found in user's persona list")
    
    # 유저 정보 저장
    await user.save()

    return {"message": "Persona deleted successfully",
            "persona_id": str(persona_id),
            "user_id": str(user_id)}


# async def create_persona(image_url: str, user_id: PydanticObjectId):
#     # feinput 변환 Persona 객체로
#     persona_data = Persona(
#         image_url=image_url,
#         voice_id=None,
#         user_id=user_id
#     )
    
#     # DB에 삽입
#     await persona_data.insert()

#     # 유저 정보에 persona_id 등록 -> 후에 유저가 가지고 있는 페르소나 조회 가능
#     user = await User.find_one({"user_id": user_id})
#     user.persona.append(persona_data.persona_id)
#     await user.save() # 유저 정보 저장

#     return {"message": "Persona added to user's persona list", "user_id": str(persona_data.user_id), "persona_id": str(persona_data.persona_id)}