from app.schema.db_schema import Persona, PersonaCreate, User
from fastapi import APIRouter, HTTPException
from beanie import PydanticObjectId

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
            "user_id": persona_data.user_id,
            "persona_id": persona_data.persona_id,
            "image_url": persona_data.image_url}

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