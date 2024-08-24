from app.schema.db_schema import Persona, RecallBook, User
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie, PydanticObjectId
from fastapi import HTTPException
DATABASE_URL = "mongodb+srv://prop30909:LAdDMTT8cYzPi2ws@cluster.2dtdf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster"
# 1. db init
async def initialize_db(db_name="Loyola_test"):
    client = AsyncIOMotorClient(DATABASE_URL) #    client = AsyncIOMotorClient(self.DATABASE_URL) # 
    await init_beanie(database=client[db_name], document_models=[Persona, RecallBook, User])

# start app
# 회원 가입을 
# front 요청(UserCreate)이 아래와 같이 넘어온다고 가정. 
fe_user_input = {
    "name": "김점례",
    "phone_number" : "01011111111",
    "password" : "11111111",
    "birth":"1968-06-18",
    "gender":"female",
}

# DB에 넣을 객체로 다음과 같이 변환해야댐
#@app.post("/users/", response_model=User) # 라우팅 되면 실행, User형식으로 데이터를 받아야 함. 
#@app.post("/users/")
async def create_user_instance(fe_input):
    existing_user = await User.find_one(User.phone_number == fe_input['phone_number'])
    # 고유값 : 폰번호 중복검사
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this phone number already exists")
    
    user_data = User(name = fe_input['name'],
            phone_number = fe_input['phone_number'],
            password = fe_input['password'],
            birth=fe_input['birth'],
            gender=fe_input['gender'],
            )
    
    await user_data.insert()

    return {"message": "User added in DB", "user_id": user_data.user_id }

# DB 조회하기

# 2. 페르소나 만들기 - 찬병이 꺼 
#현재유저 id + persona입력이 있어야 함. 
# 프론트에서 이렇게 입력 온다고 가정하고 만든 코드 
# id = PydanticObjectId('66ca1b0ea4c6cbb2eb856b92')

# fe_persona_input = {
#     "image_url": "https://cl.imagineapi.dev/assets/249f1fe7-55c9-430d-8cb3-0b53a4fc30e2/249f1fe7-55c9-430d-8cb3-0b53a4fc30e2.png",
#     "voice_id" : "voice_id_input",
#     "user_id": id
# }
# DB에 넣을 객체로 변환 
async def add_persona(fe_input):
    # feinput 변환 Persona 객체로
    persona_data = Persona(
        image_url=fe_input['image_url'],
        voice_id=fe_input['voice_id'],
        user_id=fe_input['user_id']
    )
    # persona DB에 등록. 
    existing_voice = await Persona.find_one(Persona.voice_id == fe_input['voice_id'])
    if existing_voice:
        raise HTTPException(status_code=400, detail="same voice_id exist")
    
    await persona_data.insert()
    # 유저 정보에 persona_id 등록 -> 후에 유저가 가지고 있는 페르소나 조회 가능
    user = await User.find_one({"user_id": id})
    user.persona.append(persona_data.persona_id)
    await user.save() # 유저 정보 저장

    return {"message": "Persona added to user's persona list", "user_id": persona_data.user_id, "persona_id": persona_data.persona_id}


#id = PydanticObjectId('66ca1b0ea4c6cbb2eb856b92')
#fe_input = {"user_id":id}

async def find_persona(fe_input):
    ids = fe_input["user_id"]
    user = await User.find_one({"user_id": ids}) # 유저찾고
    persona_list = user.persona # 페르소나 리스트 조회
    print(persona_list)
    
    p = await Persona.find_one({"persona_id":persona_list[0]})
    #페르소나 image_url 리턴
    return p.image_url
# 3 내가 쓸거. -> 프론트에서는 user_id 쏴주고, 내부 객체에서 image_url이랑, context 받아서 사용.

# fe_recallbook_input = {
#     "user_id": id
# }

# title = "승부를 뛰어넘은 우정"
# context ='2004년, 열 살의 나는 학교 운동장에서 축구를 하고 있었다. 그날은 가을 햇살이 따사롭게 내리쬐던 평범한 날이었다. 친구들과 함께 축구를 하던 중, 나는 박민에게 태클을 걸었다. 그 순간 박민은 갑자기 화를 내며 나를 향해 소리쳤다. 스포츠 경기 중 일어난 일이었기에 나는 왜 그가 그렇게 화를 내는지 이해할 수 없었다. 그때의 나는 당황스럽고 기분이 좋지 않았다. 친구와의 갈등은 나에게 큰 충격으로 다가왔고, 그날의 기억은 오랫동안 내 마음속에 남아 있었다.\n\n시간이 지나면서 우리는 자연스럽게 다시 가까워졌다. 어느 날, 박민이 먼저 다가와 미안하다고 사과했다. 나도 그때의 감정을 솔직하게 털어놓으며 서로의 마음을 이해하게 되었다. 그렇게 우리는 다시 친구가 되었고, 그 사건은 오히려 우리 사이를 더욱 단단하게 만들어 주었다. 그때의 경험은 나에게 갈등을 해결하는 방법과 용서의 중요성을 가르쳐 주었다. 지금도 가끔 그 시절을 떠올리며, 어린 시절의 소중한 추억으로 남아 있다.'
# paint_url ="https://cl.imagineapi.dev/assets/91daa9db-521d-4aee-905f-03ed00a98476/91daa9db-521d-4aee-905f-03ed00a98476.png"

async def add_recallbook(fe_input, title, context, paint_url):
    # existing_user = await User.find_one(RecallBook.context == context)
    # if existing_user:
        
    recallbook_data = RecallBook(
        title=title,
        context=context,
        paint_url=paint_url,
        user_id=fe_input['user_id'])
    await recallbook_data.insert()
    # 유저 조회    
    user = await User.find_one({"user_id": id})
    if recallbook_data.recallbook_id not in user.recallbooks:
        user.recallbooks.append(recallbook_data.recallbook_id)
        await user.save() # 유저 정보 저장
    else:
        return {"message":"samebook exist"}
    
    return {"message": "Recallbook added to user's recallbook list", "user_id": recallbook_data.user_id, "recallbook_id": recallbook_data.recallbook_id}

# mac = await add_recallbook(fe_recallbook_input, title, context, paint_url)