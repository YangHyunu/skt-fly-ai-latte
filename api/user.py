from app.schema.db_schema import User, UserCreate, LoginHeader
from fastapi import APIRouter, HTTPException
from config import settings

user_router = APIRouter()
login_router = APIRouter()

# fe_input_example = {
#     "name": "김점례",
#     "phone_number" : "01011111111",
#     "password" : "11111111",
#     "birth":"1968-06-18",
#     "gender":"female",
# }


@user_router.post("/user/create")
async def create_user_instance(fe_input: UserCreate):
    existing_user = await User.find_one(User.phone_number == fe_input.phone_number)
    # 고유값 : 폰번호 중복검사
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this phone number already exists")
    # User class 형태 로 frontend 입력 변환
    user_data = User(name = fe_input.name,
            phone_number = fe_input.phone_number,
            password = fe_input.password,
            birth=fe_input.birth,
            gender=fe_input.gender,
            )
    # 이미 초기화 되어있는 DB에 삽입.
    await user_data.insert()

    return {"message": "User added in DB", "user_id": user_data.user_id }

@login_router.post("/login")
async def login_user(fe_input: LoginHeader):
    # 입력받은 전화번호로 사용자 조회
    existing_user = await User.find_one(User.phone_number == fe_input.phone_number)
    print(existing_user)
    # 사용자가 존재하지 않으면 404 오류 반환
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 비밀번호가 일치하는지 확인
    if fe_input.password != existing_user.password:
        raise HTTPException(status_code=401, detail="Incorrect password")
    await settings.reminescense.prepare_to_chat(existing_user)
    await settings.refine.user = existing_user
    id = str(existing_user.user_id)
    # 비밀번호가 일치하면 user_id 반환
    return {"user_id": id}

# async def create_user(name: str, phone_number: str, password: str, birth: str, gender: str):
#     existing_user = await User.find_one(User.phone_number == phone_number)
#     # 고유값 : 폰번호 중복검사
#     if existing_user:
#         raise HTTPException(status_code=400, detail="User with this phone number already exists")
    
#     user_data = User(name=name,
#                      phone_number=phone_number,
#                      password=password,
#                      birth=birth,
#                      gender=gender,
#                      )

#     await user_data.insert()

#     return {"message": "User added in DB", "user_id": str(user_data.user_id)}