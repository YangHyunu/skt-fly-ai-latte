from fastapi import APIRouter, Depends,HTTPException
from db.database import get_db
from config import settings, Settings
from langchain.prompts import ChatPromptTemplate
from app.schema.db_schema import RecallBook, User, Recallbook_header
from beanie import PydanticObjectId

import json

recallbook_router = APIRouter()

@recallbook_router.post("/user_recallbook")
async def recall_book_search(uid: Recallbook_header):
    # uid -> db에서 조회 -> recallbooks list 반환,
    result = []
    try:
        user = await User.find_one(User.user_id == uid.user_id)
        # user = await User.find_one({"user_id":uid.user_id})

    except Exception as e:
        print("user not found")
        return {"message":"user not found"}
    for recallbook_id in user.recallbooks:
        recallbook_info = await RecallBook.find_one(RecallBook.recallbook_id == recallbook_id)
        # recallbook_info = await RecallBook.find_one({"recallbook_id":recallbook_id})
        result.append({
            "recallbook_id":recallbook_id,
            "recallbook_title": recallbook_info.title,
            "recallbook_context": recallbook_info.context,
            "recallbook_paint" : recallbook_info.paint_url
            })
    return { "recallbook_list": result }
    

async def get_recallbook(recallbook_id):
    try:
        recallbook_info = await RecallBook.find_one({"recallbook_id":recallbook_id})
        return {
                "recallbook_id":recallbook_id,
                "recallbook_title": recallbook_info.title,
                "recallbook_context": recallbook_info.context,
                "recallbook_paint" : recallbook_info.paint_url
                }
    except Exception as e:
        print("can't read recall_book")
        return {"message": "Error reading recall book"}
    
@recallbook_router.delete("/recallbook/delete/{recallbook_id}")
async def delete_recallbook(recallbook_id: PydanticObjectId, user_id: PydanticObjectId):
    
    # DB에서 recallbook_id의 페르소나 찾기
    recallbook = await RecallBook.find_one(RecallBook.recallbook_id == recallbook_id)

    if not recallbook:
        raise HTTPException(status_code=404, detail="Recallbook not found")
    
    # DB에서 recallbook 삭제
    await recallbook.delete()

    # recallbook를 가지고 있는 user 확인
    user = await User.find_one(User.user_id == user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # 리스트에서 특정 recallbook_id 삭제 (순서 유지)
        user.recallbooks.remove(recallbook_id)
    except ValueError:
        # 만약 리스트에 recallbook_id가 없으면 예외 발생
        raise HTTPException(status_code=404, detail="Recallbook ID not found in user's recallbook list")
    
    # 유저 정보 저장
    await user.save()

    return {"message": "Recallbook deleted successfully",
            "recallbook_id": str(recallbook_id),
            "user_id": str(user_id)}
