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
        user = await User.find_one({"user_id":uid.user_id})

    except Exception as e:
        print("user not found")
        return {"message":"user not found"}
    for recallbook_id in user.recallbooks:
        recallbook_info = await RecallBook.find_one({"recallbook_id":recallbook_id})
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
