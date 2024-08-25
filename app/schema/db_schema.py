from typing import Optional, List, Union
from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field

class Persona(Document):
    persona_id: PydanticObjectId = Field(default_factory=PydanticObjectId, description="id of persona instance")
    image_url: str
    voice_id: Optional[str] = None
    #chat_history: List[Union[HumanMessage, AIMessage]] = Field(default_factory=list)
    user_id: PydanticObjectId

class PersonaCreate(BaseModel):
    image_url: str
    user_id: PydanticObjectId


class RecallBook(Document):
    recallbook_id: PydanticObjectId = Field(default_factory=PydanticObjectId, description="id of recall_book instance")
    title: str
    paint_url: str
    context: str
    user_id: PydanticObjectId


class User(Document):
    user_id: PydanticObjectId = Field(default_factory=PydanticObjectId, description="User")
    name: str = Field(min_length=1, max_length=50, description="사용자 이름")
    phone_number: str = Field(min_length=10, max_length=11, description="전화번호를 - 없이 입력해주세요.")
    password: str = Field(min_length=8, max_length=128, description="비밀번호")
    birth: str = Field(min_length=8, max_length=10, pattern=r'^\d{4}-\d{2}-\d{2}$', description="생년월일은 YYYY-MM-DD 형식이어야 합니다.")
    gender: str = Field(min_length=1, max_length=10, description="성별 - 남자 or 여자")
    persona: List[PydanticObjectId] = Field(default_factory=list, description="List of persona_id")
    recallbooks: List[PydanticObjectId] = Field(default_factory=list, description="List of recall_books_id")


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50, description="사용자 이름")
    phone_number: str = Field(min_length=10, max_length=11, description="전화번호를 - 없이 입력해주세요.")
    password: str = Field(min_length=8, max_length=128, description="비밀번호")
    birth: str = Field(min_length=8, max_length=10, pattern=r'^\d{4}-\d{2}-\d{2}$', description="생년월일은 YYYY-MM-DD 형식이어야 합니다.")
    gender: str = Field(min_length=1, max_length=10, description="성별 - 남자 or 여자")

class LoginHeader(BaseModel):
    phone_number: str = Field(min_length=10, max_length=11, description="전화번호를 - 없이 입력해주세요.")
    password: str = Field(min_length=8, max_length=128, description="비밀번호")

class Recallbook_header(BaseModel):
    user_id: PydanticObjectId