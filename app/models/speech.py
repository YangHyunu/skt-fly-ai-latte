from beanie import Document, PydanticObjectId
from pydantic import BaseModel
from typing import Optional,List

# Audio endpoint에서 사용할 데이터 모델

class AudioData(BaseModel):
    data: List[int]  
    filename: str  
    
    class Config:
        from_attributes = True

class PersonaAudioData(BaseModel):
    data: List[int]  
    filename: str
    persona_id: PydanticObjectId

    class Config:
        from_attributes = True

class PersonaBase64AudioData(BaseModel):
    base64_audio: str 
    filename: str
    persona_id: PydanticObjectId

    class Config:
        from_attributes = True