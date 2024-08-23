from beanie import Document
from pydantic import BaseModel
from typing import Optional,List

# Audio endpoint에서 사용할 데이터 모델

class AudioData(BaseModel):
    data: List[int]  
    filename: str  
    
    class Config:
        from_attributes = True
