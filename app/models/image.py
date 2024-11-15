from pydantic import BaseModel
from beanie import PydanticObjectId

# Image endpoint에서 사용할 데이터 모델
class ImageData(BaseModel):
    image_url: str

class ImageBase64Data(BaseModel):
    filename: str
    base64_image: str
    user_id: PydanticObjectId