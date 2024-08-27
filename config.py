import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from dotenv import load_dotenv
from app.schema.db_schema import Persona, RecallBook, User
from classes import (ClovaSpeechClient, reminiscence_gpt, refine_gpt, image_generator, ElevenLabsClient, ReplicateClient, AzureBlobClient, SAMClient)

# .env 파일 로드
load_dotenv()

class Settings:
    def __init__(self):
        # 환경 변수 로드
        self.DATABASE_URL = os.getenv('DATABASE_URL')
        self.SECRET_KEY = os.getenv('SECRET_KEY')
        self.ALGORITHM = os.getenv('ALGORITHM', 'HS256')
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30))
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.IMA_API_KEY = os.getenv('IMA_API_KEY')
        self.CLOVA_SPEECH_INVOKE_URL = os.getenv('CLOVA_SPEECH_INVOKE_URL')
        self.CLOVA_SPEECH_SECRET = os.getenv('CLOVA_SPEECH_SECRET')
        self.ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
        self.REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
        self.REPLICATE_MODEL = os.getenv('REPLICATE_MODEL')

        # 클로바 클라이언트와 reminiscence_gpt 객체 초기화
        self.clova_client = None
        self.reminescense = None
        self.refine = None
        # 초기화 시 콘솔 메시지 출력
        self._log_initialization()

    def _log_initialization(self):
        """초기화 로그 메시지 출력."""
        print(f"Settings initialized with DATABASE_URL: {self.DATABASE_URL}")
        print(f"Settings initialized with CLOVA_SPEECH_INVOKE_URL: {self.CLOVA_SPEECH_INVOKE_URL}")

    async def initialize_database(self, db_name="Loyola_test"):
        """비동기 데이터베이스 초기화."""
        client = AsyncIOMotorClient(self.DATABASE_URL)
        await init_beanie(database=client[db_name], document_models=[Persona, RecallBook, User])  # 여기에 모델 목록 추가

    def initialize_clients(self):
        """클로바와 reminiscence GPT 클라이언트 초기화."""
        print(f"[INFO] Initializing ClovaSpeechClient...")
        self.clova_client = ClovaSpeechClient()
        print(f"[INFO] ClovaSpeechClient initialized.")
        print(f"[INFO] Initializing reminiscence_gpt...")
        self.reminescense = reminiscence_gpt()
        print(f"[INFO] reminiscence_gpt initialized.")
        self.refine = refine_gpt()
        print(f"[INFO] refine_gpt initialized.")
        self.image_generate = image_generator()
        print(f"[INFO] AzureBlobClient initialized.")
        self.azure_client = AzureBlobClient()
        print(f"[INFO] AzureBlobClient initialized.")
        self.elevenlabs = ElevenLabsClient()
        print(f"[INFO] ElevenLabsClient initialized.")
        self.replicate = ReplicateClient()
        print(f"[INFO] ReplicateClient initialized.")
        self.sam_client = SAMClient()

# 설정 객체를 전역으로 사용
settings = Settings()
