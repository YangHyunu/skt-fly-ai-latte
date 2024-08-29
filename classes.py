from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from prompts.reminisence_prompt import rem_prompt
from prompts.refine_prompt import refining_prompt
from prompts.midjourney import midjourney_prompt, title_prompt
import httpx
import aiohttp
import asyncio
from fastapi import HTTPException
import json
import os
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import replicate
import base64
import uuid
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from app.models.image import ImageBase64Data
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta, timezone
from rembg import remove
import requests
from langchain_core.tools import StructuredTool


from prompts.interview_list import interview_list
from app.schema.db_schema import User, RecallBook
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool, initialize_agent
from argparse import Namespace
import torch
import torchvision.transforms as transforms
import dlib
from pydantic import BaseModel
from SAM.datasets.augmentations import AgeTransformer
from SAM.utils.common import tensor2im
from SAM.models.psp import pSp
from SAM.scripts.align_all_parallel import align_face

class EmptyArgsSchema(BaseModel):
    pass

# .env 파일 로드
load_dotenv()

class ClovaSpeechClient:
    def __init__(self):
        self.invoke_url = os.getenv('CLOVA_SPEECH_INVOKE_URL')
        self.secret = os.getenv('CLOVA_SPEECH_SECRET')

    async def req_upload(self, file_bytes: bytes, completion: str, filename: str, callback=None, userdata=None, 
                         forbiddens=None, boostings=None, wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('media', file_bytes, filename=filename, content_type='audio/aac')
            form_data.add_field('params', json.dumps(request_body), content_type='application/json')
            
            async with session.post(self.invoke_url + '/recognizer/upload', headers=headers, data=form_data) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Clova Speech API Error")
                return await response.json()

class ReminiscenceAgent:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.temperature = 0.7
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o",
            temperature=self.temperature,
            top_p=0.6,
            seed=42
        )
        self.loaded_context = None
        self.instruction = None
        self.prompt = None
        self.chat_history = []
        self.agent = None
        self.user = None

    #login에서 같이 실행.
    async def prepare_to_chat(self, uid:User):
        self.user = uid
        self.loaded_context = await self.context(uid)
        self.instruction = self._init_instruction()
        self.tools = [self._create_retrieve_history_tool(), self._create_recomendataion_list_tool()]
        self.prompt = self.create_prompt()
        self.chat_history = []
        self.agent = self._init_agent(self.tools)
        self.agent_excutor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _init_agent(self, tools):
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt
        )
        return agent

    def _init_instruction(self):
        return [SystemMessage(content=rem_prompt.format(gender=self.user.gender, name=self.user.name, birth=self.user.birth))]

    async def context(self, uid:User):
        result = ""
        if not uid.recallbooks:
            return "No recallbooks"
        for idx, recallbook_id in enumerate(uid.recallbooks):

            idx += 1
            recallbook_info = await RecallBook.find_one({"recallbook_id":recallbook_id})
            result += f"{idx}: {recallbook_info.context}\n"
        return result

    def create_prompt(self):
        messages = self.instruction + self.chat_history + ["{user_input}"] + ["{agent_scratchpad}"]
        prompt = ChatPromptTemplate(messages=messages)
        return prompt
    
    def retrieve_history_tool(self, _=None):
            if self.loaded_context and self.loaded_context != "No recallbooks":
                return self.loaded_context
            else:
                return "저장된 이야기가 없음."
    
    def _create_retrieve_history_tool(self):
        return StructuredTool(name="RetrieveHistory", 
                    func=self.retrieve_history_tool, 
                    description=(
                    """If the user asks about previous conversations or wants to recall past dialogues, 
                    use this tool to retrieve the user's chat history. For example, questions like 'What conversations did we have before?', 
                    'What did we talk about?', or 'Tell me about our past conversations' would trigger the use of this tool.
                    사용자가 이전 대화에 대해 질문하거나, 과거에 나눈 대화를 회상하고 싶어 하는 경우, 
                        이 도구를 사용하여 사용자의 대화 기록을 가져옵니다. 예를 들어, '전에 내가 어떤 채팅들을 했었지?', 
                        '우리가 무슨 얘기를 했었나요?', '과거에 나눈 대화를 알려줘'와 같은 질문에 이 도구를 사용합니다."""),
                    args_schema=EmptyArgsSchema
                    )
    ###########
    def recomendation_list_tool(self, _=None):
        return interview_list

    def _create_recomendataion_list_tool(self):
        return StructuredTool(name="GetQuestionList",
                    func=self.recomendation_list_tool,
                    description=("""이 도구는 다양한 생애 단계에 따라 관련된 질문이나 주제가 담겨 있다., 
                                 사용자가 주제를 요청하거나 추천받기를 원할 때 해당 객체 안에 있는 내용을 이용하여 답변을 생성한다.,
                                 답변을 생성할 때에는 특정한 연령대를 먼저 말한다. ,
                                 마지막으로 사용자에게 특정한 연령대내부에 있는 주제 3가지를 제공한다 이때 한국어로만 대답한다..
                                 """
                    ),
                    args_schema=EmptyArgsSchema
                    )
    ###########

    async def get_gpt_response(self, clova_text:str):
        user_message = HumanMessage(content=clova_text)
        self.chat_history.append(user_message)
        self.prompt = self.create_prompt() # chat_history load

        response = await self.agent_excutor.ainvoke({"user_input": clova_text})

        self.chat_history.append(response['output'])
        
        return response['output']
    
    def return_history(self):
        return self.chat_history

# class reminiscence_gpt:
#     def __init__(self):
#         self.api_key = os.getenv('OPENAI_API_KEY')
#         self.chat_model = ChatOpenAI(
#             api_key=self.api_key,
#             model="gpt-4o",
#             temperature=0.6,
#             top_p=0.7,
#             seed=42
#         )
#         self.persona_message = SystemMessage(content=rem_prompt)
#         self.chat_history = []
#         prompt_template = self.create_prompt_template()
#         self.llm_chain = LLMChain(llm=self.chat_model, prompt=prompt_template)

#     def update_user_chat_history(self, clova_text: str):
#         self.chat_history.append(HumanMessage(content=clova_text))

#     def update_ai_chat_history(self, model_response:str):
#         self.chat_history.append(AIMessage(content=model_response))
        
#     def create_prompt_template(self) -> ChatPromptTemplate:
#         messages = [self.persona_message] + self.chat_history + ["{user_input}"]
#         prompt_template = ChatPromptTemplate(messages=messages)
#         return prompt_template
        
#     def get_gpt_response(self, clova_text: str) -> str:
#         self.llm_chain.prompt = self.create_prompt_template()
#         response = self.llm_chain.run({"user_input":clova_text})
#         self.update_user_chat_history(clova_text)
#         self.update_ai_chat_history(response)
#         return response
    
#     def return_history(self) -> list:
#         return self.chat_history
    
#     def get_chat_history(self) -> list:
#         return self.chat_history
    

class refine_gpt:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.refine_model = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o",
            temperature=0.3,
            top_p=0.7,
            seed=42
        )
        self.midjourney_model = ChatOpenAI(
            api_key = self.api_key,
            model="gpt-4o",
            temperature=0.0,
            top_p=0.7,
            seed=42
        )
        self.refine_prompt = PromptTemplate(input_variables=['chat_history'], template=refining_prompt)
        self.refine_chain = LLMChain(llm=self.refine_model,
                            prompt=self.refine_prompt)
        
        self.title_template = PromptTemplate(input_variables=['context'], template=title_prompt)
        self.title_chain = LLMChain(llm=self.midjourney_model,
                                    prompt=self.title_template)
        
        self.midjourney_template = PromptTemplate(input_variables=['context'], template=midjourney_prompt)
        self.midjourney_chain = LLMChain(llm=self.midjourney_model,
                                    prompt=self.midjourney_template)

    def refine(self, chat_history) -> str:
        return self.refine_chain.run({"chat_history":chat_history})
    
    def make_midjourney_prompt(self, refine_story) -> str:
        title = self.title_chain.run({"context":refine_story})
        midjourney_input = self.midjourney_chain.run({"context":refine_story})
        return title, midjourney_input


class image_generator:
    def __init__(self):
        self.api_key = os.getenv("IMA_API_KEY")
        self.headers = {
            'Authorization' : f'Bearer {self.api_key}',
            'Content-Type' : 'application/json'
        }

    async def create_image(self, context: str) -> dict:
        data = {"prompt": context}
        async with httpx.AsyncClient() as client:
            response = await client.post("https://cl.imagineapi.dev/items/images/", json=data, headers=self.headers)
        return response.json()
    
    async def check_image_status(self, image_id: str) -> list:
        async with httpx.AsyncClient() as client:
            while True:
                response = await client.get(f'https://cl.imagineapi.dev/items/images/{image_id}', headers=self.headers)
                response_data = response.json()
                if response_data['data']['status'] == 'completed':
                    if 'upscaled_urls' in response_data['data']:
                        return response_data['data']['upscaled_urls']
                    else:
                        print('image created')
                        raise HTTPException(status_code=502, detail='image created')
                    
                elif response_data['data']['status'] == 'failed':
                    print('image creation Failed')
                    raise HTTPException(status_code=503, detail='image creation Failed')

                else:
                    print('waiting for image generation')
                    asyncio.sleep(15)


class ElevenLabsClient:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)
        self.model_id = "eleven_turbo_v2_5"

    def clone_voice(self, voice_bytes: bytes, name: str):
        try:
            response = self.client.voices.add(name=name, files=[voice_bytes])
            return response
            # return {"message": f"Voice '{name}' has been added successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def text_to_cloned_voice(self, text: str, voice_id: str):
        try:
            response = self.client.text_to_speech.convert_as_stream(
                voice_id=voice_id,
                text=text,
                model_id=self.model_id,
                language_code="ko",
                voice_settings=VoiceSettings(
                    stability=0.8,
                    similarity_boost=0.9,
                ),
                seed=42,
            )

            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        

class ReplicateClient:
    def __init__(self):
        self.model = os.getenv("REPLICATE_MODEL")

    def transform_face_age(self, image: str, target_age: str):
        input = {
            "image": image,
            "target_age": target_age
        }
        try:
            output = replicate.run(self.model, input=input)
            return output
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class AzureBlobClient:
    def __init__(self):
        self.connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

    def upload_image_to_storage(self, input_face: ImageBase64Data, container_name: str) -> str:
        try:
            # base64 문자열에서 이미지 데이터를 디코딩
            image_bytes = base64.b64decode(input_face.base64_image)

            image = Image.open(BytesIO(image_bytes))

            # 업로드 파일 이름 무작위 생성
            # file_extension = os.path.splitext(input_face.filename)[1]
            # file_name = f"{uuid.uuid4()}{file_extension}"
            file_name = f"{uuid.uuid4()}.png"
            
            # 이미지 배경제거
            clean_image = remove(image, bgcolor=(255, 255, 255, 255))

            # 임시 파일로 이미지를 저장
            temp_file_path = f"./temp/{file_name}"
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            clean_image.save(temp_file_path)

            # Azure Blob Storage에 업로드
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=file_name)

            with open(temp_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            # SAS 토큰 자동 생성
            sas_token = generate_blob_sas(
                account_name=self.blob_service_client.account_name,
                container_name=container_name,
                blob_name=file_name,
                account_key=self.blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),  # 읽기 권한 부여
                expiry=datetime.now(timezone.utc) + timedelta(hours=1)  # 1시간 후 만료
            )

            # 임시 파일 삭제
            os.remove(temp_file_path)

            # SAS URL 생성
            sas_url = f"{blob_client.url}?{sas_token}"

            return sas_url

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
    def upload_image_url_to_storage(self, image_url: str, container_name: str):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))

            # 업로드 파일 이름 무작위 생성
            # file_extension = os.path.splitext(input_face.filename)[1]
            # file_name = f"{uuid.uuid4()}{file_extension}"
            file_name = f"{uuid.uuid4()}.png"

            # 임시 파일로 이미지를 저장
            temp_file_path = f"./temp/{file_name}"
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            image.save(temp_file_path)

            # Azure Blob Storage에 업로드
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=file_name)

            with open(temp_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            # SAS 토큰 자동 생성
            sas_token = generate_blob_sas(
                account_name=self.blob_service_client.account_name,
                container_name=container_name,
                blob_name=file_name,
                account_key=self.blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),  # 읽기 권한 부여
                expiry=datetime.now(timezone.utc) + timedelta(days=30)  # 30일 후 만료 - 다만 SAS 자체 한계가 7일 인듯
            )

            # 임시 파일 삭제
            os.remove(temp_file_path)

            # SAS URL 생성
            sas_url = f"{blob_client.url}?{sas_token}"

            return sas_url

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
class SAMClient:
    def __init__(self):
        self.EXPERIMENT_TYPE = 'ffhq_aging'
        self.EXPERIMENT_DATA_ARGS = {
            "ffhq_aging": {
                "model_path": "SAM/pretrained_models/sam_ffhq_aging.pt",
                "transform": transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            }
        }
        self.EXPERIMENT_ARGS = self.EXPERIMENT_DATA_ARGS[self.EXPERIMENT_TYPE]
        self.MALE_DEFAULT_REFERENCE_IMAGE_PATH = "SAM/images/reference/male_ref_image.png"
        self.FEMALE_DEFAULT_REFERENCE_IMAGE_PATH = "SAM/images/reference/female_ref_image.png"
        self.DEFAULT_LATENT_MASK = [9]
        self.TARGET_REFERENCE_LATENT_MASK = [8, 9]

    def transform_face_age(self, input_image: ImageBase64Data, target_age: str):

        # 모델 로드
        model_path = self.EXPERIMENT_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cpu')

        opts = ckpt['opts']

        # update the training options
        opts['checkpoint_path'] = model_path

        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        print('Model successfully loaded!')

        # 입력 이미지 디코딩
        image_bytes = base64.b64decode(input_image.base64_image)

        image = Image.open(BytesIO(image_bytes))

        # 업로드 파일 이름 무작위 생성
        file_name = f"{uuid.uuid4()}.png"
            
        # 이미지 배경제거
        clean_image = remove(image, bgcolor=(255, 255, 255, 255))

        # 임시 파일로 이미지를 저장
        temp_file_path = f"./temp/{file_name}"
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        clean_image.save(temp_file_path)

        # 이미지 정렬
        predictor = dlib.shape_predictor("SAM/shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath=temp_file_path, predictor=predictor)
        aligned_image.resize((256, 256))

        img_transforms = self.EXPERIMENT_ARGS['transform']
        transformed_image = img_transforms(aligned_image)

        # we'll run the image on multiple target ages
        age_transformers = [AgeTransformer(target_age=int(target_age))]

        for age_transformer in age_transformers:
            with torch.no_grad():
                input_image_age = [age_transformer(transformed_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)
                result_batch = net(input_image_age.to("cuda").float(), randomize_noise=False, resize=False)
                result_tensor = result_batch[0]
                result_image = tensor2im(result_tensor)

        # 결과 이미지를 bytes-like 객체로 변환
        byte_arr = BytesIO()
        result_image.save(byte_arr, format="PNG")
        output_image_bytes = byte_arr.getvalue()

        # 결과 이미지 Base64로 인코딩
        output_image = base64.b64encode(output_image_bytes)

        modified_image = ImageBase64Data(
            filename=file_name,
            base64_image=output_image,
            user_id=input_image.user_id
        )

        # 임시 파일 삭제
        os.remove(temp_file_path)

        return modified_image

    def transform_face_age_with_reference(self, input_image: ImageBase64Data, target_age: str, gender: str):
        
        # 모델 로드
        model_path = self.EXPERIMENT_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cpu')

        opts = ckpt['opts']

        # update the training options
        opts['checkpoint_path'] = model_path

        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        print('Model successfully loaded!')

        # 입력 이미지 디코딩
        image_bytes = base64.b64decode(input_image.base64_image)

        image = Image.open(BytesIO(image_bytes))

        # 기준 이미지 가져오기
        # 남성 여성 구분 필요
        if gender == "남성":
            ref_image = Image.open(self.MALE_DEFAULT_REFERENCE_IMAGE_PATH)
        else:
            ref_image = Image.open(self.FEMALE_DEFAULT_REFERENCE_IMAGE_PATH)    

        # 업로드 파일 이름 무작위 생성
        input_file_name = f"{uuid.uuid4()}.png"
        ref_file_name = f"{uuid.uuid4()}.png"

        # 이미지 배경제거
        input_clean_image = remove(image, bgcolor=(255, 255, 255, 255))
        ref_clean_image = remove(ref_image, bgcolor=(255, 255, 255, 255))

        # 임시 파일로 이미지를 저장
        input_temp_file_path = f"./temp/{input_file_name}"
        os.makedirs(os.path.dirname(input_temp_file_path), exist_ok=True)
        input_clean_image.save(input_temp_file_path)

        ref_temp_file_path = f"./temp/{ref_file_name}"
        os.makedirs(os.path.dirname(ref_temp_file_path), exist_ok=True)
        ref_clean_image.save(ref_temp_file_path)

        # 이미지 정렬
        predictor = dlib.shape_predictor("SAM/shape_predictor_68_face_landmarks.dat")
        
        input_aligned_image = align_face(filepath=input_temp_file_path, predictor=predictor)
        input_aligned_image.resize((256, 256))
        img_transforms = self.EXPERIMENT_ARGS['transform']
        input_norm_image = img_transforms(input_aligned_image)

        ref_aligned_image = align_face(filepath=ref_temp_file_path, predictor=predictor)
        ref_aligned_image.resize((256, 256))
        img_transforms = self.EXPERIMENT_ARGS['transform']
        ref_norm_image = img_transforms(ref_aligned_image)

        # we'll run the image on multiple target ages
        age_transformers = [AgeTransformer(target_age=int(target_age))]

        for age_transformer in age_transformers:
            with torch.no_grad():
                # latent 얻기
                ref_cuda_image = ref_norm_image.to("cuda")
                ref_latents = net.pretrained_encoder(ref_cuda_image.unsqueeze(0)) + net.latent_avg
                
                input_image_age = [age_transformer(input_norm_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)

                result_batch = net(input_image_age.to("cuda").float(),
                                   latent_mask=self.DEFAULT_LATENT_MASK,
                                   inject_latent=ref_latents,
                                   # alpha=opts.mix_alpha,
                                   resize=False)

                # result_batch = net(input_image_age.to("cuda").float(), randomize_noise=False, resize=False)
                result_tensor = result_batch[0]
                result_image = tensor2im(result_tensor)

        # 결과 이미지를 bytes-like 객체로 변환
        byte_arr = BytesIO()
        result_image.save(byte_arr, format="PNG")
        output_image_bytes = byte_arr.getvalue()

        # 결과 이미지 Base64로 인코딩
        output_image = base64.b64encode(output_image_bytes)

        modified_image = ImageBase64Data(
            filename=input_file_name,
            base64_image=output_image,
            user_id=input_image.user_id
        )

        # 임시 파일 삭제
        os.remove(input_temp_file_path)
        os.remove(ref_temp_file_path)

        return modified_image

    def transform_face_age_with_target(self, input_image: ImageBase64Data, target_age: str, target_image: ImageBase64Data):
        
        # 모델 로드
        model_path = self.EXPERIMENT_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cpu')

        opts = ckpt['opts']

        # update the training options
        opts['checkpoint_path'] = model_path

        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        print('Model successfully loaded!')

        # 입력 이미지 디코딩
        image_bytes = base64.b64decode(input_image.base64_image)
        target_bytes = base64.b64decode(target_image.base64_image)

        image = Image.open(BytesIO(image_bytes))
        target = Image.open(BytesIO(target_bytes))  

        # 업로드 파일 이름 무작위 생성
        input_file_name = f"{uuid.uuid4()}.png"
        ref_file_name = f"{uuid.uuid4()}.png"

        # 이미지 배경제거
        input_clean_image = remove(image, bgcolor=(255, 255, 255, 255))
        ref_clean_image = remove(target, bgcolor=(255, 255, 255, 255))

        # 임시 파일로 이미지를 저장
        input_temp_file_path = f"./temp/{input_file_name}"
        os.makedirs(os.path.dirname(input_temp_file_path), exist_ok=True)
        input_clean_image.save(input_temp_file_path)

        ref_temp_file_path = f"./temp/{ref_file_name}"
        os.makedirs(os.path.dirname(ref_temp_file_path), exist_ok=True)
        ref_clean_image.save(ref_temp_file_path)

        # 이미지 정렬
        predictor = dlib.shape_predictor("SAM/shape_predictor_68_face_landmarks.dat")
        
        input_aligned_image = align_face(filepath=input_temp_file_path, predictor=predictor)
        input_aligned_image.resize((256, 256))
        img_transforms = self.EXPERIMENT_ARGS['transform']
        input_norm_image = img_transforms(input_aligned_image)

        ref_aligned_image = align_face(filepath=ref_temp_file_path, predictor=predictor)
        ref_aligned_image.resize((256, 256))
        img_transforms = self.EXPERIMENT_ARGS['transform']
        ref_norm_image = img_transforms(ref_aligned_image)

        # we'll run the image on multiple target ages
        age_transformers = [AgeTransformer(target_age=int(target_age))]

        for age_transformer in age_transformers:
            with torch.no_grad():
                # latent 얻기
                ref_cuda_image = ref_norm_image.to("cuda")
                ref_latents = net.pretrained_encoder(ref_cuda_image.unsqueeze(0)) + net.latent_avg
                
                input_image_age = [age_transformer(input_norm_image.cpu()).to('cuda')]
                input_image_age = torch.stack(input_image_age)

                result_batch = net(input_image_age.to("cuda").float(),
                                   latent_mask=self.TARGET_REFERENCE_LATENT_MASK,
                                   inject_latent=ref_latents,
                                   # alpha=opts.mix_alpha,
                                   resize=False)

                # result_batch = net(input_image_age.to("cuda").float(), randomize_noise=False, resize=False)
                result_tensor = result_batch[0]
                result_image = tensor2im(result_tensor)

        # 결과 이미지를 bytes-like 객체로 변환
        byte_arr = BytesIO()
        result_image.save(byte_arr, format="PNG")
        output_image_bytes = byte_arr.getvalue()

        # 결과 이미지 Base64로 인코딩
        output_image = base64.b64encode(output_image_bytes)

        modified_image = ImageBase64Data(
            filename=input_file_name,
            base64_image=output_image,
            user_id=input_image.user_id
        )

        # 임시 파일 삭제
        os.remove(input_temp_file_path)

        return modified_image
