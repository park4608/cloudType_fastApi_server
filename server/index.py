from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import sklearn
import xgboost

app = FastAPI()

origins = [
    # "http://localhost:3000",
    # "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 파일 경로
model_path = 'model.pkl'

# 모델 파일의 크기 확인하기
if os.path.isfile(model_path):
    print('Model file size:', os.path.getsize(model_path), 'bytes')
else:
    print('Model file does not exist.')

# 입력값을 받을 모델 정의
class InputData(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float

# 예측 결과를 반환할 모델 정의
class OutputData(BaseModel):
    y: float

# API 라우팅
@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):

    # 모델 파일을 로딩합니다.
    model = joblib.load("model.pkl")
    
    type(input_data.x1)
    type(input_data.x2)
    type(input_data.x3)
    type(input_data.x4)
    type(input_data.x5)
    type(input_data.x6)

    # 입력값을 Numpy 배열로 변환합니다.
    X = np.array([input_data.x1, input_data.x2, input_data.x3, input_data.x4, input_data.x5, input_data.x6]).reshape(1, -1)
    
    print('predict')
    # 모델을 실행하고 예측 결과를 계산합니다.
    y_pred = model.predict(X)[0]
    
    print('return predict')
    # 예측 결과를 반환합니다.
    return {"y": y_pred}