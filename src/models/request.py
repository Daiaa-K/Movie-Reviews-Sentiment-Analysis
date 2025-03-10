from pydantic import BaseModel
from typing import List

class TrainingData(BaseModel):
    texts:List[str]
    Labels:List[int]
    
class TestingData(BaseModel):
    texts:List[str]
    
class QueryText(BaseModel):
    text:str