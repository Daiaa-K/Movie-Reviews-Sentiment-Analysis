from pydantic import BaseModel
from typing import List,Dict

class PredictObject(BaseModel):
    text:str
    prediction:Dict

class PredictionsObject(BaseModel):
    predictions:List[PredictObject]
    
class StatusObject(BaseModel):
    status: str
    timestamp: str
    classes:List[str]
    evaluation: Dict