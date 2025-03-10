from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import os
import joblib
import json
from datetime import datetime
from typing import List,Dict
from threading import Thread,get_ident
from src.helpers.config import STORAGE_PATH

class ModelTrainer():
    def __init__(self):
        self.storage_path = STORAGE_PATH
        #checking for storage path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self._status_path = os.path.join(self.storage_path,"model_status.json")
        #checking for status path
        if os.path.exists(self._status_path):
            with open (self._status_path,mode='r') as f:
                self.model_status = json.load(f)
        else:
            self.model_status = {
                'status': "No Model found",
                'timestamp':datetime.now().isoformat(),
                'classes':[],
                'evaluation': {}
            }
        #Checking for model path
        self._model_path = os.path.join(self.storage_path,"model.pkl")
        if os.path.exists(self._model_path):
            self.pipline = joblib.load(self._model_path)
        else:
            self.pipline = None
        
        self._running_threads = []
        
        
    def _update_status(self,status:str,classes: List[str] = [],evaluation:Dict = {}) ->None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes
        self.model_status['evaluation'] = evaluation
        
        with open(self._model_path,mode="w+") as f:
            json.dump(self.model_status,f,indent=2)
            
            
    def _train_job(self,X_train:List[str],y_train:List[int],X_test: List[str],y_test:list[int]):
        # train the model
        self.pipline.fit(X_train,y_train)
        
        #Evaluate the model
        y_pred = self.pipline.predict(X_test)
        report = classification_report(y_pred,y_test,output_dict=True)
        classes = self.pipline.classes_.tolist()
        
        #update model status
        self._update_status(status="Mode is Trained and ready",classes=classes,evaluation=report)
        
        #save the model
        joblib.dump(self.pipline,self._model_path)

        # remove thread
        id  = get_ident()
        for i,thd in enumerate(self._running_threads):
            if thd.ident == id:
                self._running_threads.pop(i)
                break
            
            
    def train(self,texts: List[str], labels: List[int]):
        if len(self._running_threads):
            raise Exception("Model is currently training...")
        
        #split data and make pipeline
        X_train,X_test,y_train,y_test = train_test_split(texts,labels,random_state=7,stratify=labels)
        clf = RandomForestClassifier(n_estimators=251,max_depth=40,min_samples_split=10)
        vect = CountVectorizer(max_features=50000,stop_words="english",min_df=0.05,max_df=0.9)
        self.pipline = Pipeline(
        [('vectorizer',vect),
        ('RandomForestClassifier',clf)]
        )
        