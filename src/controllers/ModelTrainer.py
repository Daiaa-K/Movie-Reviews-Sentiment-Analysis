from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

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
            self.pipeline = joblib.load(self._model_path)
        else:
            self.pipeline = None
        
        self._running_threads = []
        
        
    def _update_status(self,status:str,classes: List[str] = [],evaluation:Dict = {}) ->None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes
        self.model_status['evaluation'] = evaluation
        
        with open(self._status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)
            
            
    def _train_job(self,X_train:List[str],y_train:List[int],X_test: List[str],y_test:list[int]):
        # train the model
        self.pipeline.fit(X_train,y_train)
        
        #Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_pred,y_test,output_dict=True)
        classes = self.pipeline.classes_
        classes = ["positive" if label == 1 else "negative" for label in classes]
        
        #update model status
        self._update_status(status="Mode is Trained and ready",classes=classes,evaluation=report)
        
        #save the model
        joblib.dump(self.pipeline,self._model_path)

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
        rf = RandomForestClassifier(n_estimators=251,max_depth=40,min_samples_split=10)
        vect = CountVectorizer(max_features=50000,stop_words="english")
        self.pipeline = Pipeline(
        [('vectorizer',vect),
        ('RandomForestClassifier',rf)]
        )
        
        #update status
        self._update_status(status="Training...")
        
        t = Thread(target=self._train_job,args=[X_train,y_train,X_test,y_test])
        self._running_threads.append(t)
        t.start()
        
    def predict(self,texts: List[str])->List[Dict]:
        response = []
        y_probas = self.pipeline.predict_proba(texts)
        classifier = self.pipeline.named_steps['RandomForestClassifier']  # Replace 'clf' with your step name
        class_labels = classifier.classes_
        if self.pipeline:
            for i,row in enumerate(y_probas):
                row_pred = {}
                row_pred["text"] = texts[i]
                row_pred["prediction"] = dict(zip(self.model_status["classes"], map(lambda x: round(float(x), 3), row)))
                response.append(row_pred)
        else:
            raise Exception("No trained model available, please train a model first")
        
        return response
    
    
    def get_status(self)->Dict:
        return self.model_status