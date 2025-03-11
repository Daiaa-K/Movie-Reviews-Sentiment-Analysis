# **Sentiment Analysis API for Movie Reviews**

This project is a **FastAPI-based API** for **sentiment analysis** of movie reviews. It enables training and predicting sentiment (`positive` or `negative`) using a **Random Forest Classifier**. The API also includes authentication, a structured request/response format, and a model storage system.

---

## **Project Structure**

```
📂 project-root/
│-- 📂 src/
│   ├── 📂 assets/
│   │   ├── 📂 storage/             # Stores trained models and model metadata
│   ├── 📂 controllers/             # Handles model training & prediction
│   │   ├── ModelTrainer.py         # Implements training, evaluation, and prediction
│   ├── 📂 helpers/                 # Configuration helpers
│   │   ├── config.py               # Contains app settings and storage paths
│   ├── 📂 models/                  # Defines API request/response formats
│   │   ├── request.py              # Request models (training, prediction)
│   │   ├── response.py             # Response models (predictions, status)
│-- 📜 main.py                      # FastAPI application entry point
│-- 📜 .env.example                 # Example environment variables file
│-- 📜 notebook.ipynb               # Initial sentiment analysis experimentation
│-- 📜 requirements.txt              # Project dependencies
```

---

## **Installation**

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/sentiment-analysis-api.git
cd sentiment-analysis-api
```

### **2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Set up environment variables**
Create a `.env` file (or rename `.env.example`) and define:
```ini
APP_NAME=MovieReviewSentimentAPI
VERSION=1.0
API_SECRET_KEY=your_secret_key
```

---

## **Running the API**

### **Start the FastAPI server**
```bash
uvicorn main:app --reload
```

The API will be available at:
- 📌 **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 📌 **ReDoc UI:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## **API Endpoints**

### **🔹 Health Check**
**Check if the API is running**
```http
GET /
```

### **🔹 Get Model Status**
**Retrieve model training status**
```http
GET /status
```

### **🔹 Train a Model**
**Train the sentiment analysis model**
```http
POST /train
```
**Request Body:**
```json
{
  "texts": ["I loved this movie!", "It was boring and slow."],
  "Labels": [0, 1]
}
```

### **🔹 Predict Sentiment (Single Input)**
**Predict sentiment for a single review**
```http
POST /predict
```
**Request Body:**
```json
{
  "text": "This movie was fantastic!"
}
```
**Response:**
```json
{
  "text": "This movie was fantastic!",
  "prediction": {
    "positive": 0.85,
    "negative": 0.15
  }
}
```

### **🔹 Predict Sentiment (Batch)**
**Predict sentiment for multiple reviews**
```http
POST /predict-batch
```
**Request Body:**
```json
{
  "texts": [
    "Great storytelling and visuals!",
    "I didn't enjoy this film."
  ]
}
```

---

## **Model Details**

- **Algorithm:** `RandomForestClassifier`
- **Feature Extraction:** `CountVectorizer` (Bag of Words)
- **Evaluation Metrics:** `classification_report`
- **Storage:** Trained models & metadata are saved in **`src/assets/storage/`**

---

## **Future Enhancements**

✅ Improve model accuracy with deep learning (e.g., LSTM, Transformers)  
✅ Implement a frontend UI for interactive predictions  
✅ Deploy API using Docker & cloud services  

---

## **Contributors**

👤 – Diaa Kotb -

---

