# NEXA – NextGen Engineered eXpert Architecture

NEXA is a full-stack application that predicts optimal cloud data architectures using a custom Masked Autoencoder model and generates detailed pipeline documentation using Google Gemini. It includes a FastAPI backend, a browser-based frontend, and a PyTorch inference engine.

## 📌 Project Overview

NEXA helps solution architects choose the right cloud technology stack based on inputs like:
- Cloud Provider
- Source Type
- Data Ingestion Mode
- Tools
- Workflow orchestration
- ML involvement
- Use case definition

The system:
1. Collects inputs through an interactive UI
2. Uses a trained ML model to predict missing architecture components
3. Displays ranked architecture recommendations
4. Generates detailed architecture documentation via Gemini

## 🧩 System Components

### 1. Frontend (index.html + script.js)
- Dropdown-based configurator
- Progress-tracking interface
- Architecture Matrix UI
- Clickable rows to fetch full architecture descriptions
- Markdown → HTML converter for rendering Gemini responses

### 2. Backend (FastAPI – main.py)
- `/intake` → generates ranked architecture predictions
- `/gemini-conversation` → creates detailed architecture documentation
- Handles preprocessing, encoding, inference, and prompt engineering

### 3. Machine Learning Model
- PyTorch Masked Autoencoder
- Predicts values for cloud architecture columns
- Produces top-10 architecture combinations
- Uses probabilistic ranking

## 📂 Project Structure

```
├── index.html
├── script.js
├── styles.css
├── main.py
├── dgs_evaluation.py
├── encoders.pkl
├── architecture_predictor.pt
├── meta.json
├── requirements.txt
```

## ⚙️ Installation & Setup

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Create `.env`
```
GEMINI_API_KEY=your_key_here
```

### 3. Run FastAPI server
```
uvicorn main:app --reload --port 8000
```

### 4. Run frontend
Open `index.html` in browser.

## 🚀 How the System Works

1. User selects fields
2. Frontend calls `/intake`
3. Backend loads model + encoders
4. Model predicts top solutions
5. Matrix table displayed
6. User selects row → `/gemini-conversation`
7. Gemini returns architecture description
8. UI renders it

## 🛠 API Endpoints

### POST /intake
Input JSON → returns matrix + predictions

### POST /gemini-conversation
Input → selected architecture row  
Output → detailed architecture explanation

## 🧪 Offline Testing
```
python dgs_evaluation.py
```

## 💡 Tech Stack
- FastAPI
- PyTorch
- Google Gemini
- HTML/CSS/JS

## 📘 License
Internal project – organization governed.
