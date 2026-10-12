SETUP
1. Unzip your Kaggle download (indicbert2_trained.zip) so this folder has:
     app.py
     requirements.txt
     static/index.html
     indicbert2_trained/   (model_state.pt, info.json, tokenizer files)
2. pip install -r requirements.txt
3. uvicorn app:app --reload
4. Open http://127.0.0.1:8000   (API docs at /docs)

PREVIEW WITHOUT THE MODEL (fake scores):
   Windows PowerShell:  $env:DEMO_MODE="1"; uvicorn app:app --reload
   Mac/Linux:           DEMO_MODE=1 uvicorn app:app --reload
