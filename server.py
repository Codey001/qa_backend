from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import fitz
import google.generativeai as genai
from pydantic import BaseModel
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


# READ ENVIRONMENT VARIABLES
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS").split(",")
ALLOWED_METHODS = os.getenv("ALLOWED_METHODS").split(",")
ALLOWED_HEADERS = os.getenv("ALLOWED_HEADERS").split(",")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)


class TestData(BaseModel):
    filename: str
    question: str


ALLOWED_EXTENSIONS = {".pdf"}
max_file_size: int = 10  # MB


# INITIALIZE GEMINI
genai.configure(api_key=API_KEY)
model_name = MODEL_NAME

file_contents = {}  # CONTENT OF PDF FILE


def call_genai_api(context: str, question: str) -> str:
    # GEMINI RESPONSE CALL
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(f"{context}\n\nQ: {question}\nA:")
    return response.text

@app.get('/')
async def home():
    return "API working"

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # CHECK FILE EXTENSION
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="File type not supported. Only PDF files are allowed.",
            )

        # CHECK FILE SIZE
        if max_file_size is not None and file.size > (max_file_size * 1024 * 1024):
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds the maximum allowed size of {max_file_size} MB",
            )

        # STORE THE FILES
        if not os.path.exists("./files"):
            os.makedirs("./files")

        unique_filename = str(uuid.uuid4())
        file_location = f"./files/{unique_filename}.pdf"

        print("FILENAME", file_location)
        with open(file_location, "wb") as file_object:
            file_object.write(file.file.read())

        # TEXT EXTRACTION
        doc = fitz.open(file_location)
        text = ""
        for page in doc:
            text += page.get_text()
            print(text)

        file_contents[file.filename] = text

        return {"status": 200, "filename": unique_filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/")
async def ask_question(data: TestData):
    try:
        filename, question = data.filename, data.question

        # FILE EXISTS
        if filename not in file_contents:
            raise HTTPException(
                status_code=404, detail="File not found. Please upload the file first."
            )

        # API CALLED
        contents = file_contents[filename]
        answer = call_genai_api(context=contents, question=question)
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
