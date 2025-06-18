import os
import logging
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.services.search_service import find_similar_images_from_url, find_similar_images_from_path

# ----------------------------------------
# App Setup
# ----------------------------------------
app = FastAPI(title="Fashion Visual Search API")
logger = logging.getLogger("uvicorn.error")

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ----------------------------------------
# Health & UI Routes
# ----------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    return {"message": "Server is up and running!"}

@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
def render_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ----------------------------------------
# API Routes for Frontend JS
# ----------------------------------------

@app.post("/api/search", tags=["API"])
async def api_search_url(payload: dict):
    image_url = payload.get("image_url")
    if not image_url:
        return JSONResponse(status_code=400, content={"detail": "Image URL is required."})

    try:
        results = find_similar_images_from_url(image_url)
        return {"similar_images": results or []}
    except Exception as e:
        logger.exception(f"API search error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Server error occurred."})

@app.post("/api/upload-file", tags=["API"])
async def api_upload_image(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        results = find_similar_images_from_path(temp_path)
        return {"similar_images": results or []}
    except Exception as e:
        logger.exception(f"API upload error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Server error during upload."})



















