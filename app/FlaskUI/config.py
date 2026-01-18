import os

base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    API_URL = os.environ.get("API_URL", "http://localhost:8000/v1/infer_age_and_match")