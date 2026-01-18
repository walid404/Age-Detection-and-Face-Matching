from fastapi import FastAPI
from app.API.routes import face_age_route, base_route


app = FastAPI(title="Face Age and Matching API", version="1.0.0")

app.include_router(base_route.router)
app.include_router(face_age_route.router)
