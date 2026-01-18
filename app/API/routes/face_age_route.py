from fastapi import APIRouter, UploadFile, File, Depends
from app.API.core.model_loader import get_age_model, get_face_matching_model
from src.controller.age_face_inference_controller import AgeFaceMatchingInference
from PIL import Image

router = APIRouter(prefix="/v1", tags=["Face Age and Matching"])

@router.post("/infer_age_and_match")
async def infer_age_and_match(
    image_file_1: UploadFile = File(...),
    image_file_2: UploadFile = File(...),
    age_model=Depends(lambda: get_age_model("mobilenet", "src/saved_models/mobilenet_identity_bs16_lr0.0005_ep60.pt")),
    face_models=Depends(lambda: get_face_matching_model(0.25)),
):
    """
    Endpoint to infer ages and match two face images.

    Parameters
    ----------
    image_file_1 : UploadFile
        First image file.
    image_file_2 : UploadFile
        Second image file.
    age_model : torch.nn.Module
        Preloaded age prediction model.
    face_models : tuple
        Preloaded (ArcFaceExtractor, FaceMatcher).

    Returns
    -------
    dict
        Inference results including ages and match info.
    """
    arcface_extractor, face_matcher = face_models
    inference_controller = AgeFaceMatchingInference(
        age_model=age_model,
        extractor=arcface_extractor,
        matcher=face_matcher,
    )

    img1 = Image.open(image_file_1.file).convert("RGB")
    img2 = Image.open(image_file_2.file).convert("RGB")

    results = inference_controller.infer(img1, img2)

    return results