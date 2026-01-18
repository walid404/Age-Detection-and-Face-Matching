from flask import Blueprint, render_template, request, current_app
import requests
import base64


def image_to_base64(file):
    file.stream.seek(0)
    return base64.b64encode(file.stream.read()).decode("utf-8")


face_age_bp = Blueprint("face_age", __name__)


@face_age_bp.route("/", methods=["GET", "POST"])
def face_age_page():

    result = None

    if request.method == "POST":
        img1 = request.files.get("image1")
        img2 = request.files.get("image2")

        if img1 and img2:
            files = {
                "image_file_1": img1,
                "image_file_2": img2,
            }

            response = requests.post(current_app.config['API_URL'], files=files)
            response.raise_for_status()

            result = response.json()
            result["img1_b64"] = image_to_base64(img1)
            result["img2_b64"] = image_to_base64(img2)

    return render_template("face_age.html", result=result)
