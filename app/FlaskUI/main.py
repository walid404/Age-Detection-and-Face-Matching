from flask import Flask
from app.FlaskUI.BluePrint.face_age_bp import face_age_bp
from app.FlaskUI.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.register_blueprint(face_age_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
