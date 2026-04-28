from flask import Flask
from app.config import DevelopmentConfig

def create_app(config_class=DevelopmentConfig):
    app = Flask(__name__)
    app.config.from_object(config_class)

    from app.main import main_bp
    app.register_blueprint(main_bp)
    
    from app.main import shape_bp
    app.register_blueprint(shape_bp)

    return app