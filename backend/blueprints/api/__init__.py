from blueprints.api.hca import hca_bp
from blueprints.api.data_upload import data_upload_bp
from blueprints.api.data_preprocessing import data_preprocessing_bp
from blueprints.api.ml_training import ml_training_bp
from blueprints.api.ai_explanation import ai_explanation_bp
from blueprints.api.code_generation import code_generation_bp
from blueprints.api.system_status import system_status_bp
from blueprints.api.test_ai import test_ai_bp
from flask import Blueprint

api_bp = Blueprint('api', __name__, url_prefix='/api')

api_bp.register_blueprint(hca_bp)
api_bp.register_blueprint(data_upload_bp)
api_bp.register_blueprint(data_preprocessing_bp)
api_bp.register_blueprint(ml_training_bp)
api_bp.register_blueprint(ai_explanation_bp)
api_bp.register_blueprint(code_generation_bp)
api_bp.register_blueprint(system_status_bp)
api_bp.register_blueprint(test_ai_bp)

__all__ = ['api_bp']