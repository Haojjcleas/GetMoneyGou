from flask import Blueprint
from flask import jsonify

hca_bp = Blueprint('hca', __name__, url_prefix='/hca')
@hca_bp.route('/')
def hca_endpoint():
    return jsonify({"status": "ok"})

