from flask import Flask
from flask_cors import CORS
import dotenv
import os

# 加载环境变量
dotenv.load_dotenv()

# 创建Flask应用
app = Flask(__name__)

# 配置CORS
CORS(app)

# 加载配置
from config.config import Config
config = Config()

# 配置Flask应用
app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# 初始化数据库
from models import db
db.init_app(app)

# 创建数据库表
with app.app_context():
    db.create_all()

# 注册蓝图
from blueprints import all_bp
for bp in all_bp:
    app.register_blueprint(bp)
