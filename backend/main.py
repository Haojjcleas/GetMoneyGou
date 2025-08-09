from config import config
from gmg import app

if __name__ == '__main__':
    app.run(
        host=config.HOST,
        debug=config.DEBUG,
        port=config.PORT,
    )