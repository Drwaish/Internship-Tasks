from flask import Flask
from flask_smorest import Api
from resources.item import blp as ItemBluePrint

app=Flask(__name__)

app.config["PROAGATE_EXCEPTION"] = True
app.config["API_TITLE"] = "StorES OF rEST"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.20.0"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdeliver.net/npm/swagger-ui-dist"


api=Api(app)
api.register_blueprint(ItemBluePrint)

