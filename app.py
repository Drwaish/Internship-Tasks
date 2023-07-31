"""Api to predic object in image"""
from typing import Dict
import datetime
import os
from flask import jsonify, request
from apiflask import APIFlask, abort
from PIL import Image
from dotenv import load_dotenv
from predict_objects import predict


UPLOAD_FOLDER = "temp"
app = APIFlask(__name__, json_errors=False)


def check_valid_image(ext) -> bool:
    """
    Validate image type.

    Parameters
    ----------
    ext
        Extension of the image.

    Return
    ------
    bool
    """
    image_extension = ["png", "jpg", "jpeg"]
    if ext in image_extension:
        return True
    return False


def authenticate(request_auth) -> bool:
    """
    Authenticate the user using key.

    Parameters
    ----------
    request_auth
        request modeule to get data.

    Return
    ------
    bool

    """
    load_dotenv()
    key = os.getenv("key")
    headers = request_auth.headers
    auth = headers.get("X-Api-Key")
    if auth == key:
        return True
    else:
        return False


def get_image(request_api):
    """
    Extract Image from API.

    Parameters
    ----------
    request_api
        request to get data from api.

    Return
    ------
    image 
    """
    data = request_api.files["image_file"]
    if data:
        return data
    return False


def save_file(data) -> str:
    """
    Save file coming from api.

    Parameters
    ----------
    data
        Image file object received from api.
    Return
    str
    """
    unique_name = str(datetime.datetime.now().timestamp()).replace(".", "")
    file_name_split = data.filename.split(".")
    ext = file_name_split[len(file_name_split)-1]
    final_path = f"{UPLOAD_FOLDER}/{unique_name}.{ext}"
    data.save(final_path)
    return ext, final_path


@app.route("/prediction", methods=["POST"])
def predict_objects() -> Dict:
    """
    Prediction of object in image.

    Parameters
    ----------
    None

    Return
    ------
    Dict
    """
    try:
        if request.method == "POST":
            try:
                data = get_image(request)
                if data:
                    if authenticate(request_auth=request):  # authenticate
                        ext, final_path = save_file(data=data)
                        resp = {}
                        if check_valid_image(ext):
                            img_file = Image.open(final_path)
                            object_names, boxes, confidence = predict.draw_boundarybox(
                                img_file)
                            print(object_names)
                            for i, box in enumerate(boxes):
                                key = str(object_names[i]+str(i))
                                resp[key] = {"bbox": box,
                                            "confidence": confidence[i]}
                            return jsonify(resp)
                        else:
                            # if image is ot valid remove file from temp folder
                            os.remove(final_path)
                            abort(400, message="Inappropriate Parameters",
                                detail="Wrong file submited")
                    else:
                        abort(401, message="Unauthorized", detail="Invalid key or value")
                else:
                    abort(400, message="Bad Request", detail="Image or Key is incorrect")
            except KeyError:
                abort(401, message = "ImageError", detail="Not sending image properly.")
    except TypeError:
        abort(400, message= "Wrong Request", detail="Something wrong with request")


if __name__ == "__main__":
    app.run(port=3000, threaded=True)
