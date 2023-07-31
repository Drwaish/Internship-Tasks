"""Load and Predict YOLOv8s for prediction"""
from typing import Tuple
from ultralytics import YOLO


class Singleton():
    """
    Load model once while predicitions
    
    Attributes
    ----------
    model
        store model and utilize again and again
    
    """
    model = None
    def __init__(self):
        """
        __init__ to load model

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        self.model = YOLO("best.pt")


def prediction(image) -> Tuple:
    """
    Prediction of object in image.

    Parameters
    ----------
    image
        Image for prediction of object. 
    
    Return
    ------
    Tuple
    """
    load_model = Singleton()
    predict = load_model.model.predict(image)
    #print(predict)
    # Process predict list
    for result in predict:
        boxes = result.boxes  # Boxes object for bbox outputs
        names = result.names
    return boxes,names
def draw_boundarybox(img) -> Tuple:
    """
    Draw Boundry box on images

    Parameters
    ----------
    Image for detect object and print boundry box.

    Parameters
    ----------
    img
        Image use for detection.
    
    Return
    ------
    Tuple
    
    """
    boxes,names = prediction(img)
    object_name = []
    cordinates = []
    configure = []
    for box in boxes:
        class_id = names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        object_name.append(class_id)
        cordinates.append(cords)
        configure.append(conf)
    return object_name, cordinates, configure
