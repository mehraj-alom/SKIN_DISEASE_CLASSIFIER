import os 
from box.exceptions import BoxValueError
import yaml
from src.classifier import logger
import json
import joblib
from ensure import ensure_annotations
# from box import ConfigBox
from box import Box
from pathlib import Path 
from typing import Any ,List
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> Box:
    """reads yaml file and returns 
    Args:
        path_to_yaml (str) : path like input

    raises: 
        ValueError: if the file does not exist or is not a yaml file
    returns:
        Box : Box type
    """
    try: 
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully.")
            return Box(content)
    except BoxValueError :
        raise ValueError(f"File {path_to_yaml} does not exist or is not a valid YAML file.")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):

    """Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the creation of directories.
    """
    for path in path_to_directories:
        path = Path(path)
        os.makedirs(path, exist_ok = True)
        if verbose:
            logger.info(f"Directory craeted at {path}")
    


@ensure_annotations
def save_json(path :Path, data:dict):
    """save json data
    
    args:
        path(Path) : Path at which json will be saved 
        data(dict) : data to be saved in that json file    
    """
    with open(path,"w") as f:
        json.dump(data,f,indent=5)
    logger.info(f"Json file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> Box:
    """load json file
    Args:
        path (Path): Path from which json file to be load 
    returns:
        Box : Box type    
    """
    with open(path,"r") as f:
        content = json.load(f)
    logger.info(f"Json file loaded from {path}")
    return Box(content)

@ensure_annotations
def save_binary(path: Path, data:Any):
    """
    save data in binary format using joblib 
    Args:
        path (Path): Path where binary file will be saved
        data (Any) : Data to be saved in binary format
    """
    with open(path, "wb") as f:
        joblib.dump(data,f)
    logger.info(f"Binary file saved at {path}")

@ensure_annotations
def load_binary(path :Path) -> Any:
    """
    load data from binary file using joblib 
    Args:
        path(Path): Path from which binary file will be loaded                   
    returns;
        Any : Data loaded from the binary file
    """
    with open(path,"rb") as f:
        content = joblib.load(f)
    logger.info(f"Binary file loaded from {path}")
    return content

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get file size kB
    Args :
         path (Path) : Path of the file to get size
    Returns:
        int : Size of the file in kB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} kB"   

@ensure_annotations
def decode_image(imgstring : str, filename : str) -> None:
    """
    Decode a base64 encoded image and save it to a file.
    
    Args:
        imgstring (str): Base64 encoded image string.
        filename (str): Name of the file to save the decoded image.
    """
    imgdata =  base64.b64decode(imgstring)
    with open(filename, "wb") as f:
        f.write(imgdata)
        f.close()
    logger.info(f"Image decoded and saved as {filename}")

@ensure_annotations
def encode_image(cropped_image_path: Path) -> str:
    """
    Encode an image to a base64 string.
    
    Args:
        cropped_image_path (Path): Path to the image file to be encoded.
        
    Returns:
        str: Base64 encoded string of the image.
    """
    with open(cropped_image_path, "rb") as f:
        imgdata = f.read()
    encoded_string = base64.b64encode(imgdata).decode('utf-8')
    logger.info(f"Image at {cropped_image_path} encoded to base64 string.")
    return encoded_string