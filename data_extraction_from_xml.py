import pandas as pd
import numpy as np
import os
from glob import glob 
from functools import reduce
from xml.etree import ElementTree as et 
from shutil import move 


def extract_xml_files(file_path: str):
    """
    Takes a directory path and returns all .xml files inside.
    """
    xml_files = glob(os.path.join(file_path, "*.xml"))
    return xml_files

def extract_info_from_xml(filename: str):
    """Extract necessary things such as 
     --file_name 2. size (width,height) , object(name,xmin,xmax,ymin,ymax)
    args :
        xml_files : list
    return 
        ["filename",width,height,name,xmin,xmax,ymin,ymax]
    """
    try:
        tree = et.parse(filename)
        root = tree.getroot()
        image_name = root.find("filename").text
        width = root.find("size").find("width").text
        height = root.find("size").find("height").text
        objs = root.findall("object")
        parser = []
        for obj in objs:
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmax = bndbox.find("xmax").text
            xmin = bndbox.find("xmin").text
            ymax = bndbox.find("ymax").text
            ymin = bndbox.find("ymin").text
            parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])
        return parser
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return []

def create_dataFrame(data, column_name: list = None):
    """
    creates a dataframe and returns it 
    """
    df = pd.DataFrame(data, columns=column_name)
    return df 

def save_dataframe(df):
    df.to_csv("index_false.csv", index=False)
    print(f"Saved dataframe with {len(df)} rows to index_false.csv")

def type_conv(df):
    colms = ["width", "height", "xmin", "xmax", "ymin", "ymax"]
    df[colms] = df[colms].astype(int)
    return df

def relative_center_calc(df):
    df["center_x"] = ((df["xmin"] + df["xmax"]) / 2) / df["width"]
    df["center_y"] = ((df["ymin"] + df["ymax"]) / 2) / df["height"]
    df["w"] = (df["xmax"] - df["xmin"]) / df["width"]
    df["h"] = (df["ymax"] - df["ymin"]) / df["height"]
    return df 

def train_test_split(df): 
    images = df["filename"].unique()
    img_df = pd.DataFrame(images, columns=["filename"])
    img_train_df = img_df.sample(frac=0.8, random_state=42) 
    img_train = img_train_df["filename"].tolist()  
    img_test = img_df[~img_df["filename"].isin(img_train)]["filename"].tolist()  
    train_df = df[df["filename"].isin(img_train)].copy() 
    test_df = df[df["filename"].isin(img_test)].copy()  
    return train_df, test_df

def label_encoding(x):  
    labels = {"ROI": 0}
    return labels[x]

def create_folders(folder_name: str):
    """file_name : str """
    train_folder = os.path.join(folder_name, "train")
    test_folder = os.path.join(folder_name, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    return train_folder, test_folder  

def make_annotation_groups_for_each_image(train_df, test_df):
    cols = ["filename", "id", "center_x", "center_y", "w", "h"]
    groupby_obj_train = train_df[cols].groupby("filename")
    groupby_obj_test = test_df[cols].groupby("filename")
    return groupby_obj_train, groupby_obj_test

def save_data(filename, folder_path, group_obj):
    """Save each image to train/test folder and respective labels in .txt"""
    src = os.path.join("train", filename)
    dst = os.path.join(folder_path, filename)
 
    if os.path.exists(src):
        move(src, dst)

    text_filename = os.path.splitext(filename)[0] + ".txt"
    text_path = os.path.join(folder_path, text_filename)

    image_data = group_obj.get_group(filename)
    
    with open(text_path, "w") as f:
        for _, row in image_data.iterrows():
            f.write(f"{int(row['id'])} {row['center_x']:.6f} {row['center_y']:.6f} {row['w']:.6f} {row['h']:.6f}\n")


def main():
    file_path = "train"

    if not os.path.exists(file_path):
        print(f"Directory '{file_path}' not found!")
        return
    xml_files = extract_xml_files(file_path)
    if not xml_files:
        print(f"No XML files found in '{file_path}' directory!")
        return
    
    print(f"Found {len(xml_files)} XML files")
    
    parser_all = list(map(extract_info_from_xml, xml_files))
    parser_all = [p for p in parser_all if p]
    
    if not parser_all:
        print("No valid data extracted from XML files!")
        return

    data = reduce(lambda x, y: x + y, parser_all)
    
    if not data:
        print("No annotation data found in XML files!")
        return
    
    df = create_dataFrame(data, ["filename", "width", "height", "name", "xmin", "xmax", "ymin", "ymax"])

    df = type_conv(df)

    save_dataframe(df)
    new_df = relative_center_calc(df)
    if new_df is not None:
        try:
            train_df, test_df = train_test_split(new_df)
        except Exception as e:
            raise e 
    else:
        print(f"new df is empty")

    train_df["id"] = train_df["name"].apply(label_encoding)
    test_df["id"] = test_df["name"].apply(label_encoding)
    
    train_folder, test_folder = create_folders("data_images") 
    
    groupby_obj_train, groupby_obj_test = make_annotation_groups_for_each_image(train_df, test_df)
    


    for filename in groupby_obj_train.groups.keys():
        save_data(filename, train_folder, groupby_obj_train)

    for filename in groupby_obj_test.groups.keys():
        save_data(filename, test_folder, groupby_obj_test)
    
    print(f"Saved {len(groupby_obj_train.groups)} images to {train_folder}")
    print(f"Saved {len(groupby_obj_test.groups)} images to {test_folder}")


if __name__ == "__main__":
    main()