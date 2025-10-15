import os
import xml.etree.ElementTree as ET

folder = "./train"

for xml_file in os.listdir(folder):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(folder, xml_file)
        jpg_name = os.path.splitext(xml_file)[0] + ".jpg"  

        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_tag = root.find("filename")
        if filename_tag is not None:
            filename_tag.text = jpg_name

        path_tag = root.find("path")
        if path_tag is not None:
            path_tag.text = jpg_name

        tree.write(xml_path)

print("successful!.")
