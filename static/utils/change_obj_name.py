import os
import xml.etree.ElementTree as ET

PATH_TO_TEST_IMAGES_DIR = 'G:/공유 드라이브/InHandPlus/03. AI 개발/01. 복약 DB/train_image/120degree/completed/0717/train'
file_list = os.listdir(PATH_TO_TEST_IMAGES_DIR)
file_list = [file for file in file_list if file.endswith(('.xml'))]
for file in file_list:
    xml_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(root.text)
    for path in root.getiterator():
        print(path.text)
        # test to text
        path.text = path.text.replace('left_hand', 'hand')
        path.text = path.text.replace('right_hand', 'hand')
        path.text = path.text.replace('grab_bottle', 'grab_bottle_hand')
    tree.write(xml_path)
print("END")
