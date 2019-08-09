import sys
import numpy as np
import os
import tensorflow as tf
import json
import re
import static.object_list.camera as camera

from PIL import Image, ImageDraw, ImageFont
from collections import OrderedDict

from utils import ops as utils_ops
from utils import label_map_util


def check_rect(rect1, rect2):
    if rect1[0] <= rect2[0] <= rect1[2] or rect1[0] <= rect2[2] <= rect1[2]:
        if rect1[1] <= rect2[1] <= rect1[3] or rect1[1] <= rect2[3] <= rect1[3]:
            return True
    else:
        return False


def make_image_path_list(img_dir):
    PATH_TO_TEST_IMAGES_DIR = img_dir
    file_list = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    file_list = [file for file in file_list if file.endswith(('.jpeg', '.png', '.jpg'))]
    file_list = sorted(file_list, key=lambda x: re.sub('[.a-zA-Z]', '', x), reverse=False)
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, file) for file in file_list]

    return TEST_IMAGE_PATHS


def class_int_to_text(class_num, camera_degree):
    class_list_num = class_num-1
    camera120 = camera.Camera120()
    camera220 = camera.Camera220()

    if camera_degree == 120:
        return camera120.item_list()[class_list_num]
    elif camera_degree == 220:
        return camera220.item_list()[class_list_num]
    else:
        return None


def draw_bbox(draw, obj):
    # 0: ymin, 1: xmin, 2: ymax, 3: xmax
    if obj['size'][0] < 1000:
        width = 3
    else:
        width = 9
    color = get_color(obj["obj"])
    draw.line((obj["bbox"][1], obj["bbox"][2], obj["bbox"][3], obj["bbox"][2]), fill=color, width=width)
    draw.line((obj["bbox"][1], obj["bbox"][0], obj["bbox"][3], obj["bbox"][0]), fill=color, width=width)
    draw.line((obj["bbox"][1], obj["bbox"][0], obj["bbox"][1], obj["bbox"][2]), fill=color, width=width)
    draw.line((obj["bbox"][3], obj["bbox"][0], obj["bbox"][3], obj["bbox"][2]), fill=color, width=width)


def draw_class_name(_draw, obj, num=-1):
    # ymin, xmin, ymax, xmax
    if obj['size'][0] < 1000:
        font_size = 7
    else:
        font_size = 50
    font = ImageFont.truetype("static/font/HoonFuture_Bold.otf", font_size)
    if num >= 0:
        _draw.text((obj['bbox'][1], obj['bbox'][0]), f"{num + 1}", font=font, fill=(255, 255, 255, 128))
    else:
        _draw.text((obj['bbox'][1], obj['bbox'][0]), f"{obj['name']}|{100*obj['score']:.2f}%", font=font,
                   fill=(0, 0, 0, 0))


def get_color(obj_num):
    color_list = [0x30A9DE, 0xEFDC05, 0xE53A40, 0xC5E99B, 0x56445D,
                  0xDE6449, 0x41D3BD, 0xF16B6F, 0x7200DA, 0x272625,
                  0xEE6E9F, 0xFFBF00]
    return color_list[obj_num]


def real_size(detected_obj):
    # ymin, xmin, ymax, xmax
    detected_obj["bbox"][0] *= detected_obj["size"][1]
    detected_obj["bbox"][1] *= detected_obj["size"][0]
    detected_obj["bbox"][2] *= detected_obj["size"][1]
    detected_obj["bbox"][3] *= detected_obj["size"][0]
    return detected_obj


def inter_area(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea


def get_area_value(boxA):
    return (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def nms(group_data):
    del_list = set()
    threshold = 0.6
    for obj_num1 in range(len(group_data)):
        for obj_num2 in range(obj_num1 + 1, len(group_data)):
            if bb_intersection_over_union(group_data[obj_num1]["bbox"], group_data[obj_num2]["bbox"]) > threshold:
                if group_data[obj_num1]["score"] < group_data[obj_num2]["score"]:
                    if obj_num1 == 1 or obj_num1 == 2:
                        del_list.add(obj_num1)
                else:
                    if obj_num1 == 1 or obj_num1 == 2:
                        del_list.add(obj_num2)
    else:
        for i in del_list:
            del group_data[i]
    return group_data


def object_to_json_style(cls, score, bbox, image_path, camera_degree):
    obj = dict()
    obj["obj"] = int(cls)
    obj["name"] = class_int_to_text(cls, camera_degree)
    # .jpg(4words)
    obj["score"] = float("{:.4}".format(score))
    obj["bbox"] = bbox
    obj["size"] = Image.open(image_path, "r").size
    return obj


def run_inference_for_single_image(TEST_IMAGE_PATHS, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_list = list()
            image_np_list = list()
            for TEST_IMAGE in TEST_IMAGE_PATHS:
                image = Image.open(TEST_IMAGE)
                # image_np = load_image_into_numpy_array(image)
                image_np = np.asarray(image)
                image_np_list.append(image_np)

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and
                    # fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image_np, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                output_dict_list.append(output_dict)
    return output_dict_list, image_np_list


def object_detection(camera_degree):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_FROZEN_GRAPH = f"static/models/research/object_detection/inhandplus_model_{camera_degree}/" + 'frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join("static/models/research/object_detection/training/",
                                  f'labelmap_{camera_degree}.pbtxt')
    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


    score_threshold = 0.9
    img_dir = 'static/images/test_images'
    image_name_start_num = len(img_dir) + 1

    TEST_IMAGE_PATHS = make_image_path_list(img_dir)


    f = open("object_list.json", 'w')
    all_group_data = OrderedDict()
    output_dicts, image_nps = run_inference_for_single_image(TEST_IMAGE_PATHS, detection_graph)
    result_img_list = list()

    for output_dict, image_np, image_path in zip(output_dicts, image_nps, TEST_IMAGE_PATHS):
        group_data = dict()
        for score, cls, box, num in zip(output_dict['detection_scores'], output_dict['detection_classes'],
                                        output_dict['detection_boxes'], range(len(output_dict['detection_scores']))):
            if score > score_threshold:
                bbox = list()
                for point in box:
                    bbox.append(float("{:.4}".format(point)))
                group_data[num] = object_to_json_style(cls, score, bbox, image_path, camera_degree)

        # group_data = nms(group_data)
        # group_data = delete_pill_in_bottle(group_data)
        # group_data['count_pills']: count_hand_in_pill(group_data)

        im = Image.open(image_path)
        draw = ImageDraw.Draw(im)
        for num, item in group_data.items():
            img_obj = real_size(item)
            draw_bbox(draw, img_obj)
            draw_class_name(draw, img_obj)

        im.save('static/result/' + image_path[image_name_start_num:])
        result_img_list.append(image_path[image_name_start_num:])
        all_group_data[image_path[image_name_start_num:]] = group_data
    json_data = json.dumps(all_group_data, ensure_ascii=False, indent=4)
    f.write(json_data)

    f.close()
    return all_group_data


if __name__ == '__main__':
    object_detection(220)
