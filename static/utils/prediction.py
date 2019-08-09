import static.object_list.camera as camera
import math


def check_rect(rect1, rect2):
    if rect1[0] <= rect2[0] <= rect1[2] or rect1[0] <= rect2[2] <= rect1[2]:
        if rect1[1] <= rect2[1] <= rect1[3] or rect1[1] <= rect2[3] <= rect1[3]:
            return True
    else:
        return False


def check_open_bottle_hands(obj_dict):
    if ((obj_dict.keys() >= {1, 2, 3}) and
        (obj_dict[1]["bbox"][0] < obj_dict[3]["bbox"][0] < obj_dict[2]["bbox"][0])) or (
            (obj_dict.keys() >= {1, 2, 4}) and (
            obj_dict[1]["bbox"][0] < obj_dict[4]["bbox"][0] < obj_dict[2]["bbox"][0])):
        return True
    else:
        return False


def mid_point(rect):
    xmid = rect[0] + rect[2] / 2
    ymid = rect[1] + rect[3] / 2
    return [xmid, ymid]


def check_distance(rect1, rect2):
    distance = ((mid_point(rect1)[0] - mid_point(rect2)[0]) ** 2) + \
               ((mid_point(rect1)[1] - mid_point(rect2)[1]) ** 2)

    return math.sqrt(distance)


def obj_size(obj):
    # ymin, xmin, ymax, xmax
    size = (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1])
    return size


def check_obj(all_group_data, camera_type=220):
    if camera_type == 220:
        cam = camera.Camera220()
    elif camera_type == 120:
        cam = camera.Camera120()
    else:
        cam = camera.Camera()

    for value in all_group_data.values():
        for obj in value.values():
            cam[obj['name']] = True

    return cam.make_feature()


def max_face_size(all_group_data):
    max_size = 0
    for image in all_group_data.values():
        for obj in image.values():
            if obj["name"] == 'face':
                face_size = obj_size(obj) / (obj["size"][0] * obj["size"][1])
                if face_size > max_size:
                    max_size = face_size

    return round(max_size, 5)


def min_distance_hand_face(all_group_data, camera_type=220):
    min_distance = 1
    if camera_type == 220:
        face = ['face']
        hand = ['left_hand', 'right_hand']
    else:
        face = ['face']
        hand = ['hand']

    for image in all_group_data.values():
        face_list = {index: obj for index, obj in image.items() if obj["name"] in face}
        hand_list = {index: obj for index, obj in image.items() if obj["name"] in hand}
        for face_value in face_list.values():
            for hand_value in hand_list.values():
                distance = check_distance(face_value["bbox"], hand_value["bbox"]) \
                           / math.sqrt(face_value["size"][0] ** 2 + face_value["size"][1] ** 2)
                if min_distance > distance:
                    min_distance = distance
    # 가까울 수록 높은 값을 갖도록 변경
    return round(1 - min_distance, 5)


def count_hand_in_pill(group_data, camera_type=220):
    if camera_type == 220:
        cam = camera.Camera220()
    elif camera_type == 120:
        cam = camera.Camera120()
    else:
        return 0

    hand = cam.item_group_list('hand')
    pill = cam.item_group_list('pill')

    pills = set()
    del_pills = set()

    hand_group = {k: v for k, v in group_data.items() if v['name'] in hand}
    pill_group = {k: v for k, v in group_data.items() if v['name'] in pill}

    for hand_k, hand_v in hand_group.items():
        for pill_k, pill_v in pill_group.items():
            if check_rect(hand_v["bbox"], pill_v["bbox"]):
                pills.add(pill_k)

    for k, v in group_data.items():
        if v['name'] in pill:
            if k not in pills:
                del_pills.add(k)
    for k in del_pills:
        del group_data[k]

    # return count_pill_not_in_bottle(group_data)
    return len(pills)


def count_pill_not_in_bottle(group_data, camera_type=220):
    if camera_type == 220:
        cam = camera.Camera220()
    elif camera_type == 120:
        cam = camera.Camera120()
    else:
        return 0

    pill = cam.item_group_list('pill')
    bottle = cam.item_group_list('open_bottle')

    bottle_group = {k: v for k, v in group_data.items() if v['name'] in bottle}
    pill_group = {k: v for k, v in group_data.items() if v['name'] in pill}

    pills = set()

    if not bottle_group:
        return len(pill_group)
    for bottle_k, bottle_v in bottle_group.items():
        for pill_k, pill_v in pill_group.items():
            if not check_rect(bottle_v["bbox"], pill_v["bbox"]):
                pills.add(pill_k)
    return len(pills)


def get_prediction(all_group_data, camera_type=220):
    '''
    행동을 찍은 이미지를 분석한 결과와 카메라 타입을 확인하고
    행동을 분석하기 위한 데이터로 변경한다.
    :param all_group_data: 행동단위 이미지 분석을 통해서 검출된 오브젝트
    행동단위{
        이미지단위(image_name){
            오브젝트 단위(검출순서){
                오브젝트 특징{
                    obj: obj_num
                    name : name
                    score : percent
                    bbox : bounding box
                    size : image size
                }
            }
        }
    }
    :param camera_type: 촬영을 한 카메라 렌즈의 종류
    120degree, 220degree
    :return: 행동의 특징이 담긴 특징값(csv type)
    '''
    feature = list()
    if camera_type == 220:
        feature = check_obj(all_group_data, 220)
        feature.append(max_face_size(all_group_data))
        feature.append(min_distance_hand_face(all_group_data, 220))

    elif camera_type == 120:
        feature = check_obj(all_group_data, 120)

        # 임시방편임 피쳐 수가 맞지 않아서 넣어 둠(7_24_1717)
        feature.append(0)
        feature.append(0)

        feature.append(max_face_size(all_group_data))
        feature.append(min_distance_hand_face(all_group_data, 120))
    else:
        print("camera_degree_error")
    print(feature)
    return feature
