import re


class Camera:
    def __init__(self):
        self.objs = {}

    def __setitem__(self, key, value):
        if key in self.objs.keys() and type(value) is bool:
            self.objs[key] = value

    def __getitem__(self, item):
        return self.objs[item]

    def make_feature(self):
        return [1.0 if obj else 0.0 for obj in self.objs.values()]

    def item_list(self):
        return [key for key in self.objs.keys()]

    def item_group_list(self, category):
        return [obj_name for obj_name in self.item_list() if
                re.compile(f'({category})+$').search(obj_name)]


class Camera120(Camera):
    def __init__(self):
        super().__init__()
        self.objs = {
            'hand': False,
            'grab_bottle': False,
            'face': False,
            'mouth': False,
            'open_mouth': False,
            'bottle': False,
            'open_bottle': False,
            'cap': False,
            'pill': False,
            'cup': False
        }


class Camera220(Camera):
    def __init__(self):
        super().__init__()
        self.objs = {
            'left_hand': False,
            'right_hand': False,
            'open_bottle': False,
            'closed_bottle': False,
            'green_pill': False,
            'yellow_pill': False,
            'orange_pill': False,
            'pink_pill': False,
            'cup': False,
            'face': False,
            'mouth': False,
            '1_pill': False,
        }


# if __name__ == '__main__':
#     cam = Camera120()
#     cam['face'] = False
#     cam['mouth'] = True
#     print(cam.item_list()[0])
