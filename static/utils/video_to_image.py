import cv2


def video_to_image(video):
    vidcap = cv2.VideoCapture(video)
    count = 0
    imgs = list()
    frame_division = 10
    while vidcap.grab():
        ret, image = vidcap.read()

        if int(vidcap.get(1)) % frame_division == 0:
            img_num = str(count).zfill(4)
            img_name = f"{video[-27:-19]}{img_num}_{video[-19:-4]}.jpg"
            print(img_name)
            cv2.imwrite("video_frame/"+img_name, image)  # save frame as JPEG file
            imgs.append("video_frame/"+img_name)
            count += 1
    return imgs