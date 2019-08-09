from static.utils.prediction import get_prediction as prediction


def save_csv(feature_flag, file):
    for i in range(len(feature_flag) - 1):
        print(feature_flag[i], file=file, end=',')
    else:
        print(feature_flag[-1], file=file)


def train_csv(detection_result, degree, key):
    with open(f"medi{degree}.csv", "a") as f:
        feature_list = prediction(detection_result, degree)
        feature_list.append(key)
        save_csv(feature_list, f)
    return feature_list
