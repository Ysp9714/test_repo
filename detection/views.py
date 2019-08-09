import static.utils.prediction as prediction
import static.ml_models.action_recognition.ihp_action_recognition_cnn_model as model
from django.shortcuts import render,HttpResponse
from static.models.research.object_detection.object_detection_multi_img import object_detection
import json

# Create your views here.
def ActionResult(request, camera_type):
    detection_result = object_detection(camera_type)

    feature_list = prediction.get_prediction(detection_result, camera_type)
    result, hypothesis = model.get_prediction(feature_list, camera_type)

    detection_result = dict(detection_result)

    count_pill = list()
    for data in detection_result.values():
        count_pill.append(prediction.count_hand_in_pill(data))

    hypothesis = round(100 * hypothesis, 3)
    return render(request, 'detection/DetectionResult.html',
                  {'object_list': detection_result,
                   'count_pills': count_pill,
                   'result': result,
                   'hypothesis': hypothesis})


def DetectionResult(request, camera_type):
    detection_result = object_detection(camera_type)

    detection_result_json = json.dumps(detection_result, ensure_ascii=False, indent=4)
    print(type(detection_result_json))
    return HttpResponse(detection_result_json, content_type='application/json')

