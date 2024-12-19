import cv2
import torch
from sort import Sort
from ultralytics import YOLO
import numpy as np
from time import time
import random


class FirstMilitaryTracker:
    def __init__(self, output):
        self.output = output
        self.model = self.load_model()
        self.names = self.model.names
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_model(self):
        model = YOLO("path_to_weight")
        model.fuse()
        return model

    def predict(self, model, frame):
        return model.predict(frame, verbose=True, conf=0.3)

    def get_results(self, results):
        summa = 0
        detection_list = []

        for result in results[0]:
            bbox = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()

            merger_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], conf[0], class_id[0]]
            detection_list.append(merger_detection)

        return np.array(detection_list), summa

    def draw_boxes(self, frame, boxes, ids, class_id):
        for box, idx, cls in zip(boxes, ids, class_id):
            name = self.names[cls]
            label = f"{idx}:{name}"

            cv2.rectangle(frame, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.output)
        assert cap.isOpened(), "Video capture failed."

        sort = Sort(max_age=80, min_hits=4, iou_threshold=0.30)

        while True:
            start = time()
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            results = self.predict(self.model, frame)
            final_results, summa = self.get_results(results)

            if len(final_results) == 0:
                final_results = np.empty((0, 5))

            t_res = sort.update(final_results)

            bboxes = t_res[:, :-1]
            ids = t_res[:, -1].astype(int)
            class_id = final_results[:, -1].astype(int)

            for box, idx, cls in zip(bboxes, ids, class_id):

                x1, y1, x2, y2 = map(int, box)

                image = frame[y1:y2, x1:x2]

                # path = None
                name1 = random.randint(1, 10000)
                name2 = random.randint(1, 10000)

                if cls == 0:
                    path = f"PATH_TO_FOLDER/photo_{name1}_{name2}.jpg"
                    cv2.imwrite(path, image)
                if cls == 1:
                    ...

            new_frame = self.draw_boxes(frame, bboxes, ids, class_id)

            end = time()
            fps = 1 / round(end - start, 1)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('1st version', new_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) & 0xFF == ord("d"):
                break

        cap.release()
        cv2.destroyAllWindows()


path = "PASTE_PATH_HERE"
tracker = FirstMilitaryTracker(path)
tracker()













