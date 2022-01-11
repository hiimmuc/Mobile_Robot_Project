import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from onnx_engine import *
from PIL import Image

from utils import *


class ONNX_engine:
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def run(self,
            weights='path/to/weights',  # model.pt path(s)
            source=0,  # file/dir/URL/glob, 0 for webcam
            imgsz=416,  # inference size (pixels)
            conf_thres=0.7,  # confidence threshold
            iou_thres=0.5,  # NMS IOU threshold
            max_det=1,  # maximum detections per image
            device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labelsq
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric()

        # # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        device = select_device(device)
        # half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        if onnx:
            if dnn:
                # opencv-python>=4.5.4
                # check_requirements(('opencv-python>=4.5.4',))
                net = cv2.dnn.readNetFromONNX(w)
            else:
                # onnxruntime for CPU
                # check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
                import onnxruntime
                session = onnxruntime.InferenceSession(w, None)
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            # view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=onnx)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=onnx)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        # if pt and device.type != 'cpu':
        #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, img, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            if onnx:
                img = img.astype('float32')
            # else:
            #     img = torch.from_numpy(img).to(device)
            #     img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # img = np.resize(img, (1, 3, 416, 416))

            # Inference
            # if pt:
            #     visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #     pred = model(img, augment=augment, visualize=visualize)[0]
            if onnx:
                if dnn:
                    net.setInput(img)
                    pred = torch.tensor(net.forward())
                else:
                    pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                x_c = 0
                y_c = 0

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            x_c = (xyxy[0].numpy() + xyxy[2].numpy())/2
                            y_c = (xyxy[1].numpy() + xyxy[3].numpy())/2
                            print("Bounding Box Center: ({}, {})".format(x_c, y_c))

                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({1/(t3 - t2):.3f}fps)')

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.circle(im0, (int(x_c), int(y_c)), 3, (0, 0, 255), cv2.FILLED)
                    key = cv2.waitKey(1)  # 1 millisecond
                    if key & 0xFF == 27:
                        view_img = False
                    else:
                        cv2.imshow(str(p), im0)

                    # Print results
                    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, imgsz, imgsz)}' % t)
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


class PT_engine:
    def __init__(self, model_path="path/to/weight.pt"):
        super().__init__()
        self.model = torch.hub.load("ultralytics/yolov5", 'custom', path=model_path)
        self.setup_params()

    def setup_params(self):
        self.imgsz = 416
        self.model.conf = 0.7  # NMS confidence threshold
        self.model.iou = 0.5  # NMS IoU threshold
        self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1  # maximum number of detections per image
        self.x_center = []
        self.y_center = []

    def detect(self, frame):
        # time_start = time.time()
        # self.setup_params()
        results = self.model(frame, size=640)
        results.print()
        results.save()
        # self.model.conf = 0.7  # NMS confidence threshold
        # self.model.iou = 0.75  # NMS IoU threshold
        # self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        # self.model.multi_label = False  # NMS multiple labels per box
        # self.model.max_det = 1  # maximum number of detections per image
        # print('\n', results.xyxy)

        # x_min = results.xyxy[0][0][0].numpy()
        # y_min = results.xyxy[0][0][1].numpy()
        # x_max = results.xyxy[0][0][2].numpy()
        # y_max = results.xyxy[0][0][3].numpy()
        # w = x_max - x_min
        # h = y_max - y_min
        # result = [x_min, y_min, x_max, y_max]
        # # Calculate mean + rms
        # xc = (results.xyxy[0][0][0].numpy() + results.xyxy[0][0][2].numpy())/2
        # yc = (results.xyxy[0][0][1].numpy() + results.xyxy[0][0][3].numpy())/2
        # x_center.append(xc)
        # y_center.append(yc)
        # xc_mean = np.mean(xc)
        # yc_mean = np.mean(yc)
        #
        # xc_var = np.float(np.sqrt((np.sum(np.square(xc - xc_mean))) / len(x_center)))
        # yc_var = np.float(np.sqrt((np.sum(np.square(yc - yc_mean))) / len(y_center)))
        #
        # print([xc_mean, yc_mean, xc, yc, xc_var, yc_var])
        #
        # file = open("results-yolov5n.txt", "a")
        # file.writelines([str(x_min) + " ", str(y_min) + " ", str(w) + " ", str(h) + " ", "\n"])
        # fps = 1/(time.time()-time_start)
        # print(fps)


if __name__ == '__main__':
    """RUN ONNX ENGINE"""
    weights_path = r"best.onnx"
    webcam_id = 0
    model = ONNX_engine()
    model.run(weights=weights_path, source=webcam_id)

    # """RUN PT ENGINE"""
    # weight_path = r"best.pt"
    # cap = cv2.VideoCapture(webcam_id)
    # t0 = time.time()
    # solar_car_detector = PT_engine(model_path=weight_path)
    # print("Load model time:", time.time() - t0)
    # while True:
    #     success, frame = cap.read()
    #     t1 = time.time()
    #     solar_car_detector.detect(frame)
    #     print("Detect FPS:", 1/(time.time() - t1))
    #     cv2.imshow('Frame', frame)
    #     key = cv2.waitKey(1)
    #     if key & 0xFF == 27:
    #         break
