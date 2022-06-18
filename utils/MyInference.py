import torch
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
import os
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from MyMLP import MLP, MyNomalizer, args
from copy import deepcopy
import detectron2.data.transforms as T

label_name = ["standing", "sitting", "lying", "bending", "crawling"]


class MyInference:
    def __init__(self) -> None:
        #cfg 파일 읽어오기와 모델을 불러온다
        cfg = model_zoo.get_config(
            "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml", trained=True)
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        #프로젝트 최상위 디렉토리로 이동한다
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        #checkpointe로 모델에 미리 학습된 가중치를 불러온다
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        #나중에  numpy 데이터를 변환하기 위한 작업
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        #이 모델이 받아야 하는 데이터 형식을 저장 RGB or BGR
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        #자세 추론 모델을 불러온다##
        self.model_MLP = MLP(deepcopy(args))
        self.model_MLP.to(args.device)
        self.model_MLP.load_state_dict(
            torch.load('models/model_0.001_1_750.pt', map_location=args.device))
        self.model_MLP.eval()

        ##데이터를 저장할 딕셔너리##
        self.loaded_video_image = {}
        self.loaded_images = {}

    #video를 프레임 단위로 쪼개서 dict에 저장한다.
    def load_video(self, target_path: str) -> None:
        vidcap = cv2.VideoCapture(target_path)
        #count=0;
        video_name = target_path.split('/')[-1]
        self.loaded_video_image[video_name] = []
        while True:
            success, image = vidcap.read()
            if not success:
                break
            self.loaded_video_image[video_name].append(image)

    ##주어진 dir에 있는 이미지를 모두 딕셔너리에 넣는다##
    def load_images(self, target_dir: str) -> None:
        dir_path = os.getcwd()+"/"+target_dir
        self.loaded_images[target_dir] = []
        dirs = os.listdir(target_dir)
        dirs.sort()
        for image_path in dirs:
            for image_format in [".png", ".jpg", ".jpeg"]:
                if(image_format in image_path):
                    image = cv2.imread(f'{target_dir}/{image_path}')
                    self.loaded_images[target_dir].append((image_path, image))
                else:
                    continue

    ##딕셔너리에 있는 비디오를 추론하고 비디오로 저장한다##
    def inference_video(self, batch: int):
        for video_name in self.loaded_video_image.keys():
            i = 0
            inference_data = []
            while True:
                inputs = []
                for cnt in range(batch):
                    if(i >= len(self.loaded_video_image[video_name])):
                        break
                    original_image = self.loaded_video_image[video_name][i]
                    i += 1
                    height, width = original_image.shape[:2]
                    image = self.aug.get_transform(
                        original_image).apply_image(original_image)
                    image = torch.as_tensor(
                        image.astype("float32").transpose(2, 0, 1))
                    input_dict = {"image": image,
                                  "height": height, "width": width}
                    inputs.append(input_dict)
                predictions = self(inputs)
                for inferenced_data in predictions:
                    ##스코어 0.6 이상의 객체들만 남긴다##
                    inferenced_data["instances"] = inferenced_data["instances"][inferenced_data["instances"].scores > 0.6]
                    inference_data.append(inferenced_data)

                if(i >= len(self.loaded_video_image[video_name])):
                    break
            height, width = self.loaded_video_image[video_name][0].shape[:2]
            ##폴더가 없으면 만든다##
            try:
                if not os.path.exists(f'result_video'):
                    os.makedirs(f'result_video/')
            except OSError:
                pass
            cap = cv2.VideoWriter(
                f'result_video/{video_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

            for frame in range(len(inference_data)):
                visualizer = Visualizer(
                    self.loaded_video_image[video_name][frame][:, :, ::-1], self.metadata, scale=1.0)
                input = self.__get_for_mlp_keypoints__(
                    inference_data[frame]["instances"], args)
                with torch.no_grad():
                    output = self.model_MLP(input)
                val, indices = torch.max(output, dim=1)
                labels = []
                for i in indices:
                    labels.append(label_name[i])

                out = visualizer.overlay_instances(boxes=inference_data[frame]["instances"].pred_boxes.to(
                    "cpu"), labels=labels, keypoints=inference_data[frame]["instances"].pred_keypoints.to("cpu"))
                result = out.get_image()[:, :, ::-1]
                cap.write(result)

            cap.release()

        del self.loaded_video_image
        self.loaded_video_image = {}

    ##딕셔너리에 있는 이미지를 추론후 이미지로 저장한다##
    def inference_images(self, batch: int):
        for dir_name in self.loaded_images.keys():
            i = 0
            inference_data = []
            while True:
                inputs = []
                for cnt in range(batch):
                    if(i >= len(self.loaded_images[dir_name])):
                        break
                    name, original_image = self.loaded_images[dir_name][i]
                    i += 1
                    height, width = original_image.shape[:2]
                    image = self.aug.get_transform(
                        original_image).apply_image(original_image)
                    image = torch.as_tensor(
                        image.astype("float32").transpose(2, 0, 1))
                    input_dict = {"image": image,
                                  "height": height, "width": width}
                    inputs.append(input_dict)
                predictions = self(inputs)

                for inferenced_data in predictions:
                    ##스코어 0.6 이상의 객체들만 남긴다##
                    inferenced_data["instances"] = inferenced_data["instances"][inferenced_data["instances"].scores > 0.6]
                    inference_data.append(inferenced_data)

                if(i >= len(self.loaded_images[dir_name])):
                    break

            for idx in range(len(inference_data)):
                visualizer = Visualizer(
                    self.loaded_images[dir_name][idx][1][:, :, ::-1], self.metadata, scale=1.0)
                input = self.__get_for_mlp_keypoints__(
                    inference_data[idx]["instances"], args)
                with torch.no_grad():
                    output = self.model_MLP(input)
                val, indices = torch.max(output, dim=1)
                labels = []

                for i in indices:
                    labels.append(label_name[i])

                out = visualizer.overlay_instances(boxes=inference_data[idx]["instances"].pred_boxes.to(
                    "cpu"), labels=labels, keypoints=inference_data[idx]["instances"].pred_keypoints.to("cpu"))
                result = out.get_image()[:, :, ::-1]
                name, image = self.loaded_images[dir_name][idx]
                try:
                    if not os.path.exists(f'result_images'):
                        os.makedirs(f'result_images')
                except OSError:
                    pass
                relate_dir = dir_name.split('/')[-1]
                try:
                    if not os.path.exists(f'result_images/{relate_dir}'):
                        os.makedirs(f'result_images/{relate_dir}')
                except OSError:
                    pass
                cv2.imwrite(f'result_images/{relate_dir}/{name}', result)

        del self.loaded_images
        self.loaded_images = {}

    def inference_image(self, target_path: str):
        original_image = cv2.imread(target_path)
        name = target_path.split('/')[-1]
        inference_data = []
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(
            original_image).apply_image(original_image)
        image = torch.as_tensor(
            image.astype("float32").transpose(2, 0, 1))
        input_dict = {"image": image,
                      "height": height, "width": width}
        predictions = self([input_dict])
        for inferenced_data in predictions:
            ##스코어 0.6 이상의 객체들만 남긴다##
            inferenced_data["instances"] = inferenced_data["instances"][inferenced_data["instances"].scores > 0.6]
            inference_data.append(inferenced_data)
        visualizer = Visualizer(
            original_image[:, :, ::-1], self.metadata, scale=1.0)
        input = self.__get_for_mlp_keypoints__(
            inference_data[0]["instances"], args)
        with torch.no_grad():
            output = self.model_MLP(input)
        val, indices = torch.max(output, dim=1)
        labels = []
        for i in indices:
            labels.append(label_name[i])
        out = visualizer.overlay_instances(boxes=inference_data[0]["instances"].pred_boxes.to(
            "cpu"), labels=labels, keypoints=inference_data[0]["instances"].pred_keypoints.to("cpu"))
        result = out.get_image()[:, :, ::-1]
        try:
            if not os.path.exists(f'result_images'):
                os.makedirs(f'result_images')
        except OSError:
            pass
        cv2.imwrite(f'result_images/{name}', result)

    def __get_last_dir(self, absolute_dir_path: str) -> list:
        absolute_path = absolute_dir_path
        dirs = os.listdir(absolute_path)
        saved_dirs = []

        for target in dirs:
            target_path = absolute_path+"/"+target
            if os.path.isdir(target_path):
                saved_dirs.extend(self.__get_last_dir(target_path))

        for target in dirs:
            for_break = False
            for image_format in [".png", ".jpg", ".jpeg"]:
                if(image_format in target):
                    saved_dirs.append(absolute_dir_path)
                    for_break = True
                    break
            if(for_break):
                break
        return saved_dirs

    def __get_for_mlp_keypoints__(self, instances, args) -> torch.tensor:
        keypoints = instances.pred_keypoints
        return MyNomalizer(keypoints, args.device)

    def __call__(self, inputs: list) -> list:
        with torch.no_grad():
          predictions = self.model(inputs)
        return predictions
