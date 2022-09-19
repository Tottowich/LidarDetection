import argparse
import os, sys
import glob
from pathlib import Path
import time
import numpy as np
import torch
import threading
import math
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt 
from queue import Queue
from copy import copy
import yaml
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path.
# OusterTesting Should be located outside the Yolo folder.
sys.path.insert(0, '../OusterTesting') # Add path to the OusterTesting folder.
from models.common import DetectMultiBackend
from CenterDetection.CenterDetector import CenterResNet
import torch.backends.cudnn as cudnn
import utils_ouster # OusterTesting\utils_ouster.py.
import torchvision as tv
from tools.transmitter import Transmitter
from tools.open3d_live_vis import LiveVisualizer
from ouster import client
from contextlib import closing
from tools.xr_synth_utils import CSVRecorder,TimeLogger,display_predictions
from tools.xr_synth_utils import create_logger,proj_alt2
from tools.arguments import parse_config
from utils.general import (LOGGER, check_img_size, cv2,non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
# from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

class live_stream:
    """
    Class to stream data from a Sensor.
    Prep the data and send it to the model.
    """
    def __init__(self, classes,ip,stride=32,img_size=(640,1280), logger=None,auto=True):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        self.classes = classes
        self.ip = ip

        self.stride = stride
        self.img_size = img_size
        self.logger = logger
        self.auto = auto
        self.rect = True
        self.frame = 0
        self.transform = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Resize(img_size)])
    def prep(self,img0):
        """
        Prepare data from the lidar sensor.
        Args:
            img0: The image that is to be prepared.
        """
        img = np.ascontiguousarray(copy(img0))
        img = self.transform(img) # Convert to tensor and resize to model input size and CHW format for pytorch.
        if len(img.shape) == 3:
            img = img[None]
        self.frame += 1
        return img0,img
    def reshape(self,img):
        """
        Reshape the data to be compatible with the model.
        Args:
            img: the image to be reshaped.
        """
        img = cv2.resize(img,self.img_size)
        return img

def initialize_network(args,device,predict_img_shape):
    """
    Initialize the network.
    Create live streaming object to stream data from the sensor.
    Create the detection model.
    Args:
        args: Arguments from the command line.
        device: Device to run the network on (Cuda GPU if available).
    """
    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=args.data, fp16=args.half).eval()
    stride, names, pt = model.stride, model.names, model.pt
    img_size = args.imgsz
    img_size = check_img_size(imgsz=img_size, s=stride)
    model.warmup(imgsz=(1 if pt else 1, 3, *img_size))
    live = live_stream(classes=names,ip=args.OU_ip,stride=stride,auto=args.auto,img_size=img_size)
    if args.detect_center: # NOTE: Center Detection is not very refined and extensivly tested.
        model_center = CenterResNet(img_shape=predict_img_shape).to(device)
        model_center.warmup(imgsz=predict_img_shape)
        model_center.load_model("CenterModels/model_best.pt")
    else:
        model_center = None
    return model, stride, names, pt, device,live,model_center
def initialize_timer(logger:LOGGER,args,transmitter=None):
    """
    Create the timer object to keep track of all the time taken by various parts of the pipeline.
    Args:
        logger: Logger object to log the time taken.
        args: Arguments from the command line.
        transmitter: If transmitter object is available then the time taken to transmit the data is also logged.
    """
    time_logger = TimeLogger(logger,args.disp_pred)
    time_logger.create_metric("Ouster Processing")
    time_logger.create_metric("Infrence")
    time_logger.create_metric("Post Processing")
    time_logger.create_metric("Format Predictions")
    time_logger.create_metric("Projection")
    if args.detect_center:  
        time_logger.create_metric("Center Prediction")
    if args.visualize:
        time_logger.create_metric("Visualize")
    if args.save_csv:
        time_logger.create_metric("Save CSV")
    if transmitter is not None:
        if transmitter.started_udp:
            time_logger.create_metric("Transmit TD")
        if transmitter.started_ml:
            time_logger.create_metric("Transmit UE5")
    time_logger.create_metric("Full Pipeline")

    return time_logger
def preprocess_data(live:live_stream,fp16:bool,device:str,stream,scan,timelogger:TimeLogger):
    """
    Get the data from the sensor and preprocess it.
    """
    img0 = utils_ouster.signal_ref_range(stream,scan)
    pcd = utils_ouster.get_xyz(stream,scan)
    img0, img = live.prep(img0)
    img = img.to(device)
    img = img.half() if fp16 else img.float()  # uint8 to fp16/32
    img_to_vis = copy(img)
    return img0,img,img_to_vis,pcd


def visualize_yolo_2D(pred,pred_dict,img,args,centers=None,detection_area=None,names=None,logger=None):
    """
    Visualize the predictions.
    Args:
        pred: Predictions from the model.
        pred_dict: Dictionary of predictions.
        img: Image to be visualized.
        args: Arguments from the command line.
        names: Names of the classes.
        logger: Logger object to log potential predictions
    """
    detections = 0
    centers = centers.cpu().numpy() if centers is not None else None
    for i,det in enumerate(pred):
        detections += 1
        img0 = np.ascontiguousarray(copy(img).squeeze().permute(1,2,0).cpu().numpy(),dtype=np.float32)
        annotator = Annotator(img0, line_width=args.line_thickness, example=str(names)) # Create annotator object to draw on the image.
        
        if len(det):
            # Rescale boxes from being in the range of the image predictions were made on to the original image size.
            det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
        
            for j,(*xyxy, conf, cls) in enumerate(det):#reversed(det)):
                c = int(cls)  # integer class
                height = pred_dict["pred_boxes"][j,5] # Height of the bounding box in 3D space, slightly inaccurate.
                z = pred_dict["pred_boxes"][j,2] # Z coordinate of the center of the bounding box in 3D space.
                label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f} {z:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                if centers is not None: # If center detection is enabled then draw the center of object in the bounding box. TODO: Increase preformace of this model.
                    xyxy = [coordinate.cpu().numpy() for coordinate in xyxy]
                    scaled = centers[j,:]*[(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])]+[xyxy[0],xyxy[1]]
                    annotator.point(scaled)
            img0 = annotator.result()
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.imshow("Predictions",img0) # Show the image with the predictions.
            cv2.waitKey(1)
        else:
            img0  = annotator.result()
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.imshow("Predictions",img0) # Show the image without predictions.
            cv2.waitKey(1)
            
@torch.no_grad() # No grad to save memory
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, data_config = parse_config()
    range_limit = None
    init = True
    centers = None
    cudnn.benchmark = True  # set True to speed up constant image size inference
    logger = create_logger()
    predict_img_shape = (1,3,360,360) # Shape of the input image to center detection model. # TODO: Make this dynamic and increase the performance of this model.
    model, stride, names, pt, device,live,model_center = initialize_network(args,device,predict_img_shape)
    if args.OU_ip is None and args.name is None:
        raise ValueError('Please specify the ip or sensor name of the ')
    if args.save_csv: # If the user wants to save the csv file of point cloud data
        recorder = CSVRecorder(args.save_name,args.save_dir, data_config.CLASS_NAMES)
    
    if args.transmit:
        transmitter = Transmitter(reciever_ip=args.TD_ip, reciever_port=args.TD_port, classes_to_send=[9])
        transmitter.start_transmit_udp()
        transmitter.start_transmit_ml()
    else:
        transmitter = None
    try:
        [cfg_ouster, host_ouster] = utils_ouster.sensor_config(args.name if args.name is not None else args.OU_ip,args.udp_port,args.tcp_port)
    except:
        raise ConnectionError('Could not connect to the sensor')
    log_time = False # False to let the program run for one loop to warm up :)
    if args.log_time:
        time_logger = initialize_timer(logger=logger,transmitter=transmitter,args=args)
    else:
        time_logger = None


    with closing(client.Scans.stream(host_ouster, args.udp_port,complete=False)) as stream:
        logger.info(f"Streaming lidar data to: Yolov5 using {args.weights}")
         # time 
        
        start_stream = time.monotonic()
        
        for i, scan in enumerate(stream): # Ouster scan object 
            if log_time:
                time_logger.start("Ouster Processing")
            img0,img,img_to_vis,pcd = preprocess_data(stream=stream,scan=scan,device=device,live=live,fp16=model.fp16,timelogger=time_logger)  
            if log_time:
                time_logger.stop("Ouster Processing")
            if init and args.verbose:
                logger.info(f"Original Img0 shape: {img0.shape}")
                logger.info(f"Input Image shape: {img.shape}")

            if i%2 == 0 and log_time:
                time_logger.start("Full Pipeline")
            if i%2 == 1 and log_time and i != 1:
                time_logger.stop("Full Pipeline")


            if log_time:
                time_logger.start("Infrence")
            pred = model(img,augment=args.augment)
            if log_time:
                time_logger.stop("Infrence")
            if log_time:
                time_logger.start("Post Processing")
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
            if log_time:
                time_logger.stop("Post Processing")
            
            if args.detect_center:
                if log_time:
                    time_logger.start("Center Prediction")
                centers = model_center.center_predicitons(img,pred[0],predict_img_shape)
                if log_time:
                    time_logger.stop("Center Prediction")
            #print(pred)
            if log_time:
                time_logger.start("Projection")
            pred_dict,detection_area = proj_alt2(copy(pred),img[0].cpu().numpy(),xyz=pcd)
            if log_time:
                time_logger.stop("Projection")
        
            if len(pred_dict["pred_labels"]) > 0 and args.disp_pred:
                display_predictions(pred_dict,names,logger)
            
                
            
            
            if args.save_csv: # If recording, save to csv
                if log_time:
                    time_logger.start("Save CSV")
                recorder.add_frame_file(copy(pcd).cpu().numpy(),pred_dict)
                if log_time:
                    time_logger.stop("Save CSV")
            
            if args.transmit and transmitter.started_ml:
                """
                This was added incase the point cloud were to be sent to Unreal Engine for visualization with predictions.
                OBS: This is not implemented yet on the Unreal Engine side.
                """
                if log_time:
                    time_logger.start("Transmit UE5")
                transmitter.pcd = copy(pcd)
                transmitter.pred_dict = copy(pred_dict)
                transmitter.send_pcd()
                if log_time:
                    time_logger.stop("Transmit UE5")


            if args.transmit and transmitter.started_udp: # If transmitting, send to udp
                """
                Transmitting the predictions to the TD (TouchDesigner) system. These predictions are the 3D bounding boxes and position of each object.
                """
                if log_time:
                    time_logger.start("Transmit TD")
                transmitter.pred_dict = copy(pred_dict)
                transmitter.send_dict()
                if log_time :
                    time_logger.stop("Transmit TD")

            
            if args.visualize:
                """
                Visualize the predictions on the point cloud data.
                Visualizations are both available in 2D and 3D.
                """
                if log_time:
                    time_logger.start("Visualize")
                if range_limit is not None and args.pcd_vis:
                    # Limit the range of the point cloud data to visualize.
                    xyz = utils_ouster.trim_xyzr(utils_ouster.compress_mid_dim(pcd),range_limit)
                else:
                    # Point cloud visualized as a set of points, therefor the shape is (N,4) where N is the number of points.
                    xyz = utils_ouster.compress_mid_dim(pcd)
                if i == 0 and args.pcd_vis:
                    """
                    First Iteration of the loop, initialize the visualizer.
                    This is done to set the initial camera position relative to the surronding area of the point cloud data correctly.
                    If this were to be done before the first iteration, the camera would be placed in the center of the point cloud data and hard making it hard to navigate the visualization.
                    """
                    vis = LiveVisualizer("XR-SYNTHESIZER",
                                        class_names=names,
                                        first_cloud=xyz,
                                        classes_to_visualize=None
                                        )
                elif args.pcd_vis:
                    """
                    Update the point cloud data in the visualizer.
                    """
                    vis.update(points=xyz, 
                            pred_boxes=pred_dict['pred_boxes'],
                            pred_labels=pred_dict['pred_labels'],
                            pred_scores=pred_dict['pred_scores'],
                            )
                # Vislualize the predictions on the image data.
                visualize_yolo_2D(pred,pred_dict,img_to_vis,centers=centers,detection_area = detection_area,args=args,names=names,logger=logger)     
                if log_time:
                    time_logger.stop("Visualize")
            if time.monotonic()-start_stream > args.time:
                """
                If the time limit has been reached, stop the loop.
                """
                stream.close()
                break
            if log_time and args.disp_pred:
                print("\n")
            if init:
                init = False
            log_time = args.log_time
    if args.transmit:
        transmitter.stop_transmit_udp()
        transmitter.stop_transmit_ml()
    if log_time:
        # Summarize the time logging of each section. This is only done if the time logging is enabled.
        time_logger.visualize_results()
    logger.info("Stream Done")

"""
This program uses has been tested with the Ouster OS0-64 sensor and OS0-128.
Example Input:
    py live_yolo.py --weights "runs/train/ElephantSnorkeling7/weights/best.pt" --imgsz 1280 --data "Xr-Synthesizer-12/data.yaml" --iou_thres 0.8 --conf_thres 0.5 --OU_ip "192.168.200.78" --visualize --log_time --no-disp_pred --time 1000 --transmit
Best Results:
    py live_yolo.py --weights "runs/train/First-SRR-M2/weights/best.pt" --data "Xr-Synthesize-SRR-1/data.yaml" --iou_thres 0.25 --conf_thres 0.25 --OU_ip "192.168.200.79" --no-visualize --no-log_time --no-disp_pred --time 2000 --transmit --udp_port 7504 --tcp_port 7505 --no-pcd_vis --no-detect_center --img_size 640 1280
    py live_yolo.py --weights "runs/train/Frozen-WheelChair3/weights/best.pt" --data "Xr-Synthesize-SRR-1/data.yaml" --iou_thres 0.25 --conf_thres 0.25 --OU_ip "192.168.200.79" --no-visualize --no-log_time --no-disp_pred --time 2000 --transmit --udp_port 7504 --tcp_port 7505 --no-pcd_vis --no-detect_center --img_size 640 1280
    
    Using SRR.

- []: Train image segmentation model for the SRR image representation giving image segmentation in point cloud as return.
        Note:
            This could be done at a bounding box lebel but this would be slower but far easier. Training data easier to create.

TODO: Train pose estimation model. This could be used to determine the 3D pose of person in the scene surronding sensor.

TODO: Combine multiple sensors to get less occlusion and more data.
        Ideas how this could be done:
            # Appoarch 1.
                - Make 2D predictions on each sensor.
                    Note: This could be done in parallel using the batch size dimension of the model. !!!
                - Project the 2D predictions into 3D space using depth map for each sensor.
                - Combine the 3D predictions into one 3D prediction given that the relative position of the sensors are known.
                - Either 3D non max suppression on 3D positions or 2D non max suppression on the x,y coordinates of the 3D predictions.
                    Note: This would cause predictions to be lost if they stacked in the z dimension, but this is not a problem for the current use case and probably not a problem for most use cases.

TODO: Look into idea of predicting all dimensions of object in 3D space using combined sensor data via image representation.
        Ideas how this could be done:
            # Appoarch 1.
                - Stack frames from each sensor into a single image. Making predictions of image with 6 Channels instead of 3.
                    Note: Have not read anything about this but could be interesting idea for future research as multi sensor data is becoming more common and this would be cost efficient way to use it.
                    Note: Would need 3D labeling to train model.
                    Note: Could maybe use existing 3D object detection datasets to use as training data.

TODO: Create GUI for live_yolo.py to make it easier to use for non programmers.
        GUI Ideas:
            - Select sensor to use from connected sensor list.
            - Select model to use from model list.
            - Select data to use from data list.
            - Iou threshold slider.
            - Confidence threshold slider.
            - Time limit slider.
            - Enable/Disable visualization.
            - Drop down menu for selecting other command line arguments.
TODO: Create GUI for training and testing models. Good Training might not be to useful very useful.
"""

if __name__ == '__main__':
    main()

