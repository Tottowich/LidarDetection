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
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
import open3d
sys.path.insert(0, '../OusterTesting')
#sys.path.insert(1, '../OpenPCDet-linux')
from models.common import DetectMultiBackend
from CenterDetection.CenterDetector import CenterResNet
import torch.backends.cudnn as cudnn
import utils_ouster
from tools.transmitter import Transmitter
from tools.open3d_live_vis import LiveVisualizer
from ouster import client
from contextlib import closing
#from tools.visual_utils import open3d_vis_utils as V
#from pcdet.config import cfg, cfg_from_yaml_file
#from pcdet.datasets import DatasetTemplate
#from pcdet.models import build_network, load_data_to_gpu
from tools.xr_synth_utils import CSVRecorder,TimeLogger,Replayer,filter_predictions,format_predictions,display_predictions
from tools.xr_synth_utils import create_logger, proj_alt,proj_alt2
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
class live_stream:
    """
    Class to stream data from a Sensor.
    Inheritance:
        DatasetTemplate:
            Uses batch processing and collation.
    """
    def __init__(self, classes,ip,stride=32,img_size=(1280,640), logger=None,auto=True):
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
        
    def prep(self,img0):
        """
        Prepare data from the lidar sensor.
        Args:
            img0: The image that is to be prepared.
        """
        #print(self.img_size)
        #print(img0.shape)
        img = self.reshape(copy(img0))
       #print(img.shape)
        if len(img.shape) == 3:
            img = img[None]
        #img = img[..., ::-1].transpose((0,3,1,2))  # BGR to RGB, BHWC to BCHW
        img = img.transpose((0,3,1,2))  # BGR to RGB, BHWC to BCHW
        
        img = np.ascontiguousarray(img)
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
    model_center = None
    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=args.data, fp16=args.half)
    stride, names, pt = model.stride, model.names, model.pt
    #imgsz = (args.imgsz, args.imgsz//2) if isinstance(args.imgsz, int) else args.imgsz  # tuple
    imgsz = (1280,640) # Hard coded for now, must change!
    imgsz = check_img_size(imgsz=imgsz, s=stride)
    #print(imgsz)
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    if args.detect_center:
        model_center = CenterResNet(img_shape=predict_img_shape).to(device)
        model_center.warmup(imgsz=predict_img_shape)
        model_center.load_model("CenterModels/model_best.pt")
    return model, stride, names, pt, device,model_center
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
    #time_logger.create_metric("Pre Processing")
    #time_logger.create_metric("Load GPU")
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
def preprocess_data(img0):
    """
    Preprocess the image.
    """
    if isinstance(img, np.ndarray): # Faster!
        img = torch.from_numpy(copy(img0)).div(255.0)
    else:
        img = torch.tensor(copy(img0)).div(255.0)
    return img0,img


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
        annotator = Annotator(img0, line_width=args.line_thickness, example=str(names))
        
        if len(det):
            #print(img.shape[2:],img.squeeze().permute(1,2,0).shape)
            det[:,:4] = scale_coords(img.shape[2:], det[:,:4], img0.shape).round()
            
            i = 0
            for j,(*xyxy, conf, cls) in enumerate(det):#reversed(det)):
                c = int(cls)  # integer class
                height = pred_dict["pred_boxes"][j,5]
                z = pred_dict["pred_boxes"][j,2]
                label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f} {z:.2f}')
                i += 1
                annotator.box_label(xyxy, label, color=colors(c, True))
                if detection_area is not None and detection_area[j] is not None:
                    annotator.highlight_area(detection_area[j])
                if centers is not None:    
                    xyxy = [coordinate.cpu().numpy() for coordinate in xyxy]
                    scaled = centers[j,:]*[(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])]+[xyxy[0],xyxy[1]]
                    # print(f"Centers: {centers[j,:]}")
                    # print(f"Scaled: {scaled}")
                    annotator.point(scaled)
                #annotator.point(((xyxy[0]+xyxy[2])/2,((xyxy[1]+xyxy[3])/2)), color=colors(c, True))
            img0 = annotator.result()
            #logger.info(f"Det: {det}")
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            #print(f"Post viz Average img: {img0.mean()}")
            cv2.imshow("Predictions",img0)
            cv2.waitKey(1)
        else:
            #print(img.shape)
            img0  = annotator.result()
            #print(img0.shape)
            img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
            cv2.imshow("Predictions",img0)
            #print(f"Post viz Average img: {img.mean()}")
            cv2.waitKey(1)
            



def parse_config():
    """
    Parse the configuration file.
    """
    parser = argparse.ArgumentParser(description='arg parser')
    #parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
    #                    help='specify the config for demo')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=1280, help='inference size h,w')
    parser.add_argument('--recorded_dataset', type=str, default='dataset/Combined_dataset/', help='(optional) dataset path')
    parser.add_argument('--data', type=str, default='Xr-Synthesize-SRR-3/data.yaml', help='dataset configuration path')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference instead of FP32 (default)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--auto', action='store_true', help='auto size using the model')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')

    #parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--UE5_ip', type=str, default=None, help='specify the ip of the UE5 machine')
    parser.add_argument('--TD_ip', type=str, default="192.168.200.103", help='specify the ip of the TD machine')

    parser.add_argument('--TD_port', type=int, default=7002, help='specify the port of the TD machine')
    parser.add_argument('--UE5_port', type=int, default=7000, help='specify the port of the UE5 machine')
    parser.add_argument('--time', type=int, default=100
    , help='specify the time to stream data from a sensor')
    #parser.add_argument('--save_dir', type=str, default="../lidarCSV", help='specify the save directory')
    #parser.add_argument('--save_name', type=str, default="test_csv", help='specify the save name')
    if sys.version_info >= (3,9):
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
        parser.add_argument('--detect_center', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_csv', action=argparse.BooleanOptionalAction)
        parser.add_argument('--log_time', action=argparse.BooleanOptionalAction)
        parser.add_argument('--disp_pred', action=argparse.BooleanOptionalAction)
        parser.add_argument('--wait_for_key', action=argparse.BooleanOptionalAction)
        parser.add_argument('--transmit', action=argparse.BooleanOptionalAction)
        parser.add_argument('--pcd_vis', action=argparse.BooleanOptionalAction)
        

    else:
        parser.add_argument('--visualize', action='store_true')
        parser.add_argument('--no-visualize', dest='visualize', action='store_false')
        parser.add_argument('--save_csv', action='store_true')
        parser.add_argument('--no-save_csv', dest='save_csv', action='store_false')
        parser.add_argument('--log_time', action='store_true')
        parser.add_argument('--no-log_time', dest='log_time', action='store_false')
        parser.add_argument('--disp_pred', action='store_true')
        parser.add_argument('--no-disp_pred', dest='disp_pred', action='store_false')
        parser.set_defaults(visualize=True)
        parser.set_defaults(save_csv=False)
    args = parser.parse_args()
    with open(args.data,'r') as f:
        try:
            data_config = yaml.safe_load(f)
        except:
            raise ValueError(f"Invalid data config file: {args.data}")
        #cfg_from_yaml_file(args.cfg_file, cfg)

    return args,data_config#, cfg


@torch.no_grad() # No grad to save memory
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, data_config = parse_config()
    range_limit = [10,10,5]
    init = True
    centers = None
    cudnn.benchmark = True  # set True to speed up constant image size inference
    #model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    logger = create_logger()
    predict_img_shape = (1,3,360,360)
    model, stride, names, pt, device,model_center = initialize_network(args,device,predict_img_shape)
    replayer = Replayer(source = args.recorded_dataset, image_size=(640,1280))
    # Select classes to use, None -> all.
    # classes_to_use = [8]
    # Set up interactions
    #live = live_stream(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger=logger)
    
    if args.transmit:
        transmitter = Transmitter(reciever_ip=args.TD_ip, reciever_port=args.TD_port, classes_to_send=[9])
        transmitter.start_transmit_udp()
        transmitter.start_transmit_ml()
    else:
        transmitter = None
    log_time = False # False to let the program run for one loop to warm up :)
    if args.log_time:
        time_logger = initialize_timer(logger=logger,transmitter=transmitter,args=args)


        logger.info(f"Streaming lidar data to: Yolov5 using {args.weights}")
         # time 
        
        start_stream = time.monotonic()
        
    for i,(pcd,img0) in enumerate(replayer): # Ouster scan object
        if log_time:
            time_logger.start("Ouster Processing")
        if init:
            print(f"Img0: {img0.shape}")
        img0, img = replayer.prep(img0)
        img = img.to(device)
        img_to_vis = copy(img)
        if init:
            print(f"Image: {img.shape}")
            print(f"Img0: {img0.shape}")
        if log_time:
            time_logger.stop("Ouster Processing")
        
        #if range_limit is not None:
        #    xyzr = utils_ouster.trim_xyzr(xyzr,range_limit)
        #xyzr = utils_ouster.trim_data(data=xyzr,range_limit=range_limit,source=stream,scan=scan)
        #print(f"Input point cloud shape: {xyzr.shape}")
        if i%2 == 0 and log_time:
            time_logger.start("Full Pipeline")
        if i%2 == 1 and log_time and i != 1:
            time_logger.stop("Full Pipeline")
        
        

        #if log_time:
        #    time_logger.start("Load GPU")
        #load_data_to_gpu(data_dict)
        #if log_time:
        #    time_logger.stop("Load GPU")
        #print(img.shape)
        if log_time:
            time_logger.start("Infrence")
        #with torch.cuda.amp.autocast(),torch.no_grad():
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
        
            
        
        
        # if args.save_csv: # If recording, save to csv
        #     if log_time:
        #         time_logger.start("Save CSV")
        #     recorder.add_frame_file(copy(data_dict["points"][:,1:-1]).cpu().numpy(),pred_dict)
        #     if log_time:
        #         time_logger.stop("Save CSV")
        
        if args.transmit and transmitter.started_ml:
            if log_time:
                time_logger.start("Transmit UE5")
            transmitter.pcd = copy(pcd)
            transmitter.pred_dict = copy(pred_dict)
            transmitter.send_pcd()
            if log_time:
                time_logger.stop("Transmit UE5")


        if args.transmit and transmitter.started_udp: # If transmitting, send to udp
            if log_time:
                time_logger.start("Transmit TD")
            transmitter.pred_dict = copy(pred_dict)
            transmitter.send_dict()
            if log_time :
                time_logger.stop("Transmit TD")

        
        if args.visualize:
            if log_time:
                time_logger.start("Visualize")
            if range_limit is not None:
                xyz = utils_ouster.trim_xyzr(utils_ouster.compress_mid_dim(pcd),range_limit)
            else:
                xyz = utils_ouster.compress_mid_dim(pcd)
            if i == 0 and args.pcd_vis:
                vis = LiveVisualizer("XR-SYNTHESIZER",
                                    class_names=names,
                                    first_cloud=xyz,
                                    classes_to_visualize=None
                                    )
            elif args.pcd_vis:
                vis.update(points=xyz, 
                        pred_boxes=pred_dict['pred_boxes'],
                        pred_labels=pred_dict['pred_labels'],
                        pred_scores=pred_dict['pred_scores'],
                        )
            visualize_yolo_2D(pred,pred_dict,img_to_vis,centers=centers,detection_area = detection_area,args=args,names=names,logger=logger)     
            #visualize_yolo_2D_test(pred_dict,img_to_vis,args,names=names,logger=logger)
            if log_time:
                time_logger.stop("Visualize")
                            #vis = V.create_live_scene(data_dict['points'][:,1:],ref_boxes=pred_dicts[0]['pred_boxes'],
            #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'])
        # #elif args.visualize:
        #     start = time.monotonic()
        #     #V.update_live_scene(vis,pts,points=data_dict['points'][:,1:], ref_boxes=pred_dicts[0]['pred_boxes'],
        #     #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],class_names=cfg.CLASS_NAMES)
        #     if log_time:
        #         time_logger.start("Visualize")
        #     vis.update(points=data_dict['points'][:,1:], 
        #                 pred_boxes=pred_dicts['pred_boxes'],
        #                 pred_labels=pred_dicts['pred_labels'],
        #                 pred_scores=pred_dicts['pred_scores'],
        #                 )
        #     if log_time:
        #         time_logger.stop("Visualize")
        if log_time and args.disp_pred:
            print("\n")
        #if i == 6:
        #    break
        if init:
            init = False
        if args.wait_for_key:
            if logger is not None:
                logger.info("Press the \'n\' key to continue.")
            else:
                print("Press the \'n\' key to continue.")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('n'):
                    break
        log_time = args.log_time
    if args.transmit:
        transmitter.stop_transmit_udp()
        transmitter.stop_transmit_ml()
    if log_time:
        time_logger.visualize_results()
    logger.info("Stream Done")

"""
This program uses has been tested with the Ouster OS0-64 sensor and OS0-128.
Example Input:
    py live_yolo.py --weights "runs/train/ElephantSnorkeling7/weights/best.pt" --imgsz 1280 --data "Xr-Synthesizer-12/data.yaml" --iou_thres 0.8 --conf_thres 0.5 --OU_ip "192.168.200.78" --visualize --log_time --no-disp_pred --time 1000 --transmit
Best Results:
    py live_yolo.py --weights "runs/train/First-SRR-M2/weights/best.pt" --imgsz 1280 --data "Xr-Synthesize-SRR-1/data.yaml" --iou_thres 0.2 --conf_thres 0.25 --OU_ip "192.168.200.79" --visualize --log_time --no-disp_pred --time 2000 --transmit --udp_port 7504 --tcp_port 7505
    Using SRR.
"""

if __name__ == '__main__':
    main()

