import argparse
#from asyncio.format_helpers import extract_stack
import os, sys
import glob
from pathlib import Path
from telnetlib import SE
import time
from tkinter import OFF
import numpy as np
import torch
from copy import copy
import open3d
#from queue import Queue
from datetime import datetime as dt
from contextlib import closing
import torchvision as tv
import matplotlib.pyplot as plt
import pandas as pd
import math
pd.options.display.float_format = '{:,.4e}'.format
import logging
import re
import cv2
sys.path.insert(0, '../../OusterTesting')
import utils_ouster
SENSOR_HEIGHT = 0.0
OFFSET = 10
HARD_WIDTH = 0.5
HEAD_PROPORTION = 1/7 # Human head is rougnly 1/7 of the total height
HEAD_FACTOR = 1.9 # The number of "Heads" to use as distance to check area of interest for depth estimation as.
QUANTILE_FACTOR = 0.15
def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    """
    Create a logger to log the predictions made by the model.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

# def proj_alt(pred,img0,xyz,R=25,azi=90,logger=None):
#     """
#     Algorithm to project a bounding box on a image into 3D space.
#     """
#     #print(f"Image Shape: {img0.shape}")
#     img_height = img0.shape[1]
#     img_width = img0.shape[2]
#     offset = OFFSET
#     #print(f"PCD Shape: {xyz.shape}")
#     scale_y = xyz.shape[0]/img_height
#     scale_x = xyz.shape[1]/img_width 
#     obj_dim = [] # center_x, center_y, center_z, width, height, depth, rotation_x, rotation_y, rotation_z
#     heights = []
#     for det in pred[0]:
#         xyxy = det[:4].cpu().numpy()
#         x0,y0,x1,y1 = xyxy
#         x0 = max(0,x0)
#         y0 = max(0,y0)
#         x1 = min(img_width,x1)
#         y1 = min(img_height,y1)

#         pol_c_x,c_y = (x0+x1)/2,(y0+y1)/2
#         head_size = (y1-y0)*HEAD_PROPORTION*HEAD_FACTOR
#         chest_offset = ((y1-y0)/2-head_size) if c_y+((y1-y0)/2-head_size) < img_height else 0
#         pol_c_y = c_y-chest_offset
#         ix, iy = int(pol_c_x), int(pol_c_y)

#         low_y,offset_low_y = [iy-offset,iy-offset] if iy-offset>=y0 else [0,iy%offset]
#         high_y,offset_high_y = [iy+offset,iy+offset] if iy+offset<y1 else [img_height-1, img_height-iy-1]
#         low_x,offset_low_x =  [ix-offset,ix-offset] if ix-offset>=y0 else [0,iy%offset]
#         high_x,offset_high_x = [ix+offset,ix+offset] if ix+offset<x1 else [img_width-1,img_width-iy-1]
#         roi = copy(img0[2,low_y:high_y,int(x0):int(x1)])
#         poi_img = np.unravel_index(np.argmin(np.abs(np.quantile(roi,QUANTILE_FACTOR)-roi)),roi.shape) # lower_quantile
        
#         #poi_scan  = [int((poi_img[0]+int(offset_low_y))*scale_y), int((poi_img[1]+int(offset_low_x))*scale_x)]
#         poi_scan  = [int((poi_img[0]+int(offset_low_y))*scale_y), int((poi_img[1]+int(x0))*scale_x)]
#         rot = -float(poi_scan[1])/xyz.shape[1]*2*np.pi
        
#         center_x, center_y, center_z = xyz[poi_scan[0],poi_scan[1],:]
    
#         vert_dist = np.sqrt(center_x**2+center_y**2)       
#         cone_height = 2*np.tan(azi/2)*vert_dist     #     /|
#         height_cov = (y1-y0)/img_height             #    / |
#         height = height_cov*cone_height/2           # [c]  |Cone height
#                                                     #  | \ |
#                                                     #  |  \|
#         z_offset = height*HEAD_PROPORTION*HEAD_FACTOR

#         center_z = center_z-z_offset
#         circle_diam = 2*np.pi*vert_dist
#         width_cov = (x1-x0)/img_width
#         temp_width = width_cov*circle_diam
#         [width,breadth] = [temp_width,HARD_WIDTH] if temp_width>HARD_WIDTH else [HARD_WIDTH,temp_width]
        
#         #2*(center_z+SENSOR_HEIGHT)
#         obj_dim.append([center_x,center_y,center_z,breadth,width,height,rot,0,0])
#     pred_dict = {"pred_boxes": np.array(obj_dim), "pred_scores": np.array(pred[0].cpu())[:,4], "pred_labels": np.array(pred[0].cpu())[:,5].astype(np.uint8)}

#     return pred_dict,


def proj_alt2(pred,img0,xyz,R=25,azi=90,logger=None):
    """
    Algorithm to project a bounding box on a image into 3D space.
    TODO: Vectorize the code to make it faster.
    Args:
        pred: Prediction from the model
        img0: Image representation of the sensor data. Should be a 3D numpy array of shape (C,H,W). Last channels should be the depth image generated from the sensor data.
        xyz: Point cloud from the lidar sensor
    """
    img_height = img0.shape[1]
    img_width = img0.shape[2]
    offset = OFFSET
    scale_y = xyz.shape[0]/img_height # Factor to scale the predictions from the image predictions were made on to the original pseudo image
    scale_x = xyz.shape[1]/img_width
    scale_to_angle = 2*np.pi/img_width
    obj_dim = [] # center_x, center_y, center_z, width, height, depth, rotation_x, rotation_y, rotation_z
    detection_area = []
    for det in pred[0]: # Ment to be used with batch size 1
        # TODO: Try to vectorize this to increase efficiency.
        xyxy = det[:4].cpu().numpy()
        x0,y0,x1,y1 = xyxy # Bounding box coordinates
        x0 = max(0,x0) # To avoid negative indexing
        y0 = max(0,y0) # To avoid negative indexing
        x1 = min(img_width,x1) # To avoid out of bounds indexing
        y1 = min(img_height,y1) # To avoid out of bounds indexing

        x0_scaled,y0_scaled,x1_scaled,y1_scaled = x0*scale_x,y0*scale_y,x1*scale_x,y1*scale_y

        # Get the center of the bounding box in the prediction image
        pol_c_x,c_y = (x0+x1)/2,(y0+y1)/2
        head_size = (y1-y0)*HEAD_PROPORTION*HEAD_FACTOR # Calculate the head size of the person in the bounding box
        chest_offset = ((y1-y0)/2-head_size) if c_y+((y1-y0)/2-head_size) < img_height else 0 # Get the offset of the chest from the center of the bounding box
        pol_c_y = c_y-chest_offset
        ix, iy = int(pol_c_x), int(pol_c_y)
        

        # Calculate a bounding box around the chest from which the depth will be calculated.
        low_y,offset_low_y = [iy-offset,iy-offset] if iy-offset>=y0 else [0,iy%offset]
        high_y,offset_high_y = [iy+offset,iy+offset] if iy+offset<y1 else [img_height-1, img_height-iy-1]
        low_x,offset_low_x =  [ix-offset,ix-offset] if ix-offset>=y0 else [0,iy%offset]
        high_x,offset_high_x = [ix+offset,ix+offset] if ix+offset<x1 else [img_width-1,img_width-iy-1]
        # Get the depth of the chest region.
        roi = copy(img0[-1,low_y:high_y,int(x0):int(x1)])
        #poi_img = np.unravel_index(np.argmin(np.abs(np.median(roi)-roi)),roi.shape) # Closest to median approach.

        # Select the point of interest in the bounding box. This point should be on the object and not part of the background.
        # To achieve this we will select the point not with the lowest depth but close to it. This will avoid selecting points on the background and points infront of the object with some success.
        # TODO: Improve this method to select the point of interest. See live_yolo.py end of file for more elaborate methods to achieve this.
        poi_img = np.unravel_index(np.argmin(np.abs(np.quantile(roi,QUANTILE_FACTOR)-roi)),roi.shape)

        
        #poi_scan  = [int((poi_img[0]+int(offset_low_y))*scale_y), int((poi_img[1]+int(offset_low_x))*scale_x)]
        poi_scan  = [int((poi_img[0]+int(offset_low_y))*scale_y), int((poi_img[1]+int(x0))*scale_x)]
        # Calculate teh rotation of the bounding relative to the sensor.
        rot = -float(poi_scan[1])/xyz.shape[1]*2*np.pi

        detection_area.append(([int(x0),low_y,int(x1),high_y],(poi_img[0]+int(low_y),poi_img[1]+int(x0))))
        # From the point cloud with shape (horizon,vertical,3) get the 3D point that the point of interest in the bounding box corresponds to.
        center_x, center_y, center_z = xyz[poi_scan[0],poi_scan[1],:]
        # Vertical distance from the sensor to the point of interest.
        vert_dist = np.sqrt(center_x**2+center_y**2)     
        cone_height = 2*np.tan(azi/2)*vert_dist     #     /|
        height_cov = (y1-y0)/img_height             #    / |
        height = height_cov*cone_height/2           # [c]  |Cone height
                                                    #  | \ |
                                                    #  |  \|
        
        z_offset = height*HEAD_PROPORTION*HEAD_FACTOR # Offset from the center of the bounding box to the head of the person to get the center of the bounding box in 3D space.
        center_x = vert_dist*np.cos(pol_c_x*scale_to_angle+np.pi-np.pi/16) # This reduces the error in the x coordinate of the center of the bounding box.
        center_y = vert_dist*np.sin(pol_c_x*scale_to_angle-np.pi/16)
        center_z = center_z-z_offset

        circle_diam = 2*np.pi*vert_dist # Diameter of the circle that the bounding box is on.
        width_cov = (x1-x0)/img_width # Propotion of the width of the bounding box relative to the image width, reminder image is 360 degrees wide.
        temp_width = width_cov*circle_diam # Width of the bounding box in 3D space.
        [width,breadth] = [temp_width,HARD_WIDTH] if temp_width>HARD_WIDTH else [HARD_WIDTH,temp_width] # Width and breadth of the bounding box in 3D space.

        obj_dim.append([center_x,center_y,center_z,breadth,width,height,rot,0,0])
    pred_dict = {"pred_boxes": np.array(obj_dim), "pred_scores": np.array(pred[0].cpu())[:,4], "pred_labels": np.array(pred[0].cpu())[:,5].astype(np.uint8)}
    return pred_dict,detection_area
def sorted_alphanumeric(data):
    """
    Sort the given iterable in the way that humans expect.
    Args:
        data: An iterable.
    Returns: sorted version of the given iterable.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
def filter_predictions(pred_dict, classes_to_use,pcdet_format:bool=False):
    """
    Filter predictions to only include the classes we want to use.
    Args:
        pred_dict: Dictionary containing the predictions of the model.
        classes_to_use: List of classes to use (Integer indices).
        pcdet_format: If True the predictions are in the format of the PointPillars model, 
                      labels start at 1 index instead of 0 therefor should offset by 1.
    Returns: Filtered predictions correctly formatted as numpy arrays.
    """
    if isinstance(pred_dict["pred_labels"],torch.Tensor):
        pred_dict["pred_labels"] = pred_dict["pred_labels"].cpu().numpy().astype(int)
    if isinstance(pred_dict["pred_boxes"],torch.Tensor):
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].cpu().numpy()
    if isinstance(pred_dict["pred_scores"],torch.Tensor):
        pred_dict["pred_scores"] = pred_dict["pred_scores"].cpu().numpy()
    if classes_to_use is not None and len(pred_dict["pred_labels"]) > 0:
        indices = np.nonzero((sum(pred_dict["pred_labels"]-1==x for x in classes_to_use)))[0].tolist()
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].reshape(pred_dict["pred_boxes"].shape[0],-1)[indices,:]
        pred_dict["pred_labels"] = pred_dict["pred_labels"].reshape(pred_dict["pred_labels"].shape[0],-1)[indices,:]-(1 if pcdet_format else 0)
        pred_dict["pred_scores"] = pred_dict["pred_scores"].reshape(pred_dict["pred_scores"].shape[0],-1)[indices,:]
    elif len(pred_dict["pred_labels"]) > 0:
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].reshape(pred_dict["pred_boxes"].shape[0],-1)
        pred_dict["pred_labels"] = pred_dict["pred_labels"].reshape(pred_dict["pred_labels"].shape[0],-1)-(1 if pcdet_format else 0) # Subtract 1 to get the correct label indices. OBS NOT USED FOR 
        pred_dict["pred_scores"] = pred_dict["pred_scores"].reshape(pred_dict["pred_scores"].shape[0],-1)
    return pred_dict
    
def generate_distance_matrix(pred_dict):
    """
    Generate a distance matrix for the predictions. [i,j]==[j,i] is the distance between the i'th and j'th prediction.
    """
    if isinstance(pred_dict["pred_boxes"],torch.Tensor):
        pred_dict["pred_boxes"] = pred_dict["pred_boxes"].cpu().numpy()
    pred_dict["distance_matrix"] = np.zeros((pred_dict["pred_boxes"].shape[0],pred_dict["pred_boxes"].shape[0]))
    for i in range(pred_dict["pred_boxes"].shape[0]):
        for j in range(pred_dict["pred_boxes"].shape[0]):
            pred_dict["distance_matrix"][i,j] = np.linalg.norm(pred_dict["pred_boxes"][i,:3]-pred_dict["pred_boxes"][j,:3])
    return pred_dict

def get_xyz_from_predictions(pred_dict):
    """
    Get xyz coordinates from predictions.
    """
    raise NotImplementedError

def display_predictions(pred_dict, class_names, logger=None):
    """
    Display predictions.
    args:
        pred_dict: prediction dictionary. "pred_boxes", "pred_labels", "pred_scores"
        class_names: list of class names
        logger: logger
    """
    assert logger is not None, "logger is None. Please provide a logger."
    logger.info(f"Model detected: {len(pred_dict['pred_labels'])} objects.")
    for box,lbls,score in zip(pred_dict['pred_boxes'],pred_dict['pred_labels'],pred_dict['pred_scores']):
        if isinstance(lbls,list):
            lbls = lbls[0]
        if isinstance(lbls,list):
            box = box[0]
        if isinstance(lbls,list):
            score = score[0]
        logger.info(f"lbls: {lbls} score: {score}")
        logger.info(f"\t Prediciton {class_names[lbls]}, id: {lbls} with confidence: {score:.3e}.")
        logger.info(f"\t Box: {box}")
class CSVRecorder:
    """
    Class to record predictions and point clouds to a CSV file.
    """
    def __init__(self, 
                 folder_name=f"csv_folder_{dt.now().strftime('%Y%m%d_%H%M%S')}",
                 main_folder="./lidarCSV",
                 class_names=None,
                 ):
        self.main_folder = main_folder
        self.folder_name = folder_name
        self.class_names = class_names
        self.path = os.path.join(self.main_folder, self.folder_name)
        
        self.labelfile = "label"
        self.cloudfile = "cloud"
        # Create necessary folders.
        if not os.path.exists(self.main_folder):
            os.makedirs(self.main_folder)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.frames = 0
    def process_labels(self,pred_dict):
        """
        Process the labels and return as numpy array with shape (N,9) where N is the number of predictions. (x,y,z,l,w,h,cls,score)
        """
        boxes = np.array(pred_dict["pred_boxes"][:,:6])
        labels = np.array([self.class_names[int(x)] for x in pred_dict["pred_labels"]] if len(pred_dict["pred_labels"]) > 0 else []).reshape(-1,1)
        scores = np.array(pred_dict["pred_scores"]).reshape(-1,1)
    

        labels = np.concatenate((boxes,labels,pred_dict["pred_labels"].reshape(-1,1),scores),axis=1)
        return labels

    def add_frame_file(self, cloud,pred_dict):
        """
        Save point cloud and predictions to file.
        """
        cloud_name = os.path.join(self.path, f"cloud_{self.frames}.csv")
        label_name = os.path.join(self.path, f"label_{self.frames}.csv")
        np.savetxt(cloud_name, cloud, header = "x, y, z, r",delimiter=",")
        np.savetxt(label_name, self.process_labels(pred_dict=pred_dict), header = "x, y, z, rotx, roty, roz, l, w, h, label, label_idx, score",delimiter=",",fmt="%s")
        self.frames += 1
class DataHandler:
    """
    Class to handle recordings and replays of data such that it cloud be played back in real time.
    """
    def __init__(self, destination:str=None,image_name:str="image",cloud_name:str="cloud") -> None:
        self.destination = destination
        self.image_path = os.path.join(self.destination, "images")
        self.cloud_path = os.path.join(self.destination, "clouds")
        self.image_name = image_name
        self.cloud_name = cloud_name
        self.image_id = 0
        self.create_directories()
    def save_image(self, image):
        """
        Save an image to the image folder.
        """
        cv2.imwrite(os.path.join(self.image_path, f"{self.image_name}_{self.image_id}.png"), cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    def save_cloud(self, cloud):
        """
        Save a point cloud to a .npy file for later replay.
        """
        np.save(os.path.join(self.cloud_path,f"{self.cloud_name}_{self.image_id}.npy"), cloud)
    def load_cloud(self, name):
        """
        Load a cloud from a file.
        """
        return np.load(os.path.join(self.cloud_path, name))
    def load_image(self, name):
        """
        Loads an image from the image folder.
        """
        image = cv2.imread(os.path.join(self.image_path, name)).transpose((2,0,1))
        return cv2.cvtColor(cv2.imread(os.path.join(self.image_path, name)),cv2.COLOR_RGB2BGR)
    def __getitem__(self, key):
        """
        Returns the cloud and image with the given name.
        """
        return (self.load_cloud(key), self.load_image(key))
    def __iter__(self):
        """
        Iterates over the names of the clouds and images.
        """
        for cloud in os.listdir(self.cloud_path):
            id = self.extract_id(cloud)
            yield (self.load_cloud(f"{self.cloud_name}_{id}"), self.load_image(f"{self.image_name}_{id}"))
    def __len__(self):
        """
        Returns the number of clouds and images.
        """
        return len(os.listdir(self.cloud_path))
    def extract_id(self, name):
        """
        Extracts the id from the name of a cloud or image.
        """
        return name.split("_")[-1].split(".")[0]
    def create_directories(self):
        """
        Creates the necessary directories.
        """
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        if not os.path.exists(self.cloud_path):
            os.makedirs(self.cloud_path)
class Replayer(DataHandler):
    def __init__(self,source:str,image_name:str="image",cloud_name:str="cloud",image_size:tuple=(640,1280)):
        super().__init__(source,image_name,cloud_name)

        self.reshaper = tv.transforms.Resize(image_size) 
    def prep(self,img0):
        """
        Prepare data from the lidar sensor.
        Args:
            img0: The image that is to be prepared.
        """
        img = copy(img0)
        img = np.ascontiguousarray(img)
        
        img = img.transpose((2,0,1)) # BHWC to BCHW
        img = torch.from_numpy(img).div(255.0)
        img = self.reshaper(img).unsqueeze(0)
        self.image_id += 1
        return img0,img
    def sorted_files(self):
        """
        Sort the files such that when played back they are in the correct order.
        """
        return [sorted(os.listdir(self.cloud_path), key=lambda x: self.extract_id(x)), sorted(os.listdir(self.image_path), key=lambda x: self.extract_id(x))]
    def __iter__(self):
        """
        Iterates over the names of the clouds and images.
        """
        for cloud, image in zip(*self.sorted_files()):
            yield (self.load_cloud(cloud), self.load_image(image))

class OusterRecorder(DataHandler):
    """
    Class to record images and clouds to a folder.
    These images will be consisting of [Signal, Reflectivity, Range] instead of RGB.
    """
    def __init__(self,sensor_ip:str,
                    destination:str,
                    recording_time:float=0,
                    frames_to_record:int=10,
                    lidar_port:int=7504,
                    imu_port:int=7505,
                    interval:float=0.0,
                    ):
        super().__init__(destination)
        self.sensor_ip = sensor_ip
        self.interval = interval
        self.lidar_port = lidar_port
        self.imu_port = imu_port
        self.recording_time = recording_time
        self.frames_to_record = frames_to_record
        self.config = utils_ouster.sensor_config(hostname=self.sensor_ip,lidar_port=self.lidar_port,imu_port=self.imu_port)
        self.streamer = utils_ouster.streaming_object(self.sensor_ip,self.lidar_port)
        self.image_id = 0
    
    def start_recording(self):
        """
        Start recording images represenations and point clouds.
        """
        with closing(self.streamer) as stream:
            for scan in stream:
                if self.image_id >= self.frames_to_record and self.recording_time <= 0:
                    break
                else:
                    print(f"{self.image_id+1}/{self.frames_to_record}")

                img0 = utils_ouster.signal_ref_range(stream,scan)
                pcd = utils_ouster.get_xyz(stream,scan)
                print(f"Image shape: {img0.shape}")
                print(f"Cloud shape: {pcd.shape}")
                self.save_cloud(pcd)
                self.save_image(img0*255)
                self.image_id += 1
                if self.interval > 0:
                    time.sleep(self.interval)

class TimeLogger:
    """
    Class to log time for certain tasks.
    """
    def __init__(self,logger=None,disp_pred=False):
        """
        Args:
            logger: logger object to print the time.
            disp_pred: if True, display predictions.
        """
        super().__init__()
        self.time_dict = {}
        self.time_pd = None
        self.metrics_pd = None
        self.logger = logger
        self.max_timing = 5000
        if disp_pred is not None:
            self.print_log = disp_pred
        else:
            self.print_log = False

    def output_log(self,name:str):
        """
        Output the time taken at each step.
        Args:
            name: name of the step, i.e. pre_process, post_process, etc.
        """
        if self.logger is not None:
            self.logger.info(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/self.time_dict[name]['times'][-1]:.3e} Hz")
        else:
            print(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/self.time_dict[name]['times'][-1]:.3e} Hz")
    def create_metric(self, name: str):
        """
        Create a new metric beloning to the timelogger object.
        Args:
            name: name of the metric.
        """
        self.time_dict[name] = {}
        self.time_dict[name]["times"] = []
        self.time_dict[name]["start"] = 0
        self.time_dict[name]["stop"] = 0   
    def start(self, name: str):
        """
        Start the timer for a metric, should be called at each iteration before the section is started.
        """
        if name not in self.time_dict.keys():
            self.create_metric(name)
            if self.logger is not None:
                self.logger.info(f"{name} had not been initialized, initializing now.")
        self.time_dict[name]["start"] = time.monotonic()
    def stop(self, name: str):
        """
        Stop the timer for a metric, should be called at each iteration after the section is finished.
        """
        self.time_dict[name]["stop"] = time.monotonic()
        if len(self.time_dict[name]["times"]) < self.max_timing:
            self.time_dict[name]["times"].append(self.time_dict[name]["stop"] - self.time_dict[name]["start"])
        if self.print_log:
            self.output_log(name)
        if self.print_log:
            self.output_log(name)
    def log_time(self, name: str, _time: float):
        self.time_dict[name]["times"].append(_time)
    def maximum_time(self, name: str):
        """
        Returns the maximum time taken for a step given the name of that step in the pipeline.
        """
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return max(self.time_dict[name]["times"])
        return 0
    def minimum_time(self, name: str):
        """
        Returns the minimum time taken for a step given the name of that step in the pipeline.
        """
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return min(self.time_dict[name]["times"])
        return 0
    def average_time(self, name: str):
        """
        Returns the average time taken for a step in the pipeline.
        """
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return np.mean(self.time_dict[name]["times"])
        return 0
    def visualize_results(self):
        time_averages = {}
        time_max = {}
        time_min = {}
        self.time_pd = {}
        sum_ave = 0
        keys = len(self.time_dict)
        
        fig,axs = plt.subplots(keys,1)
        for i,key in enumerate(self.time_dict):
           
            if len(self.time_dict[key]["times"])>0:
                axs[i].plot(self.time_dict[key]["times"],label=key)
                axs[i].set_title(key)
                time_averages[key] = np.mean(self.time_dict[key]["times"])
                time_max[key] = self.maximum_time(key)
                time_min[key] = self.minimum_time(key)
                sum_ave += time_averages[key] if key != "Full Pipeline" else 0
            #self.time_pd[key] = self.time_dict[key]["times"]
        plt.show()
        #self.time_pd = pd.DataFrame(self.time_pd)
        
        self.metrics_pd = pd.DataFrame([time_averages,time_max,time_min],index=["average","max","min"])
        if self.logger is not None:
            self.logger.info(f"Table To summarize:\n{self.metrics_pd}\nSum of parts: {sum_ave:.3e} <=> {1/sum_ave:.3e} Hz s\nLoading time: {self.metrics_pd['Full Pipeline']['average']-sum_ave:.3e} s\nTime to spare: {1/20-sum_ave:.3e}\nFrames per second: {1/self.metrics_pd['Full Pipeline']['average']:.3e} Hz")

        else:
            print(f"Table To summarize:\n{self.metrics_pd}")
if __name__ == "__main__":
    # Example code of recording and playing one frame of the point cloud image representation along the point cloud.
    recording_name = "1_person"
    frames_to_record = 10
    sensor_ip = "192.168.200.79"
    destination = f"../dataset/Combined_Dataset/{recording_name}"
    recorder = OusterRecorder(sensor_ip=sensor_ip,destination=destination,frames_to_record=frames_to_record)
    recorder.start_recording()
    replayer = Replayer(source=destination)
    cloud, img0 = next(iter(replayer))
    print("Point Cloud shape: ",cloud.shape)
    cv2.imshow("Image Representation", img0)
    img0 = img0.transpose(2,0,1)
    img0 = torch.from_numpy(img0)
    img0 = replayer.reshaper(img0)
        