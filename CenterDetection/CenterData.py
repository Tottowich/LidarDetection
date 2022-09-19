"""
Custom image labeling tool to speed up the process of labeling images for regression model.
Also includes classes to load the dataset and create a dataloader for training.
"""

import argparse
import os, sys
import glob
from pathlib import Path
from telnetlib import SE
import time
from tkinter import OFF
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from copy import copy
import open3d
from queue import Queue
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import math
import cv2
import random
pd.options.display.float_format = '{:,.4e}'.format
import logging
import re
class DatasetCreator:
    """
    Class to correct for extracting bounding boxes from images and storing them in a new folder.
    """
    def __init__(self,dest_path,source_path,image_folder,label_folder,image_id = 0):
        if source_path == dest_path:
            raise ValueError("Source and destination path can't be the same")
        self.dest_path = dest_path
        
        self.source_path = source_path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_id = image_id
        self.image_size = (320,320)
    def create_dest_folder(self):
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
        if not os.path.exists(self.dest_path+self.image_folder):
            os.makedirs(self.dest_path+self.image_folder)
        if not os.path.exists(self.dest_path+self.label_folder):
            os.makedirs(self.dest_path+self.label_folder)
    def xywh2xyxy(self,bbox):
        # Convert bounding box to center point and width/height
        x, y, w, h = bbox
        xmin = int(x - w / 2)
        xmax = int(x + w / 2)
        ymin = int(y - h / 2)
        ymax = int(y + h / 2)
        return xmin, ymin, xmax, ymax
    def extract_bounding_boxes(self,image,labels):
        print(f"Image shape: {image.shape}")
        width = image.shape[1]
        height = image.shape[0]
        for label in labels:
            if not isinstance(label,np.ndarray):
                continue
            xmin,ymin,xmax,ymax = self.xywh2xyxy(label[1:]*[width,height,width,height])
            #print(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
            if xmin<0:
                xmin = 0
            if ymin<0:
                ymin = 0
            if xmax>width:
                xmax = width
            if ymax>height:
                ymax = height
            bounding_image = image[ymin:ymax,xmin:xmax,:]
            #print(f"bounding image shape: {bounding_image.shape}")
            bounding_image = cv2.resize(bounding_image,self.image_size)
            cv2.imwrite(self.dest_path+self.image_folder+"/"+str(self.image_id)+".jpg",cv2.cvtColor(bounding_image,cv2.COLOR_BGR2RGB))
            self.image_id += 1
    def create_dataset(self):
        self.create_dest_folder()
        for image_file in os.listdir(self.source_path+self.image_folder).sort():
            if image_file.endswith(".jpg"):
                image_name = image_file.split("/")[-1]
                label_file = self.source_path+self.label_folder+"/"+image_file[:-4]+".txt"
                if os.path.exists(label_file):
                    image = cv2.cvtColor(cv2.imread(self.source_path+self.image_folder+image_name),cv2.COLOR_BGR2RGB)
                    label = np.loadtxt(label_file)
                    if len(label)>0:
                        self.extract_bounding_boxes(image,label)

                    
                    
                else:
                    print("No label file found for image: "+image_name)
                    print(label_file)
class DataLabeler:
    """
    Label xy data from image. Store to txt file with same name as image file.
    """
    def __init__(self,source_path,image_folder,label_folder,start_id = 0):
        self.source_path = source_path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_path = self.source_path+self.image_folder
        self.label_path = self.source_path+self.label_folder
        self.current_image = ""
        self.image_size = (320,320)
        self.image_id = start_id
        self.next = False
    def label_data(self):
        # Upon mouse press, add a point to the list of points
        def on_mouse(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Point added: {x/self.image_size[0]}, {y/self.image_size[1]}")
                np.savetxt(self.label_path+"/"+self.current_image[:-4]+".txt",np.array([x/self.image_size[0],y/self.image_size[1]]),header="x,y")           
                self.next = True
            return True
        # Register the callback function on "mouse event"
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_mouse)
        # Load the images sequentially
        for image_file in os.listdir(self.source_path+self.image_folder):
            if image_file.endswith(".jpg"):
            # Show the image and wait for a keystroke in the window
                
                self.current_image = image_file
                print(f"Image: {image_file} @ {self.image_path+image_file}")
                image = cv2.imread(self.image_path+image_file)
                while not self.next:
                    cv2.imshow("image", image)
                    key = cv2.waitKey(1)
                    # If the 'q' key is pressed, exit the program
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        pass
                    if key == ord("n"):
                        self.next = True
                self.next = False

class CenterPointDataset(Dataset):
    """
    A dataset which reads from a folder of images and a folder of labels.
    Where the images are stored as .jpg and the labels are stored as .txt.
    """
    def __init__(self,source_path,image_folder,label_folder,image_size=(320,320)):
        super().__init__()
        self.source_path = source_path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_path = self.source_path+self.image_folder
        self.label_path = self.source_path+self.label_folder
        self.image_size = image_size
        self.dataset = []
    def __getitem__(self, index):
        return self.dataset[index]
    def create_dataset(self):
        corrupted = 0
        for image_file in os.listdir(self.source_path+self.image_folder):
            if image_file.endswith(".jpg"):
                image_name = image_file.split("/")[-1]
                label_file = self.source_path+self.label_folder+"/"+image_file[:-4]+".txt"
                if os.path.exists(label_file):
                    image = torch.tensor(cv2.cvtColor(cv2.imread(self.source_path+self.image_folder+image_name),cv2.COLOR_BGR2RGB),dtype=torch.float).permute(2,1,0)
                    label = torch.tensor(np.loadtxt(label_file),dtype=torch.float)
                    if len(label)>0:
                        self.dataset.append((image,label))
                else:
                    corrupted += 1
        print(f"{corrupted} corrupted images found")
        print(f"{len(self.dataset)} images found")
    def shuffle_dataset(self):
        random.shuffle(self.dataset)
    def create_dataloader(self,batch_size=1,shuffle=True,num_workers=1):
        self.create_dataset()
        train_set, test_set = self.split_dataset()
        trainloader = DataLoader(train_set,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        testloader = DataLoader(test_set,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        print(f"{len(trainloader)} training batches")
        print(f"{len(testloader)} testing batches")
        return trainloader, testloader
    
    def split_dataset(self,ratio=0.9):
        self.shuffle_dataset()
        split = int(ratio*len(self.dataset))
        train_set = self.dataset[:split]
        test_set = self.dataset[split:]
        return train_set, test_set
if __name__ == "__main__":
    dest_path = "../dataset/CenterData/"
    source_path = "../Xr-Synthesize-SRR-3/train/"
    image_folder = "images/"
    label_folder = "labels/"
    image_id = 0
    dc = DatasetCreator(dest_path,source_path,image_folder,label_folder,image_id)
    dc.create_dataset()
    dl = DataLabeler(dest_path,image_folder,label_folder)
    dl.label_data()
    # CP_data = CenterPointDataset(dest_path,image_folder,label_folder)
    # train_loader,test_loader = CP_data.create_dataloader(batch_size=16,shuffle=True,num_workers=1)
    # (image,label) = next(iter(train_loader))