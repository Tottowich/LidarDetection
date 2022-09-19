"""
Testing various models to detect the center of the person inside the bounding box.
TODO: Further develop this method.
"""

from re import A
import torch.nn as nn
import torchvision as tv
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import time
import torchvision
from copy import copy
class ModelTemplate(nn.Module):
    """
    Template for the models to be used in the CenterDetector classes.
    """
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @property
    def params(self):
        return sum(p.numel() for p in self.parameters())
    @property
    def params_grad(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def warmup(self,imgsz):
        x = torch.randn(imgsz).to(self.device)
        return self(x)
class CenterFast(ModelTemplate):
    """
    Linear model to detect center of object.
    """
    def __init__(self,img_shape) -> None:
        super().__init__()
        input_shape = img_shape[-3]*img_shape[-2]*img_shape[-1]
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(input_shape, 10)
        self.lin2 = nn.Linear(10, 2)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()
        self.norm1 = nn.LayerNorm(10)
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Sigmoid(),
            )
        
     
    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        return x
class CenterConv(ModelTemplate):
    def __init__(self,image_shape=(1,3,360,360),depth=3):
        super(CenterConv,self).__init__()
        self.seq = []
        in_channels = image_shape[-3]
        out_channels = in_channels
        kernel_size = [32,16,3] + [3]*(depth)
        self.resizer = torchvision.transforms.Resize((image_shape[-2],image_shape[-1]))

        #test_input = torch.zeros(1,3,image_shape,image_shape)
        for i in range(depth):
            self.seq.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size[i],padding=0))
            #test_input = self.seq[i](test_input)
            if i%2 == 0:
                in_channels = out_channels
                out_channels = out_channels*2
            else:
                in_channels = out_channels

            self.seq.append(nn.BatchNorm2d(in_channels))
            self.seq.append(nn.ReLU())
            self.seq.append(nn.MaxPool2d(kernel_size=2,stride=2))
            self.seq.append(nn.Dropout(0.25))
        self.seq.append(nn.Flatten())
        #test_input = self.seq[-1](test_input)
        self.seq.append(nn.Linear(in_features=7776,out_features=2))
        self.seq.append(nn.Sigmoid())
        self.seq = nn.Sequential(*self.seq)
    def forward(self,x):
        x = self.seq(x)
        return x
    def preprocess(self,x):
        x = self.resizer(x)
        return x

class CenterResNet(ModelTemplate):
    """
    Convolutional model to detect center of object.
    """
    def __init__(self,img_shape,max_det=100,device=None) -> None:
        super().__init__()
        
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        #print(backbone)
        self.max_det = max_det
        self.detections = 0
        self.in_imgs = torch.empty(size=(max_det,*img_shape[1:])).to(self.device)
        self.resizer = torchvision.transforms.Resize((img_shape[-2],img_shape[-1]))
        self.boxes = torch.empty(size=(max_det,4)).to(self.device)
        self.seq = nn.Sequential(
            *list(backbone.children())[:-1],
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid(),
            )
        self.float()
    def forward(self, x):
        x = self.seq(x)
        return x
    def center_predicitons(self,img,boxes,predict_img_shape):
        if len(boxes) != 0:
            self.imgs_from_bboxes(boxes,img[0],predict_img_shape)
            with torch.cuda.amp.autocast(),torch.no_grad():
                output = self.forward(self.in_imgs[:self.detections])
        else:
            output = torch.empty(size=(0,2)).to(self.device)
        return output
    def imgs_from_bboxes(self,bboxs,img,predict_shape):
        """
        Get images from bounding boxes.
        Args:
            bboxs (np.ndarray): bounding boxes.
            img (np.ndarray): image.
        Returns:
            np.ndarray: images.
        """
        resizer = torchvision.transforms.Resize((360,360))
        i = 0
        for bbox in bboxs:
            x1,y1,x2,y2 = bbox[:4]
            if x2-x1<=0:
                continue
            img_ = img[:,int(y1):int(y2),int(x1):int(x2)]
            img_ = resizer(torch.unsqueeze(img_,dim=0))
            self.in_imgs[i] = img_
            i+=1
        self.detections = i
    def load_model(self,model_path):
        self.load_state_dict(torch.load(model_path)) 
    # @property
    # def params(self):
    #     return sum(p.numel() for p in self.parameters())
def generate_rand_boxes(num_boxes, img_shape):
    """
    Generate random bounding boxes.
    Args:
        num_boxes (int): number of bounding boxes.
        img_shape (tuple): image shape.
    Returns:
        np.ndarray: bounding boxes.
    """

    upper_x,upper_y = img_shape[1], img_shape[0]

    x1 = np.random.randint(0,upper_x,size=num_boxes)
    y1 = np.random.randint(0,upper_y,size=num_boxes)
    x2 = np.array([np.random.randint(x1[i]+1 if x1[i]<upper_x else x1[i],upper_x) for i in range(num_boxes)])
    y2 = np.array([np.random.randint(y1[i]+1 if y1[i]<upper_x else y1[i],upper_y) for i in range(num_boxes)])
    return torch.tensor(np.stack([x1,y1,x2,y2],axis=1)).cuda()
def imgs_from_bboxes(bboxs,img,predict_shape):
    """
    Get images from bounding boxes.
    Args:
        bboxs (np.ndarray): bounding boxes.
        img (np.ndarray): image.
    Returns:
        np.ndarray: images.
    """
    imgs = torch.empty(size=(len(bboxs),*predict_shape[1:])).cuda()
    upper_x,upper_y = predict_shape[-1], predict_shape[-2]
    resizer = torchvision.transforms.Resize((360,360))
    for i,bbox in enumerate(bboxs):
        x1,y1,x2,y2 = bbox[:4]
        img_ = img[:,y1:y2,x1:x2]
        img_ = resizer(torch.unsqueeze(img_,dim=0))
        imgs[i] = img_
    return imgs

if __name__ == "__main__":
    torch.cuda.empty_cache()
    img_shape = [640,1280]
    predict_img_shape = [1,3,360,360]
    num_boxes = 50
    img = torch.randn(size=(1,3,img_shape[0],img_shape[1])).cuda()

    boxes = generate_rand_boxes(num_boxes, img.shape[-2:])
    print(f"Box shape: {boxes.shape}")
    model = CenterResNet(predict_img_shape).cuda()
    in_img = imgs_from_bboxes(boxes,img[0],predict_img_shape)
    print(f"in_img shape: {in_img.shape}")
    #print(model)
    _ = model(copy(in_img)).cpu()#warm up
    start = time.monotonic()
    in_img = imgs_from_bboxes(boxes,img[0],predict_img_shape)
    b = model(in_img)
    print(f"Output shape: {b.shape}")
    print(f"Time taken: {time.monotonic()-start:.9e}")
    print(f"Number of parameters: {model.params}")
    #print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
