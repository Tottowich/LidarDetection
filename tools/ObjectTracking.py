from re import X
import cv2
import time
import os
from sort import *
from copy import copy,deepcopy
import numpy as np
from typing import List, Tuple, Union

MAX_DIST_2D = 200
"""
File created to improve the tracking of objects in a video, both 2D and 3D.
Each object is assigned a unique ID, and the tracking is done using the Kalman Filter.
Idea is to use the two previous frames to predict which object is which.
By using Kalman Filter to predict where each object should be in the current frame based on the two previous frames,
the closest object to the predicted position is assigned the ID of the object in the previous frame.

For every object predicted by detection model.
    # Approach 1.
        - Compare the predicted position of the object to the positions calculated by the Kalman Filter.
        - Assign the ID of the object in the previous frame to the object in the current frame that is closest to the predicted position.
        - If no object is close enough to the predicted position, assign a new ID to the object in the current frame.
            TODO: Specify threshold for how close the object needs to be to the predicted position.
        - Update the position of the object.
    # Approach 2.
        - Compute the Kalman Filter for the current object with all previous objects, call this the "kalman position, short: kalman_pos".
        - 

Also provides a virtual map where the cursor can be used as a represenation of a person standing in the room.
"""
class KalmanFilter(cv2.KalmanFilter):
    def __init__(self,x,y) -> None:
        """
        Args:
            x (int): x position of the object.
            y (int): y position of the object.
        """
        super().__init__(4, 2)
        self.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
class Object2D:
    """
    An represented from bird's eye view with the origin in the top left corner.
    An objcet has an id, x position and y position.
    """
    def __init__(self, id: int, x: int, y: int):
        """
        Args:
            id (int): The id of the object.
            x (int): The x position of the object.
            y (int): The y position of the object.
        """
        self.id = id
        self.x = x 
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.kalman = KalmanFilter(x,y)
    def kalman_pos(self, x:float,y:float) -> Tuple[int,int]:
        """
        Returns the kalman predicted position of the object.
        """
        self.kalman.correct(np.array([x,y],np.float32))
        prediction = self.kalman.predict()
        return int(prediction[0]),int(prediction[1])

    def kalman_difference(self, x:float,y:float) -> float:
        """
        Returns the difference between the predicted position and the actual position.
        Args:
            x (float): The x position of the object.
            y (float): The y position of the object.
        """
        kalman_x, kalman_y = self.kalman_pos(x,y)
        return np.sqrt((kalman_x - x)**2 + (kalman_y - y)**2)
    @property
    def pos(self) -> Tuple[int,int]:
        return self.x,self.y
    @pos.setter
    def pos(self,xy:Union[tuple,np.ndarray]) -> None:
        x,y = xy
        self.prev_x = self.x
        self.prev_y = self.y
        self.x = x
        self.y = y


class ObjectTracker2D:
    def __init__(self) -> None:
        self.objects = []
        self.id = 0
        self.maximum_distance = MAX_DIST_2D
        self.removed_indexes = []
    def add_object(self, x: int, y: int) -> None:
        if len(self.removed_indexes) > 0:
            print("ADDING OBJECT @ ",x,y," WITH ID: ",self.removed_indexes[0])
            self.objects.append(Object2D(self.removed_indexes.pop(0),x,y))
        else:
            print("ADDING OBJECT @ ",x,y," WITH ID: ",self.id)
            self.objects.append(Object2D(self.id,x,y))
            self.id += 1
    def update(self, predictions:list[np.ndarray]) -> list[Object2D]:
        """
        predictions: List of predictions from detection model.
        """
        # If no objects are being tracked, add all objects from the current frame.ew
        if len(self.objects) == 0:
            print("NO OBJECTS BEING TRACKED")
            for prediction in predictions: self.add_object(prediction[0],prediction[1])
            return self.objects
        # If there are objects being tracked, update the positions of the objects.
        # The distance between the predicted position and the position calculated by the Kalman Filter is used to determine which object is which.
        # The object with the smallest distance is assigned the ID of the object in the previous frame.
        # If no object is close enough to the predicted position, assign a new ID to the object in the current frame and add it to the list of objects.
        skipped = []
        for i,prediction in enumerate(predictions):
            objs = self.objects #copy(self.objects)
            print("PREDICTION:",prediction)
            distances = [obj.kalman_difference(prediction[0],prediction[1]) for obj in objs]

            print("Distances: ",distances)
            min_dist = min(distances)
    
            if min_dist < self.maximum_distance:
                obj = self.objects[distances.index(min_dist)]
                obj.pos = np.array([prediction[0],prediction[1]],np.float32)
                print("Updating object with id: ",obj.id)
            else:
                skipped.append(i)
                self.add_object(prediction[0],prediction[1])
        return self.objects if len(self.objects) > 0 else []
            



class VirtualMap:
    """
        A map representing the real world.
        The map is a 2D array of size (width, height).
        Each cell in the map represents a square of size (cell_size, cell_size).
        The map is used to represent the real world.
        The map is updated with the position of objects in the current frame, the mouse cursor is the object.
        The predicted positions of the object are also updated on the map.
        The kalman filter is used to predict the position of the object in the current frame.

        This map is used for testing purposes of the ObjectTracker2D class.
    """
    def __init__(self, width: int=400, height: int=400, cell_size: int=1):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.map = np.zeros((width, height,3), np.uint8)
        self.mp = []
        self.objs = [] # List of objects in the current frame.
        self.measured = [] # List of positions of the objects in the current frame.
        self.mouse_down = [] # List of booleans indicating if the mouse is down or not.
        self.predicted = [] # List of predicted positions of the objects in the current frame.
        self.predicted_mouse_down = [] # List of booleans indicating if the mouse is down or not.
        self.sort_tracker = Sort(max_age=2,min_hits=2,iou_threshold=0.01)
        self.run_test()
        
    def run_test(self):
        """
            Update the map with the position of an object in the current frame.
        """        
        cv2.namedWindow("Map")
        cv2.setMouseCallback("Map", self.on_mouse)
        # self.objs.append(Object2D(1, 0, 0))
        while True:
            preds = self.get_predictions()
            self.paint_canvas()
            display_img = self.map.copy()
            if len(preds):
                boxes = preds[:,:4]
                scores = preds[:,4]
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float64, copy=False)
                trackers = self.sort_tracker.update(dets)
                boxes_track = trackers[:,:-1]
                ids = trackers[:,-1]
                for i,box in enumerate(boxes_track):
                    x1,y1,x2,y2 = box
                    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                        continue
                    xc = int((x1+x2)/2)
                    yc = int((y1+y2)/2)
                    """
                    Note:
                        Functions could be called from here to test various functions of functions reliying
                        on the detection model from here. This would greatly reduce the time needed to stand around the sensor.
                        Cursor could be used to represent a detection by the model.
                    """

                    cv2.putText(display_img,f"Obj_{ids[i]}",(xc,yc),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
                    cv2.circle(display_img,(xc,yc),5,(0,255,0),-1)
            cv2.imshow("Output",display_img)
            k = cv2.waitKey(1) &0xFF
            if k == 27: break # ESC
            if k == 32: self.reset_canvas() # SPACE
         
    def get_predictions(self) -> np.ndarray:
        """
            Get the predicted positions of the objects in the current frame.
        """
        if len(self.measured)>0:
            obj1_pos = self.mp[-1].astype(np.float32)
            obj2_pos = self.mp[-1].astype(np.float32)*[2,2,1,1,1] #self.map.shape[:2]-np.transpose(self.mp[-1][:2])
            return np.stack([obj1_pos,obj2_pos])
            return np.stack([obj1_pos])
        return []
    def paint_canvas(self):
        for i in range(len(self.measured)-1): 
            if self.mouse_down[i] and self.mouse_down[i+1]:
                cv2.line(self.map, self.measured[i], self.measured[i+1], (0,255,0))
        for i in range(len(self.predicted)-1):
            if self.predicted_mouse_down[i] and self.predicted_mouse_down[i+1]:
                cv2.line(self.map, self.predicted[i], self.predicted[i+1], (0,0,255))

    def reset_canvas(self):
        """
            Reset the map.
        """
        self.measured=[]
        self.mouse_down=[]
        self.predicted=[]
        self.predicted_mouse_down=[]
        self.map = np.zeros((self.map.shape), np.uint8)

    def on_mouse(self, event, x, y, flags, param):
        """
            Retrieve the position of the cursor in the current frame. Represents a yolo predictions.
        """
        mp = np.array([x-x/2,y-y/2,x+x/2,y+y/2,1],dtype=np.float32)
        self.mp.append(mp)
        self.measured.append((x,y))
        self.mouse_down.append(flags>0)



if __name__=="__main__":
    import cv2
    import numpy as np
    v_map = VirtualMap()