from re import L
import open3d
import torch
import numpy as np
# NuScenes Dataset
box_colormap_nuscenes = (np.array([
    [118,255,212],# Car Turquoise,aquamarine1
    [138,43,226], # Truck blueviolet
    [255,127,0], # Construction orange
    [255,215,0], # Bus yellow
    [255,106,106], # Trailer p!nk
    [127,127,127], # Barrier gray
    [0,128,128],# Motorcycle Teal
    [255,225,255], # Bicycle pinkish
    [0,255,127], # Pedestrian lime
    [0,245,255], # Trafic cone aquamarine
])/255.0).tolist() # From NuScenes dataset.

class LiveVisualizer:
    """
    Class for drawing bounding boxes on the live visualization.
    It uses the open3d library to draw the bounding boxes.
    """
    def __init__(self,
                    window_name:str='3D Viewer',
                    window_size:tuple=(1920, 1080),
                    point_size:float=1.0,
                    background_color:np.ndarray=np.array((0, 0, 0)),
                    label_colors:list=box_colormap_nuscenes,
                    draw_origin:bool=False,
                    show_labels:bool=False,
                    class_names:list=None,
                    first_cloud:np.ndarray=None,
                    classes_to_visualize:list=None,
                    max_bboxes:int=100,
                ):
        """
        Args:
            window_name (str): window name
            window_size (tuple): window dimensions, (breath, height)
            point_size (float): point size in the visualizer.
            background_color (np.ndarray): background color of the 3D window.
            label_colors (list): list of colors for the labels.
            draw_origin (bool): whether to draw origin.
            show_labels (bool): whether to show labels.
            class_names (list[str]): class names.
            first_cloud (np.ndarray): first cloud to be drawn, used to initialize the window.
            classes_to_visualize (list[int]): classes to be visualized. If None -> visualize all.
            max_bboxes (int): maximum number of bboxes to be drawn, if None => inf, or as many as provided.
        """
        self.label_colors = label_colors
        self.window_name = window_name
        self.window_size = window_size
        self.point_size = point_size
        self.background_color = background_color
        self.draw_origin = draw_origin
        self.first_cloud = first_cloud
        self.class_names = class_names
        self.max_bboxes = max_bboxes # Not truly infinite but almost :)
        self.show_labels = show_labels
        self.frame_id = 0
        self.lidar_points = open3d.geometry.PointCloud()
        self.vis = open3d.visualization.Visualizer()
        self.view_control = self.vis.get_view_control()
        if classes_to_visualize is not None:
            self.classes_to_visualize = [class_names[x] for x in classes_to_visualize]
            print(f"Classes to visualize: {self.classes_to_visualize}")
        else:
            self.classes_to_visualize = class_names
        self.initialize_visual()
        self.pred_boxes = None
        self.pred_boxes = self.initialize_bboxes()
        self.previous_num_bboxes = 0
        self.started = False

    def initialize_visual(self):
        """
        Initialize the visualizer to display the live of point clouds with bounding boxes.
        Create bounding null bounding boxes to be updated later.
        """
        if self.first_cloud is None:
            # Creates a random point cloud inorder to initialize the "camera".
            self.first_cloud = np.random.random((10000, 3))*10
        if isinstance(self.first_cloud, torch.Tensor):
            self.first_cloud = self.first_cloud.cpu().numpy()
        # Generate the window with the first cloud to make sure it is initialized
        
        self.vis.create_window(self.window_name, width=self.window_size[0], height=self.window_size[1])
        self.vis.get_render_option().point_size = self.point_size
        self.vis.get_render_option().background_color = self.background_color
        if self.draw_origin:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            self.vis.add_geometry(axis_pcd)
        if self.first_cloud is not None:
            self.lidar_points.points = open3d.utility.Vector3dVector(self.first_cloud[:, :3])
            self.lidar_points.colors = open3d.utility.Vector3dVector(np.ones((self.first_cloud.shape[0], 3)))
            self.vis.add_geometry(self.lidar_points)
        self.frame_id +=1
        self.vis.poll_events()
        self.vis.update_renderer()
        #self.vis.run()

        
    def update(self,
               points, 
               pred_boxes=None, 
               pred_labels=None, 
               pred_scores=None):
        """
        Update the visualizer with new points and bounding boxes.
        Args:
            points (np.ndarray): points to be visualized (point cloud).
            ref_boxes (np.ndarray): reference bounding boxes.
            ref_labels (np.ndarray): reference labels.
            ref_scores (np.ndarray): reference scores.
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        # Update enviroment points
        self.lidar_points.points = open3d.utility.Vector3dVector(points[:, :3])
        self.lidar_points.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        self.vis.update_geometry(self.lidar_points)
        
        # Update predicted Boxes
        self.update_bboxes(pred_boxes, scores=pred_scores,labels=pred_labels)
        self.vis.poll_events()
        self.vis.update_renderer()
    def update_bboxes(self, bboxes, scores, labels):
        """
        Update the bounding boxes.
        The bounding boxes are updated by altering the previous ones to increase efficiency.
        Args:
            bboxes (np.ndarray): bounding boxes.
            scores (np.ndarray): scores.
            labels (np.ndarray): labels.
        """
        if self.pred_boxes is not None:
            for i in range(self.max_bboxes):
                if bboxes is not None and i < bboxes.shape[0]:
                    if self.class_names[int(labels[i])] in self.classes_to_visualize: 
                        axis_angles = np.array([0, 0, bboxes[i][6] + 1e-10])

                        self.pred_boxes[i].R = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
                        self.pred_boxes[i].center = bboxes[i][:3]
                        self.pred_boxes[i].extent = bboxes[i][3:6]
                        self.pred_boxes[i].color = self.label_colors[int(labels[i])] 
                        
                        self.vis.update_geometry(self.pred_boxes[i])
                        self.vis.poll_events()
                        self.vis.update_renderer()
                elif i < self.previous_num_bboxes:
                    #self.shown_bboxes[i] = self.zero_bounding_box()
                    #print(f"Hiding Box {i}")
                    self.pred_boxes[i].center = [0,0,0]
                    self.pred_boxes[i].extent = [0,0,0]
                    self.pred_boxes[i].color = [0,0,0]
                    self.vis.update_geometry(self.pred_boxes[i])
                    self.vis.poll_events()
                    self.vis.update_renderer()
                else:
                    self.previous_num_bboxes = len(bboxes) if bboxes is not None else 0
                    break
    def zero_bounding_box(self,color=[0, 0, 0]):
        """
        A bounding box with zero-values.
        Used to more effectivly update visualizer.
        """
        axis_angles = np.array([0, 0,  1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(np.array([0,0,0]), rot,np.array([0,0,0]))
        box3d.color = [0.0,0.0,0.0]
        return box3d
    def initialize_bboxes(self):
        """
        Initializes bboxes to make it faster to update (does not continously add geometry, only updates).
        """
        self.pred_boxes = []
        for i in range(self.max_bboxes):
            box3d = self.zero_bounding_box()
            self.pred_boxes.append(box3d)
            self.vis.add_geometry(self.pred_boxes[i])
            self.vis.poll_events()
            self.vis.update_renderer()
        return self.pred_boxes

