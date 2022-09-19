#from encodings import normalize_encoding
import socket
import json
import torch
import numpy as np
from mlsocket import MLSocket
import time
import threading
class Transmitter():
    """
    Trasmitter using  mlsocket and regular udp.
    Can transmitt point clouds via mlsocket and bounding box predicitions via udp.

    """
    def __init__(self, 
                    reciever_ip:str, 
                    reciever_port:int,
                    ml_reciever_ip:str=None,
                    ml_reciever_port:int=None,
                    classes_to_send=None
                    ):
        """
        Args:
            reciever_ip: IP of the reciever.
            reciever_port: Port of the reciever.
            ml_reciever_ip: IP of the reciever for the point cloud.
            ml_reciever_port: Port of the reciever for the point cloud.
        """
        super().__init__()
        self.reciever_ip = reciever_ip
        self.reciever_port = reciever_port
        self.ml_reciever_ip = ml_reciever_ip
        self.ml_reciever_port = ml_reciever_port

        self.classes_to_send = classes_to_send
        self.pcd = None
        self.s_ml = MLSocket()
        self.s_udp = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.pred_dict = None
        self.started_udp = False
        self.started_ml = False


    def send_dict(self):
        """
        Send a dictionary.
        Args:
            None. The dictionary is stored in self.pred_dict.
        
        """
        if len(self.pred_dict["pred_labels"]) > 0:
            if isinstance(self.pred_dict["pred_labels"],torch.Tensor):
                self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].cpu().numpy()
            if isinstance(self.pred_dict["pred_boxes"],torch.Tensor):
                self.pred_dict["pred_boxes"] = self.pred_dict["pred_boxes"].cpu().numpy()
            if isinstance(self.pred_dict["pred_scores"],torch.Tensor):
                self.pred_dict["pred_scores"] = self.pred_dict["pred_scores"].cpu().numpy()
            self.pred_dict["pred_boxes"] = self.pred_dict["pred_boxes"].reshape(self.pred_dict["pred_boxes"].shape[0],-1).tolist()
            self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].reshape(self.pred_dict["pred_labels"].shape[0],-1).tolist()
            self.pred_dict["pred_scores"] = self.pred_dict["pred_scores"].reshape(self.pred_dict["pred_scores"].shape[0],-1).tolist()
        else:
            self.pred_dict["pred_boxes"] = self.pred_dict["pred_boxes"].tolist()
            self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].tolist()
            self.pred_dict["pred_scores"] = self.pred_dict["pred_scores"].tolist()
        predictions_encoded = json.dumps(self.pred_dict).encode('utf-8')
        try:
            self.s_udp.sendto(predictions_encoded, (self.reciever_ip, self.reciever_port))
        except:
            print(f"Could not send to {self.reciever_ip}")
        self.pred_dict = None

    def send_pcd(self):
        """
        Send the point cloud and the predictions.
        """
        if isinstance(self.pred_dict["pred_labels"],torch.Tensor):
            self.pred_dict["pred_labels"] = self.pred_dict["pred_labels"].cpu().numpy()
        pred_send = np.concatenate((self.pred_dict["pred_boxes"],self.pred_dict["pred_labels"],self.pred_dict["pred_scores"]),axis=1)
        self.s_ml.send(self.pcd)
        self.s_ml.send(pred_send)
        # After sending the data, we reset the variables.
        self.pred_dict = None
        self.pcd = None
    def start_transmit_ml(self):
        """
        Start the transmission.
        """
        try: 
            self.s_ml.connect((self.ml_reciever_ip,self.ml_reciever_port))
            self.started_ml = True
            return True
        except:
            return False
    def stop_transmit_ml(self):
        if self.started_ml:
            self.started_ml = False
    def _check_connection(self):
        """
        Check if the connection is alive.
        """
        try:
            self.sock.sendto(b'ping', (self.reciever_ip, self.reciever_port))
            data, addr = self.sock.recvfrom(1024)
            if data.decode('utf-8') == 'pong':
                return True
            else:
                return False
        except:
            return False
    def check_connection(self):
        """
        Check if the connection is alive.
        """
        try:
            self.connect((self.reciever_ip, self.reciever_port))
        except:
            return False
    def start_transmit_udp(self):
        """
        Start the transmission.
        """
        try: 
            self.s_udp.connect((self.reciever_ip,self.reciever_port))
            self.started_udp = True
            return True
        except:
            return False
    def stop_transmit_udp(self):
        """
        Stop the transmission.
        """
        self.started_udp = False
    def transmit_udp(self):
        while self.started_udp:
            if self.pred_dict is not None:
                self.send_dict()
        
    def transmit_ml_socket(self):
        """
        Transmit the data via MLSocket.
        TODO: Implement working version on the reciever side. This is currently working if being sent from python to python.
        TODO: Implement a way to send the pointclouds regardless of the receiver specifications.
        """
        print("Started Transmitter to {}:{}".format(self.reciever_ip, self.reciever_port))
        while self.started_ml:
            if self.pcd is not None and self.pred_dict is not None:
                print("SENDING DATA")
                self.s.send(self.pcd)
                try:
                    for key, value in (self.pred_dict.items()):
                        if isinstance(value,torch.Tensor):
                            self.pred_dict[key] = value.detach().cpu().numpy()
                    if self.classes_to_send is not None:
                        indices = self.pred_dict["pred_labels"] in self.classes_to_send
                    pred_arr = np.concatenate([np.array(self.pred_dict["pred_boxes"][:,:6]),np.array(self.pred_dict["pred_labels"]),np.array(self.pred_dict["pred_scores"])],axis=1)
                    self.s.send(pred_arr)
                    print(f"SENT : {pred_arr.shape}")
                except:
                    pass
                self.pcd = None
                self.pred_dict = None
def test_multiport_transmitter(host_udp:str=None,
                               port_udp:int=None,
                               host_ml:str=None,
                               port_ml:int=None):
    """
    Test the multiport transmitter.
    Args:
        host_udp: The host (ip-adress) of the udp socket.
        port_udp: The port of the udp socket.
        host_ml: The host (ip-adress) of the ml socket.
        port_ml: The port of the ml socket.
    """
    transmitter = Transmitter(host_udp,port_udp,host_ml,port_ml)
    transmitter.start_transmit_udp()
    transmitter.start_transmit_ml()
    # Generate random point cloud, bounding box, labels and scores:
    detections = 10
    pcd = np.random.random((100000,3))
    bboxes = np.random.random((detections,3))
    lbls = np.random.randint(0,10,(detections,1))
    score = np.random.randint(0,10,(detections,1))
    # dictionary of bounding boxes, labels and scores:
    pred_dict = {"pred_boxes":bboxes,"pred_labels":lbls,"pred_scores":score}
    transmitter.pred_dict = pred_dict
    # Send the data:
    transmitter.pcd = pcd
    transmitter.send_pcd()
    transmitter.send_dict()

    # Wait for the data to be received:
    transmitter.stop_transmit_udp()
    transmitter.stop_transmit_ml()
    transmitter.close()
    
if __name__ == "__main__":
    # Test the transmitter:
    test_multiport_transmitter("192.168.200.103",7002,"192.168.200.103",1234)



