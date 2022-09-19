import argparse
import sys
from  pathlib import Path
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
import yaml
def parse_config():
    """
    Parse the configuration file.
    """
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './TrainedModels/object/object.onnx', help='model path(s)')

    # parser.add_argument('--source', type=str, default=None, help='model path(s)')

    parser.add_argument('--ip', type=str, default=None, help='ip address')
    parser.add_argument('--port', type=int, default=None, help='port')

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=448, help='inference size h,w')
    parser.add_argument('--data', type=str, default=ROOT / "./TrainedModels/Object/data.yaml", help='(optional) dataset.yaml path')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--conf_thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels) visualizations')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference instead of FP32 (default)')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--auto', action='store_true', help='auto size using the model')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--OU_ip', type=str, default=None, help='specify the ip of the sensor')
    parser.add_argument('--name', type=str, default=None, help='specify the name of the sensor')
    parser.add_argument('--UE5_ip', type=str, default=None, help='specify the ip of the UE5 machine')
    parser.add_argument('--TD_ip', type=str, default="192.168.200.103", help='specify the ip of the TD machine')

    parser.add_argument('--udp_port', type=int, default=7502, help='specify the udp port of the sensor')
    parser.add_argument('--tcp_port', type=int, default=7503, help='specify the tcp port of the sensor')
    parser.add_argument('--TD_port', type=int, default=7002, help='specify the port of the TD machine')
    parser.add_argument('--UE5_port', type=int, default=7000, help='specify the port of the UE5 machine')
    parser.add_argument('--time', type=int, default=-1
    , help='specify the time to stream data from a sensor')
    if sys.version_info >= (3,9):
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_time_log', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_csv', action=argparse.BooleanOptionalAction)
        parser.add_argument('--log_time', action=argparse.BooleanOptionalAction)
        parser.add_argument('--disp_pred', action=argparse.BooleanOptionalAction)
        parser.add_argument('--disp_time', action=argparse.BooleanOptionalAction)
        parser.add_argument('--transmit', action=argparse.BooleanOptionalAction)
        parser.add_argument('--webcam', action=argparse.BooleanOptionalAction)
        parser.add_argument('--log_all', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if isinstance(args.imgsz, list) and len(args.imgsz) == 1:
        args.imgsz = args.imgsz[0]
    if isinstance(args.imgsz,int):
        args.imgsz = (args.imgsz, args.imgsz)
    with open(args.data,'r') as f:
        try:
            data_config = yaml.safe_load(f)
        except:
            raise ValueError(f"Invalid data config file: {args.data}")

    return args,data_config
"""
Argument descriptions:
    --weights: Path to the model weights.
    --ip: IP address of the sensor.
    --port: Port of the sensor.
    --imgsz: Size of the input image that the model will be run on.
    --data: Path to the data config file, yaml file containing list of target names.
    --max_det: Maximum number of detections per image.
    --conf_thres: Confidence threshold for the model.
    --iou_thres: IoU threshold for the model.
    --line_thickness: Thickness of the bounding box when visualized.
    --hide_labels: Hide the labels when visualized.
    --hide_conf: Hide the confidence when visualized.
    --half: Use FP16 half-precision inference instead of FP32 on model inference. ONLY AVAILABLE ON GPU.
    --dnn: Use OpenCV DNN for ONNX inference.
    --device: Device to run the model on, "cuda","0","cpu","cuda:int".
    --ckpt: Path to the model checkpoint. Not necessary
    --auto: Automatically set the input image size to the model size.
    --classes: Filter by class, only show detections of the specified class. list of integers.
    --OU_ip: IP address of the sensor.
    --name: Name of the sensor.
    --UE5_ip: IP address of the UE5 machine.
    --TD_ip: IP address of the TD machine.
    --udp_port: UDP port of the sensor.
    --tcp_port: TCP port of the sensor.
    --TD_port: Port of the TD machine.
    --UE5_port: Port of the UE5 machine.
    --time: Time to stream data from the sensor.
    --augment: Use augmented inference. VERY SLOW but more accurate.
    --agnostic-nms: Use class-agnostic NMS. Treats all classes as equal, removes overlaping detections.
    --visualize: Visualize the detections.
    --log_time: Log the time of the detections.
    --save_time_log: Save the time log of the detections.
    --save_csv: Save the detections in a csv file.
    --disp_pred: Display the predictions. @ Each frame.
    --disp_time: Display the time of the detections. @ Each frame.
"""