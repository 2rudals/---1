

#darknet open source



from ctypes import *                                           # C언어에 사용되는 타입을 파이썬에서 사용하는 타입으로 변환시키는 함수를 가져온다.
import math                                                    # math에 존재하는 내부함수를 이용하여 계산하기 위해서 가져온다.
import random                                                  # random 값을 가져오기 위함
import os


                                                               # class들을 ctype을 이용해 파이썬에 쓸 수 있도록 변환 시켜준다.

class BOX(Structure):                                          # Bounding BOX를 예측하기위한 클래스
    _fields_ = [("x", c_float),                                # x, y는 상자 중심의 좌표
                ("y", c_float),
                ("w", c_float),                                # w는 너비
                ("h", c_float)]                                # h는 높이
 

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),                                 # Bounding BOX 이미지를 찾을때 사용하는 경계상자
                ("classes", c_int),                            # 클래스의 수
                ("prob", POINTER(c_float)),                    # 확률을 의미한다.
                ("mask", POINTER(c_float)),                    # tag에 해당되는 anchor를 나타낸다.
                ("objectness", c_float),                       # confidence threshold
                ("sort_class", c_int),                         # class 정렬
                ("uc", POINTER(c_float)),                      # uncertainty(불확실성)을 나타낸다.
                ("points", c_int),                             # 객체의 포인트를 나타낸다.
                ("embeddings", POINTER(c_float)),              # 카테고리를 시각화 시켜준다.
                ("embedding_size", c_int),                     # embedding의 크기를 나타낸다.
                ("sim", c_float),                              # 유사도를 나타낸다.
                ("track_id", c_int)]                           # 객체 추적을 의미한다.

class DETNUMPAIR(Structure):                                   # detection의 포인터를 가져온다.
    _fields_ = [("num", c_int),                      
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),                                  # w는 너비  
                ("h", c_int),                                  # h는 높이
                ("c", c_int),                                  # c는 중심점
                ("data", POINTER(c_float))]                    # 이미지 데이터를 나타낸다.


class METADATA(Structure):                            
    _fields_ = [("classes", c_int),        
                ("names", POINTER(c_char_p))]                  # 객체의 수와 이름


def network_width(net):                                        # 네트워크의 너비 width은 노드의 개수로 정의된다.
    return lib.network_width(net)


def network_height(net):                                       # 네트워크의 깊이 height는 계층의 수로 정의된다. 
    return lib.network_height(net)


def bbox2points(bbox):                                         # bounding box 이미지 감지시 나타나는 상자

 
    x, y, w, h = bbox                                          # bbox의 x,y,w,h 값을 나타낸다.
    xmin = int(round(x - (w / 2)))                             # x의 최소 코너를 x와 w를 round 함수로 계산한다.
    xmax = int(round(x + (w / 2)))                             # x의 최대 코너를 x와 w를 round 함수로 계산한다.
    ymin = int(round(y - (h / 2)))                             # y의 최소 코너를 y와 h를 round 함수로 계산한다. 
    ymax = int(round(y + (h / 2)))                             # y의 최대 코너를 y와 h를 round 함수로 계산한다..
    return xmin, ymin, xmax, ymax                     


def class_colors(names):                                       # name에 임의의 RGB 색상 하나를 사용하여 딕셔너리화 시킨다.

'
    return {name: (                                            # name에 BGR 색상을 정해준다.
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


def load_network(config_file, data_file, weights, batch_size=1):                        # batch_size=1 은 한번에 한장의 사진을 처리한다.
     
    network = load_net_custom(
        config_file.encode("ascii"),
        weights.encode("ascii"), 0, batch_size)                                          # 입력요소를 아스키코드 Byte type으로 가져온다.
    metadata = load_meta(data_file.encode("ascii"))                                      # METADATA를 아스키코드 Byte type으로 가져온다.
    class_names = [metadata.names[i].decode("ascii") for i in range(metadata.classes)]   # metadata의 객체를 str형식으로 바꿔준다.
    colors = class_colors(class_names)                                                   # claas_colors 함수를 사용하여 BGR색상을 지정한다.
    return network, class_names, colors                        


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:                 # detections 내부에 존재하는 label, confidence, bbox를 for loop를 통해 반복시켜준다.
        x, y, w, h = bbox                                      # bbox입력값을 x,y,w,h로 지정해준다..
        if coordinates:                                                                                 # coordinates=True 이면
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})"
                  .format(label, confidence, x, y, w, h))                                               # 객체 이름과 신뢰도 bounding box를 나타내준다.
        else:
            print("{}: {}%".format(label, confidence))         # coodinates=False 이면 label과 confidence만 나타내고 Bounding BOX를 만들지 않는다..


def draw_boxes(detections, image, colors):                                         # 감지된 이미지에 색깔있는 Bounding BOX를 씌운다.
    import cv2                                                                     
    for label, confidence, bbox in detections:                                     # detections 내부에 존재하는 label, confidence, bbox를 for loop를 통해 반복시켜준다.
        left, top, right, bottom = bbox2points(bbox)                               # bbox2points 함수를 통해 반환된 값을 left,top,right,bottom으로 지정한다.
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)       # Bounding BOX를 만들어준다.
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),         # BOX에 text를 입력한다.
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image                                                                   
   

def decode_detection(detections):
    decoded = []                                                                   # decoded변수를 array 선언.                                                          
    for label, confidence, bbox in detections:                                     # detections 내부에 존재하는 label, confidence, bbox를 for loop를 통해 반복시켜준다.
        confidence = str(round(confidence * 100, 2))                               # confidence는 round 함수를 통해 구해준다.
        decoded.append((str(label), confidence, bbox))                             # decoded array에 label,confidence,bbox를 추가시켜준다.
    return decoded                                                                 


def remove_negatives(detections, class_names, num):                                # deteciton확률이 0이면 모든 클래스를 제거한다.

    predictions = []                                                               # predictions변수를 array 선언
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:                                        # detection 내부 prob값이 0보다 크다면
                bbox = detections[j].bbox                                          # Bounding BOX를 지정해준다.
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))        # predictions에 name, prob, bbox를 추가한다.
    return predictions                                                             


def detect_image(network, class_names, image, thresh=.5, hier_thresh=.5, nms=.45): # confidence가 가장 높은 목록과 해당 bbox를 반환시킨다.

    pnum = pointer(c_int(0))                                                       
    predict_image(network, image)                                                  # 예측 이미지
    detections = get_network_boxes(network, image.w, image.h, hresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)
    predictions = remove_negatives(detections, class_names, num)                   # remove_negatives 함수에 detections, class_names, num을 입력요소로 넣어 prediction 변수로 지정해준다.
    predictions = decode_detection(predictions)                                    # detection(predictions)을 str형으로 변환시켜준다.
    free_detections(detections, num)                                               
    return sorted(predictions, key=lambda x: x[1])                                 


#  lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#  lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True                                                                      
if os.name == "nt"                                                                 # 만약os 모듈이 windows에서 실행된다면
    cwd = os.path.dirname(__file__)                                                # 파일의 폴더 경로를 반환한다.
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']                            # 폴더 경로와 환경변수를 더해서 경로를 만든다.
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")                              # file내 path들을 묶어 하나의 경로로 만든다.
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")                      
    envKeys = list()                                                               # envKeys변수는 리스트화 시켜준다.
    for k, v in os.environ.items():                                                # os.environ모듈에 Key와 Value를 for loop를 통해 반복시킨다.
        envKeys.append(k)                                                          
    try:                                                                           # try블록에서 오류 발생시 except블록에서 대신 수행한다.
        try:                                                                       
            tmp = os.environ["FORCE_CPU"].lower()                                  # 환경 변수를 소문자로 가져온다.
            if tmp in ["1", "true", "yes", "on"]:                                  # 만약 1,true,yes,on안에 환경변수가 있다면
                raise ValueError("ForceCPU")                                       # ValueError를 나타낸다. 
            else:                                                                  
                print("Flag value {} not forcing CPU mode".format(tmp))            
        except KeyError:                                                           # try블록에서 KeyError발생시 수행한다.
          # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:                                  
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:                    # os.environ[문자]를 정수형으로 만든 값이 0보다 작다면
                    raise ValueError("ForceCPU")                                   # ValueError가 발생한다.
            try:                                                                    
                global DARKNET_FORCE_CPU                                           # 전역변수를 설정해준다.
                if DARKNET_FORCE_CPU:                                              # 만약 전역변수=True 이면
                    raise ValueError("ForceCPU")                                   # ValueError를 나타낸다.
            except NameError as cpu_error:                                         # NameError의 오류 메시지를 알려준다. 
                print(cpu_error)                                                    
        if not os.path.exists(winGPUdll):                                          # winGPUdll가 경로에 존재하지 않는다면
            raise ValueError("NoDLL")                                              # ValueError를 나타낸다.
        lib = CDLL(winGPUdll, RTLD_GLOBAL)                                         # CDLL을 통해 DLL을 로드시키고 lib에 지정해준다.
    except (KeyError, ValueError):                                                 # try에서 KeyError,ValueError오류 발생시 수행한다.
        hasGPU = False                                           
        if os.path.exists(winNoGPUdll):                                         
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)                                   
            print("Notice: CPU-only mode")                                         
        else:                                                                        
            lib = CDLL(winGPUdll, RTLD_GLOBAL)                                    
            print("Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(winNoGPUdll))            
else:#os 모듈이 windows에서 실행되지 않는다면
    lib = CDLL(os.path.join(os.environ.get('DARKNET_PATH', './'),"libdarknet.so"), RTLD_GLOBAL)
    # CDLL을 통해 DLL(환경에서 얻어낸 경로를 하나의 경로로 만든)을 로드시키고 lib에 지정해준다.
    
lib.network_width.argtypes = [c_void_p]              # 이 밑에 부분은 ctype 함수를 이용해 C언어 데이터를 파이썬에 맞는 type으로 바꾸는 코드이다.
lib.network_width.restype = c_int                                                  
lib.network_height.argtypes = [c_void_p]                                           
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes                                  
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]                                  

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)
