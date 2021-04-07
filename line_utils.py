import numpy as np                                                                
import cv2                                                                                                
import glob
import collections
import matplotlib.pyplot as plt                                                        # 필요한 패키지를 가져온다.
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from globals import ym_per_pix, xm_per_pix


class Line:                                                                             # 차선을 감지하기 위한 객체를 정해준다. 
    
    def __init__(self, buffer_len=10):

        self.detected = False                                                           # line이 마지막일때 값을 변수로 지정한다.
        self.last_fit_pixel = None                                                      # line이 마지막일때 적합한 pixel 값 저장한다.
        self.last_fit_meter = None                                                      # line이 마지막일때 적합한 meter 값 지정한다.
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)                   # deque길이를 buffer_len으로 설정해주고 저장한다. .
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)               # deque길이를 2*buffer_len으로 설정해주고 저장한다.
        self.radius_of_curvature = None                                                 # 곡률의 반지름 지정
        self.all_x = None                                                               # 차선에서 감지 된 모든 좌표(x, y)를 저장한다.
        self.all_y = None

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):  
   
        self.detected = detected                                 

        if clear_buffer:                                           # clear_buffer=True이면
            self.recent_fits_pixel = []                            # self.recent_fits_pixel,meter 를 array 선언해준다.
            self.recent_fits_meter = []                           

        self.last_fit_pixel = new_fit_pixel                        # 계속해서 새로운 값들을 self.last 변수로 넣어준다.
        self.last_fit_meter = new_fit_meter

        self.recent_fits_pixel.append(self.last_fit_pixel)         # self.last 변수들을 self.recent에 넣어준다.
        self.recent_fits_meter.append(self.last_fit_meter)         # 이 과정을 통해 update 시킨다.

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):     
        
        h, w, c = mask.shape                                                     # h,w,c를 mask.shape을 통해 변수로 선언한다.
              
        plot_y = np.linspace(0, h - 1, h)                                        # y축 간격 설정
        coeffs = self.average_fit if average else self.last_fit_pixel            # average = True 이면 self.average
                                                                                 # average = False 이면 self.last

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]   # line 너비 설정
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        pts_left = np.array(list(zip(line_left_side, plot_y)))                   # line의 왼쪽 오른쪽 좌표를 넣어준다.
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))  
        pts = np.vstack([pts_left, pts_right])

        return cv2.fillPoly(mask, [np.int32(pts)], color)                      

    @property
    def average_fit(self):                                # self.average            
        return np.mean(self.recent_fits_pixel, axis=0)

    @property                                         
    def curvature(self):                                  # line의 곡률 반지름 평균
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])
    
    @property                                                                   
    def curvature_meter(self):                            # line의 곡률 반지름 평균
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


def get_fits_by_sliding_windows(birdeye_binary, line_lt, line_rt, n_windows=9, verbose=False):

    height, width = birdeye_binary.shape                                         # 가져온 패키지에서 binary.shape으로 높이와 너비 설정한다.

    histogram = np.sum(birdeye_binary[height//2:-30, :], axis=0)                 # axis를 제외하고 계산하여 구한 값을 histogram에 넣어준다.
                                                                                 
                                                                          
    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255  # 배열을 이어 붙여 이미지로 저장한다..

    midpoint = len(histogram) // 2                                               
    leftx_base = np.argmax(histogram[:midpoint])                                 # midpoint는 왼쪽선의 시작점이 된다.
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint                     # midpoint는 오른쪽선의 시작점이 된다.
    window_height = np.int(height / n_windows)                                   # 높이를 설정한다.

    nonzero = birdeye_binary.nonzero()                                           # 이미지에서 0이 아닌 모든 x, y 좌표를 저장한다.
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    leftx_current = leftx_base                                                   # 왼쪽 시작점부터 현재 위치로 지정한다.
    rightx_current = rightx_base                                                 # 오른쪽 시작점부터 현재 위치로 지정한다.

    margin = 100                                                                 
    minpix = 50                                                                  

    left_lane_inds = []                                                          # 왼쪽, 오른쪽 lane을 array선언한다.
    right_lane_inds = []

    for window in range(n_windows):                                              
        win_y_low = height - (window + 1) * window_height                        # loop를 통해 win 경계를 구한다.
        win_y_high = height - window * window_height                             
        win_xleft_low = leftx_current - margin                            
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)         # loop에서 구한 경계를 통해 이미지 그리기
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)       # window에서 x,y0이 아닌 좌표를 식별한다.
                          & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                           & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)                                    
        right_lane_inds.append(good_right_inds)
                                 
        if len(good_left_inds) > minpix:                             # 식별한 변수에 길이가 minpix보다 크다면 current값에 0이 아닌 좌표들의 평균을 넣어준다.                                     
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)                                                      # 변수들의 배열을 하나로 합쳐준다.
    right_lane_inds = np.concatenate(right_lane_inds)

    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]                  # 왼쪽 및 오른쪽 line 위치를 추출한다.
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True                                                         
    if not list(line_lt.all_x) or not list(line_lt.all_y):                                               # 왼쪽 line에 좌표가 맞지 않다면                    
        left_fit_pixel = line_lt.last_fit_pixel                                                          # 저장했던 pixel과 meter를 새로운 변수에 저장해준다.                                 
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:                                                                                                # 왼쪽 line에 좌표가 맞다면 
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)                                     # 좌표를 2차식에 넣어 pixel과 meter를 구한다..
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):                                               # 오른쪽 line에 좌표가 맞지 않다면
        right_fit_pixel = line_rt.last_fit_pixel                                                         # 저장했던 pixel과 meter를 새로운 변수에 저장해준다.
        right_fit_meter = line_rt.last_fit_meter                                       
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)                                    # 오른쪽 line에 좌표가 맞다면
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)          # 좌표를 2차식에 넣어 pixel과 meter를 구한다.

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)                               #line좌표를 업데이트 시켜준다.
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    ploty = np.linspace(0, height - 1, height)                                                           # linspace를 통해 값들을 넣어준다.
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]           # 구한 ploty를 2차식에 넣어 위치를 구해준다.
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]
    
    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]                          # 레인의 이미지를 나타낸다.
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    if verbose:
        f, ax = plt.subplots(1, 2)                                                                       # 이미지 설정방법을 나타내준다.
        f.set_facecolor('white')
        ax[0].imshow(birdeye_binary, cmap='gray')
        ax[1].imshow(out_img)
        ax[1].plot(left_fitx, ploty, color='yellow')
        ax[1].plot(right_fitx, ploty, color='yellow')
        ax[1].set_xlim(0, 1280)
        ax[1].set_ylim(720, 0)

        plt.show()

    return line_lt, line_rt, out_img


def get_fits_by_previous_fits(birdeye_binary, line_lt, line_rt, verbose=False):  
                                                                             # 이전에 감지된 레인 라인에서 시작하여 현재 프레임에서 레인 라인 검색 속도를 높인다.

    height, width = birdeye_binary.shape

    left_fit_pixel = line_lt.last_fit_pixel                                  # 이전에 감지한 pixel을 가져온다.
    right_fit_pixel = line_rt.last_fit_pixel

    nonzero = birdeye_binary.nonzero()                                       # 패키지에서 birdeye를 가져와서 0이 아닌값을 이진분류한다.
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100            
    left_lane_inds = (                                                              
    (nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) & (
    nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
    right_lane_inds = (
    (nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) & (
    nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))

    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]                              # 왼쪽 및 오른쪽 라인 픽셀 위치를 추출한다.
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    ploty = np.linspace(0, height - 1, height)                                                       # 그래프 그리기위해서 linspace로 일정한 간격 값 설정
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255                                    
    window_img = np.zeros_like(img_fit)

    img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]                      # 왼쪽 및 오른쪽 line pixel의 색상을 나타낸다.
    img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])             # 검색 창 영역을 설명하는 다각형 생성
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))                                # x 및 y 포인트를 cv2.fillPoly ()에 사용할 수있는 형식으로 다시 캐스팅한다.
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))                                  # 이미지에 차선을 그린다.
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)

    if verbose:                                                                                      # 이미지 설정 방법을 의미한다.
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return line_lt, line_rt, img_fit


def draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state):                    # 주행 가능한 차선 영역과 감지된 차선을 왜곡되지 않은 프레임에 그린다.

    height, width, _ = img_undistorted.shape

    left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
    right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

    ploty = np.linspace(0, height - 1, height)                                                       # 그래프 그리기위해서 linspace로 일정한 간격 값 설정
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)                                       # 원래 프레임에 녹색 다각형으로 도로 그리기
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))                            # 원래의 이미지 공간으로 가져온다.

    blend_onto_road = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)

    line_warp = np.zeros_like(img_undistorted)                                                       # 강조하기 위해 별도로 실선을 그려서 표시
    line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
    line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    lines_mask = blend_onto_road.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]

    blend_onto_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.)

    return blend_onto_road


if __name__ == '__main__':

    line_lt, line_rt = Line(buffer_len=10), Line(buffer_len=10)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    for test_img in glob.glob('test_images/*.jpg'):                                                  # 테스트 이미지에 결과를 표시

        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary = binarize(img_undistorted, verbose=False)

        img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

        line_lt, line_rt, img_out = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=7, verbose=True)

