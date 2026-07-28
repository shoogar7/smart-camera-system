import cv2 as cv
import time
from camera import Camera
from roi_manager import ROIManager
from motion_detector import MotionDetector
from tracker import Tracker
from config import Config
from events import Event
import numpy as np

# orchestration of main loop
class App:    
    WAITKEY_MS = 30
    MAIN_WINDOW_NAME = 'Normal View'
    DETECTION_WINDOW_NAME = 'Detection View'
    FPS_TXT_POSITION = (100, 90)
    FPS_TXT_SCALE = 0.5
    FPS_TXT_COLOR = (0, 0, 0)
    FPS_TXT_THICKNESS = 2
    ROI_RECT_COLOR = (255, 0, 0)
    ROI_RECT_THICK = 1
    
    GAUSSIAN_KERNEL = (3, 3)
    GAUSSIAN_SIGMA = 0    
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = Camera(config)
        self.roi_manager = ROIManager(config)
        self.motion_detector = MotionDetector(config)
        self.tracker = Tracker(config)
        
        self.start = time.time()
        self.fps = None
        self.running = True
        
    def _calc_fps(self, cur_time: float) -> None:
        self.stop = cur_time
        self.fps = 1 / (self.stop - self.start)
        self.start = self.stop
        
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        blur = cv.GaussianBlur(frame, App.GAUSSIAN_KERNEL, App.GAUSSIAN_SIGMA)
        return cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    
    def _initialize(self) -> np.ndarray:
        cv.namedWindow(App.MAIN_WINDOW_NAME)
        cv.namedWindow(App.DETECTION_WINDOW_NAME)
        cv.setMouseCallback(App.MAIN_WINDOW_NAME, self.roi_manager.on_mouse)
        
        self.camera.open()  
        frame = self.camera.read()
        cur_frame = self._preprocess(frame)

        self.motion_detector.initialize(cur_frame)
        return cur_frame
        
    def _update_camera(self, time: float, frame: np.ndarray) -> None:
        if time - self.camera.last_gone_time > self.camera.gone_check_time:
            self.camera.last_gone_time = time
            self.camera.validate(frame)
    
    def _process_roi(self, frame: np.ndarray, detection_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.roi_manager.has_roi():
            return detection_frame, frame
        else:
            cv.rectangle(frame, self.roi_manager.p1, self.roi_manager.p2, App.ROI_RECT_COLOR, App.ROI_RECT_THICK)
            return self.roi_manager.get_roi(detection_frame, frame)
        
    def _process_motion(self, time: float, frame: np.ndarray, prev_frame: np.ndarray, cur_frame: np.ndarray, det_roi: np.ndarray) -> np.ndarray:
        if time - self.motion_detector.last_motion_time > self.motion_detector.motion_check_time:
            event = self.motion_detector.detect_motion(det_roi)
            
            if event:
                if not self.motion_detector.detect_camera_shift(prev_frame, cur_frame):
                    return self.tracker.tracking(frame)
        if time - self.motion_detector.last_drop_time > self.motion_detector.drop_check_time:
            self.motion_detector.detect_camera_drop(cur_frame)
        return frame
    
    def _render(self, frame: np.ndarray, detection_frame: np.ndarray) -> None:
        cv.imshow(App.DETECTION_WINDOW_NAME, detection_frame)
        cv.putText(frame, f"{frame.shape[1]}x{frame.shape[0]} FPS:{int(self.fps)}", 
                    App.FPS_TXT_POSITION, cv.FONT_HERSHEY_SIMPLEX, App.FPS_TXT_SCALE, 
                    App.FPS_TXT_COLOR, App.FPS_TXT_THICKNESS) 
        cv.imshow(App.MAIN_WINDOW_NAME, frame)
    
    def run(self) -> None:
        cur_frame = self._initialize()
        
        while self.running: 
            prev_frame = cur_frame
            frame = self.camera.read()
            cur_time = time.time()
            
            self._calc_fps(cur_time)

            cur_frame = self._preprocess(frame)

            self._update_camera(cur_time, frame)

            detection_frame = self.motion_detector.preprocess(cur_frame, prev_frame)
            det_roi, _ = self._process_roi(frame, detection_frame)
            frame = self._process_motion(cur_time, frame, prev_frame, cur_frame, det_roi)

            self._render(frame, detection_frame)

            if cv.waitKey(App.WAITKEY_MS) == ord('q'): # in ms - 1/1000s 
                self.camera.release()
                self.running = False