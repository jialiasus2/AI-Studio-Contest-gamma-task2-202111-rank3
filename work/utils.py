import os
import cv2
import datetime

def read_image(img_path):
    img = cv2.imread(img_path)
    img[:,:,::-1] = img
    return img

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).squeeze()

def save_image(img_path, img):
    cv2.imwrite(img_path, img[:,:,::-1])

def get_all_paths(folder, ext=None):
    paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
    if ext:
        paths = [p for p in paths if p.endswith(ext)]
    return paths

def get_datetime():
    return datetime.datetime.now().strftime('%m%d_%H%M')


class LogWriter():
    def __init__(self, log_path=None, print_out=True, log_time=True, clear_pre_content=True):
        self.log_path = log_path
        self.print_out = print_out
        self.log_time = log_time
        if log_path and clear_pre_content:
            os.system('rm -f '+log_path)

    def __call__(self, log_content):
        log_content = self.add_info(log_content)
        if self.print_out:
            self.Print(log_content)
        if self.log_path:
            self.Save(self.log_path, log_content)

    def add_info(self, log_content):
        if self.log_time:
            log_content = ('LOG[%s]:  '%get_datetime())+log_content
        return log_content

    def Print(self, log_content):
        print(log_content)

    def Save(self, log_path, log_content):
        if log_path:
            with open(log_path, 'a') as f:
                f.write(log_content+'\n')
