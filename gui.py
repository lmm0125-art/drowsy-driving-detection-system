import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
import torch
import threading
from pathlib import Path
import os
from ultralytics import YOLO
import sys
import cv2
import numpy as np
import time
import pyttsx3

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk  # 位置不同可能报错

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

warnings.filterwarnings("ignore")

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class WinGUI(Tk):
    """界面布局类"""

    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_frame_control_frame = self.__tk_frame_control_frame(self)
        self.tk_button_upload_image = self.__tk_button_upload_image(self.tk_frame_control_frame)
        self.tk_button_video = self.__tk_button_video(self.tk_frame_control_frame)
        self.tk_button_cam = self.__tk_button_cam(self.tk_frame_control_frame)
        self.tk_label_show_image = self.__tk_label_show_image(self)
        self.tk_button_close = self.__tk_button_close(self.tk_frame_control_frame)
        self.tk_frame_result_frame = self.__tk_frame_result_frame(self)
        self.tk_label_time = self.__tk_label_time(self.tk_frame_result_frame)
        self.tk_label_time_show = self.__tk_label_time_show(self.tk_frame_result_frame)
        self.tk_label_count = self.__tk_label_count(self.tk_frame_result_frame)
        self.tk_label_count_show = self.__tk_label_count_show(self.tk_frame_result_frame)
        self.tk_label_conf = self.__tk_label_conf(self.tk_frame_result_frame)
        self.tk_label_conf_show = self.__tk_label_conf_show(self.tk_frame_result_frame)
        self.tk_label_class = self.__tk_label_class(self.tk_frame_result_frame)
        self.tk_label_class_show = self.__tk_label_class_show(self.tk_frame_result_frame)
        self.tk_label_xmin = self.__tk_label_xmin(self.tk_frame_result_frame)
        self.tk_label_xmin_show = self.__tk_label_xmin_show(self.tk_frame_result_frame)
        self.tk_label_ymin = self.__tk_label_ymin(self.tk_frame_result_frame)
        self.tk_label_ymin_show = self.__tk_label_ymin_show(self.tk_frame_result_frame)
        self.tk_label_xmax = self.__tk_label_xmax(self.tk_frame_result_frame)
        self.tk_label_xmax_show = self.__tk_label_xmax_show(self.tk_frame_result_frame)
        self.tk_label_ymax = self.__tk_label_ymax(self.tk_frame_result_frame)
        self.tk_label_ymax_show = self.__tk_label_ymax_show(self.tk_frame_result_frame)
        self.tk_label_select_obj = self.__tk_label_select_obj(self.tk_frame_result_frame)
        self.tk_select_box_select_obj_show = self.__tk_select_box_select_obj_show(self.tk_frame_result_frame)
        self.tk_label_driving_flag = self.__tk_label_driving_flag(self.tk_frame_result_frame)
        self.tk_label_driving_show = self.__tk_label_driving_show(self.tk_frame_result_frame)

    def __win(self):
        self.title("基于深度学习的疲劳驾驶检测系统")
        # 设置窗口大小、居中
        width = 560
        height = 710
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)

        self.resizable(width=False, height=False)

    def scrollbar_autohide(self, vbar, hbar, widget):
        """自动隐藏滚动条"""

        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)

        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def __tk_frame_control_frame(self, parent):
        frame = Frame(parent, borderwidth=2, relief="solid", )
        frame.place(x=60, y=10, width=445, height=50)
        return frame

    def __tk_button_upload_image(self, parent):
        btn = Button(parent, text="上传图片", takefocus=False, )
        btn.place(x=15, y=8, width=79, height=30)
        return btn

    def __tk_button_video(self, parent):
        btn = Button(parent, text="上传视频", takefocus=False, )
        btn.place(x=110, y=8, width=84, height=30)
        return btn

    def __tk_button_cam(self, parent):
        btn = Button(parent, text="打开摄像头", takefocus=False, )
        btn.place(x=207, y=8, width=90, height=30)
        return btn

    def __tk_button_close(self, parent):
        btn = Button(parent, text="关闭视频或摄像", takefocus=False, )
        btn.place(x=312, y=8, width=112, height=30)
        return btn

    def __tk_label_show_image(self, parent):
        label = Label(parent, text="检测窗口", anchor="center", background="white", font=("Arial", 16))
        label.place(x=60, y=68, width=440, height=440)
        return label

    def __tk_frame_result_frame(self, parent):
        frame = Frame(parent, borderwidth=2, relief="solid", )
        frame.place(x=60, y=515, width=445, height=165)
        return frame

    def __tk_label_time(self, parent):
        label = Label(parent, text="用时：", anchor="center", background="white")
        label.place(x=10, y=8, width=70, height=30)
        return label

    def __tk_label_time_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=83, y=8, width=80, height=30)
        return label

    def __tk_label_count(self, parent):
        label = Label(parent, text="目标数目：", anchor="center", background="white")
        label.place(x=178, y=8, width=70, height=30)
        return label

    def __tk_label_count_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=250, y=8, width=90, height=30)
        return label

    def __tk_label_conf(self, parent):
        label = Label(parent, text="置信度：", anchor="center", background="white")
        label.place(x=10, y=45, width=70, height=30)
        return label

    def __tk_label_conf_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=83, y=45, width=80, height=30)
        return label

    def __tk_label_class(self, parent):
        label = Label(parent, text="类别：", anchor="center", background="white")
        label.place(x=178, y=45, width=70, height=30)
        return label

    def __tk_label_class_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=250, y=45, width=90, height=30)
        return label

    def __tk_label_xmin(self, parent):
        label = Label(parent, text="xmin:", anchor="center", background="white")
        label.place(x=10, y=82, width=45, height=30)
        return label

    def __tk_label_xmin_show(self, parent):
        label = Label(parent, text="111", anchor="center", background="white")
        label.place(x=58, y=82, width=45, height=30)
        return label

    def __tk_label_ymin(self, parent):
        label = Label(parent, text="ymin:", anchor="center", background="white")
        label.place(x=115, y=82, width=45, height=30)
        return label

    def __tk_label_ymin_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=163, y=82, width=45, height=30)
        return label

    def __tk_label_xmax(self, parent):
        label = Label(parent, text="xmax:", anchor="center", background="white")
        label.place(x=220, y=82, width=45, height=30)
        return label

    def __tk_label_xmax_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=268, y=82, width=45, height=30)
        return label

    def __tk_label_ymax(self, parent):
        label = Label(parent, text="ymax:", anchor="center", background="white")
        label.place(x=325, y=82, width=45, height=30)
        return label

    def __tk_label_ymax_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=373, y=82, width=60, height=30)
        return label

    def __tk_label_select_obj(self, parent):
        label = Label(parent, text="目标选择：", anchor="center", background="white")
        label.place(x=355, y=8, width=80, height=30)
        return label

    def __tk_select_box_select_obj_show(self, parent):
        cb = Combobox(parent, state="readonly", background="white")
        cb['values'] = ("")
        cb.place(x=355, y=42, width=80, height=30)
        return cb

    def __tk_label_driving_flag(self, parent):
        label = Label(parent, text="驾驶员状态：", anchor="center", background="white")
        label.place(x=10, y=120, width=120, height=31)
        return label

    def __tk_label_driving_show(self, parent):
        label = Label(parent, text="正常", anchor="center", borderwidth=1, relief="groove", background="white")
        label.place(x=136, y=120, width=300, height=30)
        return label


class Win(WinGUI):
    """界面事件绑定类"""

    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)

    def __event_bind(self):
        self.tk_button_upload_image.bind('<Button-1>', self.ctl.upload_image)
        self.tk_button_video.bind('<Button-1>', self.ctl.upload_video)
        self.tk_button_cam.bind('<Button-1>', self.ctl.open_cam)
        self.tk_button_close.bind('<Button-1>', self.ctl.close_video_cam)
        self.tk_select_box_select_obj_show.bind('<<ComboboxSelected>>', self.ctl.combox_change)
        pass

    def __style_config(self):
        pass


class Controller:
    """逻辑处理类，图片检测、视频检测、摄像头检测的逻辑处理"""
    ui: Win

    def __init__(self):
        # todo 修改模型权重路径
        model_path = os.path.join(BASE_DIR, 'runs/detect/train/weights/best.pt')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.stop = False
        self.file_path = ""
        # 图片读取进程
        self.output_size = 640
        self.img2predict = ""
        # 语音相关初始化
        self.engine_lock = threading.Lock()
        self.last_warning_time = 0
        self.warning_cooldown = 3  # 语音提醒冷却时间(秒)

        # 更新视频图像
        self.cap = None
        self.is_camera_open = False
        self.stopEvent = threading.Event()
        self.frame = None
        self.is_playing = False
        self.is_capturing = False
        self.roll = 0
        self.roll_eye = 0
        self.roll_mouth = 0
        # 加载检测模型
        self.model = YOLO(model_path, task='detect')
        self.model(np.zeros((48, 48, 3)))  # 预先加载推理模型
        self.delay = 15  # 视频帧延迟时间

    def speak_warning(self):
        """在独立线程中播放警告语音"""

        def _speak():
            try:
                with self.engine_lock:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)
                    engine.say("疲劳驾驶，注意安全")
                    engine.runAndWait()
                    engine.stop()
            except Exception as e:
                print(f"语音播放错误: {e}")

        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_warning_time > self.warning_cooldown:
            self.last_warning_time = current_time
            threading.Thread(target=_speak, daemon=True).start()

    def init(self, ui):
        """
        得到UI实例，对组件进行初始化配置
        """
        self.ui = ui
        # TODO 组件初始化 赋值操作
        self.ui.tk_button_close.config(state=DISABLED)

    def display_image(self, file_path):
        """显示图片"""
        # 打开图片并调整大小
        image = Image.open(file_path)
        image = image.resize((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.ui.tk_label_show_image.config(image=photo)
        self.ui.tk_label_show_image.image = photo

    def upload_image(self, evt):
        """图像目标检测"""
        self.ui.tk_button_close.config(state=DISABLED)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.NORMAL)

        # 打开文件对话框选择图片文件
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if self.image_path:
            """检测图片"""
            org_path = self.image_path
            # 目标检测
            t1 = time.time()
            results = self.model(org_path)[0]
            names = results.names
            t2 = time.time()
            self.ui.tk_label_time_show.config(text='{:.3f} s'.format(t2 - t1))
            now_img = results.plot()
            resize_scale = self.output_size / now_img.shape[0]
            im0 = cv2.resize(now_img, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite(os.path.join(BASE_DIR, 'images/tmp/single_result.jpg'), im0)
            self.display_image(os.path.join(BASE_DIR, "images/tmp/single_result.jpg"))

            location_list = results.boxes.xyxy.tolist()
            location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            cls_list = [int(i) for i in cls_list]
            conf_list = results.boxes.conf.tolist()
            conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

            total_nums = len(location_list)
            self.ui.tk_label_count_show.config(text="{}".format(total_nums))

            choose_list = ['全部']
            target_names = [names[id] + '_' + str(index) for index, id in enumerate(cls_list)]
            detected_names = [names[index] for index in cls_list]
            if 'closed_eye' in detected_names or 'open_mouth' in detected_names:
                self.ui.tk_label_driving_show.config(text="疲劳驾驶", background="red", foreground="black")
                self.speak_warning()
            else:
                self.ui.tk_label_driving_show.config(text='正常驾驶', background="white", foreground="black")
            choose_list = choose_list + target_names
            self.ui.tk_select_box_select_obj_show.config(values=choose_list)

            self.results = results
            self.names = names
            self.cls_list = cls_list
            self.conf_list = conf_list
            self.location_list = location_list

            if total_nums >= 1:
                self.ui.tk_label_class_show.config(text="{}".format(names[self.cls_list[0]]))
                self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
                #   设置坐标位置值
                #   默认显示第一个目标框坐标
                self.ui.tk_label_xmin_show.config(text="{}".format(self.location_list[0][0]))
                self.ui.tk_label_ymin_show.config(text="{}".format(self.location_list[0][1]))
                self.ui.tk_label_xmax_show.config(text="{}".format(self.location_list[0][2]))
                self.ui.tk_label_ymax_show.config(text="{}".format(self.location_list[0][3]))
            else:
                self.ui.tk_label_class_show.config(text=" ")
                self.ui.tk_label_conf_show.config(text=" ")
                self.ui.tk_label_xmin_show.config(text=" ")
                self.ui.tk_label_ymin_show.config(text=" ")
                self.ui.tk_label_xmax_show.config(text=" ")
                self.ui.tk_label_ymax_show.config(text=" ")

    def combox_change(self, evt):
        com_text = self.ui.tk_select_box_select_obj_show.get()
        if com_text == '全部':
            cur_box = self.location_list
            cur_img = self.results.plot()
            self.ui.tk_label_class_show.config(text="{}".format(self.names[self.cls_list[0]]))
            self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            cur_img = self.results[index].plot()
            self.ui.tk_label_class_show.config(text="{}".format(self.names[self.cls_list[index]]))
            self.ui.tk_label_conf_show.config(text="{}".format(str(self.conf_list[index])))

        # 设置坐标位置值
        self.ui.tk_label_xmin_show.config(text="{}".format(str(cur_box[0][0])))
        self.ui.tk_label_ymin_show.config(text="{}".format(str(cur_box[0][1])))
        self.ui.tk_label_xmax_show.config(text="{}".format(str(cur_box[0][2])))
        self.ui.tk_label_ymax_show.config(text="{}".format(str(cur_box[0][3])))

        cv2.imwrite(os.path.join(BASE_DIR, "images/tmp/single_result.jpg"), cur_img)
        self.display_image(os.path.join(BASE_DIR, "images/tmp/single_result.jpg"))

    def upload_video(self, evt):
        """视频目标检测"""
        self.ui.tk_button_upload_image.config(state=tk.DISABLED)
        self.ui.tk_button_video.config(state=tk.DISABLED)
        self.ui.tk_button_cam.config(state=tk.DISABLED)
        self.ui.tk_button_close.config(state=tk.NORMAL)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.DISABLED)
        self.ui.tk_label_driving_show.config(text='正常驾驶', background="white", foreground="black")

        """开启视频文件检测事件"""
        self.video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_playing = True
            self.update_frame()

    def update_frame(self):
        if self.is_playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 目标检测
                t1 = time.time()
                results = self.model(frame)[0]
                names = results.names
                t2 = time.time()
                self.ui.tk_label_time_show.config(text='{:.3f} s'.format(t2 - t1))
                now_img = results.plot()
                cv2.imwrite(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"), now_img)
                self.display_image(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"))

                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                conf_list = results.boxes.conf.tolist()
                conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

                total_nums = len(location_list)
                self.ui.tk_label_count_show.config(text="{}".format(total_nums))

                detected_names = [names[index] for index in cls_list]
                if 'closed_eye' in detected_names:
                    self.roll_eye = self.roll_eye + 1
                if 'open_mouth' in detected_names:
                    self.roll_mouth = self.roll_mouth + 1
                self.roll += 1
                # 计算疲劳驾驶
                if self.roll == 10:
                    perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                    if perclos > 0.36:
                        self.ui.tk_label_driving_show.config(text="疲劳驾驶", background="red", foreground="black")
                        self.speak_warning()
                    else:
                        self.ui.tk_label_driving_show.config(text='正常驾驶', background="white", foreground="black")
                    self.roll = 0
                    self.roll_eye = 0
                    self.roll_mouth = 0

                self.results = results
                self.names = names
                self.cls_list = cls_list
                self.conf_list = conf_list
                self.location_list = location_list

                if total_nums >= 1:
                    self.ui.tk_label_class_show.config(text="{}".format(names[self.cls_list[0]]))
                    self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
                    #   设置坐标位置值
                    #   默认显示第一个目标框坐标
                    self.ui.tk_label_xmin_show.config(text="{}".format(self.location_list[0][0]))
                    self.ui.tk_label_ymin_show.config(text="{}".format(self.location_list[0][1]))
                    self.ui.tk_label_xmax_show.config(text="{}".format(self.location_list[0][2]))
                    self.ui.tk_label_ymax_show.config(text="{}".format(self.location_list[0][3]))
                else:
                    self.ui.tk_label_class_show.config(text=" ")
                    self.ui.tk_label_conf_show.config(text=" ")
                    self.ui.tk_label_xmin_show.config(text=" ")
                    self.ui.tk_label_ymin_show.config(text=" ")
                    self.ui.tk_label_xmax_show.config(text=" ")
                    self.ui.tk_label_ymax_show.config(text=" ")
            else:
                self.is_playing = False
                self.cap.release()
        if self.is_playing:
            self.ui.after(self.delay, self.update_frame)

    def open_cam(self, evt):
        """打开摄像头目标检测"""
        self.ui.tk_button_upload_image.config(state=tk.DISABLED)
        self.ui.tk_button_video.config(state=tk.DISABLED)
        self.ui.tk_button_cam.config(state=tk.DISABLED)
        self.ui.tk_button_close.config(state=tk.NORMAL)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.DISABLED)
        self.ui.tk_label_driving_show.config(text='正常驾驶', background="white", foreground="black")

        """开启摄像头检测事件"""
        if not self.is_capturing:
            self.is_capturing = True
            self.cap = cv2.VideoCapture(0)
            self.update_camera_frame()

    def update_camera_frame(self):
        if self.is_capturing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 目标检测
                t1 = time.time()
                results = self.model(frame)[0]
                names = results.names
                t2 = time.time()
                self.ui.tk_label_time_show.config(text='{:.3f} s'.format(t2 - t1))
                now_img = results.plot()
                cv2.imwrite(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"), now_img)
                self.display_image(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"))

                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                conf_list = results.boxes.conf.tolist()
                conf_list = ['%.2f %%' % (each * 100) for each in conf_list]

                total_nums = len(location_list)
                self.ui.tk_label_count_show.config(text="{}".format(total_nums))

                detected_names = [names[index] for index in cls_list]
                if 'closed_eye' in detected_names:
                    self.roll_eye = self.roll_eye + 1
                if 'open_mouth' in detected_names:
                    self.roll_mouth = self.roll_mouth + 1
                self.roll += 1
                # 计算疲劳驾驶
                if self.roll == 10:
                    perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                    if perclos > 0.36:
                        self.ui.tk_label_driving_show.config(text="疲劳驾驶", background="red", foreground="black")
                        self.speak_warning()
                    else:
                        self.ui.tk_label_driving_show.config(text='正常驾驶', background="white", foreground="black")
                    self.roll = 0
                    self.roll_eye = 0
                    self.roll_mouth = 0

                self.results = results
                self.names = names
                self.cls_list = cls_list
                self.conf_list = conf_list
                self.location_list = location_list

                if total_nums >= 1:
                    self.ui.tk_label_class_show.config(text="{}".format(names[self.cls_list[0]]))
                    self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
                    #   设置坐标位置值
                    #   默认显示第一个目标框坐标
                    self.ui.tk_label_xmin_show.config(text="{}".format(self.location_list[0][0]))
                    self.ui.tk_label_ymin_show.config(text="{}".format(self.location_list[0][1]))
                    self.ui.tk_label_xmax_show.config(text="{}".format(self.location_list[0][2]))
                    self.ui.tk_label_ymax_show.config(text="{}".format(self.location_list[0][3]))
                else:
                    self.ui.tk_label_class_show.config(text=" ")
                    self.ui.tk_label_conf_show.config(text=" ")
                    self.ui.tk_label_xmin_show.config(text=" ")
                    self.ui.tk_label_ymin_show.config(text=" ")
                    self.ui.tk_label_xmax_show.config(text=" ")
                    self.ui.tk_label_ymax_show.config(text=" ")
                self.ui.after(30, self.update_camera_frame)
            else:
                self.close_video_cam(None)

    def close_video_cam(self, evt):
        """关闭摄像头和视屏检测事件"""
        self.ui.tk_button_upload_image.config(state=tk.NORMAL)
        self.ui.tk_button_video.config(state=tk.NORMAL)
        self.ui.tk_button_cam.config(state=tk.NORMAL)
        self.ui.tk_button_close.config(state=tk.DISABLED)
        """窗口关闭事件"""
        if self.is_playing or self.is_capturing:
            self.cap.release()
            self.is_playing = False
            self.is_capturing = False


if __name__ == "__main__":
    controller = Controller()
    app = Win(controller)
    app.mainloop()
