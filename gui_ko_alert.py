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
try:
    import winsound
except Exception:
    winsound = None

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
        self.title("딥러닝 기반 졸음운전 감지 시스템")
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
        btn = Button(parent, text="이미지 업로드", takefocus=False, )
        btn.place(x=15, y=8, width=79, height=30)
        return btn

    def __tk_button_video(self, parent):
        btn = Button(parent, text="동영상 업로드", takefocus=False, )
        btn.place(x=110, y=8, width=84, height=30)
        return btn

    def __tk_button_cam(self, parent):
        btn = Button(parent, text="카메라 열기", takefocus=False, )
        btn.place(x=207, y=8, width=90, height=30)
        return btn

    def __tk_button_close(self, parent):
        btn = Button(parent, text="영상/카메라 종료", takefocus=False, )
        btn.place(x=312, y=8, width=112, height=30)
        return btn

    def __tk_label_show_image(self, parent):
        label = Label(parent, text="감지 화면", anchor="center", background="white", font=("Arial", 16))
        label.place(x=60, y=68, width=440, height=440)
        return label

    def __tk_frame_result_frame(self, parent):
        frame = Frame(parent, borderwidth=2, relief="solid", )
        frame.place(x=60, y=515, width=445, height=165)
        return frame

    def __tk_label_time(self, parent):
        label = Label(parent, text="소요시간:", anchor="center", background="white")
        label.place(x=10, y=8, width=70, height=30)
        return label

    def __tk_label_time_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=83, y=8, width=80, height=30)
        return label

    def __tk_label_count(self, parent):
        label = Label(parent, text="객체 수:", anchor="center", background="white")
        label.place(x=178, y=8, width=70, height=30)
        return label

    def __tk_label_count_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=250, y=8, width=90, height=30)
        return label

    def __tk_label_conf(self, parent):
        label = Label(parent, text="신뢰도:", anchor="center", background="white")
        label.place(x=10, y=45, width=70, height=30)
        return label

    def __tk_label_conf_show(self, parent):
        label = Label(parent, text=" ", anchor="center", background="white")
        label.place(x=83, y=45, width=80, height=30)
        return label

    def __tk_label_class(self, parent):
        label = Label(parent, text="분류:", anchor="center", background="white")
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
        label = Label(parent, text="객체 선택:", anchor="center", background="white")
        label.place(x=355, y=8, width=80, height=30)
        return label

    def __tk_select_box_select_obj_show(self, parent):
        cb = Combobox(parent, state="readonly", background="white")
        cb['values'] = ("")
        cb.place(x=355, y=42, width=80, height=30)
        return cb

    def __tk_label_driving_flag(self, parent):
        label = Label(parent, text="운전자 상태:", anchor="center", background="white")
        label.place(x=10, y=120, width=120, height=31)
        return label

    def __tk_label_driving_show(self, parent):
        label = Label(parent, text="정상", anchor="center", borderwidth=1, relief="groove", background="white")
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
        self.warning_cooldown = 3  # 음성/팝업 알림 쿨다운(초)
        self.last_alert_level = None

        # 피로 단계별 알림 자료 경로
        # 사용자가 직접 파일을 넣으면 팝업 이미지/효과음이 자동으로 바뀝니다.
        self.alert_assets = {
            "suspected": {
                "ko": "의심 피로",
                "message": "의심 피로 상태입니다. 주의해서 운전하세요.",
                "image": os.path.join(BASE_DIR, "images/alerts/suspected.png"),
                "sound": os.path.join(BASE_DIR, "sounds/suspected.wav"),
                "color": "#FFD966"
            },
            "mild": {
                "ko": "경도 피로",
                "message": "경도 피로가 감지되었습니다. 잠시 휴식하세요.",
                "image": os.path.join(BASE_DIR, "images/alerts/mild.png"),
                "sound": os.path.join(BASE_DIR, "sounds/mild.wav"),
                "color": "#F4B183"
            },
            "severe": {
                "ko": "중증 피로",
                "message": "중증 피로 상태입니다. 즉시 휴식하세요!",
                "image": os.path.join(BASE_DIR, "images/alerts/severe.png"),
                "sound": os.path.join(BASE_DIR, "sounds/severe.wav"),
                "color": "#FF0000"
            }
        }
        self.alert_images = {}

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

    def play_alert_sound(self, level):
        """피로 단계별 경고음 또는 TTS 음성을 재생"""
        info = self.alert_assets[level]
        sound_path = info["sound"]

        def _play():
            try:
                # wav 파일이 있으면 먼저 재생
                if winsound and os.path.exists(sound_path):
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    return

                # wav 파일이 없으면 pyttsx3로 한국어 문구 재생
                with self.engine_lock:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)
                    engine.say(info["message"])
                    engine.runAndWait()
                    engine.stop()
            except Exception as e:
                print(f"알림음 재생 오류: {e}")

        threading.Thread(target=_play, daemon=True).start()

    def show_alert_popup(self, level):
        """피로 단계별 이미지 팝업을 표시"""
        info = self.alert_assets[level]

        popup = tk.Toplevel(self.ui)
        popup.title(info["ko"])
        popup.geometry("360x420")
        popup.resizable(False, False)
        popup.attributes("-topmost", True)

        title_label = tk.Label(
            popup,
            text=info["ko"],
            font=("Arial", 22, "bold"),
            bg=info["color"],
            fg="black"
        )
        title_label.pack(fill="x", pady=(0, 10))

        image_path = info["image"]
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img = img.resize((260, 220))
                photo = ImageTk.PhotoImage(img)
                self.alert_images[level] = photo
                img_label = tk.Label(popup, image=photo)
                img_label.pack(pady=8)
            except Exception as e:
                tk.Label(popup, text=f"이미지 표시 오류: {e}", wraplength=300).pack(pady=20)
        else:
            # 이미지 파일이 아직 없을 때도 프로그램이 꺼지지 않도록 기본 안내 표시
            tk.Label(
                popup,
                text="알림 이미지 파일을 넣어주세요.\n" + image_path,
                wraplength=300,
                justify="center"
            ).pack(pady=45)

        msg_label = tk.Label(
            popup,
            text=info["message"],
            font=("Arial", 13),
            wraplength=320,
            justify="center"
        )
        msg_label.pack(pady=10)

        tk.Button(popup, text="확인", command=popup.destroy).pack(pady=8)

        # 2.5초 뒤 자동 닫힘
        popup.after(2500, popup.destroy)

    def trigger_alert(self, level):
        """쿨다운을 적용하여 팝업 + 경고음을 실행"""
        current_time = time.time()
        if current_time - self.last_warning_time > self.warning_cooldown:
            self.last_warning_time = current_time
            self.last_alert_level = level
            self.show_alert_popup(level)
            self.play_alert_sound(level)

    def get_fatigue_level_by_names(self, detected_names):
        """이미지 1장 기준 피로 단계 판정"""
        has_closed_eye = 'closed_eye' in detected_names
        has_open_mouth = 'open_mouth' in detected_names

        if has_closed_eye and has_open_mouth:
            return "severe"
        elif has_closed_eye:
            return "mild"
        elif has_open_mouth:
            return "suspected"
        else:
            return "normal"

    def get_fatigue_level_by_perclos(self, perclos):
        """영상/카메라 10프레임 기준 피로 단계 판정"""
        if perclos >= 0.70:
            return "severe"
        elif perclos >= 0.50:
            return "mild"
        elif perclos > 0.36:
            return "suspected"
        else:
            return "normal"

    def update_driving_status(self, level):
        """운전자 상태 라벨 갱신 및 알림 실행"""
        if level == "normal":
            self.ui.tk_label_driving_show.config(text='정상 운전', background="white", foreground="black")
        else:
            info = self.alert_assets[level]
            self.ui.tk_label_driving_show.config(
                text=info["ko"],
                background=info["color"],
                foreground="black"
            )
            self.trigger_alert(level)

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

            choose_list = ['전체']
            target_names = [names[id] + '_' + str(index) for index, id in enumerate(cls_list)]
            detected_names = [names[index] for index in cls_list]

            level = self.get_fatigue_level_by_names(detected_names)
            self.update_driving_status(level)

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
        if com_text == '전체':
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
        self.ui.tk_label_driving_show.config(text='정상 운전', background="white", foreground="black")

        """开启동영상 파일检测事件"""
        self.video_path = filedialog.askopenfilename(title="동영상 파일 선택", filetypes=[("동영상 파일", "*.mp4 *.avi")])
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
                # 计算졸음운전
                if self.roll == 10:
                    perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                    level = self.get_fatigue_level_by_perclos(perclos)
                    self.update_driving_status(level)
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
        """카메라 열기目标检测"""
        self.ui.tk_button_upload_image.config(state=tk.DISABLED)
        self.ui.tk_button_video.config(state=tk.DISABLED)
        self.ui.tk_button_cam.config(state=tk.DISABLED)
        self.ui.tk_button_close.config(state=tk.NORMAL)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.DISABLED)
        self.ui.tk_label_driving_show.config(text='정상 운전', background="white", foreground="black")

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
                # 计算졸음운전
                if self.roll == 10:
                    perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                    level = self.get_fatigue_level_by_perclos(perclos)
                    self.update_driving_status(level)
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
