import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from PIL import Image, ImageTk, ImageOps

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 결과 이미지를 임시 저장할 폴더가 없으면 자동 생성
os.makedirs(os.path.join(BASE_DIR, "images", "tmp"), exist_ok=True)


class WinGUI(tk.Tk):
    """界面布局类"""

    # 전체 UI 선 색상 조정: 검은 굵은 선 -> 얇은 회색 선
    BORDER_COLOR = "#C9C9C9"
    BUTTON_BORDER = "#D3D3D3"
    PANEL_BG = "#F7F7F7"
    FIELD_BG = "#FFFFFF"
    TEXT_COLOR = "#333333"

    def __init__(self):
        super().__init__()
        self.__win()

        # 전체 창 배경 이미지
        self.tk_label_window_bg = self.__tk_window_background(self)

        self.tk_frame_control_frame = self.__tk_frame_control_frame(self)
        self.tk_button_upload_image = self.__tk_button_upload_image(self.tk_frame_control_frame)
        self.tk_button_video = self.__tk_button_video(self.tk_frame_control_frame)
        self.tk_button_cam = self.__tk_button_cam(self.tk_frame_control_frame)
        self.tk_button_close = self.__tk_button_close(self.tk_frame_control_frame)

        self.tk_label_show_image = self.__tk_label_show_image(self)

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
        width = 560
        height = 710
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = "%dx%d+%d+%d" % (
            width,
            height,
            (screenwidth - width) / 2,
            (screenheight - height) / 2,
        )
        self.geometry(geometry)
        self.resizable(width=False, height=False)

    def __get_background_path(self):
        """전체 UI 배경 이미지 경로 찾기"""
        bg_candidates = [
            os.path.join(BASE_DIR, "images", "ui_bg.jpg"),
            os.path.join(BASE_DIR, "images", "ui_bg.jpeg"),
            os.path.join(BASE_DIR, "images", "ui_bg.png"),
            os.path.join(BASE_DIR, "ui_bg.jpg"),
            os.path.join(BASE_DIR, "ui_bg.jpeg"),
            os.path.join(BASE_DIR, "ui_bg.png"),
        ]
        for path in bg_candidates:
            if os.path.exists(path):
                return path
        return None

    def __tk_window_background(self, parent):
        """창 전체에 배경 이미지를 표시"""
        bg_path = self.__get_background_path()
        if bg_path:
            try:
                bg = Image.open(bg_path).convert("RGB")
                self.window_bg_full_image = ImageOps.fit(bg, (560, 710), method=Image.LANCZOS)
                self.window_bg_photo = ImageTk.PhotoImage(self.window_bg_full_image)
                label = tk.Label(parent, image=self.window_bg_photo, bd=0)
                label.place(x=0, y=0, width=560, height=710)
                label.lower()
                return label
            except Exception as e:
                print("전체 배경 이미지 로드 실패:", e)

        parent.configure(bg="#F2F2F2")
        label = tk.Label(parent, bg="#F2F2F2", bd=0)
        label.place(x=0, y=0, width=560, height=710)
        label.lower()
        return label

    def _panel(self, parent):
        """얇은 회색 테두리용 Frame"""
        return tk.Frame(
            parent,
            bg=self.PANEL_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.BORDER_COLOR,
        )

    def _button(self, parent, text):
        """얇은 회색 테두리용 Button"""
        return tk.Button(
            parent,
            text=text,
            takefocus=False,
            bg="#F8F8F8",
            fg=self.TEXT_COLOR,
            activebackground="#EFEFEF",
            activeforeground=self.TEXT_COLOR,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=self.BUTTON_BORDER,
            highlightcolor=self.BUTTON_BORDER,
            font=("Malgun Gothic", 10),
        )

    def _field_label(self, parent, text=" ", width_border=True):
        """결과 영역의 작은 박스들을 얇은 회색 선으로 표시"""
        return tk.Label(
            parent,
            text=text,
            anchor="center",
            bg=self.FIELD_BG,
            fg=self.TEXT_COLOR,
            relief="flat",
            bd=0,
            highlightthickness=1 if width_border else 0,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.BORDER_COLOR,
            font=("Malgun Gothic", 9),
        )

    def scrollbar_autohide(self, vbar, hbar, widget):
        """自动隐藏滚动条"""

        def show():
            if vbar:
                vbar.lift(widget)
            if hbar:
                hbar.lift(widget)

        def hide():
            if vbar:
                vbar.lower(widget)
            if hbar:
                hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar:
            vbar.bind("<Enter>", lambda e: show())
            vbar.bind("<Leave>", lambda e: hide())
        if hbar:
            hbar.bind("<Enter>", lambda e: show())
            hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor="ne")

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor="sw")

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = ttk.Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = ttk.Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def __tk_frame_control_frame(self, parent):
        frame = self._panel(parent)
        frame.place(x=60, y=10, width=445, height=50)
        return frame

    def __tk_button_upload_image(self, parent):
        btn = self._button(parent, "이미지 업로드")
        btn.place(x=15, y=8, width=79, height=30)
        return btn

    def __tk_button_video(self, parent):
        btn = self._button(parent, "동영상 업로드")
        btn.place(x=110, y=8, width=84, height=30)
        return btn

    def __tk_button_cam(self, parent):
        btn = self._button(parent, "카메라 열기")
        btn.place(x=207, y=8, width=90, height=30)
        return btn

    def __tk_button_close(self, parent):
        btn = self._button(parent, "영상/카메라 종료")
        btn.place(x=312, y=8, width=112, height=30)
        return btn

    def __tk_label_show_image(self, parent):
        label = tk.Label(
            parent,
            text="감지 화면",
            anchor="center",
            bg="white",
            fg=self.TEXT_COLOR,
            font=("Malgun Gothic", 16),
            bd=0,
            highlightthickness=1,
            highlightbackground=self.BORDER_COLOR,
            highlightcolor=self.BORDER_COLOR,
        )
        label.place(x=60, y=68, width=440, height=440)
        return label

    def __tk_frame_result_frame(self, parent):
        frame = self._panel(parent)
        frame.place(x=60, y=515, width=445, height=165)
        return frame

    def __tk_label_time(self, parent):
        label = self._field_label(parent, "소요시간:")
        label.place(x=10, y=8, width=70, height=30)
        return label

    def __tk_label_time_show(self, parent):
        label = self._field_label(parent, " ")
        label.place(x=83, y=8, width=80, height=30)
        return label

    def __tk_label_count(self, parent):
        label = self._field_label(parent, "객체 수:")
        label.place(x=178, y=8, width=70, height=30)
        return label

    def __tk_label_count_show(self, parent):
        label = self._field_label(parent, " ")
        label.place(x=250, y=8, width=90, height=30)
        return label

    def __tk_label_conf(self, parent):
        label = self._field_label(parent, "신뢰도:")
        label.place(x=10, y=45, width=70, height=30)
        return label

    def __tk_label_conf_show(self, parent):
        label = self._field_label(parent, " ")
        label.place(x=83, y=45, width=80, height=30)
        return label

    def __tk_label_class(self, parent):
        label = self._field_label(parent, "분류:")
        label.place(x=178, y=45, width=70, height=30)
        return label

    def __tk_label_class_show(self, parent):
        label = self._field_label(parent, " ")
        label.place(x=250, y=45, width=90, height=30)
        return label

    def __tk_label_xmin(self, parent):
        label = self._field_label(parent, "xmin:")
        label.place(x=10, y=82, width=45, height=30)
        return label

    def __tk_label_xmin_show(self, parent):
        label = self._field_label(parent, "111")
        label.place(x=58, y=82, width=45, height=30)
        return label

    def __tk_label_ymin(self, parent):
        label = self._field_label(parent, "ymin:")
        label.place(x=115, y=82, width=45, height=30)
        return label

    def __tk_label_ymin_show(self, parent):
        label = self._field_label(parent, " ")
        label.place(x=163, y=82, width=45, height=30)
        return label

    def __tk_label_xmax(self, parent):
        label = self._field_label(parent, "xmax:")
        label.place(x=220, y=82, width=45, height=30)
        return label

    def __tk_label_xmax_show(self, parent):
        label = self._field_label(parent, " ")
        label.place(x=268, y=82, width=45, height=30)
        return label

    def __tk_label_ymax(self, parent):
        label = self._field_label(parent, "ymax:")
        label.place(x=325, y=82, width=45, height=30)
        return label

    def __tk_label_ymax_show(self, parent):
        label = self._field_label(parent, " ")
        label.place(x=373, y=82, width=60, height=30)
        return label

    def __tk_label_select_obj(self, parent):
        label = self._field_label(parent, "객체 선택:")
        label.place(x=355, y=8, width=80, height=30)
        return label

    def __tk_select_box_select_obj_show(self, parent):
        cb = ttk.Combobox(parent, state="readonly")
        cb["values"] = ("")
        cb.place(x=355, y=42, width=80, height=30)
        return cb

    def __tk_label_driving_flag(self, parent):
        label = self._field_label(parent, "운전자 상태:")
        label.place(x=10, y=120, width=120, height=31)
        return label

    def __tk_label_driving_show(self, parent):
        label = self._field_label(parent, "정상")
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
        self.tk_button_upload_image.bind("<Button-1>", self.ctl.upload_image)
        self.tk_button_video.bind("<Button-1>", self.ctl.upload_video)
        self.tk_button_cam.bind("<Button-1>", self.ctl.open_cam)
        self.tk_button_close.bind("<Button-1>", self.ctl.close_video_cam)
        self.tk_select_box_select_obj_show.bind("<<ComboboxSelected>>", self.ctl.combox_change)

    def __style_config(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            borderwidth=1,
            relief="flat",
            arrowsize=12,
        )


class Controller:
    """逻辑处理类，图片检测、视频检测、摄像头检测的逻辑处理"""

    ui: Win

    def __init__(self):
        # todo 修改模型权重路径
        model_path = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.stop = False
        self.file_path = ""
        self.output_size = 640
        self.img2predict = ""

        # 语音相关初始化
        self.engine_lock = threading.Lock()
        self.last_warning_time = 0
        self.warning_cooldown = 3
        self.last_alert_level = None

        # 피로 단계별 알림 자료 경로
        self.alert_assets = {
            "suspected": {
                "ko": "졸음 의심",
                "message": "졸음이 의심됩니다. 주의해서 운전하세요.",
                "image": os.path.join(BASE_DIR, "images/alerts/suspected.png"),
                "sound": os.path.join(BASE_DIR, "sounds/suspected.wav"),
                "color": "#FFD966",
            },
            "mild": {
                "ko": "경미한 졸음",
                "message": "경미한 졸음 상태가 감지되었습니다. 잠시 휴식하세요.",
                "image": os.path.join(BASE_DIR, "images/alerts/mild.png"),
                "sound": os.path.join(BASE_DIR, "sounds/mild.wav"),
                "color": "#F4B183",
            },
            "severe": {
                "ko": "심한 졸음",
                "message": "심한 졸음 상태입니다. 즉시 휴식하세요!",
                "image": os.path.join(BASE_DIR, "images/alerts/severe.png"),
                "sound": os.path.join(BASE_DIR, "sounds/severe.wav"),
                "color": "#FF0000",
            },
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
        self.model = YOLO(model_path, task="detect")
        self.model(np.zeros((48, 48, 3)))  # 预先加载推理模型
        self.delay = 15

    def play_alert_sound(self, level):
        """피로 단계별 경고음 또는 TTS 음성을 재생"""
        info = self.alert_assets[level]
        sound_path = info["sound"]

        def _play():
            try:
                if winsound and os.path.exists(sound_path):
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    return

                with self.engine_lock:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 150)
                    engine.setProperty("volume", 1.0)
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
        popup.geometry("360x300")
        popup.resizable(False, False)
        popup.attributes("-topmost", True)

        title_label = tk.Label(
            popup,
            text=info["ko"],
            font=("Arial", 22, "bold"),
            bg=info["color"],
            fg="black",
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
            except Exception:
                # 이미지 파일에 문제가 있어도 경로 안내 문구는 표시하지 않음
                pass
        else:
            # 이미지 파일이 없을 때도 경로 안내 문구를 표시하지 않음
            pass

        msg_label = tk.Label(
            popup,
            text=info["message"],
            font=("Arial", 13),
            wraplength=320,
            justify="center",
        )
        msg_label.pack(pady=10)

        tk.Button(popup, text="확인", command=popup.destroy).pack(pady=8)
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
        has_closed_eye = "closed_eye" in detected_names
        has_open_mouth = "open_mouth" in detected_names

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
            self.ui.tk_label_driving_show.config(text="정상 운전", bg="white", fg="black")
        else:
            info = self.alert_assets[level]
            self.ui.tk_label_driving_show.config(text=info["ko"], bg=info["color"], fg="black")
            self.trigger_alert(level)

    def show_home_background(self):
        """중앙 감지 화면에 기본 배경 이미지를 표시"""
        bg_candidates = [
            os.path.join(BASE_DIR, "images", "ui_bg.jpg"),
            os.path.join(BASE_DIR, "images", "ui_bg.jpeg"),
            os.path.join(BASE_DIR, "images", "ui_bg.png"),
            os.path.join(BASE_DIR, "ui_bg.jpg"),
            os.path.join(BASE_DIR, "ui_bg.jpeg"),
            os.path.join(BASE_DIR, "ui_bg.png"),
        ]

        # 창 전체 배경과 자연스럽게 이어지도록 중앙 감지 영역 부분만 잘라서 표시
        if hasattr(self.ui, "window_bg_full_image"):
            try:
                image = self.ui.window_bg_full_image.crop((60, 68, 500, 508))
                photo = ImageTk.PhotoImage(image)
                self.ui.tk_label_show_image.config(image=photo, text="")
                self.ui.tk_label_show_image.image = photo
                return
            except Exception as e:
                print("중앙 배경 이미지 로드 실패:", e)

        bg_path = None
        for path in bg_candidates:
            if os.path.exists(path):
                bg_path = path
                break

        if bg_path:
            try:
                image = Image.open(bg_path).convert("RGB")
                image = ImageOps.fit(image, (440, 440), method=Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.ui.tk_label_show_image.config(image=photo, text="")
                self.ui.tk_label_show_image.image = photo
            except Exception as e:
                self.ui.tk_label_show_image.config(
                    image="",
                    text="감지 화면",
                    font=("Arial", 16),
                    bg="white",
                )
                print("배경 이미지 로드 실패:", e)
        else:
            self.ui.tk_label_show_image.config(
                image="",
                text="감지 화면",
                font=("Arial", 16),
                bg="white",
            )

    def init(self, ui):
        """得到UI实例，对组件进行初始化配置"""
        self.ui = ui
        self.ui.tk_button_close.config(state=tk.DISABLED)

        # 프로그램 시작 시 중앙 감지 화면에 기본 배경 표시
        self.show_home_background()

    def display_image(self, file_path):
        """显示图片"""
        image = Image.open(file_path).convert("RGB")
        image = ImageOps.fit(image, (440, 440), method=Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.ui.tk_label_show_image.config(image=photo, text="")
        self.ui.tk_label_show_image.image = photo

    def upload_image(self, evt):
        """图像目标检测"""
        self.ui.tk_button_close.config(state=tk.DISABLED)
        self.ui.tk_select_box_select_obj_show.delete(0, tk.END)
        self.ui.tk_select_box_select_obj_show.config(state=tk.NORMAL)

        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if self.image_path:
            org_path = self.image_path

            t1 = time.time()
            results = self.model(org_path)[0]
            names = results.names
            t2 = time.time()

            self.ui.tk_label_time_show.config(text="{:.3f} s".format(t2 - t1))
            now_img = results.plot()
            resize_scale = self.output_size / now_img.shape[0]
            im0 = cv2.resize(now_img, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite(os.path.join(BASE_DIR, "images/tmp/single_result.jpg"), im0)
            self.display_image(os.path.join(BASE_DIR, "images/tmp/single_result.jpg"))

            location_list = results.boxes.xyxy.tolist()
            location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            cls_list = [int(i) for i in cls_list]
            conf_list = results.boxes.conf.tolist()
            conf_list = ["%.2f %%" % (each * 100) for each in conf_list]

            total_nums = len(location_list)
            self.ui.tk_label_count_show.config(text="{}".format(total_nums))

            choose_list = ["전체"]
            target_names = [names[id] + "_" + str(index) for index, id in enumerate(cls_list)]
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
        if not hasattr(self, "location_list") or not self.location_list:
            return

        if com_text == "전체":
            cur_box = self.location_list
            cur_img = self.results.plot()
            self.ui.tk_label_class_show.config(text="{}".format(self.names[self.cls_list[0]]))
            self.ui.tk_label_conf_show.config(text="{}".format(self.conf_list[0]))
        else:
            index = int(com_text.split("_")[-1])
            cur_box = [self.location_list[index]]
            cur_img = self.results[index].plot()
            self.ui.tk_label_class_show.config(text="{}".format(self.names[self.cls_list[index]]))
            self.ui.tk_label_conf_show.config(text="{}".format(str(self.conf_list[index])))

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
        self.ui.tk_label_driving_show.config(text="정상 운전", bg="white", fg="black")

        self.video_path = filedialog.askopenfilename(
            title="동영상 파일 선택",
            filetypes=[("동영상 파일", "*.mp4 *.avi")],
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_playing = True
            self.update_frame()
        else:
            # 파일 선택을 취소했을 때 버튼 복원
            self.close_video_cam(None)

    def update_frame(self):
        if self.is_playing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                t1 = time.time()
                results = self.model(frame)[0]
                names = results.names
                t2 = time.time()

                self.ui.tk_label_time_show.config(text="{:.3f} s".format(t2 - t1))
                now_img = results.plot()
                cv2.imwrite(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"), now_img)
                self.display_image(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"))

                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                conf_list = results.boxes.conf.tolist()
                conf_list = ["%.2f %%" % (each * 100) for each in conf_list]

                total_nums = len(location_list)
                self.ui.tk_label_count_show.config(text="{}".format(total_nums))

                detected_names = [names[index] for index in cls_list]
                if "closed_eye" in detected_names:
                    self.roll_eye = self.roll_eye + 1
                if "open_mouth" in detected_names:
                    self.roll_mouth = self.roll_mouth + 1
                self.roll += 1

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
                if self.cap:
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
        self.ui.tk_label_driving_show.config(text="정상 운전", bg="white", fg="black")

        if not self.is_capturing:
            self.is_capturing = True
            self.cap = cv2.VideoCapture(0)
            self.update_camera_frame()

    def update_camera_frame(self):
        if self.is_capturing and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                t1 = time.time()
                results = self.model(frame)[0]
                names = results.names
                t2 = time.time()

                self.ui.tk_label_time_show.config(text="{:.3f} s".format(t2 - t1))
                now_img = results.plot()
                cv2.imwrite(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"), now_img)
                self.display_image(os.path.join(BASE_DIR, "images/tmp/single_result_vid.jpg"))

                location_list = results.boxes.xyxy.tolist()
                location_list = [list(map(int, e)) for e in location_list]
                cls_list = results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                conf_list = results.boxes.conf.tolist()
                conf_list = ["%.2f %%" % (each * 100) for each in conf_list]

                total_nums = len(location_list)
                self.ui.tk_label_count_show.config(text="{}".format(total_nums))

                detected_names = [names[index] for index in cls_list]
                if "closed_eye" in detected_names:
                    self.roll_eye = self.roll_eye + 1
                if "open_mouth" in detected_names:
                    self.roll_mouth = self.roll_mouth + 1
                self.roll += 1

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

        if self.is_playing or self.is_capturing:
            if self.cap:
                self.cap.release()
            self.is_playing = False
            self.is_capturing = False

        self.roll = 0
        self.roll_eye = 0
        self.roll_mouth = 0

        # 영상/카메라 종료 후 중앙 감지 화면을 다시 기본 배경으로 복원
        if hasattr(self, "ui"):
            self.show_home_background()


if __name__ == "__main__":
    controller = Controller()
    app = Win(controller)
    app.mainloop()
