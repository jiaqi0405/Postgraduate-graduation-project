from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPixmap
import random

from shape import Square, Rectangle, Circle, Ellipse, Triangle, FreeCurve
from customer import BG_COLOR, FG_COLOR, BTN_COLOR, create_btn, create_btn2, create_combo, create_label, create_input, Page
from shapes_page import ShapesPage

class ExperimentWindow(Page):
    def __init__(self, vars):
        super().__init__(vars)
        self.setStyleSheet(f"background: {BG_COLOR};")
        self.setWindowTitle("Experiment Display")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        # 保存原始尺寸
        self.original_width = vars.get('frame').width()
        self.original_height = vars.get('frame').height()
        self.resize(self.original_width, self.original_height)
        self.stack = QStackedWidget()
        self.is_maximized = False
        self._init_layout()

        exp_page = self.vars.get('pages').get('experiment')
        exp_page.finish_clicked_signal.connect(self.close)

    def _init_layout(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 创建一个容器widget来包含stack，使其可以居中
        self.container = QWidget()
        self.container_layout = QVBoxLayout()
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(0)
        self.container_layout.addWidget(self.stack)
        self.container.setLayout(self.container_layout)
        
        # 创建一个居中容器
        center_layout = QHBoxLayout()
        center_layout.addStretch(1)
        center_layout.addWidget(self.container)
        center_layout.addStretch(1)
        
        self.main_layout.addStretch(1)
        self.main_layout.addLayout(center_layout)
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        self.main = WinMain(self.vars, self.show_exp)
        self.stack.addWidget(self.main)
        self.exp = WinExperiment(self.vars, self.show_block)
        self.stack.addWidget(self.exp)
        self.block = WinBlockFinish(self.vars, self.show_exp)
        self.stack.addWidget(self.block)

        self.main.start_exp_signal.connect(self.block.create_block_layout)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateLayout()
        
    def updateLayout(self):
        # 获取当前窗口尺寸
        current_width = self.width()
        current_height = self.height()
        
        # 计算缩放比例
        width_ratio = current_width / self.original_width
        height_ratio = current_height / self.original_height
        scale_ratio = min(width_ratio, height_ratio)
        
        # 计算缩放后的尺寸
        scaled_width = int(self.original_width * scale_ratio)
        scaled_height = int(self.original_height * scale_ratio)
        
        # 设置固定大小
        self.container.setFixedSize(scaled_width, scaled_height)
        
        # 更新边距以保持居中
        margin_h = (current_width - scaled_width) // 2
        margin_v = (current_height - scaled_height) // 2
        self.main_layout.setContentsMargins(margin_h, margin_v, margin_h, margin_v)

    def showEvent(self, event):
        super().showEvent(event)
        self.updateLayout()  # 更新布局

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if not self.is_maximized:
                self.showMaximized()
                self.is_maximized = True
            else:
                self.showNormal()
                self.is_maximized = False
            self.updateLayout()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event) 
    
    def show_main(self):
        self.stack.setCurrentWidget(self.main)
    def show_exp(self):
        self.stack.setCurrentWidget(self.exp)
    def show_block(self):
        self.stack.setCurrentWidget(self.block)

class WinMain(Page):
    start_exp_signal = Signal()
    def __init__(self, vars, show_exp):
        super().__init__(vars)
        self.show_exp = show_exp
        self._init_layout()

    def _init_layout(self):
        layout = QVBoxLayout()
        
        # 顶部栏
        top_bar = QHBoxLayout()
        icon_label = QLabel()
        icon_pixmap = QPixmap("icon.png")
        icon_label.setPixmap(icon_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_label.setFixedSize(200, 100)
        top_bar.addWidget(icon_label, alignment=Qt.AlignLeft)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        label = create_label(f"{self.vars.get('type')} experiment", 42)
        layout.addWidget(label)

        blocks = create_label(f"Total blocks: {int(self.vars.get('blocks'))}")
        layout.addWidget(blocks)

        btn = create_btn2('Start', "#000000")
        btn.clicked.connect(self.start_experiment)
        btn.setFixedWidth(300)
        btn.setFixedHeight(80)
        hbox = QHBoxLayout()
        hbox.addWidget(btn)

        layout.addLayout(hbox)
        self.setLayout(layout)
    def start_experiment(self):
        self.start_exp_signal.emit()
        self.show_exp()
        
    def showEvent(self, event):
        super().showEvent(event)
        

class WinExperiment(ShapesPage):
    def __init__(self, vars, show_block):
        super().__init__(vars, 1)
        self.show_block = show_block

        exp_page = self.vars.get('pages').get('experiment')
        exp_page.finish_block_signal.connect(self.show_block)

        block_page = self.vars.get('pages').get('blocks')
        block_page.confirm_setting_signal.connect(self.update_frame)

        self.bind_exp_page_signal(exp_page)
        
        self._init_layout()

        
        
    def _init_layout(self):
        hbox = self.create_shapes_layout()
        self.setLayout(hbox)

    def showEvent(self, event):
        super().showEvent(event)



    
class WinBlockFinish(Page):
    def __init__(self, vars, show_exp):
        super().__init__(vars)
        self.show_exp = show_exp

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.change_time_text)

        rest_timer = self.vars.get('rest_timer')
        rest_timer.timeout.connect(self.show_exp)

        
        exp_page = self.vars.get('pages').get('experiment')
        exp_page.pause_signal.connect(lambda: self.timer.stop())
        exp_page.resume_signal.connect(self.show_exp)
        exp_page.finish_block_signal.connect(self.dispaly_block_finish)
        exp_page.finish_signal.connect(self.update_finished_info)

        self.block_arr = []

        self._init_layout()

    def _init_layout(self):
        layout = QVBoxLayout()
        
        # 顶部栏
        top_bar = QHBoxLayout()
        icon_label = QLabel()
        icon_pixmap = QPixmap("icon.png")
        icon_label.setPixmap(icon_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_label.setFixedSize(200, 100)
        top_bar.addWidget(icon_label, alignment=Qt.AlignLeft)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        label = create_label(f"Block {self.vars.get('process').get('block')} finished", 42)
        self.finish_label = label
        title_layout = QHBoxLayout()
        title_layout.addStretch()
        title_layout.addWidget(label)
        title_layout.addSpacing(50)
        # title_layout.addLayout(self.create_block_layout())
        title_layout.addStretch()
        layout.addLayout(title_layout)

        label = create_label("Please rest", 42)
        layout.addWidget(label)

        self.reset_rest_time()
        self.time_label = create_label(self.convert_time(self.rest), 42)
        layout.addWidget(self.time_label)

        self.setLayout(layout)
        
    def showEvent(self, event):
        super().showEvent(event)

        # print('block finish', self.vars)
        self.finish_label.setText(f"Block {self.vars.get('process').get('block') - 1} finished")

        self.reset_rest_time()
        self.time_label.setText(self.convert_time(self.rest))
        self.timer.start(1000)

    def change_time_text(self):
        self.rest = self.rest - 1
        self.time_label.setText(self.convert_time(self.rest))

        if self.rest == 20:
            audio_path = r'C:/Users/yjq/Desktop/guifinal/gui1/Twenty.wav'
            print(f"Playing audio file: {audio_path}")
            try:
                self.vars.get('player').play(audio_path)
            except Exception as e:
                print(f"Error playing audio: {str(e)}")

        if self.rest <= 0:
            self.timer.stop()
            if int(self.vars.get('process').get('block')) < int(self.vars.get('blocks')):
                self.show_exp()

    def convert_time(self, seconds):
        minutes, seconds = divmod(seconds, 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        return time_str

    def reset_rest_time(self):
        self.rest = max(0, int(self.vars.get('rest')) - 1)

    def create_block_layout(self):
        blocks = int(self.vars.get('blocks'))
        hbox = QHBoxLayout()

        self.block_arr = []
        for _ in range(blocks):
            shape = Rectangle()
            shape.set_size(150, 80)
            shape.set_border_color(FG_COLOR)
            shape.set_color(BG_COLOR)
            shape.set_text_type("nothing")
            self.block_arr.append(shape)
            hbox.addWidget(shape)
        return hbox

    def dispaly_block_finish(self):
        block = int(self.vars.get('process').get('block'))
        # print("get block finish..", block)
        if block > 0 and block <= len(self.block_arr):
            self.block_arr[block - 1].set_color(FG_COLOR)
            # print("set block color..", block)

    def update_finished_info(self):
        self.time_label.setText("All blocks finished.")

        audio_path = r'C:/Users/yjq/Desktop/guifinal/gui1/All.wav'
        print(f"Playing audio file: {audio_path}")
        try:
            self.vars.get('player').play(audio_path)
        except Exception as e:
            print(f"Error playing audio: {str(e)}")

    def say(self, text):
        self.vars.get('engine').say(text)
        self.vars.get('engine').runAndWait()
