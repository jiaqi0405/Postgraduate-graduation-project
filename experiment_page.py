from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QFrame, QLineEdit, QLayout, QLayoutItem, QCheckBox
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from shape import ShapeWidget, Square, Rectangle, Circle, Ellipse, Triangle, FreeCurve
from customer import BG_COLOR, FG_COLOR, BTN_COLOR, FSIZE, Title_size, create_btn, create_btn2, create_combo, create_label, create_input, UtilsWidget, Page
from experiment_window import WinMain
from shapes_page import ShapesPage
from trca_integration import TRCAIntegration
import random

class ExperimentPage(ShapesPage):
    receive_start_signal = Signal()
    cue_signal = Signal()
    finish_trial_signal = Signal()
    finish_block_signal = Signal()
    pause_signal = Signal()
    resume_signal = Signal()
    finish_signal = Signal()
    finish_clicked_signal = Signal()
    def __init__(self, vars, switch_to_blocks, switch_to_main):
        super().__init__(vars)
        self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR};")
        self.switch_to_blocks = switch_to_blocks
        self.switch_to_main = switch_to_main
        self.selected_index = None
        self.shape_widgets = []
        self.size_hint_widget = None

        self.running = True
        
        # TRCA集成
        self.trca_integration = TRCAIntegration(vars)
        self.recognition_flag = 0  # 0: 只收集数据, 1: 实时识别+数据存储
        self.current_target_index = None
        self.recognition_result_char = None
        
        # 连接TRCA信号
        self.trca_integration.recognition_result.connect(self.handle_recognition_result)
        self.trca_integration.data_saved.connect(self.handle_data_saved)
        self.trca_integration.status_update.connect(self.handle_status_update)

        self.cue_timer = self.vars.get('cue_timer')
        self.flash_timer = self.vars.get('flash_timer')
        self.pause_timer = self.vars.get('pause_timer')
        self.rest_timer = self.vars.get('rest_timer')

        self.cue_timer.timeout.connect(self.start_flash_timer)
        self.flash_timer.timeout.connect(self.start_pause_timer)
        self.pause_timer.timeout.connect(self.finish_trial)
        self.rest_timer.timeout.connect(self.start_experiment)

        self.bind_exp_page_signal(self)

        self._init_layout()

    def _init_layout(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(10)

        # 上部标题
        top_widget = QWidget()
        title_layout = QHBoxLayout()
        title = QLabel("Experiment")
        self.title = title
        title.setStyleSheet(f"font-size: { Title_size}px; color: {FG_COLOR};")
        title_layout.addWidget(title)
        title_layout.addSpacing(200)

        title_info = QLabel("")
        self.title_info = title_info
        title_info.setStyleSheet(f"font-size: { Title_size}px; color: #FF0000;")
        title_layout.addWidget(title_info)

        title_layout.addStretch()
        top_widget.setLayout(title_layout)
        self.layout.addWidget(top_widget, alignment=Qt.AlignTop)

        # 中部左右布局
        self.center = QHBoxLayout()
        # 左2/3
        vbox = QVBoxLayout()
        self.left_frame = QFrame()
        self.left_frame.setStyleSheet(f"border: 2px solid {FG_COLOR}; background: {BG_COLOR}; padding:0;")
        vbox.addLayout(self.create_info_layout())
        vbox.addWidget(self.left_frame, stretch=1)
        self.center.addLayout(vbox, stretch=2)
        # 右1/3
        right = QVBoxLayout()
        
        # 上方按钮
        btn_row = QVBoxLayout()
        pause_btn = create_btn2("Pause")
        continue_btn = create_btn2("Continue")
        pause_btn.clicked.connect(self.pause_experiment)
        continue_btn.clicked.connect(self.resume_experiment)

        btn_row.addWidget(pause_btn)
        btn_row.addWidget(continue_btn)
        btn_layout = self.lay_h_center(QVBoxLayout(), btn_row)
        right.addLayout(btn_layout)

        # TRCA控制区域
        trca_frame = QFrame()
        trca_frame.setStyleSheet(f"border: 1px solid {FG_COLOR}; background: {BG_COLOR}; padding: 5px;")
        trca_layout = QVBoxLayout()
        
        # TRCA模式选择
        self.recognition_checkbox = QCheckBox("Real-time recognition mode")
        self.recognition_checkbox.setStyleSheet(f"color: {FG_COLOR}; font-size: {FSIZE}px;")
        self.recognition_checkbox.stateChanged.connect(self.toggle_recognition_mode)
        trca_layout.addWidget(self.recognition_checkbox)
        
        # TRCA状态显示
        self.trca_status_label = self.label("Mode: Collect data only")
        self.trca_status_label.setStyleSheet(f"color: #00FF00; font-size: {FSIZE-2}px;")
        trca_layout.addWidget(self.trca_status_label)
        
        # 识别结果显示
        self.recognition_result_label = self.label("Identification results: --")
        self.recognition_result_label.setStyleSheet(f"color: #FFFF00; font-size: {FSIZE}px; font-weight: bold;")
        trca_layout.addWidget(self.recognition_result_label)
        
        trca_frame.setLayout(trca_layout)
        right.addWidget(trca_frame)
        
        right.addSpacing(10)
        
        # 图标区
        right.addLayout(self.icon_layout("#FF0000", "target"))
        right.addLayout(self.icon_layout("#FFFF00", "result"))
        
        # 信息显示区域
        self.cue_time = self.label(f"Cue time (s):          {self.vars.get('cue')}")
        right.addWidget(self.cue_time)
        
        self.flash_time = self.label(f"Flashing time (s):   {self.vars.get('flash')}")
        right.addWidget(self.flash_time)
        
        self.pause_time = self.label(f"Pause time (s):       {self.vars.get('pause')}")
        right.addWidget(self.pause_time)
        
        right.addSpacing(20)
        
        self.rest_time = self.label(f"Rest time between blocks (s): {self.vars.get('rest')}")
        right.addWidget(self.rest_time)
        
        self.window_time = self.label(f"Time window (s):      {self.vars.get('window')}")
        right.addWidget(self.window_time)
        right.addStretch()
        
        # 下方 Finish 按钮
        finish_btn = create_btn2("Finish")
        finish_btn.clicked.connect(self.handle_finish_clicked)
        right.addWidget(finish_btn, alignment=Qt.AlignCenter)
        
        right_widget = QWidget()
        right_widget.setLayout(right)
        self.right_widget = right_widget
        self.center.addWidget(self.right_widget, stretch=1)
        self.layout.addLayout(self.center, stretch=1)

        # 下部 Back/Next
        bottom_bar = QHBoxLayout()
        back_btn = create_btn("<< Back")
        back_btn.clicked.connect(self.switch_to_blocks)
        bottom_bar.addWidget(back_btn, alignment=Qt.AlignLeft)
        bottom_bar.addStretch()
        self.layout.addLayout(bottom_bar)
        self.setLayout(self.layout)

    def label(self, text):
        return create_label(text, FSIZE, Qt.AlignLeft)
    def create_info_layout(self):
        hbox = QHBoxLayout()
        self.block_info = create_label('Block 1/5')
        self.trial_info = create_label('Trial 1/3')
        hbox.addStretch()
        hbox.addWidget(self.block_info)
        hbox.addStretch()
        hbox.addWidget(self.trial_info)
        hbox.addStretch()
        self.update_status_display()
        return hbox
    def showEvent(self, event):
        super().showEvent(event)
        self.title.setText(f"{self.vars.get('type')}")
        self.refresh_left_rects()

        
        self.cue_time.setText(f"Cue time (s):          {self.vars.get('cue')}")
        self.flash_time.setText(f"Flashing time (s):   {self.vars.get('flash')}")
        self.pause_time.setText(f"Pause time (s):       {self.vars.get('pause')}")
        self.rest_time.setText(f"Rest time between blocks (s): {self.vars.get('rest')}")
        self.window_time.setText(f"Time window (s):      {self.vars.get('window')}")

        self.title_info.setText("")
        self.update_status_display()

    def refresh_left_rects(self):
        hbox = self.create_shapes_layout()
        self.left_frame.setLayout(hbox)


    def icon_layout(self, color, text):
        hbox = QHBoxLayout()
        square = Square()
        square.set_text_type('nothing')
        square.set_color(BG_COLOR)
        square.set_border_color(color)
        square.set_size(15)
        hbox.addWidget(square)
        hbox.addWidget(self.label(text))
        return hbox

    def handle_start_signal(self):
        self.receive_start_signal.emit()
        self.flashed_index = []
        self.start_experiment()

    def start_experiment(self):
        self.running = True
        self.title_info.setText("Experiment doing")
        # 更新页面信息
        self.update_status_display()
        self.set_random_target()
        
        # 获取当前目标索引并启动TRCA trial
        process = self.vars.get('process')
        current_index = process.get('curr_index', 0)
        current_block = process.get('block', 1)
        current_trial = process.get('trial', 1)
        
        self.current_target_index = current_index
        self.trca_integration.start_trial(current_index, current_block, current_trial)
        
        self.cue_signal.emit()
        self.cue_timer.start()
        # print('cue ----')
    def start_flash_timer(self):
        if self.running:
            self.flash_timer.start()
            # print('flash ----')
    def start_pause_timer(self):
        if self.running:
            self.pause_timer.start()
            # print('pause ----')
    # def start_rest_timer(self):
    #     self.finish_trial()
    #     if self.running:
    #         self.rest_timer.start()
            # print('rest ----')
    def finish_trial(self):
        # 结束TRCA trial
        self.trca_integration.end_trial()
        
        # 更新 trails 记录
        process = self.vars.get('process')
        # print('process1', process)
        # print('var1', self.vars)
        trial = int(process.get('trial')) + 1
        process.update({'trial':trial})
        self.vars.update({'process':process})
        # print('process2', process)
        # print('var2', self.vars)

        # 全部结束就更新 block 记录
        count = int(self.vars.get('count'))
        block = int(process.get('block'))
        blocks = int(self.vars.get('blocks'))
        if trial > count:
            block = block + 1
            process.update({'block':block})
            process.update({'trial':1})
            self.vars.update({'process':process})
            # block 结束事件
            self.finish_block_signal.emit()
            self.flashed_index = []
            
            if self.running:
                self.rest_timer.start()
        else:
            self.start_experiment()

        # print('update trial', trial, 'block ', block, 'count', count)
        # 全部结束 
        if block > blocks:
            self.running = False
            self.finish_signal.emit()
            self.finish_experiment()
            
        self.finish_trial_signal.emit()

        self.update_status_display()

    def pause_experiment(self):
        self.running = False
        self.vars.get('cue_timer').stop()
        self.vars.get('flash_timer').stop()
        self.vars.get('pause_timer').stop()
        self.vars.get('rest_timer').stop()

        self.title_info.setText("Experiment pause")

        self.pause_signal.emit()

        

    def resume_experiment(self):
        self.title_info.setText("Experiment doing")
        self.resume_signal.emit()
        self.start_experiment()

    def finish_experiment(self):
        self.pause_experiment()
        self.title_info.setText("Experiment finished")
        self.finish_signal.emit()

    def handle_finish_clicked(self):
        QTimer.singleShot(30, self.vars.get('player').stop)
        self.finish_clicked_signal.emit()
        self.finish_experiment()
        self.switch_to_main()

    def update_status_display(self):
        v = self.vars
        curr_block = int(v.get('process').get('block'))
        blocks = int(v.get('blocks'))
        curr_trial = int(v.get('process').get('trial'))
        trials = int(v.get('count'))
        self.block_info.setText(f"Block {min(curr_block, blocks)}/{blocks}")
        self.trial_info.setText(f"Trial {curr_trial}/{trials}")

    def set_random_target(self):
        trial = int(self.vars.get('process').get('trial'))
        if trial <= len(self.flashed_index): # resume 时， 不用设置
            return
        max_count = int(self.vars.get('count'))
        if max_count == len(self.flashed_index):
            self.finish_experiment()
            return
        index = random.randint(0, max_count - 1)
        while index in self.flashed_index:
            index = random.randint(0, max_count - 1)
            if max_count == len(self.flashed_index):
                break
        self.flashed_index.append(index)
        self.update_process_value("curr_index", index)

    def update_process_value(self, key, value, add=False):
        process = self.vars.get('process')
        temp = int(process.get(key))
        if add:
            process.update({key: temp + value})
        else:
            process.update({key:value})
        self.vars.update({'process':process})
        
    # TRCA related methods
    def toggle_recognition_mode(self, state):
        """Switch recognition mode"""
        self.recognition_flag = 1 if state == 2 else 0  # Qt.Checked = 2
        self.trca_integration.set_recognition_flag(self.recognition_flag)
        
        # Update status display
        if self.recognition_flag == 1:
            self.trca_status_label.setText("Mode: Real-time recognition + data storage")
            self.trca_status_label.setStyleSheet(f"color: #FF8800; font-size: {FSIZE-2}px;")
        else:
            self.trca_status_label.setText("Mode: Collect data only")
            self.trca_status_label.setStyleSheet(f"color: #00FF00; font-size: {FSIZE-2}px;")
            self.recognition_result_label.setText("Identification results: --")
            
    def handle_recognition_result(self, predicted_char, confidence):
        """Processing recognition results"""
        self.recognition_result_char = predicted_char
        result_text = f"Identification results: {predicted_char} ({confidence:.2f})"
        self.recognition_result_label.setText(result_text)
        
        # If recognition mode is on, update the shape display
        if self.recognition_flag == 1:
            self.update_recognition_display(predicted_char)
            
    def handle_data_saved(self, filename):
        """Handling data saving events"""
        print(f"Experimental data has been saved: {filename}")
        
    def handle_status_update(self, status):
        """Handling Status Updates"""
        print(f"TRCA Status Update: {status}")
        
    def update_recognition_display(self, predicted_char):
        """Updated visual display of recognition results"""
        try:
            # Parse the character to get the index
            if predicted_char.startswith('T'):
                predicted_index = int(predicted_char[1:]) - 1
                self.highlight_shape(predicted_index, "#FFFF00")
        except (ValueError, IndexError):
            pass
            
    def finish_experiment(self):
        """Rewrite finish_experiment to save TRCA data"""
        self.pause_experiment()
        self.title_info.setText("Experiment finished")
        
        # Saving TRCA Experiment Data
        self.trca_integration.save_experiment_data()
        
        self.finish_signal.emit()
