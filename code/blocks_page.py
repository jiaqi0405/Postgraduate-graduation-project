from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QFrame
from PySide6.QtCore import Qt, Signal

from customer import BG_COLOR, FG_COLOR, BTN_COLOR, Title_size, create_btn, create_btn2, create_combo, create_label, create_input, Page, simple_input
from experiment_page import ExperimentPage
from experiment_window import ExperimentWindow
from utils import parse_size

class BlocksPage(Page):

    confirm_setting_signal = Signal()
    def __init__(self, vars, switch_to_appearance, switch_to_experiment, switch_to_motion):
        super().__init__(vars)
        self.vars = vars
        self.switch_to_appearance = switch_to_appearance
        self.switch_to_experiment = switch_to_experiment
        self.switch_to_motion = switch_to_motion
        self.experiment_page = None
        self.experiment_window = None
        self._init_layout()


    def _init_layout(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(0)

        # 上部标题
        title = QLabel("Blocks")
        self.title = title
        title.setStyleSheet(f"font-size: { Title_size}px; color: {FG_COLOR};")
        layout.addWidget(title, alignment=Qt.AlignTop)

        # 中部左右布局
        center = QHBoxLayout()
        # 左2/3
        left = QFrame()
        left.setStyleSheet(f" background: {BG_COLOR};")
        
        block_hbox = QHBoxLayout()
        self.blocks_input = self.custom_input("Number of blocks: ", lambda v: self.vars.update({"blocks": parse_size(v)}))
        block_hbox.addStretch()
        block_hbox.addLayout(self.blocks_input['layout'])
        block_hbox.addStretch()
        left.setLayout(block_hbox)
        center.addWidget(left, stretch=2)
        # 右1/3
        right = QVBoxLayout()
        right.addStretch()
        right.addWidget(create_label("Single trial", 72))
        right.addSpacing(70)

        self.cue_input = self.custom_input("Cue time(s): ", lambda v: self.vars.update({"cue": parse_size(v)}))
        right.addLayout(self.cue_input['layout'])
        self.flash_input = self.custom_input("Flash time(s): ", lambda v: self.vars.update({"flash": parse_size(v)}))
        right.addLayout(self.flash_input['layout'])
        self.pause_input = self.custom_input("Pause time(s): ", lambda v: self.vars.update({"pause": parse_size(v)}))
        right.addLayout(self.pause_input['layout'])

        right.addSpacing(100)

        self.rest_input = self.custom_input("Rest time between blocks(s): ", lambda v: self.vars.update({"rest": parse_size(v)}))
        right.addLayout(self.rest_input['layout'])

        # self.window_input = self.custom_input("Time window (s): ", lambda v: self.vars.update({"window": parse_size(v)}))
        # right.addLayout(self.window_input['layout'])

        right.addStretch()


        right_widget = QWidget()
        right_widget.setLayout(right)
        center.addWidget(right_widget, stretch=1)
        layout.addLayout(center, stretch=1)

        # 下部 Back/Next
        bottom_bar = QHBoxLayout()
        back_btn = create_btn("<< Back")
        self.back_btn = back_btn
        back_btn.clicked.connect(self.switch_to_appearance)
        next_btn = create_btn2("Confirm settings", "#FF0000", 300)
        # next_btn.setStyleSheet(f"background: {FG_COLOR}; color: {'#ff0000'}; border: none; font-size: 24px;")

        next_btn.clicked.connect(self.confirm_settings)  # 可扩展
        bottom_bar.addWidget(back_btn, alignment=Qt.AlignLeft)
        bottom_bar.addStretch()
        bottom_bar.addWidget(next_btn, alignment=Qt.AlignRight)
        layout.addLayout(bottom_bar)
        self.setLayout(layout) 

        
    def showEvent(self, event):
        super().showEvent(event)
        self.title.setText((f"{self.vars.get('type')} >> Setting: number >> Appearance >> Blocks"))

        self.back_btn.clicked.disconnect()
        if self.vars.get('type') == 'SSVEP':
            self.back_btn.clicked.connect(self.switch_to_appearance)
        else:
            self.back_btn.clicked.connect(self.switch_to_motion)


        self.blocks_input['widget'].setText(str(self.vars.get('blocks')))
        self.cue_input['widget'].setText(str(self.vars.get('cue')))
        self.flash_input['widget'].setText(str(self.vars.get('flash')))
        self.pause_input['widget'].setText(str(self.vars.get('pause')))
        self.rest_input['widget'].setText(str(self.vars.get('rest')))
        # self.window_input['widget'].setText(str(self.vars.get('window')))

    def custom_input(self, text, handler):
        
        label = create_label(text)
        edit = simple_input(None, 100)
        layout = self.lay_left(label, edit)
        edit.textChanged.connect(handler)
        return {'layout':layout, 'widget':edit}

    def confirm_settings(self):
        cue_timer = self.vars.get('cue_timer')
        cue_timer.setInterval(1000 * float(self.vars.get('cue')))
        cue_timer.setSingleShot(True)
        
        flash_timer = self.vars.get('flash_timer')
        flash_timer.setInterval(1000 * float(self.vars.get('flash')))
        flash_timer.setSingleShot(True)

        pause_timer = self.vars.get('pause_timer')
        pause_timer.setInterval(1000 * float(self.vars.get('pause')))
        pause_timer.setSingleShot(True)

        rest_timer = self.vars.get('rest_timer')
        rest_timer.setInterval(1000 * float(self.vars.get('rest')))
        rest_timer.setSingleShot(True)

        # 初始化记录
        self.vars.update({'process':{
                'block': 1, 'trial':1, 'curr_index':0
            }})

        # self.vars.get('frame').print_shapes()

        self.confirm_setting_signal.emit()
        
        # 创建并显示弹出窗口
        if not self.experiment_window:
            self.experiment_window = ExperimentWindow(self.vars)

            exp_page = self.vars.get('pages').get('experiment')
            self.experiment_window.main.start_exp_signal.connect(exp_page.handle_start_signal)
        self.experiment_window.show_main()
        self.center_on_screen(self.experiment_window)
        self.experiment_window.show()

        self.switch_to_experiment()

    def center_on_screen(self, widget):
        screen = widget.screen() or QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        widget_geometry = widget.frameGeometry()
        center_point = screen_geometry.center()
        widget_geometry.moveCenter(center_point)
        widget.move(widget_geometry.topLeft())
