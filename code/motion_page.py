from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QFrame, QLineEdit, QLayout, QLayoutItem
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from shape import ShapeWidget, Square, Rectangle, Circle, Ellipse, Triangle, FreeCurve
from customer import BG_COLOR, FG_COLOR, BTN_COLOR, FSIZE, Title_size, create_btn, create_btn2, create_combo, create_label, create_input, UtilsWidget, Page, simple_combo, simple_input

class MotionPage(Page):
    def __init__(self, vars, switch_to_appearance, switch_to_blocks):
        super().__init__(vars)
        self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR};")
        self.switch_to_appearance = switch_to_appearance
        self.switch_to_blocks = switch_to_blocks
        self.selected_index = None
        self.shape_widgets = []
        self.size_hint_widget = None
        self._init_layout()

    def _init_layout(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(10)

        # 上部标题
        title = QLabel("SSVEP >> Setting: number >> Appearance >> Status")
        self.title = title
        title.setStyleSheet(f"font-size: { Title_size}px; color: {FG_COLOR};")
        self.layout.addWidget(title, alignment=Qt.AlignTop)

        # 中部左右布局
        self.center = QHBoxLayout()
        # 左2/3
        self.left_frame = QFrame()
        self.center.addWidget(self.left_frame, stretch=2)
        # 右1/3
        right = QVBoxLayout()
        
        right.addStretch()

        status_label = create_label("Status: ")
        status_combo = simple_combo(["dynamic", "static"])
        status_layout = self.lay_h_center(status_label, status_combo)

        right.addLayout(status_layout)
        self.status_combo = status_combo
        self.status_combo.currentTextChanged.connect(lambda v: self.on_status_changed(v))
        
        # right.addStretch()
        right.addLayout(self.dest_layout())
        right.addStretch()
        
        right_widget = QWidget()
        right_widget.setLayout(right)
        self.right_widget = right_widget
        self.center.addWidget(self.right_widget, stretch=1)
        self.layout.addLayout(self.center, stretch=1)

        # 下部 Back/Next
        bottom_bar = QHBoxLayout()
        back_btn = create_btn("<< Back")
        back_btn.clicked.connect(self.switch_to_appearance)
        next_btn = create_btn(">> Next")
        next_btn.clicked.connect(self.to_next_page)
        
        bottom_bar.addWidget(back_btn, alignment=Qt.AlignLeft)
        bottom_bar.addStretch()
        bottom_bar.addWidget(next_btn, alignment=Qt.AlignRight)

        self.layout.addLayout(bottom_bar)
        self.setLayout(self.layout)

    def dest_layout(self):
        hbox = QHBoxLayout()
        vbox0 = QVBoxLayout()
        vbox0.addStretch()
        vbox0.addWidget(self.label("Objective: "))
        # hbox.addLayout(vbox0)

        vbox1 = QVBoxLayout()
        vbox1.addStretch()
        vbox1.addWidget(self.label('X'))
        self.dest_x = simple_input("x", 100)
        self.dest_x.textChanged.connect(self.set_dest_flag)
        vbox1.addWidget(self.dest_x)
        hbox.addLayout(vbox1)

        vbox2 = QVBoxLayout()
        vbox2.addStretch()
        vbox2.addWidget(self.label('Y'))
        self.dest_y = simple_input()
        self.dest_y.textChanged.connect(self.set_dest_flag)
        vbox2.addWidget(self.dest_y)
        hbox.addLayout(vbox2)

        layout = self.lay_h_center(vbox0, hbox)
        return layout

    def label(self, text):
        return create_label(text, FSIZE, Qt.AlignCenter)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_left_rects()

        self.on_status_changed(self.status_combo.currentText())

    def set_dest_flag(self):
        
        if not self.left_frame.selected_shape:
            return
        selected_shape = self.left_frame.selected_shape
            
        try:
            x = float(self.dest_x.text())
        except Exception:
            return
        try:
            y = float(self.dest_y.text())
            y = self.left_frame.height() - y - selected_shape.height()
        except Exception:
            return

        # print("dest is..", x, y)

        
        x = max(0, x)
        x = min(x, self.left_frame.width() - selected_shape.width())

        y = max(0, y)
        y = min(y, self.left_frame.height() - selected_shape.height())

        self.left_frame.selected_shape.set_dest_position(x - selected_shape.width() / 2, y - selected_shape.height() / 2)
        # print("set dest flag", x, y)
        self.left_frame.add_dest_flag(x, y)
        

    def refresh_left_rects(self):
        self.center.removeWidget(self.left_frame)
        self.center.removeWidget(self.right_widget)

        self.left_frame = self.vars.get('frame').copy();
        self.left_frame.select_action = self.select_shape
        self.left_frame = self.left_frame.copy();
        self.left_frame.dest_signal.connect(self.update_dest)

        self.left_frame.setStyleSheet(f"border: 2px solid {FG_COLOR}; background: {BG_COLOR};")
        self.select_shape(0)

        
        self.center.addWidget(self.left_frame, stretch=2)
        self.center.addWidget(self.right_widget, stretch=1)

    def on_status_changed(self, value):
        self.left_frame.update_selected_motion(value)

        if value == 'static':
            self.enable(self.dest_x, False)
            self.enable(self.dest_y, False)
        else:
            self.enable(self.dest_x, True)
            self.enable(self.dest_y, True)

    def enable(self, widget, enable):
        widget.setEnabled(enable)
        fsize = 24
        if enable:
            widget.setStyleSheet(f"background: {'#FFFFFF'}; color: black;font-size:{fsize}px")
        else:
            widget.setStyleSheet(f"background: {'#aeaeaf'}; color: black;font-size:{fsize}px")
    
    def select_shape(self, idx):
        self.left_frame.select_shape(idx)

        # 更新下拉框和预览
        self.update_controls_for_shape()
    
    
    def update_controls_for_shape(self):
        shape = self.left_frame.selected_shape
        if not shape:
            return
        # 更新 Shape 下拉框
        motion_status = shape.motion_status
        index = self.status_combo.findText(motion_status)
        if index >= 0:
            self.status_combo.setCurrentIndex(index)

            
        # 更新 dest locartion 
        dest_y = self.left_frame.height() - shape.dest_y - shape.height()
        self.update_dest(shape.dest_x, dest_y)

    def update_dest(self, x, y):
        self.block_edit_value(self.dest_x, str(x))
        self.block_edit_value(self.dest_y, str(y))

    def to_next_page(self):
        self.vars.update({'frame': self.left_frame})
        self.switch_to_blocks()
