from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit
from PySide6.QtCore import Qt

from customer import BG_COLOR, FG_COLOR, Title_size, create_btn, Page, create_label, simple_input
from utils import parse_size


class SettingPage(Page):
    def __init__(self, vars, switch_to_main, switch_to_appearance):
        super().__init__(vars)
        self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR}; font-size:")
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(0)

        # 左上角标题
        title = QLabel(f"{vars.get('type')} >> Setting")
        self.title = title
        title.setStyleSheet(f"font-size: { Title_size}px; color: {FG_COLOR};")
        layout.addWidget(title, alignment=Qt.AlignLeft)
        layout.addStretch()

        # 中间输入框
        center = QHBoxLayout()
        center.setAlignment(Qt.AlignCenter)
        label = create_label("Number of stimuli: ")
        center.addWidget(label, alignment=Qt.AlignCenter)
        input_box = simple_input("Enter number...", 200, 24)
        input_box.textChanged.connect(lambda v: self.vars.update({"count": parse_size(v)}))
        self.input_box = input_box
        center.addWidget(input_box, alignment=Qt.AlignCenter)
        layout.addLayout(center)
        layout.addStretch()

        # 底部 Back/Next
        bottom_bar = QHBoxLayout()
        back_btn = create_btn("<< Back")
        back_btn.clicked.connect(switch_to_main)
        next_btn = create_btn(">> Next")
        next_btn.clicked.connect(switch_to_appearance)
        bottom_bar.addWidget(back_btn, alignment=Qt.AlignLeft)
        bottom_bar.addStretch()
        bottom_bar.addWidget(next_btn, alignment=Qt.AlignRight)
        layout.addLayout(bottom_bar)
        self.setLayout(layout) 



    def showEvent(self, event):
        super().showEvent(event)
        self.title.setText(f"{self.vars.get('type')} >> Setting: number")
        self.input_box.setText(str(int(self.vars.get('count'))))
