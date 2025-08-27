from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox
from PySide6.QtGui import QPixmap, QCursor
from PySide6.QtCore import Qt

from customer import BG_COLOR, FG_COLOR, create_btn, Page, create_label, simple_combo


class MainPage(Page):
    def __init__(self, vars,switch_to_setting):
        super().__init__(vars)
        # self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR};")
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(0)

        # 顶部栏
        top_bar = QHBoxLayout()
        # 左上角 icon
        icon_label = QLabel()
        icon_path = "C:/Users/yjq/Desktop/guifinal/gui1/icon.png"  
        icon_pixmap = QPixmap(icon_path)
        if not icon_pixmap.isNull():
            icon_label.setPixmap(icon_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            print(f"Failed to load icon from {icon_path}")
        icon_label.setFixedSize(200, 100)
        top_bar.addWidget(icon_label, alignment=Qt.AlignLeft)
        top_bar.addStretch()
        layout.addLayout(top_bar)
        layout.addStretch()

        # 中间 Paradigm + 下拉
        center = QHBoxLayout()
        center.setAlignment(Qt.AlignCenter)
        paradigm_label = create_label("Paradigm: ")
        paradigm_combo = simple_combo(["SSVEP + motion"], 300)
        paradigm_combo.currentTextChanged.connect(lambda v: self.vars.update({"type": v}))

        label_combo = self.lay_h_center(paradigm_label, paradigm_combo)
        center.addLayout(label_combo)
        layout.addLayout(center)
        layout.addStretch()

        # 右下角 Setting 链接按钮
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        setting_btn = create_btn(">> Setting")
        setting_btn.clicked.connect(switch_to_setting)
        bottom_bar.addWidget(setting_btn, alignment=Qt.AlignRight)
        layout.addLayout(bottom_bar)
        self.setLayout(layout) 
