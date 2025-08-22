from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QDialogButtonBox
from PySide6.QtCore import Qt
from customer import BG_COLOR, FG_COLOR, create_btn, create_btn2, create_input

class ShapePropertiesDialog(QDialog):
    def __init__(self, frequency=0, angle=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shape Properties")
        self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR};")
        # self.setModal(True)
        self.setFixedSize(400, 200)
        
        self.frequency = frequency
        self.angle = angle
        
        self._init_layout()
        
    def _init_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        
        # Frequency 输入框
        freq_input = create_input("Frequency (Hz): ")
        layout.addLayout(freq_input['layout'])
        self.freq_edit = freq_input['widget']
        
        # Angle 输入框
        angle_input = create_input("Phase (π): ")
        layout.addLayout(angle_input['layout'])
        self.angle_edit = angle_input['widget']
        
        # 按钮
        btn_layout = QHBoxLayout()
        ok_btn = create_btn2("OK")
        cancel_btn = create_btn2("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def get_values(self):
        try:
            freq = float(self.freq_edit.text())
        except ValueError:
            freq = 0
        try:
            angle = float(self.angle_edit.text())
        except ValueError:
            angle = 0
        return freq, angle 