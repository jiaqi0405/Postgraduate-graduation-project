from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QFrame, QLayout
from PySide6.QtGui import QPixmap, QCursor, QKeyEvent
from PySide6.QtCore import Qt, QSize, Signal
from typing import Optional, Union

BG_COLOR = "#111111"
FG_COLOR = "#FFFFFF"
BTN_COLOR = "#111111"

FSIZE = 32
Title_size = FSIZE


def create_btn(text, fsize=32):
    btn = QPushButton(text)
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    btn.setStyleSheet(f"background: {BTN_COLOR}; color: {FG_COLOR}; border: none; font-size: {fsize}px;")
    return btn
def create_btn2(text, color_text='#FF0000', size=150):
    btn = QPushButton(text)
    btn.setCursor(QCursor(Qt.PointingHandCursor))
    btn.setStyleSheet(f"background: {FG_COLOR}; color: {color_text}; border: none; font-size: 24px; padding: 10px")
    btn.setFixedWidth(size)
    return btn

def create_combo(text, items):
    hbox = QHBoxLayout()
    hbox.setAlignment(Qt.AlignCenter)
    label = QLabel(text)
    label.setStyleSheet(f"color: {FG_COLOR}; font-size: 24px;")
    label.setAlignment(Qt.AlignCenter)
    label.setFixedWidth(200)
    hbox.addWidget(label)
    combo = QComboBox()
    combo.addItems(items)
    
    combo.setStyleSheet(f"background: {FG_COLOR}; color: {BG_COLOR}; font-size: 24px;")
    combo.setFixedWidth(200)

    hbox.addWidget(combo)
    return {"layout": hbox, "widget": combo}

def simple_combo(items, size=200, font_size=24):
    combo = QComboBox()
    combo.addItems(items)
    combo.setStyleSheet(f"background: {FG_COLOR}; color: {BG_COLOR}; font-size: {font_size}px;")
    combo.setFixedWidth(size)

    return combo

def create_label(text, fontsize=FSIZE, align=Qt.AlignCenter):
    text = QLabel(text)
    text.setStyleSheet(f"color: {FG_COLOR}; font-size: {fontsize}px; font-weight:bolder; border: none;")
    text.setAlignment(align)
    return text

def create_input(text=None, width=100):
        
    edit = QLineEdit()  
    edit.setStyleSheet(f"background: {FG_COLOR}; color: {BG_COLOR}; border: 1px solid {FG_COLOR};")
    edit.setFixedWidth(width)
    hbox = None
    if text:
        label = create_label(text)
        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(edit)
    return {"layout": hbox, "widget": edit}

def simple_input(hint=None, width=100, fsize=24):
        
    edit = QLineEdit()  
    if hint:
        edit.setPlaceholderText(hint)
    edit.setStyleSheet(f"background: {FG_COLOR}; color: {BG_COLOR}; font-size:{fsize}px")
    edit.setFixedWidth(width)
    return edit


class DirectionalLineEdit(QLineEdit):
    direct_signal = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Up:
            self.direct_signal.emit(0, -1)
        elif event.key() == Qt.Key_Down:
            self.direct_signal.emit(0, 1)
        elif event.key() == Qt.Key_Left:
            self.direct_signal.emit(-1, 0)
            super().keyPressEvent(event)
        elif event.key() == Qt.Key_Right:
            self.direct_signal.emit(1, 0)
        super().keyPressEvent(event)



class UtilsWidget(QWidget):
    def __init__(self):
        super().__init__()

    def get_direct_layout(self, widget: QWidget) -> Optional[QLayout]:
        """获取直接包含 widget 的 layout（递归方式）"""
        parent = widget.parentWidget()
        if not parent:
            return None

        parent_layout = parent.layout()
        if not parent_layout:
            return None

        return self.get_direct_layout_in_layout(parent_layout, widget)

    def get_direct_layout_in_layout(self, layout: QLayout, widget: QWidget) -> Optional[QLayout]:
        """在给定的 layout 中递归查找 widget 的直接父 layout"""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() == widget:
                return layout
            if item.layout():  # 如果是嵌套 layout，递归查找
                result = self.get_direct_layout_in_layout(item.layout(), widget)
                if result:
                    return result
        return None

        
    def clear_layout(self, layout):
        """递归清空布局中的所有控件"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def block_edit_value(self, widget: QLineEdit, value: str):
        """如果 widget:QLineEdit 的值修改时绑定了函数， 先关闭信号，修改值后再打开
        可以防止程序触发值修改与值修改触发程序之间的死循环调用"""
        widget.blockSignals(True)
        widget.setText(value)
        widget.blockSignals(False)

    def block_combo_value(self, widget: QComboBox, index: int):
        """如果 widget:QComboBox 的值修改时绑定了函数， 先关闭信号，修改值后再打开
        可以防止程序触发值修改与值修改触发程序之间的死循环调用"""
        widget.blockSignals(True)
        widget.setCurrentIndex(index)
        widget.blockSignals(False)


class Page(UtilsWidget):
    def __init__(self, vars = None):
        super().__init__()
        self.vars = vars
        self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR}; font-size:bolder;")

    def lay_h_center(self, wid1, wid2):
        hbox = QHBoxLayout()
        h1 = QHBoxLayout()
        h2 = QHBoxLayout()
        if isinstance(wid1, QWidget):
            h1.addWidget(wid1, alignment=Qt.AlignRight, stretch=1)
        else:
            h1helper = QHBoxLayout()
            h1helper.addStretch()
            h1helper.addLayout(wid1)
            h1.addLayout(h1helper)

        if isinstance(wid2, QWidget):
            h2.addWidget(wid2, alignment=Qt.AlignLeft, stretch=1)
        else:
            h2helper = QHBoxLayout()
            h2helper.addLayout(wid2)
            h2helper.addStretch()
            h2.addLayout(h2helper)
        hbox.addLayout(h1, stretch=1)
        hbox.addLayout(h2, stretch=1)
        return hbox

    def lay_left(self, wid1, wid2):
        hbox = QHBoxLayout()
        h1 = QHBoxLayout()
        h2 = QHBoxLayout()
        if isinstance(wid1, QWidget):
            h1.addWidget(wid1, alignment=Qt.AlignLeft)
        else:
            h1.addLayout(wid1)

        if isinstance(wid2, QWidget):
            h2.addWidget(wid2, alignment=Qt.AlignLeft)
        else:
            h2helper = QHBoxLayout()
            h2helper.addLayout(wid2)
            h2helper.addStretch()
            h2.addLayout(h2helper)
        hbox.addLayout(h1)
        hbox.addLayout(h2, stretch=1)
        return hbox
    def lay_center_1(self, wid):
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addLayout(wid)
        hbox.addStretch()
        return hbox
    