from PySide6.QtWidgets import QWidget, QDialog
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPolygon
from PySide6.QtCore import QPoint, Qt, Signal, QTimer, QSize

import time
import random
import traceback
import math

from shape_dialog import ShapePropertiesDialog
from customer import BG_COLOR, FG_COLOR, BTN_COLOR, create_btn, create_combo, create_label, create_input
from utils import parse_size, parse_a_b, distance

SIZE = 100

BORDER_SIZE = 5

class ShapeWidget(QWidget):
    drag_stop_signal = Signal()
    def __init__(self, x=0, y=0, color='#FFFFFF', border_color=QColor(220,0,0), parent=None):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.pos_x = 0
        self.pos_y = 0
        self.dest_x = 0
        self.dest_y = 0
        self.src_x = 0
        self.src_y = 0
        self.color_text = color
        self.color = QColor(self.color_text)
        self.src_color = QColor(self.color_text)
        self.border_color = border_color

        self.target_radius = 30
        
        self.setMinimumSize(20, 20)
        self.setMaximumSize(300, 300)
        self.options = {}

        self.frequency = 20
        self.angle = 0
        self.frame_index = 0

        self.state = "init" # init selected target result flashing
        self.state_borders = {'init':BG_COLOR, 'selected':'#FF8C00', 'result':'#FFFF00', 'target':'#FF0000'}

        # --- 闪烁相关 ---
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self.change_luminance)
        self.flash_on = True

        self.proxy = None

        self.motion_status = 'static'

        self.text_type = 'normal'  # normal flag center target nothing

        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.move_to_dest)
        self.speed_x = 0
        self.speed_y = 0

        self.dragging = False
        self.double_clicked = False

        self.click_x = 0
        self.click_y = 0

        self.id = f'{time.time()} - {random.randint(0, 10000)}'
        # traceback.print_stack()

        self.dest_shape = None

        self.set_state('init')
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_proxy(self, proxy):
        self.proxy = proxy

    def set_position(self, x, y):
        if self.dest_x == self.pos_x and self.dest_y == self.pos_y:
            self.dest_x = x
            self.desy_y = y

        self.pos_x = x
        self.pos_y = y
        if self.proxy:
            self.proxy.setPos(x, y)
        self.update()

    def set_dest_position(self, x, y):
        self.dest_x = x
        self.dest_y = y
        # print("set dest pos", x, y)

    def set_color(self, color):
        if isinstance(color, str):
            self.color_text = color
            self.color = QColor(color)
            self.src_color = QColor(color)
        else:
            print('set_color must use text color', color)
        self.update()

    def set_border_color(self, border_color):
        if isinstance(border_color, str):
            self.border_color = QColor(border_color)
        else:
            self.border_color = border_color
        self.update()

    def set_size(self, *args):
        pass

    def update_options(self, options):
        for key, value in options.items():
            if hasattr(self, key):
                setattr(self, key, value)
    def get_options(self):
        return {
            'pos_x':self.pos_x, 'pos_y':self.pos_y, 'dest_x':self.dest_x, 'dest_y':self.dest_y,
            'frequency': self.frequency, 'angle':self.angle, 'color':self.color, 'src_color':self.src_color,
            'color_text':self.color_text, 'border_color':self.border_color,
            'motion_status':self.motion_status,
            'dest_shape': self.dest_shape
        }

    def set_state(self, state):
        self.state = state
        self.set_border_color(self.state_borders.get(state, BG_COLOR))

        if self.dest_shape:
            if state == 'selected':
                self.dest_shape.set_color(FG_COLOR)
            else:
                self.dest_shape.set_color("#6d6d6d")
        

    def paintEvent(self, event):
        # 闪烁时切换明暗
        # if self.state == 'flashing' and not self.flash_on:
        #     painter = QPainter(self)
        #     painter.setRenderHint(QPainter.Antialiasing)
        #     painter.fillRect(self.rect(), QColor(BG_COLOR))
        #     return
        self.draw_shape(event)
        if self.text_type == 'normal':
            self.draw_text(event)
        elif self.text_type == 'flag':
            self.draw_flag(event)
        elif self.text_type == 'center':
            self.draw_center(event, 6)
        elif self.text_type == 'target':
            self.draw_center(event, self.target_radius)
        else:
            pass
    def draw_flag(self, event):
        pass
    def draw_shape(self, event):
        pass
    def draw_center(self, event, r):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor("#FF0000"), 3)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor("#FF0000")))
        x = self.x if self.x else (self.width() - 2*r) // 2
        y = self.y if self.y else (self.height() - 2*r) // 2
        painter.drawEllipse(x, y, 2*r, 2*r)
    
    def draw_text(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(0, 0, 0), 1))  # 黑色文字
        
        # 设置字体
        font = painter.font()
        font.setPointSize(8)
        font.setFamily("Comic Sans MS")
        font.setBold(True)
        painter.setFont(font)
        
        # 计算中心位置
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # 绘制 frequency 和 angle
        text1 = f"{self.frequency} Hz"
        text2 = f"{self.angle} π"
        
        # 计算文字位置（竖直排列）
        text_rect1 = painter.fontMetrics().boundingRect(text1)
        text_rect2 = painter.fontMetrics().boundingRect(text2)
        
        x1 = center_x - text_rect1.width() // 2
        y1 = center_y - text_rect1.height() - 2
        x2 = center_x - text_rect2.width() // 2
        y2 = center_y + 2
        
        painter.drawText(x1, y1, text1)
        painter.drawText(x2, y2, text2)
    def setup_painter(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(0, 0, 0), 1))  # 黑色文字
        
        # 设置字体
        font = painter.font()
        font.setPointSize(16)
        font.setFamily("Comic Sans MS")
        font.setBold(True)
        painter.setFont(font)
        
        # 计算中心位置
        center_x = self.width() // 2
        center_y = self.height() // 2
        return painter, center_x, center_y

    def flash(self):
        if self.frequency <= 0:
            return
        self.frame_index = 0
        interval = int(500 / self.frequency)  # ms, 一明一暗
        interval = max(10, interval) # 防止闪得太快
        self.flash_on = True
        self.flash_timer.stop()
        self.flash_timer.start(interval)
        self.set_state('flashing')
        self.update()

    def stop_flash(self):
        self.flash_timer.stop()
        self.flash_on = True
        self.update()

    def change_luminance(self):
        self.flash_on = not self.flash_on
        refresh_rate = 60.0
        luminance = 0.5 *(1 + math.sin(2 * math.pi * self.frequency * (self.frame_index / refresh_rate) + self.angle))
        self.frame_index = self.frame_index + 1

        self.color = self.change_color(self.src_color, luminance)
        self.update()

    def change_color(self, qcolor: QColor, target_lightness_0_1: float) -> QColor:
        hsl = qcolor.toHsl()
        h = hsl.hue()
        s = hsl.saturation()
        l = int(target_lightness_0_1 * 255)
        return QColor.fromHsl(h, s, l)


    def copy(self):
        copy = self.__class__()
        options = self.get_options()
        copy.update_options(options)
        return copy

    def parse_size_and_update(self, text):
        pass
    def get_size_text(self):
        return ''
    def set_motion_status(self, motion):
        self.motion_status = motion
        if motion != 'dynamic':
            if self.dest_shape:
                self.dest_shape.deleteLater()
                self.dest_shape = None
    def set_text_type(self, text_type):
        self.text_type = text_type

    def stop_move(self):
        self.move_timer.stop()

    def start_move(self, flash_time):
        # print('start move', self.dest_x, self.pos_x, self.dest_y, self.pos_y)
        if self.motion_status == 'static':
            return
        if self.dest_x == self.pos_x and self.dest_y == self.pos_y:
            return
        interval = 20.0
        steps = flash_time * 1000 / interval
        self.speed_x = (self.dest_x - self.pos_x) / steps
        self.speed_y = (self.dest_y - self.pos_y) / steps
        self.move_timer.start(interval)
        # print('start speed', self.speed_x, self.speed_y)
    def move_to_dest(self):
        """向目标移动"""
        if distance(self.pos_x, self.pos_y, self.dest_x, self.dest_y) < distance(0, 0, self.speed_x, self.speed_y):
            self.pos_x = self.dest_x
            self.pos_y = self.dest_y
            self.move_timer.stop()

        self.pos_x = self.pos_x + self.speed_x
        self.pos_y = self.pos_y + self.speed_y

        if self.proxy:
            self.proxy.setPos(self.pos_x, self.pos_y)

    def handle_drag_signal(self, x, y):
        if self.dragging:
            self.set_position(x - self.click_x, y - self.click_y)

    def handle_release_signal(self):
        self.dragging = False
        # print("shape release", self.pos_x, self.pos_y, self.dest_x, self.dest_y)
        # 拖动停止更新信息
        self.drag_stop_signal.emit()
    def update_size11(self):
        pass

    def store_position(self):
        self.src_x = self.pos_x
        self.src_y = self.pos_y

    def recover_position(self):
        self.set_position(self.src_x, self.src_y)

    def recover_color(self):
        self.color = self.src_color
        self.update()

    def center_from(self, x, y):
        return distance(self.pos_x, self.pos_y, x, y)

    def close_to(self, x, y):
        return self.center_from(x, y) < self.target_radius



class ClickableShape(ShapeWidget):
    clicked = Signal()
    
    def mousePressEvent(self, event):
        # print('shape click', event.pos())
        if self.double_clicked:
            self.double_clicked = False
            return
        else:
            self.dragging = True
        # self.grabMouse()  # 强制捕获鼠标事件
        self.click_x = event.pos().x()
        self.click_y = event.pos().y()
        self.clicked.emit()
        super().mousePressEvent(event)
    

    def mouseDoubleClickEvent(self, event):
        self.double_clicked = True
        # 双击时弹出属性对话框
        dialog = ShapePropertiesDialog(self.frequency, self.angle)
        if dialog.exec() == QDialog.Accepted:
            freq, angle = dialog.get_values()
            self.frequency = freq
            self.angle = angle
            self.update()
        super().mouseDoubleClickEvent(event)

class Square(ClickableShape):
    def __init__(self, side=SIZE, **kwargs):
        super().__init__(**kwargs)
        self.set_size(side)

    def set_size(self, side):
        self.side = side
        self.target_radius = self.side * 0.45

        self.setMaximumSize(side, side)
        self.resize(side, side)  # 手动放大

        self.update()

    def draw_shape(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.border_color, BORDER_SIZE)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color))
        s = self.side
        x = self.x if self.x else (self.width() - s) // 2
        y = self.y if self.y else (self.height() - s) // 2
        painter.drawRect(x, y, s, s)
    def parse_size_and_update(self, text):
            side = parse_size(text)
            if side > 0:
                self.set_size(side)
    def get_size_text(self):
        return str(self.side)
    def draw_flag(self, event):
        painter, center_x, center_y = self.setup_painter()
        
        # 绘制 frequency 和 angle
        text1 = "side"
        
        # 计算文字位置（竖直排列）
        text_rect1 = painter.fontMetrics().boundingRect(text1)
        
        x1 = center_x - text_rect1.width() // 2
        y1 = self.height() - 10
        
        painter.drawText(x1, y1, text1)
    def get_options(self):
        options = super().get_options()
        options.update({
            'side':self.side
        })
        return options
    # def update_options(self, options):
    #     super().update_options(options)
    #     for key, value in options.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
                
    def update_size11(self):
        self.set_size(self.side)

    def center_from(self, x, y):
        return distance(self.pos_x + self.side/2, self.pos_y + self.side/2, x, y)

class Rectangle(ClickableShape):
    def __init__(self, width=SIZE, height=SIZE * 0.8, **kwargs):
        super().__init__(**kwargs)
        self.set_size(width, height)

    def set_size(self, width, height):
        self.rect_width = width
        self.rect_height = height
        self.target_radius = min(self.rect_width, self.rect_height) * 0.45
        self.setMaximumSize(width, height)
        self.resize(width, height) 
        self.update()

    def draw_shape(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.border_color, BORDER_SIZE)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color))
        w = self.rect_width
        h = self.rect_height
        x = self.x if self.x else (self.width() - w) // 2
        y = self.y if self.y else (self.height() - h) // 2
        painter.drawRect(x, y, w, h)
    def parse_size_and_update(self, text):
            width, height = parse_a_b(text)
            if width > 0 and height > 0:
                self.set_size(width, height)
    def get_size_text(self):
        return f'{self.rect_width} x {self.rect_height}'

    def draw_flag(self, event):
        painter, center_x, center_y = self.setup_painter()
        
        # 绘制 frequency 和 angle
        text1 = "a"
        
        # 计算文字位置（竖直排列）
        text_rect1 = painter.fontMetrics().boundingRect(text1)
        
        x1 = center_x - text_rect1.width() // 2
        y1 = self.height() - 10

        x2 = self.height() - 20
        y2 = center_y
        
        painter.drawText(x1, y1, 'a')
        painter.drawText(x2, y2, 'b')
    def get_options(self):
        options = super().get_options()
        options.update({
            'rect_width':self.rect_width,
            'rect_height':self.rect_height
        })
        return options
    def update_size11(self):
        self.set_size(self.rect_width, self.rect_height)
    def center_from(self, x, y):
        return distance(self.pos_x + self.rect_width/2, self.pos_y + self.rect_height/2, x, y)

class Circle(ClickableShape):
    def __init__(self, radius=SIZE/2, **kwargs):
        super().__init__(**kwargs)
        self.set_size(radius)

    def set_size(self, radius):
        self.radius = radius
        self.target_radius = self.radius * 0.9
        delta = 6
        self.setMaximumSize(radius * 2 + delta, radius * 2 + delta)
        self.resize(radius * 2 + delta, radius * 2 + delta) 
        self.update()

    def draw_shape(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(self.border_color, BORDER_SIZE)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color))
        r = self.radius
        x = self.x if self.x else (self.width() - 2*r) // 2
        y = self.y if self.y else (self.height() - 2*r) // 2
        painter.drawEllipse(x, y, 2*r, 2*r)
    def parse_size_and_update(self, text):
            radius = parse_size(text)
            if radius > 0:
                self.set_size(radius)
    def get_size_text(self):
        return str(self.radius)
    def draw_flag(self, event):
        painter, center_x, center_y = self.setup_painter()
        
        # 绘制 frequency 和 angle
        text1 = "r"
        
        # 计算文字位置（竖直排列）
        text_rect1 = painter.fontMetrics().boundingRect(text1)
        
        x1 = center_x + 20
        y1 = center_y
        
        painter.drawText(x1, y1, text1)
    def get_options(self):
        options = super().get_options()
        options.update({
            'radius':self.radius
        })
        return options
    def update_size11(self):
        self.set_size(self.radius)
    def center_from(self, x, y):
        return distance(self.pos_x + self.radius, self.pos_y + self.radius, x, y)

class Ellipse(ClickableShape):
    def __init__(self, a=SIZE/2, b=SIZE*0.4, **kwargs):
        super().__init__(**kwargs)
        self.set_size(a, b)

    def set_size(self, a, b):
        self.a = a  # 长轴
        self.b = b  # 短轴
        self.target_radius = min(self.a, self.b) * 0.9
        delta = 6
        self.setMaximumSize(a * 2 + delta, b * 2 + delta)
        self.resize(a * 2 + delta, b * 2 + delta) 
        self.update()

    def draw_shape(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.border_color, BORDER_SIZE)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color))
        a = self.a
        b = self.b
        x = self.x if self.x else (self.width() - 2*a) // 2
        y = self.y if self.y else (self.height() - 2*b) // 2
        painter.drawEllipse(x, y, 2*a, 2*b)
    def parse_size_and_update(self, text):
            width, height = parse_a_b(text)
            if width > 0 and height > 0:
                self.set_size(width, height)
    def get_size_text(self):
        return f'{self.a} x {self.b}'
    def draw_flag(self, event):
        painter, center_x, center_y = self.setup_painter()
        
        # 绘制 frequency 和 angle
        text1 = "a"
        text2 = "b"
        
        # 计算文字位置（竖直排列）
        text_rect1 = painter.fontMetrics().boundingRect(text1)
        
        x1 = center_x + 20
        y1 = center_y
        x2 = center_x
        y2 = center_y + 20
        
        painter.drawText(x1, y1, text1)
        painter.drawText(x2, y2, text2)
    def get_options(self):
        options = super().get_options()
        options.update({
            'a':self.a,
            'b':self.b
        })
        return options
    def update_size11(self):
        self.set_size(self.a, self.b)
    def center_from(self, x, y):
        return distance(self.pos_x + self.a, self.pos_y + self.b, x, y)

class Triangle(ClickableShape):
    def __init__(self, side=SIZE, **kwargs):
        super().__init__(**kwargs)
        self.set_size(side)

    def set_size(self, side):
        self.side = side
        self.target_radius = self.side * 0.2
        delta = 6
        self.setMaximumSize(side + delta, side + delta)
        self.resize(side + delta, side + delta) 
        self.update()

    def draw_shape(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # painter.fillRect(self.rect(), QColor(FG_COLOR))
        pen = QPen(self.border_color, BORDER_SIZE)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color))
        s = self.side
        x = self.x if self.x else self.width() // 2
        y = self.y if self.y else (self.height() - s) // 2
        # 画等边三角形
        points = [
            QPoint(x, y),
            QPoint(x - s//2, y + int(s * 0.866)),
            QPoint(x + s//2, y + int(s * 0.866))
        ]
        painter.drawPolygon(QPolygon(points))
    def parse_size_and_update(self, text):
            side = parse_size(text)
            if side > 0:
                self.set_size(side)
    def get_size_text(self):
        return str(self.side)
    def draw_flag(self, event):
        painter, center_x, center_y = self.setup_painter()
        
        # 绘制 frequency 和 angle
        text1 = "side"
        
        # 计算文字位置（竖直排列）
        text_rect1 = painter.fontMetrics().boundingRect(text1)
        
        x1 = center_x - text_rect1.width() // 2
        y1 = self.height() - 20
        
        painter.drawText(x1, y1, text1)
    def get_options(self):
        options = super().get_options()
        options.update({
            'side':self.side
        })
        return options
    def update_size11(self):
        self.set_size(self.side)
    def center_from(self, x, y):
        return distance(self.pos_x + self.side /2, self.pos_y + self.side/2, x, y)
        
    def close_to(self, x, y):
        return self.center_from(x, y) < self.target_radius * 3

class FiveStar(ClickableShape):
    def __init__(self, side=SIZE*0.8, **kwargs):
        super().__init__(**kwargs)
        self.set_size(side)

    def set_size(self, side):
        self.side = side
        self.setMaximumSize(int(side * 1), int(side * 1))
        self.resize(side, side) 
        self.update()

    def draw_shape(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.border_color, BORDER_SIZE)
        painter.setPen(pen)
        painter.setBrush(QBrush(self.color))

        from math import sin, cos, pi

        s = self.side
        cx = self.x if self.x else self.width() // 2
        cy = self.y if self.y else self.height() // 2

        r_outer = s // 2
        r_inner = int(r_outer * 0.5)

        points = []
        for i in range(10):
            angle_deg = 36 * i - 90
            angle_rad = angle_deg * pi / 180
            r = r_outer if i % 2 == 0 else r_inner
            x = cx + int(r * cos(angle_rad))
            y = cy + int(r * sin(angle_rad))
            points.append(QPoint(x, y))

        painter.drawPolygon(QPolygon(points))
    def get_options(self):
        options = super().get_options()
        options.update({
            'side':self.side
        })
        return options
    def update_size11(self):
        self.set_size(self.side)


class FreeCurve(ClickableShape):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = []
        self.closed = False
        self.drawing = False
        self.setMouseTracking(True)
        self.setMinimumSize(20, 20)
        # self.setMaximumSize(SIZE, SIZE)
        

    def mousePressEvent(self, event):
        if not self.drawing:
            if not self.closed and event.button() == Qt.RightButton:
                self.drawing = True
                self.points = []
                self.closed = False
                self.update()
            else:
                self.clicked.emit()
        else:
            if event.button() == Qt.LeftButton:
                p = event.position() if hasattr(event, 'position') else event.pos()
                self.points.append(QPoint(int(p.x()), int(p.y())))
                self.update()
            elif event.button() == Qt.RightButton and len(self.points) > 2:
                self.closed = True
                self.drawing = False
                self.update()
        super().mousePressEvent(event)

    def draw_shape(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.drawing and not self.closed:
            # 初始灰色矩形
            painter.setPen(QPen(QColor(180,180,180), 2))
            painter.setBrush(QBrush(QColor(122,122,122)))
            rect = self.rect().adjusted(8, 8, -8, -8)
            painter.drawRect(rect)
        else:
            pen = QPen(self.border_color if not self.closed else QColor(255,140,0) if getattr(self, 'selected', False) else self.border_color, 3)
            painter.setPen(pen)
            painter.setBrush(QBrush(self.color if self.closed else QColor(255,255,255,80)))
            if self.points:
                if self.closed:
                    painter.drawPolygon(QPolygon(self.points))
                else:
                    for i in range(1, len(self.points)):
                        painter.drawLine(self.points[i-1], self.points[i])
                    for pt in self.points:
                        painter.drawEllipse(pt, 3, 3)

    def area(self):
        # 粗略像素面积（闭合后才有意义）
        if not self.closed or len(self.points) < 3:
            return 0
        # 使用多边形面积公式
        pts = self.points
        n = len(pts)
        area = 0
        for i in range(n):
            x1, y1 = pts[i].x(), pts[i].y()
            x2, y2 = pts[(i+1)%n].x(), pts[(i+1)%n].y()
            area += (x1 * y2 - x2 * y1)
        return abs(area) // 2

    def get_options(self):
        options = super().get_options()
        options.update({
            'points':self.points,
            'closed':self.closed,
            'drawing':self.drawing
        })
        return options
    # def set_selected(self, selected: bool):
    #     self.selected = selected
    #     self.update()
    def parse_size_and_update(self, text):
            side = parse_size(text)
            if side > 0:
                self.set_size(side)
    def get_size_text(self):
        return str(self.area())