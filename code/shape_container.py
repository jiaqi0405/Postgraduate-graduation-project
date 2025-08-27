from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QFrame, QLineEdit, QLayout, QLayoutItem, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget
from PySide6.QtCore import Qt, Signal, QObject, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QMouseEvent, QCursor

from shape import ShapeWidget, Square, Rectangle, Circle, Ellipse, Triangle, FreeCurve, FiveStar
from utils import distance

class MyGraphicsView(QGraphicsView):
    drag_signal = Signal(float, float)
    def __init__(self, *args):
        super().__init__(*args)
        # self.setMouseTracking(False)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.trigger_drag_signal(event.pos().x(), event.pos().y())

        boundary = self.rect()
        if not boundary.contains(event.pos()):
            clamped_x = max(min(event.pos().x(), boundary.width()), 0)
            clamped_y = max(min(event.pos().y(), boundary.height()), 0)
            QCursor.setPos(self.mapToGlobal(QPoint(clamped_x, clamped_y)))
        super().mouseMoveEvent(event)

    def trigger_drag_signal(self, x, y):
        self.drag_signal.emit(x, y)

class CustomFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.view = MyGraphicsView(self.scene, self)
        self.view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.view.setSceneRect(self.scene.sceneRect())
        # view.setRenderHints(view.renderHints())
    
    def resizeEvent(self, event):
        self.view.setGeometry(0, 0, self.width(), self.height())
        self.view.setSceneRect(0, 0, self.width() - 5, self.height() - 5)
        super().resizeEvent(event)

    def add(self, widget):
        proxy = self.scene.addWidget(widget)

        proxy.setAcceptHoverEvents(True)
        proxy.setAcceptedMouseButtons(Qt.AllButtons)
        proxy.setFocusPolicy(Qt.StrongFocus)

        widget.set_proxy(proxy)
        proxy.setPos(widget.pos_x, widget.pos_y)
        return proxy

    def get_view(self):
        return self.view

class BoxFrame(CustomFrame):
    dest_signal = Signal(float, float)
    release_signal = Signal()

    def __init__(self, vars, select_action = None, parent=None):
        super().__init__(parent)
        self.vars = vars
        self.select_action = select_action or self.select_shape
        self.shape_widgets = []
        self.selected_shap = None
        
        self.dest = None

    def refresh_shapes(self, n):
        
        self.scene.clear()
        self.shape_widgets = []
        
        for i in range(n):
            shape = Square()
            x, y = self.calculate_position(n, i, shape.side, shape.side)
            shape.set_position(x, y)
            shape.set_border_color(QColor(220,0,0))
            shape.set_color("#FFFFFF")
            shape.clicked.connect(lambda idx=i: self.select_action(idx))
            self.view.drag_signal.connect(shape.handle_drag_signal)
            self.release_signal.connect(shape.handle_release_signal)
            self.shape_widgets.append(shape)
            proxy = self.add(shape)

        
    def select_shape(self, idx):
        self.selected_index = idx
        self.selected_shape = None
        for i, shape in enumerate(self.shape_widgets):
            if i == idx:
                shape.set_state('selected')  # 橙色
                self.selected_shape = shape
            else:
                shape.set_state('init')
            shape.update()

    def update_selected_color(self, color_text):
        if self.selected_shape:
            self.selected_shape.set_color(color_text)

    def replace_selected(self, new_shape):
        self.shape_widgets[self.selected_index] = new_shape
        self.selected_shape = new_shape
        self.add(new_shape)

    def update_selected_size(self, text):
        if not self.selected_shape:
            return
        self.selected_shape.parse_size_and_update(text)

    def copy(self, copy_type=0):
        """
        copy_type:
            0 normal
            1 without text, result, dest shape
        """
        copy = BoxFrame(self.vars)
        for i in range(len(self.shape_widgets)):
            shape = self.shape_widgets[i]
            shape_copy = shape.copy()
            # print('shape     options', shape.get_options())
            # print('copy----- options', shape_copy.get_options())

            if copy_type == 1:
                shape_copy.set_text_type('nothing')

            shape_copy.clicked.connect(lambda idx=i: self.select_action(idx))

            copy.shape_widgets.append(shape_copy)
            shape_copy.update_size11()
            copy.add(shape_copy)

        if copy_type == 0:
            # copy dest shape
            for i in range(len(self.shape_widgets)):
                shape = self.shape_widgets[i]
                if shape.motion_status == 'dynamic' and shape.dest_shape:
                    # print('src pos', shape.dest_shape.pos_x, shape.dest_shape.pos_y)
                    dest_copy = shape.dest_shape.copy()
                    dest_copy.set_text_type('nothing')
                    dest_copy.update_size11()
                    proxy = copy.add(dest_copy)
                    # print('dest_copy pos', dest_copy.pos_x, dest_copy.pos_y)
                    proxy.setPos(dest_copy.pos_x, dest_copy.pos_y)

        return copy

    def set_selected_state(self, state):
        if self.selected_shape:
            self.selected_shape.set_state(state)

    def get_shape(self, index):
        if index >= 0 and index < len(self.shape_widgets):
            return self.shape_widgets[index]
        else:
            return None

    def update_selected_motion(self, motion):
        if not self.selected_shape:
            return
        self.selected_shape.set_motion_status(motion)

    def all_flash(self):
        for shape in self.shape_widgets:
            shape.flash()
    def all_stop_flash(self):
        for shape in self.shape_widgets:
            shape.stop_flash()
    
    def all_start_move(self, flash_time):
        for shape in self.shape_widgets:
            shape.start_move(flash_time)
    
    def all_stop_move(self):
        for shape in self.shape_widgets:
            shape.stop_move()

    def mousePressEvent(self, event):
        self.click_handler(event)
        super().mousePressEvent(event)
    def mouseReleaseEvent(self, event):
        # print('container release', event.pos())
        self.release_signal.emit()
        super().mouseReleaseEvent(event)

    def click_handler(self, event):
        pos = event.position()
        # print("got clicked", pos.x(), pos.y())
        if not self.selected_shape:
            return
        selected = self.selected_shape
        selected.set_dest_position(pos.x() - selected.width() / 2, pos.y() - selected.height() / 2)
        if selected.motion_status == 'dynamic':
            self.add_dest_flag(pos.x(), pos.y())

            dest_y = self.height() - pos.y()
            self.trigger_dest_signal(pos.x(), dest_y)


    def add_dest_flag(self, x, y):
        selected = self.selected_shape

        # 离自己太近是不行的
        # print('distance', selected.center_from(x, y), 'radius', selected.target_radius)
        if selected.close_to(x, y):
            return

        if selected.dest_shape:
            selected.dest_shape.deleteLater()
        dest = FiveStar()
        selected.dest_shape = dest
        size = 32
        dest.set_text_type('nothing')
        dest.set_size(size)
        dest.setMaximumSize(size, size)

        delta = size * 0.5
        dest.set_position(x - delta, y - delta)
        self.add(dest)
        # proxy = self.scene.addWidget(dest)
        # proxy.setPos(x - delta, y - delta)
    
    def trigger_dest_signal(self, x, y):
        self.dest_signal.emit(x, y)


    def calculate_position3(self, n, i, width, height):
        
        unit_x = int(self.width() / 3)
        unit_y = int(self.height() / int((n+3)/3))
    
        x = (i%3)*unit_x
        y = int(i/3)*unit_y

        return x, y
        
    def calculate_position(self, n, i, width, height):
        # 统一使用网格布局，不再特殊处理n=4的情况
        if n <= 4:
            # 对于1-4个图形，使用2x2网格
            col = 2
        elif n <= 6:
            # 对于5-6个图形，使用3x2网格
            col = 3
        elif n <= 9:
            # 对于7-9个图形，使用3x3网格
            col = 3
        else:
            # 更多图形使用更大的网格
            col = 4
        
        # 计算行数
        rows = int((n + col - 1) / col)
        
        # 计算单元格大小
        unit_x = int(self.width() / (col + 1))
        unit_y = int(self.height() / (rows + 1))
        
        # 计算当前图形的行列位置
        row = int(i / col)
        col_pos = i % col
        
        # 计算实际位置（居中对齐）
        x = (col_pos + 1) * unit_x - width // 2
        y = (row + 1) * unit_y - height // 2
        
        # 确保位置在边界内
        x = max(0, min(x, self.width() - width))
        y = max(0, min(y, self.height() - height))

        return x, y
        
    def calculate_position4(self, n, i, width, height):
        
        unit_x = int(self.width() / 6)
        unit_y = int(self.height() / 6)
    
        x = (i%2)*unit_x*2 + unit_x + (unit_x - width / 2)
        y = int(i/2)*unit_y * 2 + unit_y + (unit_y - height / 2)

        return x, y

    def store_all_position(self):
        for shape in self.shape_widgets:
            shape.store_position()

    def recover_all_position(self):
        for shape in self.shape_widgets:
            shape.recover_position()

    def recover_all_color(self):
        for shape in self.shape_widgets:
            shape.recover_color()
            
    def print_shapes(self):
        print('frame shapes ------------------------------')
        for shape in self.shape_widgets:
            print(shape.get_options())
