from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QFrame, QLineEdit, QLayout, QLayoutItem, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QGridLayout
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
import json
import os

from shape import ShapeWidget, Square, Rectangle, Circle, Ellipse, Triangle, FreeCurve
from customer import BG_COLOR, FG_COLOR, BTN_COLOR, Title_size, create_btn, create_combo, create_label, create_input, UtilsWidget, DirectionalLineEdit, Page, simple_combo, simple_input
from utils import find_key, parse_size
from shape_container import BoxFrame


class AppearancePage(Page):
    def __init__(self, vars, switch_to_setting, switch_to_blocks, switch_to_motion):
        super().__init__(vars)
        self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR};")
        self.switch_to_setting = switch_to_setting
        self.switch_to_blocks = switch_to_blocks
        self.switch_to_motion = switch_to_motion
        self.shape_widgets = []
        # 图形颜色映射
        self.color_map = {
            '#FFFFFF': 'White',
            '#FF0000': 'Red',
            '#00FF00': 'Green',
            '#0000FF': 'Blue',
            '#808080': 'Gray',
            '#FFFF00': 'Yellow',
            '#FFA500': 'Orange',
            '#FFC0CB': 'Pink',
            '#800080': 'Purple',
            '#A52A2A': 'Brown',
            '#00FFFF': 'CYan',
            '#FFD700': 'Gold'
        }
        self._init_layout()

    def _init_layout(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(0)

        # 上部标题
        title = QLabel("Appearance")
        self.title = title
        title.setStyleSheet(f"font-size: { Title_size}px; color: {FG_COLOR};")
        self.layout.addWidget(title, alignment=Qt.AlignTop)

        # 中部左右布局
        self.center = QHBoxLayout()
        # 左2/3
        self.left_frame = BoxFrame(self.vars, self.select_shape)
        self.left_frame.setStyleSheet(f"border: 2px solid {FG_COLOR}; background: {BG_COLOR};")
        self.center.addWidget(self.left_frame, stretch=2)
        self.vars.update({'frame': self.left_frame})
        # 右1/3
        right = QVBoxLayout()

        right.addStretch()
        
        shape_label = create_label("Shape: ")
        shape_combo = simple_combo(["Square", "Rectangle", "Circle", "Ellipse", "Triangle", "Free curve"])
        shape_layout = self.lay_h_center(shape_label, shape_combo)

        right.addLayout(shape_layout)
        self.shape_combo = shape_combo
        self.shape_combo.currentTextChanged.connect(lambda _: self.on_shape_changed())

        right.addStretch()
        color_label = create_label("Color: ")
        color_combo = simple_combo(self.color_map.values())
        # color_combo = simple_combo(["White", "Red", "Green", "Blue"])
        color_layout = self.lay_h_center(color_label, color_combo)
        right.addLayout(color_layout)
        self.color_combo = color_combo
        self.color_combo.currentTextChanged.connect(lambda _: self.on_color_changed())

        # right.addSpacing(20)
        right.addStretch()

        self.size_hint, size_layout = self.create_hint_layout()
        right.addLayout(size_layout)

        size_label = create_label("Size: ")
        size_edit = simple_input("size", 200)
        size_layout = self.lay_h_center(size_label, size_edit)
        self.size_edit = size_edit
        right.addLayout(size_layout)
        self.size_edit.textChanged.connect(lambda v: self.left_frame.update_selected_size(v))

        right.addStretch()

        right.addSpacing(20)
        # self.move_hint, move_layout = self.create_hint_layout()
        self.move_hint = QVBoxLayout()
        move_layout = self.lay_center_1(self.move_hint)
        # right.addLayout(move_layout)

        text = create_label("Location: ")
        input_x = self.direct_input()
        input_y = self.direct_input()
        # btn_row = QHBoxLayout()
        # btn_row.addWidget(input_x)
        # btn_row.addWidget(input_y)
        # input_layout = self.lay_h_center(text, btn_row)

        self.loc_x = input_x
        self.loc_y = input_y
        input_x.textChanged.connect(self.set_selected_pos)
        input_y.textChanged.connect(self.set_selected_pos)
        input_x.direct_signal.connect(self.move_by_key)
        input_y.direct_signal.connect(self.move_by_key)

        # text_row = QHBoxLayout()
        # text1 = QLabel("         ")
        text2 = create_label("X")
        text3 = create_label("Y")
        # text_row.addWidget(text2, stretch=1)
        # text_row.addWidget(text3, stretch=1)
        # text_layout = self.lay_h_center(text1, text_row)
        # right.addLayout(input_layout)
        # right.addLayout(text_layout)

        grid = QGridLayout()
        grid.addLayout(move_layout, 0, 1, 1, 2)
        grid.addWidget(text, 1, 0)
        grid.addWidget(input_x, 1, 1)
        grid.addWidget(input_y, 1, 2)
        grid.addWidget(text2, 2, 1)
        grid.addWidget(text3, 2, 2)
        grid_hbox = QHBoxLayout()
        grid_hbox.addStretch()
        grid_hbox.addLayout(grid)
        grid_hbox.addStretch()


        right.addLayout(grid_hbox)
        right.addStretch()
        right_widget = QWidget()


        right_widget.setLayout(right)
        self.center.addWidget(right_widget, stretch=1)
        self.layout.addLayout(self.center, stretch=1)

        # 下部 Back/Next
        bottom_bar = QHBoxLayout()

        back_btn = create_btn("<< Back")
        back_btn.clicked.connect(self.switch_to_setting)
        
        # 添加保存和加载按钮
        save_btn = create_btn("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        load_btn = create_btn("Load Settings")
        load_btn.clicked.connect(self.load_settings)
        
        next_btn = create_btn(">> Next")
        self.next_btn = next_btn
        self.next_btn.clicked.connect(self.to_next_page)
        
        bottom_bar.addWidget(back_btn, alignment=Qt.AlignLeft)
        bottom_bar.addStretch()
        bottom_bar.addWidget(save_btn, alignment=Qt.AlignCenter)
        bottom_bar.addSpacing(20) 
        bottom_bar.addWidget(load_btn, alignment=Qt.AlignCenter)
        bottom_bar.addStretch()
        bottom_bar.addWidget(next_btn, alignment=Qt.AlignRight)
        self.layout.addLayout(bottom_bar)
        self.setLayout(self.layout)

    def showEvent(self, event):
        super().showEvent(event)
        self.title.setText((f"{self.vars.get('type')} >> Setting: number >> Appearance"))
        self.refresh_left_rects()
        
        # 尝试加载上次保存的设置
        self.load_settings()

    def to_next_page(self):
        self.vars.update({'frame': self.left_frame})
        if self.vars.get('type') == 'SSVEP':
            self.switch_to_blocks()
        else:
            self.switch_to_motion()

    def refresh_left_rects(self):
        try:
            n = int(self.vars.get("count", 0))
        except Exception:
            n = 0
        n = max(1, n)
        n = min(n, 9)

        self.left_frame.refresh_shapes(n)
        for shape in self.left_frame.shape_widgets:
            shape.drag_stop_signal.connect(self.update_controls_for_shape)
        # 默认选中第一个
        self.select_shape(0)

    def select_shape(self, idx):
        self.left_frame.select_shape(idx)

        # 更新下拉框和预览
        self.update_controls_for_shape()

    def update_controls_for_shape(self):
        shape = self.left_frame.selected_shape
        if not shape:
            return
        # 更新 Shape 下拉框
        shape_type = type(shape).__name__
        index = self.shape_combo.findText(shape_type)
        if index >= 0:
            # self.shape_combo.setCurrentIndex(index)
            self.block_combo_value(self.shape_combo, index)

        # 更新 Color 下拉框
        color_name = self.color_map[shape.color_text]
        index = self.color_combo.findText(color_name)
        if index >= 0:
            # self.color_combo.setCurrentIndex(index)
            self.block_combo_value(self.color_combo, index)

        # 更新 size
        self.block_edit_value(self.size_edit, shape.get_size_text())

        # 更新 location 
        self.loc_x.setFocus()
        self.update_location_display(shape, shape.pos_x, shape.pos_y)
        # 更新预览图形
        self.update_size_hint(shape)

    def update_location_display(self, shape, x, y):
        self.block_edit_value(self.loc_x, str(x + shape.width()/2))
        self.block_edit_value(self.loc_y, str(self.left_frame.height() - shape.height()/2 - y))

    def update_size_hint(self, shape):

        shape_type = type(shape).__name__
        preview = self.create_shape_by_type(shape_type)
        preview.set_text_type('flag')
        # print('shape_type', shape_type, 'pre', type(preview).__name__)
        # preview.set_color(shape.color_text)
        preview.set_border_color(shape.border_color)
        preview.setFixedSize(100, 100)

        # 找到 size_hint 布局并替换
        self.clear_layout(self.size_hint)
        self.size_hint.addWidget(preview)

        self.update_move_hint(shape)

    def update_move_hint(self, shape):

        shape_type = type(shape).__name__
        preview = self.create_shape_by_type(shape_type)
        preview.set_text_type('center')
        # print('shape_type', shape_type, 'pre', type(preview).__name__)
        # preview.set_color(shape.color_text)
        preview.set_border_color(shape.border_color)
        preview.setFixedSize(100, 100)

        # 找到 size_hint 布局并替换
        self.clear_layout(self.move_hint)
        self.move_hint.addWidget(preview)

    def create_shape_by_type(self, shape_type):
            if shape_type == "Square":
                new_shape = Square()
            elif shape_type == "Rectangle":
                new_shape = Rectangle()
            elif shape_type == "Circle":
                new_shape = Circle()
            elif shape_type == "Ellipse":
                new_shape = Ellipse()
            elif shape_type == "Triangle":
                new_shape = Triangle()
            elif shape_type == "Free curve":
                new_shape = FreeCurve()
            else:
                new_shape = Square()
            return new_shape

    def on_shape_changed(self):
        if self.left_frame.selected_shape:
            shape_type = self.shape_combo.currentText()
            old = self.left_frame.selected_shape
            color_text = old.color_text
            border_color = old.border_color
            new_shape = self.create_shape_by_type(shape_type)
            new_shape.set_color(color_text)
            new_shape.set_border_color(border_color)
            # print('options', old.get_options())
            new_shape.update_options(old.get_options())


            new_shape.clicked.connect(lambda idx=self.left_frame.selected_index: self.select_shape(idx))
            self.left_frame.view.drag_signal.connect(new_shape.handle_drag_signal)
            self.left_frame.release_signal.connect(new_shape.handle_release_signal)
            new_shape.drag_stop_signal.connect(self.update_controls_for_shape)
            
            self.left_frame.replace_selected(new_shape)
            old.deleteLater()

            # self.update_size_hint(new_shape)
            self.update_controls_for_shape()


    def on_color_changed(self):
        color_text = find_key(self.color_map, self.color_combo.currentText(), "#FFFFFF")
        self.left_frame.update_selected_color(color_text)

        if self.left_frame.selected_shape:
            self.update_size_hint(self.left_frame.selected_shape)


    def set_selected_pos(self):
        if self.left_frame.selected_shape:
            selected_shape = self.left_frame.selected_shape
        else:
            return
            
        try:
            read_x = float(self.loc_x.text())
            x = read_x - selected_shape.width()/2
        except Exception:
            print('set selected pos Error x', )
            x = selected_shape.pos_x
        try:
            read_y = float(self.loc_y.text())
            y = self.left_frame.height() - read_y - selected_shape.height()/2
        except Exception:
            print('set selected pos Error y', )
            y = selected_shape.pos_y

        x = max(0, x)
        x = min(x, self.left_frame.width() - selected_shape.width())

        y = max(0, y)
        y = min(y, self.left_frame.height() - selected_shape.height())

        self.update_location_display(selected_shape,x, y)

        # print("set selected pos 003", x, y)
        selected_shape.set_position(x, y)

    
    def direct_input(self, width=100, fsize=24):
            
        edit = DirectionalLineEdit()  
        edit.setStyleSheet(f"background: {FG_COLOR}; color: {BG_COLOR}; font-size:{fsize}px")
        edit.setFixedWidth(width)
        return edit
    
    
    def move_by_key(self, x, y):
        # print('move_by key.', x, y)
        if not self.left_frame.selected_shape:
            return
        s = self.left_frame.selected_shape
        # loc_y = self.left_frame.height() - s.height() - s.pos_y
        # s.set_position(s.pos_x + x, s.pos_y + y)
        self.update_location_display(s, s.pos_x + x, s.pos_y + y)
        self.set_selected_pos()
        # self.loc_x.setText(str(s.pos_x + x))
        # self.loc_y.setText(str(loc_y))

    def create_hint_layout(self):
        size_hint = QVBoxLayout()

        size_hbox =QHBoxLayout()
        size_hbox.addSpacing(50)
        size_hbox.addLayout(size_hint)
        size_hbox.addStretch()

        size_layout = self.lay_h_center(create_label(""), size_hbox)
        return size_hint, size_layout

    def get_settings_path(self):
        # 在用户目录下创建配置目录
        config_dir = os.path.expanduser("~/.ssvep_config")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        return os.path.join(config_dir, f"appearance_settings_{self.vars.get('count')}.json")

    def save_settings(self):
        settings = {
            'count': self.vars.get('count'),
            'shapes': []
        }
        
        # 保存每个图形的设置
        for shape in self.left_frame.shape_widgets:
            shape_data = {
                'type': type(shape).__name__,
                'color': shape.color_text,
                'size': shape.get_size_text(),
                'position': {
                    'x': shape.pos_x + shape.width()/2,
                    'y': self.left_frame.height() - shape.height()/2 - shape.pos_y
                },
                'frequency': shape.frequency,
                'angle': shape.angle
            }
            settings['shapes'].append(shape_data)

        # 保存到文件
        try:
            settings_path = self.get_settings_path()
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"Settings saved successfully to {settings_path}")
        except Exception as e:
            print(f"Error saving settings: {str(e)}")

    def load_settings(self):
        settings_path = self.get_settings_path()
        if not os.path.exists(settings_path):
            print(f"No saved settings found at {settings_path}")
            return
            
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)

            saved_count = settings.get('count')
            current_count = self.vars.get('count')
            
            # 确保保存的设置和当前的刺激数量匹配
            if saved_count != current_count:
                print(f"Saved settings are for {saved_count} stimuli, but current setup has {current_count} stimuli")
                return

            shapes_data = settings.get('shapes', [])
            if len(shapes_data) != len(self.left_frame.shape_widgets):
                print(f"Saved settings have {len(shapes_data)} shapes, but current setup has {len(self.left_frame.shape_widgets)} shapes")
                return

            # 应用保存的设置到每个图形
            for i, (shape, shape_data) in enumerate(zip(self.left_frame.shape_widgets, shapes_data)):
                # 选择当前图形
                self.select_shape(i)
                
                # 更新图形类型
                if type(shape).__name__ != shape_data['type']:
                    self.shape_combo.setCurrentText(shape_data['type'])
                    self.on_shape_changed()
                    # 重新获取更新后的图形对象
                    shape = self.left_frame.shape_widgets[i]
                
                # 更新颜色
                color_name = self.color_map.get(shape_data['color'], 'White')
                self.color_combo.setCurrentText(color_name)
                self.on_color_changed()
                
                # 更新大小
                self.size_edit.setText(str(shape_data['size']))
                self.left_frame.update_selected_size(str(shape_data['size']))
                
                # 更新位置
                self.loc_x.setText(str(shape_data['position']['x']))
                self.loc_y.setText(str(shape_data['position']['y']))
                self.set_selected_pos()
                
                # 更新频率和相位
                if 'frequency' in shape_data:
                    shape.frequency = shape_data['frequency']
                    print(f"图形 {i+1} 频率设置为: {shape.frequency}")
                if 'angle' in shape_data:
                    shape.angle = shape_data['angle']
                    print(f"图形 {i+1} 相位设置为: {shape.angle}")
                
                # 强制更新图形显示
                shape.update()
                
            # 最后取消选择
            self.left_frame.selected_index = -1
            self.left_frame.selected_shape = None
            for shape in self.left_frame.shape_widgets:
                shape.set_state('init')
                shape.update()

            print("Settings loaded successfully")
        except Exception as e:
            print(f"Error loading settings: {str(e)}")