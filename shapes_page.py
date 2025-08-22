

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPixmap
import random

from shape import Square, Rectangle, Circle, Ellipse, Triangle, FreeCurve
from customer import BG_COLOR, FG_COLOR, BTN_COLOR, create_btn, create_btn2, create_combo, create_label, create_input, Page


class ShapesPage(Page):
    def __init__(self, vars, shape_copy_type=0):
        super().__init__(vars)
        self.shape_copy_type = shape_copy_type
        
        self.frame_container = None

        

        # self.vars.update({'pause_fun':self.pause_fun})

        self.vars.get('cue_timer').timeout.connect(self.cue_timer_handler)
        self.vars.get('flash_timer').timeout.connect(self.flash_timer_handler)

        
        
    def _init_layout(self):
        hbox = self.create_shapes_layout()
        self.setLayout(hbox)

    def bind_exp_page_signal(self, exp_page):
        exp_page.receive_start_signal.connect(self.setup_start_info)
        exp_page.cue_signal.connect(self.set_target_shape)
        exp_page.finish_trial_signal.connect(self.set_result_shape)
        exp_page.pause_signal.connect(self.pause_fun)

    def start_experiment(self, event):

        self.flashed_index = []
        process = self.vars.get('process')
        block = int(process.get('block'))
        process.update({'block':block + 1, 'trial':0})
        self.vars.update({'process':process})
        self.timer_pause()

    
    def create_shapes_layout(self):
        hbox = QHBoxLayout()
        self.frame_container = hbox
        hbox.addWidget(self.copy_new_frame())
        return hbox
    
    def copy_new_frame(self):
        frame = self.vars.get('frame').copy(self.shape_copy_type)
        self.frame = frame
        self.frame.setStyleSheet(f"border: none; background: {BG_COLOR}; padding:0; margin:0;")
        return frame

    def cue_timer_handler(self):
        
        index = int(self.vars.get('process').get('curr_index'))
        if self.shape_copy_type == 0:
            self.frame.get_shape(index).set_text_type('normal')
        else:
            self.frame.get_shape(index).set_text_type('nothing')
            
        self.frame.all_flash()
        self.frame.all_start_move(float(self.vars.get('flash')))


    def flash_timer_handler(self):

        self.frame.all_stop_flash()
        self.frame.all_stop_move()

        

    def pause_fun(self):
        self.frame.all_stop_flash()
        self.frame.all_stop_move()

    def set_target_shape(self):
        self.frame.recover_all_position()
        self.frame.recover_all_color()

        # QTimer.singleShot(3000, ....)
        self.later_set_target_shape()

    def later_set_target_shape(self):
        index = int(self.vars.get('process').get('curr_index'))
        self.frame.get_shape(index).set_state('target')
        self.frame.get_shape(index).set_text_type('target')

    def set_result_shape(self):
        if self.shape_copy_type == 1: #用户界面不显示  result
            return 
        index = int(self.vars.get('process').get('curr_index'))
        self.frame.get_shape(index).set_state('result')

    def setup_start_info(self):
        self.frame.store_all_position()

    def update_frame(self):
        if self.frame_container:
            self.clear_layout(self.frame_container)
            self.frame_container.addWidget(self.copy_new_frame())

