import sys
import json
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from main_page import MainPage
from setting_page import SettingPage
from appearance_page import AppearancePage
from blocks_page import BlocksPage
from motion_page import MotionPage
from experiment_page import ExperimentPage

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl
# import pyttsx3


BG_COLOR = "#111111"
FG_COLOR = "#FFFFFF"


class AudioPlayer:
    def __init__(self):
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
    def play(self, file_path):
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.player.play()
    def stop(self):
        self.player.stop()

import pygame
import os

class GameAudioPlayer:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.current_file = None

    def play(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        self.current_file = file_path
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def stop(self):
        pygame.mixer.music.stop()



class MainWindow(QMainWindow):
    direction_signal = Signal(float, float)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paradigm GUI")
        self.setStyleSheet(f"background: {BG_COLOR}; color: {FG_COLOR};")
        self.resize(1000, 600) 
        # self.setWindowState(Qt.WindowMaximized)
        self.is_fullscreen = True
        self.showFullScreen()
        # 变量
        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # 1秒间隔
        self.timer.setSingleShot(True)  # 关键设置：只触发一次
        # self.engine = pyttsx3.init()
        # self.player = AudioPlayer()
        self.player = GameAudioPlayer()
        self.vars = {
            'type':'SSVEP+motion', 'count': 5, 'blocks':2, 'cue':0.5, 'flash':1, 'pause':0.2, 'rest':4, 'window':0.8,
            'timer': self.timer,
            'pages':{},
            'process':{
                'block': 0, 'trial':0, 'curr_index':0
            },
            'cue_timer':QTimer(self),
            'flash_timer':QTimer(self),
            'pause_timer':QTimer(self),
            'rest_timer':QTimer(self),
            'pause_fun':None,
            # 'engine':self.engine,
            'player': self.player,
            'mainWindow':self
        }
        # 页面堆栈
        self.stack = QStackedWidget()
        self.main_page = MainPage(self.vars, self.show_setting )
        self.setting_page = SettingPage(self.vars, self.show_main, self.show_appearance)
        self.appearance_page = AppearancePage(self.vars, self.show_setting, self.show_blocks, self.show_motion)
        self.blocks_page = BlocksPage(self.vars, self.show_appearance, self.show_experiment, self.show_motion)
        self.motion_page = MotionPage(self.vars, self.show_appearance, self.show_blocks)
        self.experiment_page = ExperimentPage(self.vars, self.show_blocks, self.show_main)
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.setting_page)
        self.stack.addWidget(self.appearance_page)
        self.stack.addWidget(self.blocks_page)
        self.stack.addWidget(self.motion_page)
        self.stack.addWidget(self.experiment_page)
        self.setCentralWidget(self.stack)

        self.blocks_page.confirm_setting_signal.connect(self.experiment_page.update_frame)

        pages = {
            'main':self.main_page,
            'setting':self.setting_page,
            'appearance':self.appearance_page,
            'blocks': self.blocks_page,
            'motion':self.motion_page,
            'experiment':self.experiment_page
        }

        self.vars.update({'pages':pages})



    def show_main(self):
        self.stack.setCurrentWidget(self.main_page)

    def show_setting(self):
        self.stack.setCurrentWidget(self.setting_page)

    def show_appearance(self):
        self.stack.setCurrentWidget(self.appearance_page)

    def show_blocks(self):
        self.stack.setCurrentWidget(self.blocks_page)

    def show_motion(self):
        self.stack.setCurrentWidget(self.motion_page)

    def show_experiment(self):
        self.stack.setCurrentWidget(self.experiment_page)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            self.showFullScreen()
            self.is_fullscreen = True
        elif event.key() == Qt.Key_Escape:
            if self.is_fullscreen:
                # self.showMaximized()
                self.showNormal()
                self.is_fullscreen = False
            else:
                sys.exit()
        else:
            super().keyPressEvent(event)

def main():

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # window.is_fullscreen = True
    # window.showFullScreen()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
