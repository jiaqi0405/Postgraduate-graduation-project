#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试TRCA集成功能
"""

import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QCheckBox
from PySide6.QtCore import QTimer, Qt

# 添加路径
sys.path.append('../TRCA')

from trca_integration import TRCAIntegration
from eeg_processor import RealTimeEEGProcessor, EEGWaveformDisplay

class TRCATestWindow(QMainWindow):
    """
    TRCA集成测试窗口
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TRCA集成测试")
        self.setGeometry(100, 100, 1200, 800)
        
        # 模拟vars字典
        self.vars = {
            'type': 'SSVEP+motion',
            'count': 5,
            'blocks': 2,
            'cue': 0.5,
            'flash': 1.0,
            'pause': 0.2,
            'rest': 4.0,
            'window': 0.8,
            'process': {
                'block': 1,
                'trial': 1,
                'curr_index': 0
            }
        }
        
        # 初始化TRCA集成
        self.trca_integration = TRCAIntegration(self.vars)
        
        # 连接信号
        self.trca_integration.recognition_result.connect(self.handle_recognition_result)
        self.trca_integration.data_saved.connect(self.handle_data_saved)
        self.trca_integration.status_update.connect(self.handle_status_update)
        
        # 初始化EEG处理器
        self.eeg_processor = RealTimeEEGProcessor(
            time_window=1.0,
            sampling_rate=250,
            num_channels=9
        )
        
        # 连接EEG处理器信号
        self.eeg_processor.data_received.connect(self.handle_eeg_data)
        self.eeg_processor.status_update.connect(self.handle_eeg_status)
        
        # 测试状态
        self.current_trial = 0
        self.max_trials = 10
        self.recognition_flag = 0
        
        # 初始化界面
        self.init_ui()
        
        print("TRCA测试窗口初始化完成")
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        # 右侧波形显示
        self.waveform_display = EEGWaveformDisplay(
            num_channels=9,
            sampling_rate=250,
            display_duration=5.0
        )
        main_layout.addWidget(self.waveform_display, stretch=2)
        
        central_widget.setLayout(main_layout)
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("TRCA Integration Test")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        layout.addWidget(title)
        
        # TRCA状态显示
        self.trca_status_label = QLabel("TRCA status: Initializing")
        self.trca_status_label.setStyleSheet("color: #666; margin: 10px 0;")
        layout.addWidget(self.trca_status_label)
        
        # EEG状态显示
        self.eeg_status_label = QLabel("EEG Status: Not Connected")
        self.eeg_status_label.setStyleSheet("color: #666; margin: 10px 0;")
        layout.addWidget(self.eeg_status_label)
        
        # 识别模式选择
        self.recognition_checkbox = QCheckBox("Enable real-time recognition")
        self.recognition_checkbox.stateChanged.connect(self.toggle_recognition_mode)
        layout.addWidget(self.recognition_checkbox)
        
        # 识别结果显示
        self.result_label = QLabel("识Identification results: --")
        self.result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0066CC; margin: 10px 0;")
        layout.addWidget(self.result_label)
        
        # 试验信息
        self.trial_info_label = QLabel(f"Experiment: {self.current_trial}/{self.max_trials}")
        layout.addWidget(self.trial_info_label)
        
        # 控制按钮
        layout.addWidget(QLabel("Control Operation:"))
        
        # 连接EEG设备按钮
        self.connect_btn = QPushButton("Connecting EEG equipment (simulation)")
        self.connect_btn.clicked.connect(self.connect_eeg_device)
        layout.addWidget(self.connect_btn)
        
        # 开始数据采集按钮
        self.start_recording_btn = QPushButton("Start data collection")
        self.start_recording_btn.clicked.connect(self.start_eeg_recording)
        self.start_recording_btn.setEnabled(False)
        layout.addWidget(self.start_recording_btn)
        
        # 开始试验按钮
        self.start_trial_btn = QPushButton("Start the experiment")
        self.start_trial_btn.clicked.connect(self.start_trial)
        layout.addWidget(self.start_trial_btn)
        
        # 下一个试验按钮
        self.next_trial_btn = QPushButton("Next experiment")
        self.next_trial_btn.clicked.connect(self.next_trial)
        layout.addWidget(self.next_trial_btn)
        
        # 保存数据按钮
        self.save_data_btn = QPushButton("Save experimental data")
        self.save_data_btn.clicked.connect(self.save_experiment_data)
        layout.addWidget(self.save_data_btn)
        
        # 重置实验按钮
        self.reset_btn = QPushButton("Reset Experiment")
        self.reset_btn.clicked.connect(self.reset_experiment)
        layout.addWidget(self.reset_btn)
        
        # 停止采集按钮
        self.stop_recording_btn = QPushButton("Stop data collection")
        self.stop_recording_btn.clicked.connect(self.stop_eeg_recording)
        self.stop_recording_btn.setEnabled(False)
        layout.addWidget(self.stop_recording_btn)
        
        layout.addStretch()
        
        # 统计信息
        layout.addWidget(QLabel("Statistics:"))
        self.stats_label = QLabel("Accuracy: --%\nAverage confidence: --")
        self.stats_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.stats_label)
        
        panel.setLayout(layout)
        return panel
        
    def toggle_recognition_mode(self, state):
        """切换识别模式"""
        self.recognition_flag = 1 if state == 2 else 0
        self.trca_integration.set_recognition_flag(self.recognition_flag)
        
        mode_text = "实时识别+数据存储" if self.recognition_flag == 1 else "只收集数据"
        self.trca_status_label.setText(f"TRCA Status: {mode_text}")
        
        if self.recognition_flag == 0:
            self.result_label.setText("识别结果: --")
            
    def connect_eeg_device(self):
        """连接EEG设备"""
        success = self.eeg_processor.connect_device('simulation')
        
        if success:
            self.connect_btn.setText("EEG设备已连接")
            self.connect_btn.setEnabled(False)
            self.start_recording_btn.setEnabled(True)
            self.eeg_status_label.setText("EEG状态: 已连接(仿真模式)")
        else:
            self.eeg_status_label.setText("EEG状态: 连接失败")
            
    def start_eeg_recording(self):
        """开始EEG数据采集"""
        success = self.eeg_processor.start_recording()
        
        if success:
            self.start_recording_btn.setEnabled(False)
            self.stop_recording_btn.setEnabled(True)
            self.eeg_status_label.setText("EEG状态: 正在采集数据")
            
            # 连接波形显示
            self.eeg_processor.waveform_display = self.waveform_display
        else:
            self.eeg_status_label.setText("EEG状态: 启动采集失败")
            
    def stop_eeg_recording(self):
        """停止EEG数据采集"""
        self.eeg_processor.stop_recording()
        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        self.eeg_status_label.setText("EEG状态: 已停止采集")
        
    def start_trial(self):
        """开始一个试验"""
        if self.current_trial >= self.max_trials:
            self.result_label.setText("所有试验已完成!")
            return
            
        # 随机选择目标
        target_index = np.random.randint(0, 40)
        
        # 更新试验信息
        self.current_trial += 1
        self.trial_info_label.setText(f"试验: {self.current_trial}/{self.max_trials}")
        
        # 启动TRCA试验
        self.trca_integration.start_trial(
            target_index=target_index,
            block_num=1,
            trial_num=self.current_trial
        )
        
        self.result_label.setText(f"试验进行中... 目标: T{target_index+1}")
        
        # 模拟试验持续时间
        QTimer.singleShot(2000, self.end_current_trial)  # 2秒后结束试验
        
    def end_current_trial(self):
        """结束当前试验"""
        self.trca_integration.end_trial()
        
    def next_trial(self):
        """开始下一个试验"""
        self.start_trial()
        
    def save_experiment_data(self):
        """保存实验数据"""
        self.trca_integration.save_experiment_data()
        
        # 显示统计信息
        stats = self.trca_integration.calculate_statistics()
        if stats:
            stats_text = f"准确率: {stats['accuracy']:.1f}%\n平均置信度: {stats['avg_confidence']:.3f}"
            self.stats_label.setText(stats_text)
            
    def reset_experiment(self):
        """重置实验"""
        self.trca_integration.reset_experiment()
        self.current_trial = 0
        self.trial_info_label.setText(f"试验: {self.current_trial}/{self.max_trials}")
        self.result_label.setText("识别结果: --")
        self.stats_label.setText("准确率: --%\n平均置信度: --")
        
    def handle_recognition_result(self, predicted_char, confidence):
        """处理识别结果"""
        result_text = f"识别结果: {predicted_char} (置信度: {confidence:.3f})"
        self.result_label.setText(result_text)
        print(f"识别结果: {predicted_char}, 置信度: {confidence:.3f}")
        
    def handle_data_saved(self, filename):
        """处理数据保存事件"""
        print(f"数据已保存: {filename}")
        
    def handle_status_update(self, status):
        """处理状态更新"""
        print(f"TRCA状态: {status}")
        
    def handle_eeg_data(self, data):
        """处理EEG数据"""
        # 这里可以添加实时数据处理逻辑
        pass
        
    def handle_eeg_status(self, status):
        """处理EEG状态更新"""
        print(f"EEG状态: {status}")
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止数据采集
        if self.eeg_processor.is_recording:
            self.eeg_processor.stop_recording()
            
        # 断开设备连接
        self.eeg_processor.disconnect_device()
        
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 2px;
            border-radius: 4px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QLabel {
            color: #333333;
            font-size: 12px;
        }
        QCheckBox {
            color: #333333;
            font-size: 12px;
        }
    """)
    
    # 创建测试窗口
    window = TRCATestWindow()
    window.show()
    
    print("TRCA集成测试程序启动")
    print("使用说明:")
    print("1. 点击'连接EEG设备(仿真)'连接仿真设备")
    print("2. 点击'开始数据采集'开始采集仿真数据")
    print("3. 勾选'启用实时识别'开启识别模式")
    print("4. 点击'开始试验'进行SSVEP识别测试")
    print("5. 点击'保存实验数据'保存结果")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
