import numpy as np
import time
import threading
from queue import Queue, Empty
import os
import csv
from datetime import datetime
import scipy.io
from PySide6.QtCore import QObject, Signal, QTimer

# 导入TRCA相关模块
import sys
sys.path.append('../TRCA')
from TRCA_benchmark import train_model, test_model, filterbank, ftrca
from eeg_processor import TRCAModel, RealTimeEEGProcessor

class TRCAIntegration(QObject):
    """
    TRCA算法与GUI集成类
    实现数据存储和实时识别功能
    """
    
    # 信号定义
    recognition_result = Signal(str, float)  # 识别结果信号 (字符, 置信度)
    data_saved = Signal(str)  # 数据保存信号
    status_update = Signal(str)  # 状态更新信号
    
    def __init__(self, vars_dict):
        super().__init__()
        self.vars = vars_dict
        
        # 实验控制参数
        self.recognition_flag = 0  # 0: 只收集数据, 1: 实时识别+数据存储
        self.current_block = 1
        self.current_trial = 1
        self.target_char = None
        
        # 数据存储
        self.experiment_data = []
        self.trial_data = []
        self.recognition_results = []
        
        # TRCA模型和数据处理
        self.trca_model = None
        self.simulated_data = None
        self.trained_model = None
        
        # 仿真参数 (使用S1.mat数据)
        self.use_simulation = True
        self.simulation_data = None
        self.simulation_block_index = 5  # 使用第6个block作为测试
        
        # 初始化
        self.init_trca_system()
        self.load_simulation_data()
        
        print("TRCA集成系统初始化完成")
        
    def init_trca_system(self):
        """初始化TRCA系统"""
        try:
            # 定义SSVEP频率参数 (模拟40个目标)
            self.beta_params = {}
            self.keyboard_layout = []
            
            # 生成40个频率 (8-15.8Hz, 间隔0.2Hz)
            frequencies = []
            for offset in [0, 0.2, 0.4, 0.6, 0.8]:
                frequencies.extend(np.arange(8, 16) + offset)
            
            # 创建5x8的键盘布局
            chars = [f'T{i+1}' for i in range(40)]
            for i in range(5):
                row = chars[i*8:(i+1)*8]
                self.keyboard_layout.append(row)
            
            # 设置频率参数
            for i, char in enumerate(chars):
                self.beta_params[char] = {
                    'frequency': frequencies[i],
                    'phase': 0.0
                }
            
            print(f"TRCA系统初始化: {len(chars)}个目标, 频率范围: {min(frequencies):.1f}-{max(frequencies):.1f}Hz")
            
        except Exception as e:
            print(f"TRCA系统初始化失败: {e}")
            
    def load_simulation_data(self):
        """加载S1.mat仿真数据"""
        try:
            data_path = '../TRCA/data/S1.mat'
            if not os.path.exists(data_path):
                print(f"警告: 找不到数据文件 {data_path}")
                return
                
            print("加载S1.mat仿真数据...")
            mat_data = scipy.io.loadmat(data_path)
            data = mat_data['data']  # [channels, samples, targets, blocks]
            
            # 重新排列维度: [targets, channels, samples, blocks]
            eeg = np.transpose(data, (2, 0, 1, 3))
            
            # 选择感兴趣的通道 (MATLAB索引转Python索引)
            chan_idx = np.array([48, 54, 55, 56, 57, 58, 61, 62, 63]) - 1
            eeg = eeg[:, chan_idx, :, :]
            
            self.simulation_data = eeg
            num_targets, num_channels, num_samples, num_blocks = eeg.shape
            
            print(f"仿真数据加载成功:")
            print(f"- 目标数: {num_targets}")
            print(f"- 通道数: {num_channels}")
            print(f"- 样本数: {num_samples}")
            print(f"- 块数: {num_blocks}")
            
            # 训练TRCA模型 (使用前5个block)
            self.train_trca_model()
            
        except Exception as e:
            print(f"加载仿真数据失败: {e}")
            
    def train_trca_model(self):
        """训练TRCA模型"""
        try:
            if self.simulation_data is None:
                print("没有仿真数据，无法训练模型")
                return
                
            print("开始训练TRCA模型...")
            
            # 使用前5个block进行训练
            train_blocks = list(range(5))  # blocks 0-4
            traindata = self.simulation_data[:, :, :, train_blocks]
            
            # 训练参数
            fs = 250  # 采样率
            num_fbs = 5  # 子频带数
            
            # 训练模型
            self.trained_model = train_model(traindata, fs, num_fbs, 'ftrca')
            
            print("TRCA模型训练完成")
            print(f"- 子频带数: {self.trained_model['num_fbs']}")
            print(f"- 目标数: {self.trained_model['num_targs']}")
            
        except Exception as e:
            print(f"TRCA模型训练失败: {e}")
            
    def set_recognition_flag(self, flag):
        """设置识别标志
        Args:
            flag (int): 0=只收集数据, 1=实时识别+数据存储
        """
        self.recognition_flag = flag
        status = "实时识别+数据存储" if flag == 1 else "只收集数据"
        print(f"识别模式设置为: {status}")
        self.status_update.emit(f"模式: {status}")
        
    def start_trial(self, target_index, block_num, trial_num):
        """开始一个trial
        Args:
            target_index (int): 目标索引 (0-based)
            block_num (int): 块编号
            trial_num (int): trial编号
        """
        self.current_block = block_num
        self.current_trial = trial_num
        
        # 获取目标字符
        chars = [f'T{i+1}' for i in range(40)]
        if 0 <= target_index < len(chars):
            self.target_char = chars[target_index]
        else:
            self.target_char = 'T1'  # 默认目标
            
        # 记录trial开始信息
        trial_info = {
            'block': block_num,
            'trial': trial_num,
            'target_char': self.target_char,
            'target_index': target_index,
            'start_time': time.time(),
            'recognition_results': [],
            'final_result': None
        }
        
        self.trial_data.append(trial_info)
        
        print(f"开始Trial {trial_num}, Block {block_num}, 目标: {self.target_char}")
        
        # 如果是识别模式，启动仿真识别
        if self.recognition_flag == 1:
            self.start_simulation_recognition(target_index)
            
    def start_simulation_recognition(self, target_index):
        """启动仿真识别过程"""
        try:
            if self.simulation_data is None or self.trained_model is None:
                print("仿真数据或训练模型不可用")
                return
                
            # 使用第6个block (索引5) 作为测试数据
            test_block = self.simulation_block_index
            testdata = self.simulation_data[:, :, :, test_block:test_block+1]
            testdata = testdata[:, :, :, 0]  # 移除block维度
            
            # 模拟实时识别过程
            def simulate_recognition():
                try:
                    # 等待一段时间模拟数据收集
                    time.sleep(0.5)
                    
                    # 执行TRCA识别
                    results = test_model(testdata, self.trained_model, is_ensemble=True)
                    
                    # 获取目标的识别结果
                    if target_index < len(results):
                        predicted_index = results[target_index] - 1  # 转换为0-based索引
                        confidence = np.random.uniform(0.6, 0.95)  # 模拟置信度
                        
                        # 判断是否正确识别
                        is_correct = (predicted_index == target_index)
                        
                        # 获取预测的字符
                        chars = [f'T{i+1}' for i in range(40)]
                        predicted_char = chars[predicted_index] if 0 <= predicted_index < len(chars) else 'Unknown'
                        
                        # 记录识别结果
                        result_info = {
                            'timestamp': time.time(),
                            'predicted_char': predicted_char,
                            'predicted_index': predicted_index,
                            'confidence': confidence,
                            'is_correct': is_correct
                        }
                        
                        # 更新trial数据
                        if self.trial_data:
                            self.trial_data[-1]['recognition_results'].append(result_info)
                            self.trial_data[-1]['final_result'] = result_info
                            
                        # 发送识别结果信号
                        self.recognition_result.emit(predicted_char, confidence)
                        
                        status = "正确" if is_correct else "错误"
                        print(f"识别结果: {predicted_char} (置信度: {confidence:.2f}, {status})")
                        
                    else:
                        print(f"目标索引 {target_index} 超出范围")
                        
                except Exception as e:
                    print(f"仿真识别过程出错: {e}")
                    
            # 在独立线程中运行识别
            recognition_thread = threading.Thread(target=simulate_recognition, daemon=True)
            recognition_thread.start()
            
        except Exception as e:
            print(f"启动仿真识别失败: {e}")
            
    def end_trial(self):
        """结束当前trial"""
        if self.trial_data:
            current_trial = self.trial_data[-1]
            current_trial['end_time'] = time.time()
            
            # 计算trial持续时间
            duration = current_trial['end_time'] - current_trial['start_time']
            current_trial['duration'] = duration
            
            print(f"Trial结束, 持续时间: {duration:.2f}秒")
            
    def save_experiment_data(self, filename=None):
        """保存实验数据"""
        try:
            if not self.trial_data:
                print("没有实验数据可保存")
                return
                
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"experiment_data_{timestamp}.csv"
                
            # 保存trial数据到CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Block', 'Trial', 'Target_Char', 'Target_Index',
                    'Predicted_Char', 'Predicted_Index', 'Confidence',
                    'Is_Correct', 'Trial_Duration', 'Recognition_Flag'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for trial in self.trial_data:
                    final_result = trial.get('final_result', {})
                    if final_result is None:
                        final_result = {}
                    
                    row = {
                        'Block': trial.get('block', 0),
                        'Trial': trial.get('trial', 0),
                        'Target_Char': trial.get('target_char', ''),
                        'Target_Index': trial.get('target_index', -1),
                        'Predicted_Char': final_result.get('predicted_char', ''),
                        'Predicted_Index': final_result.get('predicted_index', -1),
                        'Confidence': final_result.get('confidence', 0.0),
                        'Is_Correct': final_result.get('is_correct', False),
                        'Trial_Duration': trial.get('duration', 0.0),
                        'Recognition_Flag': self.recognition_flag
                    }
                    
                    writer.writerow(row)
                    
            print(f"实验数据已保存到: {filename}")
            self.data_saved.emit(filename)
            
            # 计算统计信息
            self.calculate_statistics()
            
        except Exception as e:
            print(f"保存实验数据失败: {e}")
            
    def calculate_statistics(self):
        """计算实验统计信息"""
        try:
            if not self.trial_data:
                print("没有实验数据可计算统计信息")
                return None
                
            total_trials = len(self.trial_data)
            correct_trials = 0
            total_confidence = 0
            total_duration = 0
            
            for trial in self.trial_data:
                final_result = trial.get('final_result', {})
                if final_result is None:
                    final_result = {}
                    
                if final_result.get('is_correct', False):
                    correct_trials += 1
                    
                total_confidence += final_result.get('confidence', 0)
                total_duration += trial.get('duration', 0)
                
            accuracy = (correct_trials / total_trials) * 100 if total_trials > 0 else 0
            avg_confidence = total_confidence / total_trials if total_trials > 0 else 0
            avg_duration = total_duration / total_trials if total_trials > 0 else 0
            
            print(f"\n实验统计:")
            print(f"- 总试验数: {total_trials}")
            print(f"- 正确试验数: {correct_trials}")
            print(f"- 准确率: {accuracy:.1f}%")
            print(f"- 平均置信度: {avg_confidence:.3f}")
            print(f"- 平均试验时长: {avg_duration:.2f}秒")
            
            return {
                'total_trials': total_trials,
                'correct_trials': correct_trials,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'avg_duration': avg_duration
            }
            
        except Exception as e:
            print(f"计算统计信息失败: {e}")
            return None
            
    def get_current_status(self):
        """获取当前状态信息"""
        return {
            'recognition_flag': self.recognition_flag,
            'current_block': self.current_block,
            'current_trial': self.current_trial,
            'target_char': self.target_char,
            'total_trials': len(self.trial_data),
            'model_loaded': self.trained_model is not None,
            'simulation_ready': self.simulation_data is not None
        }
        
    def reset_experiment(self):
        """重置实验数据"""
        self.trial_data = []
        self.recognition_results = []
        self.current_block = 1
        self.current_trial = 1
        self.target_char = None
        
        print("实验数据已重置")
        self.status_update.emit("实验数据已重置")