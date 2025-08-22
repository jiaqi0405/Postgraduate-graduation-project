import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import time
from queue import Queue, Empty
import csv
from datetime import datetime
import h5py
from scipy import signal
from scipy.signal import butter, filtfilt, cheby1
from PySide6.QtCore import QObject, Signal, QTimer, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout
import pygds

class EEGWaveformDisplay(QWidget):
    """
    实时EEG波形显示组件
    """
    def __init__(self, num_channels=9, sampling_rate=250, display_duration=5.0):
        super().__init__()
        
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.display_duration = display_duration
        self.buffer_size = int(sampling_rate * display_duration)
        
        # 数据缓冲区
        self.data_buffer = np.zeros((num_channels, self.buffer_size))
        self.time_axis = np.linspace(-display_duration, 0, self.buffer_size)
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 初始化绘图
        self.init_plot()
        
        # 更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(50)  # 20Hz更新频率
        
    def init_plot(self):
        """初始化绘图"""
        self.figure.clear()
        
        # 创建子图
        self.axes = []
        channel_names = [f'Ch{i+1}' for i in range(self.num_channels)]
        
        for i in range(self.num_channels):
            ax = self.figure.add_subplot(self.num_channels, 1, i+1)
            ax.set_xlim(-self.display_duration, 0)
            ax.set_ylim(-50, 50)  # μV范围
            ax.set_ylabel(f'{channel_names[i]} (μV)', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if i == self.num_channels - 1:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticklabels([])
                
            self.axes.append(ax)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_data(self, new_data):
        """更新数据缓冲区
        Args:
            new_data: shape (num_channels, num_samples)
        """
        if new_data.shape[0] != self.num_channels:
            return
            
        num_new_samples = new_data.shape[1]
        
        # 滚动缓冲区
        self.data_buffer = np.roll(self.data_buffer, -num_new_samples, axis=1)
        self.data_buffer[:, -num_new_samples:] = new_data
        
    def update_plot(self):
        """更新绘图"""
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.plot(self.time_axis, self.data_buffer[i, :], 'b-', linewidth=0.8)
            ax.set_xlim(-self.display_duration, 0)
            ax.set_ylim(-50, 50)
            ax.set_ylabel(f'Ch{i+1} (μV)', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if i == self.num_channels - 1:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticklabels([])
                
        self.canvas.draw()

class TRCAModel:
    """
    TRCA算法模型类
    """
    def __init__(self, sampling_rate=250, num_harmonics=5, num_fbs=5):
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.num_fbs = num_fbs
        
        # 模型参数
        self.spatial_filters = None
        self.reference_templates = None
        self.ensemble_weights = None
        self.frequencies = None
        
        # 滤波器组参数
        self.filterbank_params = {
            'num_fbs': num_fbs,
            'passband': [6, 90],
            'stopband': [4, 100],
            'gpass': 3,
            'gstop': 40
        }
        
    def design_filterbank(self):
        """设计滤波器组"""
        filters = []
        
        for i in range(self.num_fbs):
            # 计算通带频率
            passband_start = self.filterbank_params['passband'][0] + i * 2
            passband_end = min(self.filterbank_params['passband'][1], 
                             self.sampling_rate / 2 - 1)
            
            if passband_start >= passband_end:
                break
                
            # 设计Chebyshev Type I滤波器
            try:
                sos = signal.cheby1(4, self.filterbank_params['gpass'], 
                                  [passband_start, passband_end], 
                                  btype='band', fs=self.sampling_rate, output='sos')
                filters.append(sos)
            except Exception as e:
                print(f"滤波器设计失败 (子带 {i}): {e}")
                break
                
        return filters
        
    def generate_reference_signals(self, frequencies, time_window, phase=0):
        """生成参考信号
        Args:
            frequencies: 刺激频率列表
            time_window: 时间窗口长度(秒)
            phase: 相位偏移
        Returns:
            reference_signals: shape (num_targets, num_harmonics*2, num_samples)
        """
        num_samples = int(time_window * self.sampling_rate)
        t = np.arange(num_samples) / self.sampling_rate
        
        reference_signals = []
        
        for freq in frequencies:
            signals = []
            
            # 生成谐波信号
            for harmonic in range(1, self.num_harmonics + 1):
                # 正弦和余弦分量
                sin_signal = np.sin(2 * np.pi * harmonic * freq * t + phase)
                cos_signal = np.cos(2 * np.pi * harmonic * freq * t + phase)
                signals.extend([sin_signal, cos_signal])
                
            reference_signals.append(np.array(signals))
            
        return np.array(reference_signals)
        
    def filterbank(self, eeg_data):
        """应用滤波器组
        Args:
            eeg_data: shape (num_channels, num_samples)
        Returns:
            filtered_data: list of filtered data for each sub-band
        """
        filters = self.design_filterbank()
        filtered_data = []
        
        for sos in filters:
            try:
                # 应用滤波器
                filtered = signal.sosfiltfilt(sos, eeg_data, axis=1)
                filtered_data.append(filtered)
            except Exception as e:
                print(f"滤波过程出错: {e}")
                # 如果滤波失败，使用原始数据
                filtered_data.append(eeg_data.copy())
                
        return filtered_data
        
    def load_pretrained_model(self, model_path):
        """加载预训练模型
        Args:
            model_path: 模型文件路径
        """
        try:
            import scipy.io
            model_data = scipy.io.loadmat(model_path)
            
            self.spatial_filters = model_data.get('spatial_filters')
            self.reference_templates = model_data.get('templates')
            self.ensemble_weights = model_data.get('ensemble_weights')
            self.frequencies = model_data.get('frequencies', [])
            
            print(f"成功加载预训练模型: {model_path}")
            return True
            
        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            return False
            
    def init_random_weights(self, num_channels, num_targets):
        """初始化随机权重(用于测试)"""
        self.spatial_filters = np.random.randn(self.num_fbs, num_targets, num_channels)
        self.ensemble_weights = np.ones(self.num_fbs) / self.num_fbs
        
        # 生成测试频率
        self.frequencies = np.arange(8, 16, 0.2)[:num_targets]
        
        print(f"初始化随机权重: {num_targets}个目标, {num_channels}个通道")
        
    def compute_trca_score(self, eeg_data, target_frequencies, time_window=1.0):
        """计算TRCA分数
        Args:
            eeg_data: shape (num_channels, num_samples)
            target_frequencies: 目标频率列表
            time_window: 时间窗口长度
        Returns:
            scores: 各目标的分数
        """
        try:
            # 输入验证
            if eeg_data.ndim != 2:
                raise ValueError(f"EEG数据维度错误: {eeg_data.shape}")
                
            num_channels, num_samples = eeg_data.shape
            
            # 应用滤波器组
            filtered_data = self.filterbank(eeg_data)
            
            # 生成参考信号
            reference_signals = self.generate_reference_signals(
                target_frequencies, time_window)
            
            scores = []
            
            for target_idx, freq in enumerate(target_frequencies):
                target_scores = []
                
                # 对每个子频带计算分数
                for fb_idx, filtered_eeg in enumerate(filtered_data):
                    if (self.spatial_filters is not None and 
                        target_idx < self.spatial_filters.shape[1]):
                        
                        # 应用空间滤波器
                        spatial_filter = self.spatial_filters[fb_idx, target_idx, :]
                        filtered_signal = np.dot(spatial_filter, filtered_eeg)
                        
                        # 计算与参考信号的相关性
                        ref_signal = reference_signals[target_idx]
                        
                        # 确保信号长度匹配
                        min_len = min(filtered_signal.shape[0], ref_signal.shape[1])
                        
                        correlations = []
                        for ref_component in ref_signal:
                            corr = np.corrcoef(filtered_signal[:min_len], 
                                             ref_component[:min_len])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr ** 2)
                                
                        if correlations:
                            target_scores.append(np.mean(correlations))
                        else:
                            target_scores.append(0.0)
                    else:
                        target_scores.append(0.0)
                        
                # 集成多个子频带的分数
                if self.ensemble_weights is not None:
                    ensemble_score = np.average(target_scores, weights=self.ensemble_weights)
                else:
                    ensemble_score = np.mean(target_scores)
                    
                scores.append(ensemble_score)
                
            return np.array(scores)
            
        except Exception as e:
            print(f"TRCA分数计算失败: {e}")
            return np.zeros(len(target_frequencies))

class RealTimeEEGProcessor(QObject):
    """
    实时EEG数据处理器
    """
    
    # 信号定义
    data_received = Signal(np.ndarray)  # 新数据接收信号
    recognition_result = Signal(str, float)  # 识别结果信号
    status_update = Signal(str)  # 状态更新信号
    error_occurred = Signal(str)  # 错误信号
    
    def __init__(self, beta_params=None, keyboard_layout=None, 
                 time_window=1.0, sampling_rate=250, num_channels=9):
        super().__init__()
        
        # 基本参数
        self.beta_params = beta_params or {}
        self.keyboard_layout = keyboard_layout or []
        self.time_window = time_window
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        
        # 数据缓冲区
        self.buffer_size = int(sampling_rate * time_window * 2)  # 2倍时间窗口
        self.data_buffer = np.zeros((num_channels, self.buffer_size))
        self.buffer_index = 0
        
        # 设备连接
        self.device = None
        self.is_connected = False
        self.is_recording = False
        
        # 数据处理
        self.trca_model = TRCAModel(sampling_rate=sampling_rate)
        self.waveform_display = None
        
        # 数据记录
        self.recorded_data = []
        self.experiment_log = []
        
        # 信号处理参数
        self.notch_freq = 50  # 工频陷波
        self.bandpass_low = 1.0
        self.bandpass_high = 100.0
        
        # 初始化滤波器
        self.init_filters()
        
        print("实时EEG处理器初始化完成")
        
    def init_filters(self):
        """初始化信号处理滤波器"""
        try:
            # 陷波滤波器 (50Hz)
            nyquist = self.sampling_rate / 2
            notch_low = (self.notch_freq - 1) / nyquist
            notch_high = (self.notch_freq + 1) / nyquist
            
            self.notch_sos = signal.butter(4, [notch_low, notch_high], 
                                         btype='bandstop', output='sos')
            
            # 带通滤波器
            bp_low = self.bandpass_low / nyquist
            bp_high = min(self.bandpass_high / nyquist, 0.99)
            
            self.bandpass_sos = signal.butter(4, [bp_low, bp_high], 
                                            btype='band', output='sos')
            
            print("信号处理滤波器初始化成功")
            
        except Exception as e:
            print(f"滤波器初始化失败: {e}")
            self.notch_sos = None
            self.bandpass_sos = None
            
    def connect_device(self, device_type='simulation'):
        """连接EEG设备
        Args:
            device_type: 设备类型 ('simulation', 'gtec', 'neuracle')
        """
        try:
            if device_type == 'simulation':
                # 仿真模式
                self.device = 'simulation'
                self.is_connected = True
                self.status_update.emit("已连接到仿真设备")
                return True
                
            elif device_type in ['gtec', 'neuracle']:
                # 真实设备连接
                try:
                    # 创建GDS设备实例
                    self.device = pygds.GDS()
                    
                    # 初始化设备
                    self.device.Initialize()
                    
                    # 连接到设备
                    connected_devices = self.device.GetConnectedDevices()
                    if not connected_devices:
                        raise Exception("未找到连接的设备")
                    
                    # 选择第一个可用设备
                    device_info = connected_devices[0]
                    self.device.Connect(device_info)
                    
                    # 获取设备信息
                    device_name = self.device.GetDeviceName()
                    sampling_rate = self.device.GetSamplingRate()
                    channel_count = self.device.GetNumberOfChannels()
                    
                    print(f"已连接设备: {device_name}")
                    print(f"采样率: {sampling_rate} Hz")
                    print(f"通道数: {channel_count}")
                    
                    # 更新采样率和通道数（如果需要）
                    if sampling_rate != self.sampling_rate:
                        print(f"警告: 设备采样率({sampling_rate})与配置采样率({self.sampling_rate})不匹配")
                    
                    if channel_count != self.num_channels:
                        print(f"警告: 设备通道数({channel_count})与配置通道数({self.num_channels})不匹配")
                        self.num_channels = min(channel_count, self.num_channels)
                    
                    self.is_connected = True
                    self.status_update.emit(f"已连接到{device_type}设备: {device_name}")
                    return True
                    
                except Exception as device_error:
                    raise Exception(f"设备连接失败: {device_error}")
                
            else:
                raise Exception(f"不支持的设备类型: {device_type}")
                
        except Exception as e:
            error_msg = f"设备连接失败: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False
            
    def start_recording(self):
        """开始数据采集"""
        if not self.is_connected:
            self.error_occurred.emit("设备未连接")
            return False
            
        try:
            self.is_recording = True
            
            if self.device == 'simulation':
                # 启动仿真数据生成
                self.simulation_thread = threading.Thread(
                    target=self._simulation_data_loop, daemon=True)
                self.simulation_thread.start()
            else:
                # 启动真实设备数据采集
                self.recording_thread = threading.Thread(
                    target=self._real_data_loop, daemon=True)
                self.recording_thread.start()
                
            self.status_update.emit("数据采集已开始")
            return True
            
        except Exception as e:
            error_msg = f"启动数据采集失败: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False
            
    def stop_recording(self):
        """停止数据采集"""
        self.is_recording = False
        
        # 保存数据
        self.save_recorded_data()
        
        self.status_update.emit("数据采集已停止")
        
    def _simulation_data_loop(self):
        """仿真数据生成循环"""
        # 增加数据块大小，确保足够进行滤波处理
        # 原来是40ms (0.04s)，现在改为120ms (0.12s)
        chunk_size = int(self.sampling_rate * 0.12)  # 120ms chunks
        
        print(f"仿真数据块大小: {chunk_size} 样本")
        
        while self.is_recording:
            try:
                # 生成仿真EEG数据
                t = np.arange(chunk_size) / self.sampling_rate + time.time()
                
                # 基础噪声
                data = np.random.randn(self.num_channels, chunk_size) * 10
                
                # 添加SSVEP信号 (仅用于测试)
                for i, freq in enumerate([10, 12, 15]):
                    if i < self.num_channels:
                        ssvep_signal = 5 * np.sin(2 * np.pi * freq * t)
                        data[i, :] += ssvep_signal
                        
                # 添加工频干扰
                powerline_noise = 2 * np.sin(2 * np.pi * 50 * t)
                data += powerline_noise
                
                # 处理数据
                self._process_new_data(data)
                
                time.sleep(0.12)  # 120ms间隔
                
            except Exception as e:
                print(f"仿真数据生成错误: {e}")
                break
                
    def _real_data_loop(self):
        """真实设备数据采集循环"""
        if self.device is None:
            self.error_occurred.emit("PYGDS库不可用或设备未连接")
            return
            
        try:
            # 获取支持的采样率
            sr = self.device.GetSupportedSamplingRates()[0]
            self.device.SamplingRate = max(sr.keys())
            
            # 根据设备类型进行配置
            if self.device.DeviceType == pygds.DEVICE_TYPE_GHIAMP:
                for i, ch in enumerate(self.device.Channels):
                    if i >= 40:
                        ch.Acquire = 0
            elif self.device.DeviceType == pygds.DEVICE_TYPE_GUSBAMP:
                self.device.SamplingRate = 1200  # >1200 no internal signal
                
            # 设置扫描数
            self.device.NumberOfScans = self.device.SamplingRate
            
            # 配置设备
            self.device.SetConfiguration()
            
            # 计算每次读取的数据量（每1/3秒读取一次）
            cnt = self.device.SamplingRate // 3
            
            # 数据采集循环
            while self.is_recording:
                try:
                    # 获取数据
                    raw_data = self.device.GetData(cnt)
                    if raw_data is not None and len(raw_data) > 0:
                        data = self.device.GetDataAsFloat(raw_data)
                        
                        # 转换数据格式为 (channels, samples)
                        if isinstance(data, list):
                            data = np.array(data)
                        
                        if data.ndim == 1:
                            # 如果是一维数组，重塑为 (channels, samples)
                            data = data.reshape(self.num_channels, -1)
                        elif data.ndim == 2:
                            # 如果是二维数组，确保格式正确
                            if data.shape[0] != self.num_channels:
                                data = data.T  # 转置
                        
                        # 只取前num_channels个通道
                        if data.shape[0] > self.num_channels:
                            data = data[:self.num_channels, :]
                            
                        # 处理新数据
                        self._process_new_data(data)
                    else:
                        # 没有数据时短暂等待
                        time.sleep(0.01)
                        
                except Exception as e:
                    print(f"数据读取错误: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.error_occurred.emit(f"设备配置或启动失败: {e}")
            return
        finally:
            # 停止数据采集
            try:
                self.device.StopAcquisition()
            except Exception as e:
                print(f"停止数据采集失败: {e}")
            
            # 关闭设备
            try:
                self.device.Close()
            except Exception as e:
                print(f"关闭设备失败: {e}")
        
    def _process_new_data(self, new_data):
        """处理新接收的数据
        Args:
            new_data: shape (num_channels, num_samples)
        """
        try:
            # 预处理
            processed_data = self._preprocess_data(new_data)
            
            # 更新缓冲区
            self._update_buffer(processed_data)
            
            # 记录数据
            self.recorded_data.append({
                'timestamp': time.time(),
                'data': processed_data.copy()
            })
            
            # 发送数据信号
            self.data_received.emit(processed_data)
            
            # 更新波形显示
            if self.waveform_display is not None:
                self.waveform_display.update_data(processed_data)
                
        except Exception as e:
            print(f"数据处理错误: {e}")
            
    def _preprocess_data(self, data):
        """数据预处理
        Args:
            data: 原始EEG数据
        Returns:
            processed_data: 预处理后的数据
        """
        processed = data.copy()
        
        try:
            # 检查数据长度是否足够进行滤波
            # sosfiltfilt需要数据长度大于padlen，padlen通常是滤波器阶数的2倍+1
            # 对于4阶滤波器，padlen约为27
            min_data_length = 30  # 安全起见设置为30
            
            if data.shape[1] < min_data_length:
                # 数据长度不足，跳过滤波
                print(f"数据长度不足({data.shape[1]} < {min_data_length})，跳过滤波处理")
                return processed
            
            # 陷波滤波 (去除工频干扰)
            if self.notch_sos is not None:
                processed = signal.sosfiltfilt(self.notch_sos, processed, axis=1)
                
            # 带通滤波
            if self.bandpass_sos is not None:
                processed = signal.sosfiltfilt(self.bandpass_sos, processed, axis=1)
                
        except Exception as e:
            print(f"预处理失败: {e}")
            
        return processed
        
    def _update_buffer(self, data):
        """更新数据缓冲区"""
        num_samples = data.shape[1]
        
        # 滚动缓冲区
        self.data_buffer = np.roll(self.data_buffer, -num_samples, axis=1)
        self.data_buffer[:, -num_samples:] = data
        
    def get_current_window_data(self):
        """获取当前时间窗口的数据
        Returns:
            window_data: shape (num_channels, window_samples)
        """
        window_samples = int(self.time_window * self.sampling_rate)
        return self.data_buffer[:, -window_samples:].copy()
        
    def perform_recognition(self, target_frequencies):
        """执行SSVEP识别
        Args:
            target_frequencies: 目标频率列表
        Returns:
            predicted_index: 预测的目标索引
            confidence: 置信度
        """
        try:
            # 获取当前窗口数据
            window_data = self.get_current_window_data()
            
            # 计算TRCA分数
            scores = self.trca_model.compute_trca_score(
                window_data, target_frequencies, self.time_window)
            
            # 找到最高分数的目标
            predicted_index = np.argmax(scores)
            confidence = scores[predicted_index] / np.sum(scores) if np.sum(scores) > 0 else 0
            
            return predicted_index, confidence
            
        except Exception as e:
            print(f"识别过程失败: {e}")
            return 0, 0.0
            
    def save_recorded_data(self, filename=None):
        """保存记录的数据"""
        if not self.recorded_data:
            return
            
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"eeg_data_{timestamp}.h5"
                
            with h5py.File(filename, 'w') as f:
                # 保存EEG数据
                for i, record in enumerate(self.recorded_data):
                    grp = f.create_group(f'record_{i}')
                    grp.create_dataset('timestamp', data=record['timestamp'])
                    grp.create_dataset('data', data=record['data'])
                    
                # 保存元数据
                f.attrs['sampling_rate'] = self.sampling_rate
                f.attrs['num_channels'] = self.num_channels
                f.attrs['time_window'] = self.time_window
                
            print(f"EEG数据已保存: {filename}")
            
        except Exception as e:
            print(f"保存EEG数据失败: {e}")
            
    def get_device_info(self):
        """获取设备信息"""
        return {
            'connected': self.is_connected,
            'recording': self.is_recording,
            'device_type': str(self.device),
            'sampling_rate': self.sampling_rate,
            'num_channels': self.num_channels,
            'buffer_size': self.buffer_size
        }
        
    def disconnect_device(self):
        """断开设备连接"""
        self.stop_recording()
        
        if self.device and self.device != 'simulation':
            try:
                 # 停止数据采集
                 self.device.StopAcquisition()
                 
                 # 断开真实设备连接
                 self.device.Disconnect()
            except Exception as e:
                 print(f"断开设备连接失败: {e}")
                
        self.is_connected = False
        self.device = None
        self.status_update.emit("设备已断开连接")

# 辅助函数
def filterbank(eeg_data, fs=250, num_fbs=5):
    """
    滤波器组函数 (与TRCA_benchmark.py兼容)
    """
    model = TRCAModel(sampling_rate=fs, num_fbs=num_fbs)
    return model.filterbank(eeg_data)

def ftrca(eeg_data, num_fbs=5, is_ensemble=True):
    """
    快速TRCA函数 (与TRCA_benchmark.py兼容)
    """
    # 这里实现快速TRCA算法
    # 返回格式应与TRCA_benchmark.py一致
    pass