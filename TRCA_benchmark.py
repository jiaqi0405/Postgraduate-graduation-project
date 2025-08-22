import numpy as np
import scipy.io
import scipy.signal
import scipy.linalg
from scipy import stats
import warnings
from datetime import datetime
import os

def filterbank(eeg, fs, idx_fb):
    """
    Filter bank design for decomposing EEG data into sub-band components

    Parameters:
    -----------
    eeg : ndarray
        Input EEG data (n_channels, n_samples, n_trials)
    fs : float
        Sampling rate
    idx_fb : int
        Index of filters in filter bank analysis (1-10)

    Returns:
    --------
    y : ndarray
        Sub-band components decomposed by filter bank
    """
    if idx_fb is None:
        warnings.warn('Missing filter index. Default value (idx_fb = 1) will be used.')
        idx_fb = 1
    elif idx_fb < 1 or idx_fb > 10:
        raise ValueError('The number of sub-bands must be 0 < idx_fb <= 10.')

    # Get data dimensions
    if eeg.ndim == 2:
        num_chans, num_smpls = eeg.shape
        num_trials = 1
        eeg = eeg[:, :, np.newaxis]
    else:
        num_chans, num_smpls, num_trials = eeg.shape

    # Nyquist frequency - 注意原始代码中的fs=fs/2
    fs_nyq = fs / 2

    # Filter bank parameters
    passband = np.array([6, 14, 22, 30, 38, 46, 54, 62, 70, 78])
    stopband = np.array([4, 10, 16, 24, 32, 40, 48, 56, 64, 72])

    # Filter design parameters (注意：MATLAB索引从1开始，Python从0开始)
    Wp = [passband[idx_fb - 1] / fs_nyq, 90 / fs_nyq]
    Ws = [stopband[idx_fb - 1] / fs_nyq, 100 / fs_nyq]

    # Design Chebyshev Type I filter
    N, Wn = scipy.signal.cheb1ord(Wp, Ws, 3, 40)
    B, A = scipy.signal.cheby1(N, 0.5, Wn, btype='band')

    # Apply filter
    y = np.zeros_like(eeg)
    if num_trials == 1:
        for ch_i in range(num_chans):
            y[ch_i, :, 0] = scipy.signal.filtfilt(B, A, eeg[ch_i, :, 0])
    else:
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[ch_i, :, trial_i] = scipy.signal.filtfilt(B, A, eeg[ch_i, :, trial_i])

    # Remove singleton dimension if only one trial
    if num_trials == 1:
        y = y[:, :, 0]

    return y


def ftrca(eeg):
    """
    Fast Task-Related Component Analysis (fTRCA)

    Parameters:
    -----------
    eeg : ndarray
        Input EEG data (n_channels, n_samples, n_trials)

    Returns:
    --------
    W : ndarray
        Weight coefficients for spatial filtering
    """
    num_chans, num_smpls, num_trials = eeg.shape

    # Center data (remove mean)
    for trial_i in range(num_trials):
        x1 = eeg[:, :, trial_i]
        eeg[:, :, trial_i] = x1 - np.mean(x1, axis=1, keepdims=True)

    # Compute covariance matrices
    SX = np.sum(eeg, axis=2)
    S = SX @ SX.T

    UX = eeg.reshape(num_chans, num_smpls * num_trials)
    Q = UX @ UX.T

    # Solve generalized eigenvalue problem
    eigenvalues, W = scipy.linalg.eigh(S, Q)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    W = W[:, idx]

    return W


def trca(eeg):
    """
    Task-Related Component Analysis (TRCA)

    Parameters:
    -----------
    eeg : ndarray
        Input EEG data (n_channels, n_samples, n_trials)

    Returns:
    --------
    W : ndarray
        Weight coefficients for spatial filtering
    """
    num_chans, num_smpls, num_trials = eeg.shape
    S = np.zeros((num_chans, num_chans))

    # Compute inter-trial covariance
    for trial_i in range(num_trials - 1):
        x1 = eeg[:, :, trial_i]
        x1 = x1 - np.mean(x1, axis=1, keepdims=True)

        for trial_j in range(trial_i + 1, num_trials):
            x2 = eeg[:, :, trial_j]
            x2 = x2 - np.mean(x2, axis=1, keepdims=True)
            S = S + x1 @ x2.T + x2 @ x1.T

    # Compute total covariance
    UX = eeg.reshape(num_chans, num_smpls * num_trials)
    UX = UX - np.mean(UX, axis=1, keepdims=True)
    Q = UX @ UX.T

    # Solve generalized eigenvalue problem
    eigenvalues, W = scipy.linalg.eigh(S, Q)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    W = W[:, idx]

    return W


def sscor(eeg):
    """
    Sum of Squared Correlations (SSCOR)

    Parameters:
    -----------
    eeg : ndarray
        Input EEG data (n_channels, n_samples, n_trials)

    Returns:
    --------
    W : ndarray
        Weight coefficients for spatial filtering
    """
    num_chans, _, num_trials = eeg.shape

    # Compute average template
    X_n = np.mean(eeg, axis=2)
    X_n = X_n - np.mean(X_n, axis=1, keepdims=True)

    # Cholesky decomposition
    K_1 = np.linalg.cholesky(X_n @ X_n.T)

    # Compute sum of squared correlations
    G_T_G = np.zeros((num_chans, num_chans))
    for trial_i in range(num_trials):
        x_i = eeg[:, :, trial_i]
        x_i = x_i - np.mean(x_i, axis=1, keepdims=True)

        K_i = np.linalg.cholesky(x_i @ x_i.T)
        C_1i = X_n @ x_i.T
        G_i = np.linalg.solve(K_1.T, C_1i) @ np.linalg.inv(K_i)
        G_T_G = G_T_G + G_i.T @ G_i

    # Get largest eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(G_T_G)
    v_1 = eigenvectors[:, -1]

    # Compute spatial filter
    W = np.linalg.solve(K_1, v_1.reshape(-1, 1))

    return W


def itr(n, p, t):
    """
    Calculate information transfer rate (ITR) for BCI

    Parameters:
    -----------
    n : int
        Number of targets
    p : float
        Target identification accuracy (0 <= p <= 1)
    t : float
        Averaged time for a selection [s]

    Returns:
    --------
    itr : float
        Information transfer rate [bits/min]
    """
    if p < 0 or p > 1:
        raise ValueError('Accuracy needs to be between 0 and 1.')
    elif p < 1 / n:
        warnings.warn('The ITR might be incorrect because accuracy < chance level.')
        return 0
    elif p == 1:
        return np.log2(n) * 60 / t
    else:
        return (np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))) * 60 / t


def train_model(eeg, fs, num_fbs, algorithm='ftrca'):
    """
    Training model based on template-matching method for SSVEP detection

    Parameters:
    -----------
    eeg : ndarray
        Input EEG data (n_targets, n_channels, n_samples, n_trials)
    fs : float
        Sampling rate
    num_fbs : int
        Number of sub-bands
    algorithm : str
        Spatial filtering method ('ftrca', 'trca', or 'sscor')

    Returns:
    --------
    model : dict
        Learning model for testing phase
    """
    if algorithm not in ['ftrca', 'trca', 'sscor']:
        raise ValueError('Algorithm should be selected from ftrca, trca, or sscor.')

    print(f'The algorithm, {algorithm}, will be used.')

    if num_fbs is None:
        num_fbs = 3

    num_targs, num_chans, num_smpls, _ = eeg.shape

    # Initialize arrays
    trains = np.zeros((num_targs, num_fbs, num_chans, num_smpls))
    W = np.zeros((num_fbs, num_targs, num_chans))

    # Train for each target and filter bank
    for targ_i in range(num_targs):
        eeg_tmp = eeg[targ_i, :, :, :]

        for fb_i in range(num_fbs):
            # Apply filter bank
            eeg_filtered = filterbank(eeg_tmp, fs, fb_i + 1)  # MATLAB索引从1开始

            # Store average template
            trains[targ_i, fb_i, :, :] = np.mean(eeg_filtered, axis=2)

            # Compute spatial filter
            if algorithm == 'ftrca':
                w_tmp = ftrca(eeg_filtered)
            elif algorithm == 'trca':
                w_tmp = trca(eeg_filtered)
            elif algorithm == 'sscor':
                w_tmp = sscor(eeg_filtered)

            W[fb_i, targ_i, :] = w_tmp[:, 0]

    # Create model dictionary
    model = {
        'trains': trains,
        'W': W,
        'num_fbs': num_fbs,
        'fs': fs,
        'num_targs': num_targs
    }

    return model


def test_model(eeg, model, is_ensemble=True):
    """
    Test model based on template-matching method for SSVEP detection

    Parameters:
    -----------
    eeg : ndarray
        Input EEG data (n_targets, n_channels, n_samples)
    model : dict
        Learning model from training phase
    is_ensemble : bool
        True for ensemble TRCA method, False for standard TRCA

    Returns:
    --------
    results : ndarray
        Estimated target indices (1-based to match MATLAB)
    """
    num_targs = model['num_targs']
    num_fbs = model['num_fbs']

    # Filter bank coefficients
    fb_coefs = np.arange(1, num_fbs + 1) ** (-1.25) + 0.25

    results = np.zeros(num_targs, dtype=int)

    for targ_i in range(num_targs):
        test_tmp = eeg[targ_i, :, :]
        r = np.zeros((num_fbs, num_targs))

        for fb_i in range(num_fbs):
            # Apply filter bank
            testdata = filterbank(test_tmp, model['fs'], fb_i + 1)

            for class_i in range(num_targs):
                traindata = model['trains'][class_i, fb_i, :, :]

                if not is_ensemble:
                    # Standard TRCA: use class-specific spatial filter
                    w = model['W'][fb_i, class_i, :]
                else:
                    # Ensemble TRCA: average spatial filters across all classes
                    w = np.mean(model['W'][fb_i, :, :], axis=0)

                # Compute correlation
                r_tmp = np.corrcoef(testdata.T @ w, traindata.T @ w)
                r[fb_i, class_i] = r_tmp[0, 1]

        # Weighted combination across filter banks
        rho = fb_coefs @ r

        # Find maximum (add 1 for MATLAB-style indexing)
        results[targ_i] = np.argmax(rho) + 1

    return results


# ==================== 新增功能：模型保存和加载 ====================

def save_trca_pretrained_weights(model, filename='trca_pretrained_weights.mat', subject_id=1):
    """
    保存TRCA模型为预训练权重文件
    
    Parameters:
    -----------
    model : dict
        训练好的TRCA模型
    filename : str
        保存的文件名
    subject_id : int
        被试编号
    """
    
    print(f"正在保存TRCA预训练权重到: {filename}")
    
    # 计算ensemble权重（每个子频带的平均权重）
    ensemble_weights = {}
    for fb_i in range(model['num_fbs']):
        # 对每个子频带，计算所有目标的平均空间滤波器
        ensemble_w = np.mean(model['W'][fb_i, :, :], axis=0)
        ensemble_weights[f'W_ensemble_band{fb_i+1}'] = ensemble_w.reshape(-1, 1)
    
    # 分离每个子频带的训练模板
    templates = {}
    for fb_i in range(model['num_fbs']):
        templates[f'trains_band{fb_i+1}'] = model['trains'][:, fb_i, :, :]
    
    # 分离每个子频带的空间滤波器权重
    spatial_filters = {}
    for fb_i in range(model['num_fbs']):
        spatial_filters[f'W_band{fb_i+1}'] = model['W'][fb_i, :, :]
    
    # 滤波器银行系数
    fb_coefs = np.arange(1, model['num_fbs'] + 1) ** (-1.25) + 0.25
    
    # 构建保存的数据字典
    save_data = {
        # Ensemble权重（用于实时识别）
        **ensemble_weights,
        
        # 训练模板（每个子频带）
        **templates,
        
        # 完整的空间滤波器权重（每个子频带×每个目标）
        **spatial_filters,
        
        # 滤波器银行系数
        'fb_coefs': fb_coefs,
        
        # 模型参数
        'fs': np.array([[model['fs']]]),
        'num_fbs': np.array([[model['num_fbs']]]),
        'num_targs': np.array([[model['num_targs']]]),
        
        # 模型信息
        'model_info': {
            'algorithm': 'ensemble_trca',
            'subject_id': subject_id,
            'save_time': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'TRCA pretrained weights for single subject'
        }
    }
    
    # 创建保存目录
    save_dir = os.path.dirname(filename) if os.path.dirname(filename) else '.'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存为MATLAB格式
    try:
        scipy.io.savemat(filename, save_data)
        print(f"✅ TRCA预训练权重保存成功!")
        print(f"   文件位置: {os.path.abspath(filename)}")
        print(f"   文件大小: {os.path.getsize(filename) / 1024:.2f} KB")
        
        # 显示保存的内容摘要
        print(f"\n保存内容摘要:")
        print(f"- Ensemble权重: {model['num_fbs']} 个子频带")
        print(f"- 训练模板: {model['num_targs']} 个目标 × {model['num_fbs']} 个子频带")
        print(f"- 空间滤波器: {model['num_fbs']} × {model['num_targs']} 个权重矩阵")
        print(f"- 采样率: {model['fs']} Hz")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def load_trca_pretrained_weights(filename='trca_pretrained_weights.mat'):
    """
    加载TRCA预训练权重文件
    
    Parameters:
    -----------
    filename : str
        权重文件名
        
    Returns:
    --------
    model : dict
        重建的TRCA模型
    """
    
    print(f"正在加载TRCA预训练权重: {filename}")
    
    try:
        # 加载MATLAB文件
        data = scipy.io.loadmat(filename)
        
        # 提取基本参数
        fs = int(data['fs'][0, 0])
        num_fbs = int(data['num_fbs'][0, 0])
        num_targs = int(data['num_targs'][0, 0])
        
        # 重建trains数组
        trains = np.zeros((num_targs, num_fbs, data['trains_band1'].shape[1], data['trains_band1'].shape[2]))
        for fb_i in range(num_fbs):
            trains[:, fb_i, :, :] = data[f'trains_band{fb_i+1}']
        
        # 重建W数组
        W = np.zeros((num_fbs, num_targs, data['W_band1'].shape[1]))
        for fb_i in range(num_fbs):
            W[fb_i, :, :] = data[f'W_band{fb_i+1}']
        
        # 重建模型字典
        model = {
            'trains': trains,
            'W': W,
            'num_fbs': num_fbs,
            'fs': fs,
            'num_targs': num_targs
        }
        
        print(f"✅ TRCA预训练权重加载成功!")
        print(f"   目标数量: {num_targs}")
        print(f"   子频带数量: {num_fbs}")
        print(f"   采样率: {fs} Hz")
        
        return model
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


def verify_model_save_load(original_model, filename):
    """
    验证模型保存和加载的正确性
    
    Parameters:
    -----------
    original_model : dict
        原始模型
    filename : str
        保存的文件名
    """
    
    print("\n" + "="*50)
    print("验证模型保存和加载功能")
    print("="*50)
    
    # 加载保存的模型
    loaded_model = load_trca_pretrained_weights(filename)
    
    if loaded_model is None:
        print("❌ 模型加载失败，无法验证")
        return False
    
    # 验证关键参数
    checks = []
    
    # 检查基本参数
    checks.append(('采样率', original_model['fs'] == loaded_model['fs']))
    checks.append(('子频带数量', original_model['num_fbs'] == loaded_model['num_fbs']))
    checks.append(('目标数量', original_model['num_targs'] == loaded_model['num_targs']))
    
    # 检查数组形状
    checks.append(('trains形状', original_model['trains'].shape == loaded_model['trains'].shape))
    checks.append(('W形状', original_model['W'].shape == loaded_model['W'].shape))
    
    # 检查数值精度（允许小的数值误差）
    trains_close = np.allclose(original_model['trains'], loaded_model['trains'], rtol=1e-10)
    W_close = np.allclose(original_model['W'], loaded_model['W'], rtol=1e-10)
    checks.append(('trains数值', trains_close))
    checks.append(('W数值', W_close))
    
    # 显示验证结果
    all_passed = True
    for check_name, passed in checks:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{check_name:12} : {status}")
        all_passed = all_passed and passed
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 模型保存和加载验证完全通过！")
        print("   预训练权重文件可以安全使用。")
    else:
        print("⚠️  模型保存和加载存在问题，请检查。")
    print("="*50)
    
    return all_passed


def main():
    """
    Main function - Implementation of TRCA-based SSVEP detection with model saving
    """
    print("TRCA-based SSVEP Detection System with Model Saving")
    print("=" * 60)

    # Parameters for analysis
    filename = 'data/S1.mat'  # 只处理S1被试
    len_gaze_s = 0.5 # Data length for target identification [s]
    len_delay_s = 0.13  # Visual latency [s]
    num_fbs = 5  # Number of sub-bands
    is_ensemble = 1  # 0: TRCA, 1: Ensemble TRCA
    alpha_ci = 0.05  # Significance level for confidence intervals

    # Fixed parameters
    fs = 250  # Sampling rate [Hz]
    len_shift_s = 0.5  # Duration for gaze shifting [s]

    # Stimulus frequencies
    list_freqs = []
    for offset in [0, 0.2, 0.4, 0.6, 0.8]:
        list_freqs.extend(np.arange(8, 16) + offset)
    list_freqs = np.array(list_freqs)

    num_targs = len(list_freqs)
    labels = np.arange(1, num_targs + 1)  # 1-based labels

    # Calculate derived parameters
    len_gaze_smpl = int(np.round(len_gaze_s * fs))
    len_delay_smpl = int(np.round(len_delay_s * fs))
    len_sel_s = len_gaze_s + len_shift_s
    ci = 100 * (1 - alpha_ci)

    # Load data
    print(f"\n加载数据: {filename}")
    try:
        mat_data = scipy.io.loadmat(filename)
        data = mat_data['data']  # Assuming variable name is 'data'

        # Rearrange dimensions: [channels, samples, targets, blocks] -> [targets, channels, samples, blocks]
        eeg = np.transpose(data, (2, 0, 1, 3))

        # Select channels of interest (convert MATLAB 1-based to Python 0-based)
        chan_idx = np.array([48, 54, 55, 56, 57, 58, 61, 62, 63]) - 1
        eeg = eeg[:, chan_idx, :, :]

        # Get data dimensions
        _, num_chans, _, num_blocks = eeg.shape

        print(f"✅ 数据加载成功!")
        print(f"   数据形状: {eeg.shape}")
        print(f"   使用通道数: {num_chans}")
        print(f"   实验块数: {num_blocks}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # Perform Leave-One-Out Cross-Validation
    print(f"\n开始 {'Ensemble' if is_ensemble else 'Standard'} TRCA 训练和测试:")
    print("-" * 50)

    accs = np.zeros(num_blocks)
    itrs = np.zeros(num_blocks)
    trained_model = None  # 保存训练好的模型

    for loocv_i in range(num_blocks):
        # Training stage
        train_idx = list(range(num_blocks))
        train_idx.remove(loocv_i)
        traindata = eeg[:, :, :, train_idx]

        model = train_model(traindata, fs, num_fbs, 'ftrca')
        
        # 保存第一次训练的模型用于后续保存
        if trained_model is None:
            trained_model = model

        # Test stage
        testdata = eeg[:, :, :, loocv_i]
        estimated = test_model(testdata, model, is_ensemble)

        # Evaluation
        is_correct = (estimated == labels)
        accs[loocv_i] = np.mean(is_correct) * 100
        itrs[loocv_i] = itr(num_targs, np.mean(is_correct), len_sel_s)

        print(f'Block {loocv_i + 1}: Accuracy = {accs[loocv_i]:5.2f}%, ITR = {itrs[loocv_i]:5.2f} bpm')

    # Summarize results
    print("\n" + "=" * 50)
    print("交叉验证结果汇总")
    print("=" * 50)

    # Calculate confidence intervals
    acc_mean = np.mean(accs)
    acc_ci = stats.t.interval(1 - alpha_ci, len(accs) - 1, loc=acc_mean, scale=stats.sem(accs))

    itr_mean = np.mean(itrs)
    itr_ci = stats.t.interval(1 - alpha_ci, len(itrs) - 1, loc=itr_mean, scale=stats.sem(itrs))

    print(f'平均准确率 = {acc_mean:5.2f}% ({ci:.0f}% CI: {acc_ci[0]:5.2f} - {acc_ci[1]:5.2f}%)')
    print(f'平均ITR = {itr_mean:5.2f} bpm ({ci:.0f}% CI: {itr_ci[0]:5.2f} - {itr_ci[1]:5.2f} bpm)')

    # ========== 新增功能：保存预训练权重 ==========
    print("\n" + "=" * 60)
    print("保存TRCA预训练权重")
    print("=" * 60)
    
    if trained_model is not None:
        # 保存预训练权重
        weight_filename = 'trca_pretrained_weights.mat'
        subject_id = 1  # S1被试
        
        success = save_trca_pretrained_weights(trained_model, weight_filename, subject_id)
        
        if success:
            # 验证保存和加载功能
            verify_model_save_load(trained_model, weight_filename)
            
            print(f"\n🎉 任务完成！")
            print(f"   TRCA预训练权重已保存为: {weight_filename}")
            print(f"   现在可以在eeg_processor.py中加载使用此模型")
            print(f"   平均识别准确率: {acc_mean:.2f}%")
        else:
            print(f"\n❌ 预训练权重保存失败")
    else:
        print("❌ 没有训练好的模型可以保存")


if __name__ == "__main__":
    main()