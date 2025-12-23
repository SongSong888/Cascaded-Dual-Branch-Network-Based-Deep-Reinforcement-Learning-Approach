# -------------------- 系统常量 --------------------
import os


class SystemConfig:
    # 基础参数 
    NUM_FACTORIES = 6  # 工厂数量
    NUM_EDGES = 7  # 边缘服务器数量
    CLOUD_SERVERS = 1  # 云服务器数量
    TIME_SLOT = 1.0  # 时间槽长度(ms)
    SIMULATION_TIME = 3 * 60 * 1000  # 一轮时间3分钟(ms)
    SAVE_MODEL_PATH = './saved_model/'  # 模型保存路径
    ALL_EPISODES = 100  # 总训练轮数
    TEST_EPISODES = 20  # 测试轮数

    # 计算资源 
    END_CPU = [1.5, 3.0]  # 本地终端总CPU频率(GHz) 1.5~2.0异构
    EDGE_CPU = [10.0, 20.0]  # 边缘服务总CPU频率(GHz)
    CLOUD_CPU = 50.0  # 边缘服务总CPU频率(GHz)
    TD_K = 10e-25  # 本地设备的能耗系数 κ
    TD_TRANS_PTX = 0.2  # 传输功率0.2w
    EDGE_K = 1e-28  # 边缘设备的能耗系数 κ
    POWER_COEFFICIENT = 0.01  # 能耗权重

    # 任务参数 
    TASK_TYPES = ['deterministic', 'non-deterministic']
    TASK_FEATURE_DIM = 4  # (数据大小，截止时间，是否非确定性任务，需要的cpu频率，(任务优先级，哪个时段的任务))
    TASK_DETE_CYCLE_MEAN = 5e2  # 确定性任务计算周期均值
    TASK_NON_DETE_CYCLE_MEAN = 1e4  # 非确定性任务计算周期均值
    TASK_DETE_DATA = 1.5  # 确定性任务数据大小均值
    TASK_NON_DETE_DATA = 2

    # 5G-A模拟环境参数
    # eMBB参数
    eMBB_subcarrier_spacing = 60e3  # 60 kHz子载波间隔
    eMBB_symbols_per_slot = 14  # 常规时隙结构
    eMBB_bandwidth = 400e6  # 400MHz带宽
    eMBB_scheduling_latency = 10e-3  # 10ms调度时延
    eMBB_code_rate = 0.78  # 较高码率
    eMBB_modulation_order = 8  # 256QAM（高吞吐量）
    eMBB_symbol_duration = 1 / eMBB_subcarrier_spacing
    eMBB_slot_duration = eMBB_symbols_per_slot * eMBB_symbol_duration
    eMBB_bits_per_re = eMBB_modulation_order * eMBB_code_rate
    # URLLC参数
    URLLC_subcarrier_spacing = 120e3  # 120 kHz子载波间隔
    URLLC_symbols_per_slot = 2  # 微时隙结构（快速传输）
    URLLC_bandwidth = 400e6  # 400MHz带宽
    URLLC_scheduling_latency = 0.25e-3  # 0.25ms超低调度时延
    URLLC_code_rate = 0.45  # 中等码率
    URLLC_modulation_order = 6  # 64QAM
    URLLC_symbol_duration = 1 / URLLC_subcarrier_spacing
    URLLC_slot_duration = URLLC_symbols_per_slot * URLLC_symbol_duration
    URLLC_bits_per_re = URLLC_modulation_order * URLLC_code_rate

    TO_CLOUD_LATENCY = 40  # 任务传输到云端的时间(ms)

    # RL参数
    UPPER_STATE_DIM = 2 + NUM_EDGES * 3 + CLOUD_SERVERS * 3 + TASK_FEATURE_DIM  # 上层状态维度
    UPPER_ACTION_SPACE = 1 + NUM_EDGES + CLOUD_SERVERS
    LOWER_STATE_DIM = UPPER_ACTION_SPACE + UPPER_STATE_DIM  # 下层状态维度
    LOWER_ACTION_DIM = 2  # (带宽比例, CPU比例)
    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    START_TRAIN_SIZE = 1000

    # DQN参数
    DQN_GAMMA = 0.1
    DQN_TAU = 0.005
    DQN_LR = 3e-4
    # SASAC参数
    FEATURE_NET_LR = 3e-3
    ETA = 0.01
    SASAC_ACTOR_LR = 1e-3
    SASAC_CRITIC_LR = 1e-2
    SASAC_ALPHA_LR = 3e-4
    SASAC_ALPHA = 1
    SASAC_GAMMA = 0.1
    SASAC_TAU = 0.005
    # SAC参数
    SAC_ACTOR_LR = 1e-3
    SAC_CRITIC_LR = 1e-2
    SAC_ALPHA_LR = 3e-4
    SAC_ALPHA = 1
    SAC_GAMMA = 0.1
    SAC_TAU = 0.005
    # DDPG参数
    SIGMA = 0.15
    DDPG_ACTOR_LR = 1e-3
    DDPG_CRITIC_LR = 1e-2
    DDPG_TAU = 0.005
    DDPG_GAMMA = 0.1

    UPDATE_FREQUENCY = 10000  # 更新频率(ms) 10s*1000


class FileConfig:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def create_file(self, method):
        # 构建基础路径：log/save/method
        base_dir = os.path.join(self.base_dir, method)
        # 确保基础目录存在
        os.makedirs(base_dir, exist_ok=True)

        # 获取所有数字子目录的编号
        existing_nums = []
        # 遍历目录中的每个条目
        for name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, name)
            # 检查是否是目录且名称全为数字
            if os.path.isdir(dir_path) and name.isdigit():
                existing_nums.append(int(name))

        # 确定新目录的编号
        new_num = max(existing_nums) + 1 if existing_nums else 1

        # 创建新目录
        new_dir = os.path.join(base_dir, str(new_num))
        os.makedirs(new_dir, exist_ok=True)
        print(f'Created directory: {new_dir}')
        return new_dir, new_num
