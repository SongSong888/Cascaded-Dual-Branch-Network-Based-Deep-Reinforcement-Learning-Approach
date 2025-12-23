import random
import numpy as np
from Conf import SystemConfig
from scipy.stats import poisson


# 任务生成器类
class TaskGenerator:
    def __init__(self, task_lambda=15.3):
        self.lambda_map = {
            'rush_hours': task_lambda,  # 高峰时段任务率
            'normal_time': task_lambda,  # 普通时段任务率
            'night_time': task_lambda  # 夜间时段任务率
        }
        self.last_gen_time = 0
        self.task_counter = 0
        # 设置随机种子
        random.seed(88)
        np.random.seed(88)

    def clear(self):
        self.last_gen_time = 0
        self.task_counter = 0

    def test_fix_random_seed(self, episode):
        random.seed(episode)
        np.random.seed(episode)

    def generate(self, current_time):
        """基于时间段的泊松过程生成任务批次"""
        tasks = []
        time_slot, time_type = self._get_time_slot(current_time)
        lambda_param = self.lambda_map[time_slot]
        # 计算时间间隔内的事件数
        delta_t = current_time - self.last_gen_time
        n_tasks = poisson.rvs(lambda_param * delta_t / 1000)  # 转换为毫秒级

        # 生成任务批次
        for _ in range(n_tasks):
            factory_name = np.random.randint(0, SystemConfig.NUM_FACTORIES)
            task = IIoTTask(factory_name, current_time)
            task.priority = self._calc_priority(time_slot, task.task_type)
            task.time_type = time_type
            tasks.append(task)
            self.task_counter += 1

        self.last_gen_time = current_time
        return tasks

    def _get_time_slot(self, timestamp):
        """根据仿真时间判断时段"""
        # hour = (timestamp // 3600000)  # 转换为小时
        if 0 <= timestamp < 1*60*1000:  # 0-1分钟
            return 'normal_time', 0
        elif 1*60*1000 <= timestamp < 2*60*1000:  # 1-2分钟
            return 'rush_hours', 1
        else:
            return 'night_time', 2

    def _calc_priority(self, time_slot, task_type):
        """动态优先级计算"""
        priority_matrix = {
            'rush_hours': {'deterministic': 3, 'non-deterministic': 2},
            'normal_time': {'deterministic': 2, 'non-deterministic': 1},
            'night_time': {'deterministic': 1, 'non-deterministic': 1}
        }
        return priority_matrix[time_slot][task_type]


# 任务类
class IIoTTask:
    def __init__(self, factory_name, gen_time):
        self.task_from = factory_name
        self.task_id = self._generate_uid(gen_time)
        self.gen_time = gen_time
        self.task_type = self._set_task_type()
        self.data_size = self._set_data_size()
        self.cycles = self._set_compute_cycles()
        self.deadline = self._set_deadline(gen_time)
        self.priority = 0  # 优先级
        self.time_type = 0  # 时间类型, 0: normal_time, 1: rush_hours, 2: night_time
        self.task_transfer_time = 0  # 传输完成时间
        self.finish_time = 0  # 完成时间

    def _generate_uid(self, timestamp):
        """生成唯一任务ID"""
        return hash(f"IIoT_TASK_{timestamp}_{random.getrandbits(64)}")

    def _set_data_size(self):
        """基于任务类型设置数据量"""
        if self.task_type == 'deterministic':
            return np.clip(np.random.weibull(SystemConfig.TASK_DETE_DATA), 0.1, 3)  # 1-3MB
        else:
            return np.clip(np.random.exponential(SystemConfig.TASK_NON_DETE_DATA) * 5, 1, 50)  # 1-50MB

    def _set_compute_cycles(self):
        """设置计算量（兆周期）"""
        if self.task_type == 'deterministic':
            return np.clip(np.random.normal(SystemConfig.TASK_DETE_CYCLE_MEAN, 2e2), 1e1, 1e3)  # 10-1000
        else:
            return np.clip(np.random.normal(SystemConfig.TASK_NON_DETE_CYCLE_MEAN, 5e3), 1e2, 2e4)  # 100-20000

    def _set_deadline(self, gen_time):
        """设置动态时限"""
        base_deadline = {
            'deterministic': np.random.randint(100, 200),  # 确定性任务的完成时间要求在20-200ms之间
            'non-deterministic': np.random.randint(1000, 5000)  # 1000-5000ms的截止时间
        }[self.task_type]
        return gen_time + base_deadline

    def _set_task_type(self):
        """基于概率分布设置任务类型"""
        return np.random.choice(
            SystemConfig.TASK_TYPES,
            p=[0.65, 0.35]
        )

    def __str__(self):
        return f"Task ID: {self.task_id}, Type: {self.task_type}, Size: {self.data_size}MB, Cycles: {self.cycles}M, Deadline: {self.deadline}"


if __name__ == '__main__':
    gen = TaskGenerator()
    for i in range(3600000):
        gen.generate(i)
