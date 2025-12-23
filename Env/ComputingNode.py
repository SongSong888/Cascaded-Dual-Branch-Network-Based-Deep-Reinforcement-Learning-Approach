from Conf import SystemConfig
import numpy as np


def fix_random_seed():
    np.random.seed(88)


# -------------------- 计算节点类 --------------------
class ComputingNode:
    def __init__(self, node_type):
        self.node_type = node_type
        if "local" in node_type:
            self.dynamic_cpu = np.round(np.random.uniform(SystemConfig.END_CPU[0], SystemConfig.END_CPU[1]), 3)
            self.k = SystemConfig.TD_K
        elif "edge" in node_type:
            self.dynamic_cpu = np.round(np.random.uniform(SystemConfig.EDGE_CPU[0], SystemConfig.EDGE_CPU[1]), 3)
            self.k = SystemConfig.EDGE_K
        else:
            self.dynamic_cpu = SystemConfig.CLOUD_CPU
            self.k = 0.0
        self.total_bw = SystemConfig.eMBB_bandwidth
        self.trans_power = SystemConfig.TD_TRANS_PTX
        self.allocated = {}  # {task_id: [bw_ratio, cpu_ratio]}

    def allocate_resources(self, task_id, bw_ratio, cpu_ratio):
        """分配资源并返回实际分配值"""
        available_bw = 1.0 - sum([v[0] for v in self.allocated.values()])
        available_cpu = 1.0 - sum([v[1] for v in self.allocated.values()])

        actual_bw = min(bw_ratio, available_bw)
        actual_cpu = min(cpu_ratio, available_cpu)

        self.allocated[task_id] = [actual_bw, actual_cpu]
        return (actual_bw * self.total_bw,
                actual_cpu * self.dynamic_cpu)

    def release_resources(self, task_id):
        if task_id in self.allocated:
            del self.allocated[task_id]

    def update_bw(self, task_id):
        if task_id in self.allocated:
            self.allocated[task_id][0] = 0.0

    def clear_allocated(self):
        self.allocated.clear()

    def calculate_upper_resource(self, bw_ratio, cpu_ratio):
        """计算上层动作的可用资源"""
        available_bw = 1.0 - sum([v[0] for v in self.allocated.values()])
        available_cpu = 1.0 - sum([v[1] for v in self.allocated.values()])
        actual_bw = min(bw_ratio, available_bw)
        actual_cpu = min(cpu_ratio, available_cpu)
        return (actual_bw * self.total_bw,
                actual_cpu * self.dynamic_cpu)
