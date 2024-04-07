"""
# -*- coding: utf-8 -*-
# Time : 2023/9/11 16:41
# File : DMCModel.py
# Author : Wang Hai
# Code Specification : PEP8
# Description :
"""
import numpy as np


class DMCModel:
    """DMC预测控制"""

    def __init__(self, step_param, dmc_step):
        """
        初始化阶跃参数、特征个数、最大输入长度、预测步、控制步；
        初始化计算得到：各特征阶跃持续步长、阶跃a矩阵、滚动优化的dt矩阵；
        """
        self.step_param = step_param
        self.params_count = int(len(self.step_param) / 3)
        self.input_max_step = 500
        self.predict_step = dmc_step[0]
        self.control_step = dmc_step[1]
        self.time_lists, self.a_matrix_cut, self.a_matrix_array = self.get_matrix()
        self.dt_matrix_ = self.dt_matrix()

    def get_matrix(self):
        """
        计算各特征持续影响时间步长和a矩阵
        """
        time_lists = []
        a_matrix_cut = []
        a_matrix_array = []
        # 遍历每个影响量
        for pc in range(self.params_count):
            # 取系数k、时间常数t、时滞d
            k = self.step_param[pc]
            t = self.step_param[pc + self.params_count]
            d = self.step_param[pc + self.params_count * 2]
            # 取矩阵1-500
            steps = np.arange(0, self.input_max_step)
            # 按照一阶系统阶跃函数构建动态信息a矩阵
            a_matrix = ((1 - np.exp(-steps / t)) * k)
            # 保存所有动态信息a矩阵，用于滚动优化dt_matrix()使用
            a_matrix_array.append(a_matrix)
            # 遍历A矩阵
            for i in range(len(a_matrix)):
                # 按阶跃到达系数的0.95视为影响结束，截断a矩阵并加上时滞，作为预测使用的a矩阵，保存在a_matrix_cut，
                # predict_values中调用
                if k > 0 and a_matrix[i] > 0.95 * k:
                    time_lists.append(i + d)
                    a_matrix_cut.append(np.append(np.array([0 for _ in range(int(d))]), a_matrix[:i]))
                    break
                if k < 0 and a_matrix[i] < 0.95 * k:
                    time_lists.append(i + d)
                    a_matrix_cut.append(np.append(np.array([0 for _ in range(int(d))]), a_matrix[:i]))
                    break
        return time_lists, a_matrix_cut, a_matrix_array

    def predict_values(self, u_matrix, dp_last):
        """
        初始化一个结果矩阵，用于叠加所有特征输入的增量u_matrix和动态信息a_matrix矩阵点积的矩阵下三角对角线之和，并加上初始的料层值，
        得到未来预测步的料层波动
        """
        result_matrix = np.zeros(self.predict_step)
        # 遍历每个特征
        record_s_matrix = []
        for pc in range(self.params_count):
            # 取当前特征影响步
            cl = self.time_lists[pc]
            # 取动态信息a矩阵
            a_matrix_i = self.a_matrix_cut[pc]
            # 取U矩阵
            u_matrix_i = u_matrix[-int(cl):, pc]
            # 点积运算，并差分
            a_u_dot = np.diff(np.dot(u_matrix_i[:, np.newaxis], a_matrix_i[np.newaxis, :]))[1:, :]
            # a_u_shape = a_u_dot.shape[0]
            # 从a_u_dot右上角-左下角对角线开始对每个对角线求和，遍历会线性叠加所有特征的影响
            record_list = []
            for i in range(len(result_matrix)):
                result_matrix[i] += np.trace(a_u_dot[i:, i:][:, ::-1])
                record_list.append(np.trace(a_u_dot[i:, i:][:, ::-1]))
            record_s_matrix.append(record_list)
        # 通过当前的料层差压值叠加未来100步的影响，得到未来100步的预测值
        y_p = [dp_last + result_matrix[0]]
        # 料层每步的预测值要累加前面所有的影响
        for rm_ in range(len(result_matrix) - 1):
            y_p.append(y_p[rm_] + result_matrix[rm_ + 1])
        return y_p, record_s_matrix

    def dt_matrix(self):
        """
        计算dt_matrix 用于滚动优化输出增量
        """
        # 构建Q矩阵，误差权重矩阵
        q_matrix = np.diag([0.98 for _ in range(self.predict_step)])
        # 构建R矩阵，控制权重矩阵
        r_matrix = np.diag([0.02 for _ in range(self.control_step)])
        # 初始化A矩阵
        a_matrix = np.zeros((self.predict_step, self.control_step))
        # 遍历控制步 构建A矩阵
        for cs in range(self.control_step):
            a_matrix[:, cs] = np.append([0 for _ in range(cs)], self.a_matrix_array[-1][:self.predict_step - cs])
        # 计算得到dt矩阵
        dt_matrix = np.dot(np.dot((1 / (np.dot(np.dot(a_matrix.T, q_matrix), a_matrix) + r_matrix)), a_matrix.T),
                           q_matrix)
        return dt_matrix

    def target_value_flow(self, dp_last, target_value, a):
        """
        平滑目标值
        """
        target_value_list = []
        yr = dp_last
        for i in range(self.predict_step):
            yr = a * yr + (1 - a) * target_value
            target_value_list.append(yr)
        return target_value_list

    def cal_increments(self, input_data, dp_last, target_value):
        """
        :param input_data: 输入特征历史增量变化时序数据
        :param dp_last: 当前计算步料层差异值
        :param target_value: 控制目标值
        :return: 增量、预测值
        """
        # 预测
        target_value_flow = self.target_value_flow(dp_last, target_value, 0)
        predict_value, record_s_matrix = self.predict_values(input_data[-self.input_max_step:], dp_last)
        # 反馈矫正
        error = dp_last - predict_value[0]
        predict_value += 1 * error
        # 滚动优化
        increments = np.dot(self.dt_matrix_, (np.array(target_value_flow) - predict_value))
        return increments[0], predict_value, record_s_matrix

    def cal_increments_spare(self, input_data, dp_last, target_value):
        """
        备用方案：反馈调节，以防DMC效果不佳！
        :param input_data: 输入特征历史增量变化时序数据
        :param dp_last: 当前计算步料层差异值
        :param target_value: 控制目标值
        :return: 增量、预测值
        """
        target_value_flow = self.target_value_flow(dp_last, target_value, 0)
        predict_value = [dp_last for _ in range(10)]
        increments = sum(target_value_flow[:10]) - sum(predict_value)
        return increments, predict_value
