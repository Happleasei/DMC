# -*- coding: utf-8 -*-
# @Time : 2023/4/10 15:13
# @Author : Wang Hai
# @Email : nicewanghai@163.com
# @Code Specification : PEP8
# @File : DMCtrl.py
# @Project :
import numpy as np
import pandas as pd


class DMCCtrl:

    def __init__(self):
        pass

    @staticmethod
    def data_process(data_list):
        """
        选取预测模型的输入列，拼接为一个二维数组
        输入只需要保证 最后一个是控制量即可，增加输入后对应修改step_param
        一次风风量 炉瞬时总给煤量 是控制目标的影响量
        冷渣机变频器转速总反馈 是实际控制的设备
        """
        dl_df = pd.DataFrame(data_list)
        input_df = pd.DataFrame([])
        # 一次风风量
        input_df["first_wind"] = dl_df.iloc[:, [60]]
        # 炉瞬时总给煤量
        input_df["coal_flow"] = dl_df.iloc[:, [30]]
        """////----------可在下一行开始添加新的输入，需与阶跃参数对应即可----------////"""
        # 冷渣机变频器转速总反馈
        input_df["cold_slag"] = dl_df.iloc[:, [6, 11, 16, 21]].sum(axis=1)
        input_data = input_df.diff().iloc[1:, :].values
        # 当前料层差压值
        dp_last = dl_df.iloc[:, [0, 1]].mean(axis=1).iloc[-1]
        return input_data, dp_last

    def cal_best_ctrl_value(self, dmc_model, data_list, target_value, increments, safe_limit, limit_dp, multi_value):
        """
        :param dmc_model: 调用DMC模型
        :param data_list: 数据list
        :param target_value: 目标值
        :param increments: 增量
        :param safe_limit: 安全约束
        :param limit_dp: 沼泽区区间
        :param multi_value: 增量系数
        :return:
        """
        limit_dp_step = 10
        input_data, dp_last = self.data_process(data_list)
        out_increment, predict_value = dmc_model.cal_increments(input_data, dp_last, target_value)
        print("预测非控未来0-10步料层差压：{}".format(np.round(predict_value[0:10], 4)))
        if sum([(predict_value[i] - target_value) ** 2 for i in range(limit_dp_step)]) < (limit_dp ** 2) * limit_dp_step:
            safe_limit += "未来料层差压在目标值上下{}波动不控制！".format(limit_dp)
            out_increment = 0
        out_increment = out_increment * multi_value
        print(out_increment)
        if out_increment <= increments[0]:
            out_increment = increments[0]
        elif out_increment >= increments[1]:
            out_increment = increments[1]
        return out_increment, predict_value[:10], [], safe_limit
