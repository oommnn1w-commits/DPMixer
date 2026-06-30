"""
M4 Summary
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import os

from data_provider.m4 import M4Dataset, M4Meta


def group_values(values, groups, group_name):
    """根据组名筛选时间序列，并去掉 NaN"""
    return [v[~np.isnan(v)] for v, g in zip(values, groups) if g == group_name]


def mase(forecast, insample, outsample, frequency):
    """Mean Absolute Scaled Error"""
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def smape_2(forecast, target):
    """sMAPE"""
    forecast = np.array(forecast)
    target = np.array(target)
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast, target):
    """MAPE"""
    forecast = np.array(forecast)
    target = np.array(target)
    denom = np.abs(target)
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


class M4Summary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')

    def evaluate(self):
        grouped_owa = OrderedDict()

        # 读取 Naive2 预测
        naive2_forecasts_raw = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = [v[~np.isnan(v)] for v in naive2_forecasts_raw]

        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}

        for group_name in M4Meta.seasonal_patterns:
            file_name = os.path.join(self.file_path, f"{group_name}_forecast.csv")
            if os.path.exists(file_name):
                model_forecast_raw = pd.read_csv(file_name).values
                model_forecast = [v[~np.isnan(v)] for v in model_forecast_raw]
            else:
                print(f"Warning: forecast file not found: {file_name}")
                continue

            naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)

            # 获取每组频率
            frequency = self.training_set.frequencies[[i for i, g in enumerate(self.test_set.groups) if g == group_name][0]]

            # 计算 MASE
            model_mases[group_name] = np.mean([
                mase(forecast=model_forecast[i],
                     insample=insample[i],
                     outsample=target[i],
                     frequency=frequency)
                for i in range(len(model_forecast))
            ])
            naive2_mases[group_name] = np.mean([
                mase(forecast=naive2_forecast[i],
                     insample=insample[i],
                     outsample=target[i],
                     frequency=frequency)
                for i in range(len(model_forecast))
            ])

            # 计算 sMAPE
            naive2_smapes[group_name] = np.mean([
                smape_2(naive2_forecast[i], target[i])
                for i in range(len(naive2_forecast))
            ])
            grouped_smapes[group_name] = np.mean([
                smape_2(model_forecast[i], target[i])
                for i in range(len(model_forecast))
            ])

            # 计算 MAPE
            grouped_mapes[group_name] = np.mean([
                mape(model_forecast[i], target[i])
                for i in range(len(model_forecast))
            ])

        # 计算 OWA
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_smapes_summary = self.summarize_groups(grouped_smapes)

        for k in grouped_model_mases.keys():
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes_summary[k] / grouped_naive2_smapes[k]) / 2

        def round_all(d):
            return {k: round(v, 3) for k, v in d.items()}

        return round_all(grouped_smapes_summary), round_all(grouped_owa), round_all(grouped_mapes), round_all(grouped_model_mases)

    def summarize_groups(self, scores):
        """重新按照 M4 分组规则汇总"""
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len([g for g in self.test_set.groups if g == group_name])

        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        scores_summary['Others'] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary['Average'] = average

        return scores_summary
