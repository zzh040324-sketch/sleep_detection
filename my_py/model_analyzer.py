import numpy as np
import pandas as pd
import polars as pl

def analyze_sleep_model_performance(rf_submission, ground_truth, val_data, val_ids, rf_classifier=None, min_sleep_duration=30):
    """
    全新的模型判断函数，不使用my_funs.py中现有的任何函数
    
    参数:
    rf_submission: 模型预测结果（pandas DataFrame），包含series_id, event, step列
    ground_truth: 真实事件数据（pandas DataFrame），包含series_id, event, step列
    val_data: 验证时间序列数据（Polars DataFrame）
    val_ids: 验证集的series_id列表
    rf_classifier: 训练好的随机森林分类器模型，用于计算内存占用
    min_sleep_duration: 最小睡眠周期长度（分钟），默认30分钟
    
    返回:
    dict: 包含模型性能统计信息
    """
    # 计算模型内存占用
    if rf_classifier is not None:
        import pickle
        import io
        
        # 将模型序列化为字节流以计算内存占用
        buffer = io.BytesIO()
        pickle.dump(rf_classifier, buffer)
        model_size = len(buffer.getvalue()) / (1024 * 1024)  # 转换为MB
        print(f"\n模型内存占用: {model_size:.2f} MB")
    # 1. 计算验证数据总天数
    def count_days(data):
        """统计数据中的天数，按series_id分组统计"""
        days_by_series = {}
        series_ids = data['series_id'].unique()
        
        for series_id in series_ids:
            series_data = data.filter(pl.col('series_id') == series_id)
            unique_dates = series_data.select(
                pl.col('timestamp').dt.date()
            ).unique()
            days_by_series[series_id] = len(unique_dates)
        
        return days_by_series
    
    days_by_series = count_days(val_data)
    total_val_days = sum(days_by_series.values())
    
    # 2. 计算预测事件总数和预测天数
    total_predicted_onset = len(rf_submission[rf_submission['event'] == 'onset'])
    total_predicted_wakeup = len(rf_submission[rf_submission['event'] == 'wakeup'])
    
    # 计算成功预测的天数：一个onset在后面24小时内有一个wakeup事件，且中间没有其他onset事件
    total_predicted_days = 0
    # 按series_id分组处理
    grouped = rf_submission.groupby('series_id')
    
    for series_id, group in grouped:
        # 按step排序
        sorted_events = group.sort_values('step')
        # 获取所有事件的step和event类型
        steps = sorted_events['step'].tolist()
        events = sorted_events['event'].tolist()
        
        # 遍历onset事件
        i = 0
        while i < len(events):
            if events[i] == 'onset':
                # 记录当前onset的step
                onset_step = steps[i]
                # 计算24小时后的step阈值（1步=5秒，24小时=24*60*60秒=24*60*60/5步=24*60*12步）
                threshold_step = onset_step + 24 * 60 * 12
                
                # 检查后续事件
                has_valid_wakeup = False
                j = i + 1
                
                while j < len(events) and steps[j] <= threshold_step:
                    if events[j] == 'onset':
                        # 中间有其他onset事件，当前onset无效
                        break
                    elif events[j] == 'wakeup':
                        # 找到有效的wakeup事件
                        has_valid_wakeup = True
                        break
                    j += 1
                
                if has_valid_wakeup:
                    total_predicted_days += 1
                    # 跳过已匹配的wakeup事件
                    i = j + 1
                else:
                    # 没有找到有效的wakeup事件，继续下一个onset
                    i += 1
            else:
                # 跳过wakeup事件
                i += 1
    
    prediction_rate = total_predicted_days / total_val_days if total_val_days > 0 else 0
    onset_event_rate = total_predicted_onset / total_val_days if total_val_days > 0 else 0
    wakeup_event_rate = total_predicted_wakeup / total_val_days if total_val_days > 0 else 0
    # 3. 定义不同的睡眠匹配时间（分钟）
    match_times = [5, 10, 15, 30, 60, 120]
      
    results = {
        'model_info': {
            'model_size_mb': model_size if rf_classifier is not None else 0,
            'predicted_onset_count': total_predicted_onset,
            'predicted_wakeup_count': total_predicted_wakeup
        },
        'prediction_stats': {
            'total_val_days': total_val_days,
            'total_predicted_days': total_predicted_days,
            'prediction_rate': prediction_rate,
            'onset_event_rate': onset_event_rate,
            'wakeup_event_rate': wakeup_event_rate
        },
        'match_time_stats': {}
    }
    
    # 遍历每个匹配时间
    for match_time in match_times:
        # 计算匹配时间对应的步数差异阈值（1步=5秒）
        step_threshold = (match_time * 60) // 5
        
        # 准备数据
        pred_sorted = rf_submission.sort_values(['series_id', 'event', 'step'])
        gt_sorted = ground_truth.sort_values(['series_id', 'event', 'step'])
        
        # 存储统计结果
        onset_matches = 0
        wakeup_matches = 0
        onset_step_diffs = []
        wakeup_step_diffs = []
        sleep_duration_errors = []
        
        # 遍历每个series_id
        for series_id in val_ids:
            # 获取当前series_id的真实事件
            gt_onsets = gt_sorted[(gt_sorted['series_id'] == series_id) & (gt_sorted['event'] == 'onset')]['step'].tolist()
            gt_wakeups = gt_sorted[(gt_sorted['series_id'] == series_id) & (gt_sorted['event'] == 'wakeup')]['step'].tolist()
            
            # 获取当前series_id的预测事件
            pred_onsets = pred_sorted[(pred_sorted['series_id'] == series_id) & (pred_sorted['event'] == 'onset')]['step'].tolist()
            pred_wakeups = pred_sorted[(pred_sorted['series_id'] == series_id) & (pred_sorted['event'] == 'wakeup')]['step'].tolist()
            
            # 匹配onset事件（优化算法）
            o_val_idx, o_pred_idx = 0, 0
            while o_val_idx < len(gt_onsets) and o_pred_idx < len(pred_onsets):
                gt_step = gt_onsets[o_val_idx]
                pred_step = pred_onsets[o_pred_idx]
                step_diff = abs(gt_step - pred_step)
                
                if step_diff <= step_threshold:
                    onset_matches += 1
                    onset_step_diffs.append(step_diff)
                    o_val_idx += 1
                    o_pred_idx += 1
                elif gt_step < pred_step:
                    o_val_idx += 1
                else:
                    o_pred_idx += 1
            
            # 匹配wakeup事件（优化算法）
            w_val_idx, w_pred_idx = 0, 0
            while w_val_idx < len(gt_wakeups) and w_pred_idx < len(pred_wakeups):
                gt_step = gt_wakeups[w_val_idx]
                pred_step = pred_wakeups[w_pred_idx]
                step_diff = abs(gt_step - pred_step)
                
                if step_diff <= step_threshold:
                    wakeup_matches += 1
                    wakeup_step_diffs.append(step_diff)
                    w_val_idx += 1
                    w_pred_idx += 1
                elif gt_step < pred_step:
                    w_val_idx += 1
                else:
                    w_pred_idx += 1
            
            # 计算睡眠周期时长误差
            # 确保onset和wakeup事件成对
            min_cycles = min(len(gt_onsets), len(gt_wakeups), len(pred_onsets), len(pred_wakeups))
            for i in range(min_cycles):
                # 真实睡眠周期时长
                gt_duration = gt_wakeups[i] - gt_onsets[i]
                # 预测睡眠周期时长
                pred_duration = pred_wakeups[i] - pred_onsets[i]
                # 计算绝对误差
                duration_error = abs(gt_duration - pred_duration)
                sleep_duration_errors.append(duration_error)
        
        # 计算统计指标
        total_gt_onset = len(gt_sorted[gt_sorted['event'] == 'onset'])
        total_gt_wakeup = len(gt_sorted[gt_sorted['event'] == 'wakeup'])
        
        onset_match_rate = onset_matches / total_gt_onset if total_gt_onset > 0 else 0
        wakeup_match_rate = wakeup_matches / total_gt_wakeup if total_gt_wakeup > 0 else 0
        
        # 将步数差异转换为时间差异（秒）：步数 * 5
        onset_time_diffs = [diff * 5 for diff in onset_step_diffs]
        wakeup_time_diffs = [diff * 5 for diff in wakeup_step_diffs]
        all_time_diffs = onset_time_diffs + wakeup_time_diffs
        
        onset_avg_time_diff = np.mean(onset_time_diffs) if onset_time_diffs else 0
        onset_std_time_diff = np.std(onset_time_diffs) if onset_time_diffs else 0
        wakeup_avg_time_diff = np.mean(wakeup_time_diffs) if wakeup_time_diffs else 0
        wakeup_std_time_diff = np.std(wakeup_time_diffs) if wakeup_time_diffs else 0
        
        total_matches = onset_matches + wakeup_matches
        total_gt_events = total_gt_onset + total_gt_wakeup
        total_match_rate = total_matches / total_gt_events if total_gt_events > 0 else 0
        
        total_avg_time_diff = np.mean(all_time_diffs) if all_time_diffs else 0
        total_std_time_diff = np.std(all_time_diffs) if all_time_diffs else 0
        
        # 将睡眠周期误差转换为时间差异（秒）：步数 * 5
        sleep_duration_errors_sec = [error * 5 for error in sleep_duration_errors]
        avg_duration_error_sec = np.mean(sleep_duration_errors_sec) if sleep_duration_errors_sec else 0
        std_duration_error_sec = np.std(sleep_duration_errors_sec) if sleep_duration_errors_sec else 0
        
        # 存储结果
        results['match_time_stats'][match_time] = {
            'onset': {
                'matches': onset_matches,
                'match_rate': onset_match_rate,
                'avg_time_diff': onset_avg_time_diff,
                'std_time_diff': onset_std_time_diff
            },
            'wakeup': {
                'matches': wakeup_matches,
                'match_rate': wakeup_match_rate,
                'avg_time_diff': wakeup_avg_time_diff,
                'std_time_diff': wakeup_std_time_diff
            },
            'total': {
                'matches': total_matches,
                'match_rate': total_match_rate,
                'avg_time_diff': total_avg_time_diff,
                'std_time_diff': total_std_time_diff
            },
            'sleep_duration': {
                'avg_error': avg_duration_error_sec,
                'std_error': std_duration_error_sec
            }
        }
    
    # 5. 计算总体统计指标（不按匹配时间）
    total_pred_onset = len(rf_submission[rf_submission['event'] == 'onset'])
    total_pred_wakeup = len(rf_submission[rf_submission['event'] == 'wakeup'])
    total_gt_onset = len(ground_truth[ground_truth['event'] == 'onset'])
    total_gt_wakeup = len(ground_truth[ground_truth['event'] == 'wakeup'])
    
    # 6. 打印结果（优化输出内容）
    print(f"验证数据总天数: {total_val_days}")
    print(f"预测天数: {total_predicted_days}")
    print(f"预测率: {prediction_rate:.4f}")
    print(f"onset事件率: {onset_event_rate:.4f}")
    print(f"wakeup事件率: {wakeup_event_rate:.4f}")
    print(f"预测onset事件数: {total_pred_onset}")
    print(f"预测wakeup事件数: {total_pred_wakeup}")
    print(f"真实onset事件数: {total_gt_onset}")
    print(f"真实wakeup事件数: {total_gt_wakeup}")
    
    # 整理不同睡眠匹配时间的统计数据
    match_times_sorted = sorted(results['match_time_stats'].keys())
    
    # 提取各种指标的数据
    onset_matches = []
    onset_match_rates = []
    onset_avg_diffs = []
    onset_std_diffs = []
    
    wakeup_matches = []
    wakeup_match_rates = []
    wakeup_avg_diffs = []
    wakeup_std_diffs = []
    
    total_matches = []
    total_match_rates = []
    total_avg_diffs = []
    total_std_diffs = []
    
    sleep_duration_avg_errors = []
    sleep_duration_std_errors = []
    
    for match_time in match_times_sorted:
        stats = results['match_time_stats'][match_time]
        
        # Onset数据
        onset_matches.append(stats['onset']['matches'])
        onset_match_rates.append(stats['onset']['match_rate'])
        onset_avg_diffs.append(stats['onset']['avg_time_diff'])
        onset_std_diffs.append(stats['onset']['std_time_diff'])
        
        # Wakeup数据
        wakeup_matches.append(stats['wakeup']['matches'])
        wakeup_match_rates.append(stats['wakeup']['match_rate'])
        wakeup_avg_diffs.append(stats['wakeup']['avg_time_diff'])
        wakeup_std_diffs.append(stats['wakeup']['std_time_diff'])
        
        # 总数据
        total_matches.append(stats['total']['matches'])
        total_match_rates.append(stats['total']['match_rate'])
        total_avg_diffs.append(stats['total']['avg_time_diff'])
        total_std_diffs.append(stats['total']['std_time_diff'])
        
        # 睡眠周期误差
        sleep_duration_avg_errors.append(stats['sleep_duration']['avg_error'])
        sleep_duration_std_errors.append(stats['sleep_duration']['std_error'])
    
    # 格式化输出
    match_times_str = '/'.join(map(str, match_times_sorted))
    
    # Onset事件统计
    print(f"Onset匹配数({match_times_str})分钟：{'/'.join(map(str, onset_matches))}")
    print(f"Onset匹配率({match_times_str})分钟：{'/'.join([f'{rate:.4f}' for rate in onset_match_rates])}")
    print(f"Onset平均差(秒)({match_times_str})分钟：{'/'.join([f'{diff:.2f}' for diff in onset_avg_diffs])}")
    print(f"Onset标准差(秒)({match_times_str})分钟：{'/'.join([f'{diff:.2f}' for diff in onset_std_diffs])}")
    print()
    
    # Wakeup事件统计
    print(f"Wakeup匹配数({match_times_str})分钟：{'/'.join(map(str, wakeup_matches))}")
    print(f"Wakeup匹配率({match_times_str})分钟：{'/'.join([f'{rate:.4f}' for rate in wakeup_match_rates])}")
    print(f"Wakeup平均差(秒)({match_times_str})分钟：{'/'.join([f'{diff:.2f}' for diff in wakeup_avg_diffs])}")
    print(f"Wakeup标准差(秒)({match_times_str})分钟：{'/'.join([f'{diff:.2f}' for diff in wakeup_std_diffs])}")
    print()
    
    # 总事件统计
    print(f"总和匹配数({match_times_str})分钟：{'/'.join(map(str, total_matches))}")
    print(f"总和匹配率({match_times_str})分钟：{'/'.join([f'{rate:.4f}' for rate in total_match_rates])}")
    print(f"总和平均差(秒)({match_times_str})分钟：{'/'.join([f'{diff:.2f}' for diff in total_avg_diffs])}")
    print(f"总和标准差(秒)({match_times_str})分钟：{'/'.join([f'{diff:.2f}' for diff in total_std_diffs])}")
    print()

    return results
