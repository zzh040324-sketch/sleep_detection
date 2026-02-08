import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

# 定义时间格式常量
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

def load_and_preprocess_data(train_series_path='/home/zhuangzhuohan/sleep_data/train_series/train_series.parquet', 
                           train_events_path='/home/zhuangzhuohan/sleep_data/train_events/train_events.csv',
                           test_series_path='/home/zhuangzhuohan/sleep_data/test_series/test_series.parquet'):
    """
    加载和预处理数据
    
    参数:
    train_series_path: 训练时间序列数据路径
    train_events_path: 训练事件数据路径
    test_series_path: 测试时间序列数据路径
    
    返回:
    train_series: 预处理后的训练时间序列数据
    train_events: 预处理后的训练事件数据
    test_series: 预处理后的测试时间序列数据
    """
    # 创建时间转换表达式
    timestamp_expr = pl.col('timestamp').str.to_datetime(format=TIME_FORMAT, time_zone='UTC')
    
    dt_transforms = [
        timestamp_expr.alias('timestamp'),  # 转换为UTC时区的datetime
        (timestamp_expr.dt.year() - 2000).cast(pl.UInt8).alias('year'),  # 提取年份（减去2000以节省空间）
        timestamp_expr.dt.month().cast(pl.UInt8).alias('month'),  # 提取月份
        timestamp_expr.dt.day().cast(pl.UInt8).alias('day'),  # 提取日期
        timestamp_expr.dt.hour().cast(pl.UInt8).alias('hour')  # 提取小时
    ]
    
    data_transforms = [
        pl.col('anglez').cast(pl.Int16), # 将anglez转换为16位整数以节省空间
        (pl.col('enmo')*1000).cast(pl.UInt16), # 将enmo乘以1000并转换为16位无符号整数
    ]
    
    # 读取训练数据（使用lazy loading提高效率）
    train_series = pl.scan_parquet(train_series_path).with_columns(
        dt_transforms + data_transforms
        )
    
    # 读取训练事件数据
    train_events = pl.read_csv(train_events_path).with_columns(
        dt_transforms
        ).drop_nulls()  # 删除空值
    
    # 读取测试数据（使用lazy loading提高效率）
    test_series = pl.scan_parquet(test_series_path).with_columns(
        dt_transforms + data_transforms
        )
    
    # 移除事件数量不匹配的夜晚（确保每个onset对应一个wakeup）
    mismatches = train_events.drop_nulls().group_by(['series_id', 'night']).agg([
        ((pl.col('event') == 'onset').sum() == (pl.col('event') == 'wakeup').sum()).alias('balanced')
        ]).sort(by=['series_id', 'night']).filter(~pl.col('balanced'))
    
    # 移除不匹配的数据
    for mm in mismatches.to_numpy(): 
        train_events = train_events.filter(~((pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1])))
    
    # 获取唯一的series_id列表
    series_ids = train_events['series_id'].unique(maintain_order=True).to_list()
    
    # 更新train_series，只保留有事件数据的series_id
    train_series = train_series.filter(pl.col('series_id').is_in(series_ids))
    
    return train_series, train_events, test_series, series_ids

def create_features():
    """
    创建特征和特征列名列表
    
    返回:
    features: 特征变换列表
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    """
    # 初始化特征列表，先加入小时特征
    features, feature_cols = [pl.col('hour')], ['hour']  
    
    # 为不同时间窗口创建特征
    for mins in [5, 30, 60*2, 60*8] :  # 5分钟、30分钟、2小时、8小时
        
        for var in ['enmo', 'anglez'] :  # 对enmo和anglez两个变量创建特征
            
            # 创建基础统计特征
            features += [
                # 计算滚动平均值（绝对值）
                pl.col(var).rolling_mean(12 * mins, center=True, min_samples=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_mean'),
                # 计算滚动最大值（绝对值）
                pl.col(var).rolling_max(12 * mins, center=True, min_samples=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_max'),
                # 计算滚动标准差（绝对值）
                pl.col(var).rolling_std(12 * mins, center=True, min_samples=1).abs().cast(pl.UInt16).alias(f'{var}_{mins}m_std')
            ]
            
            # 更新特征列名列表
            feature_cols += [ 
                f'{var}_{mins}m_mean', f'{var}_{mins}m_max', f'{var}_{mins}m_std'
            ]
            
            # 创建一阶差分特征（衡量变化率）
            features += [
                # 计算一阶差分的滚动平均值（绝对值）
                (pl.col(var).diff().abs().rolling_mean(12 * mins, center=True, min_samples=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_mean'),
                # 计算一阶差分的滚动最大值（绝对值）
                (pl.col(var).diff().abs().rolling_max(12 * mins, center=True, min_samples=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_max'),
                # 计算一阶差分的滚动标准差（绝对值）
                (pl.col(var).diff().abs().rolling_std(12 * mins, center=True, min_samples=1)*10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_std')
            ]
            
            # 更新特征列名列表
            feature_cols += [ 
                f'{var}_1v_{mins}m_mean', f'{var}_1v_{mins}m_max', f'{var}_1v_{mins}m_std'
            ]
    
    id_cols = ['series_id', 'step', 'timestamp']  # 标识列
    
    return features, feature_cols, id_cols

def apply_features(data, features, id_cols, feature_cols):
    """
    应用特征变换到数据
    
    参数:
    data: 要应用特征变换的数据
    features: 特征变换列表
    id_cols: 标识列列表
    feature_cols: 特征列名列表
    
    返回:
    应用了特征变换的数据
    """
    return data.with_columns(
        features
    ).select(id_cols + feature_cols)  # 只保留需要的列

def make_train_dataset(train_data, train_events, feature_cols, id_cols, drop_nulls=False):
    """
    创建训练数据集的改进版本，修复了一些问题
    
    参数:
    train_data: 训练时间序列数据
    train_events: 训练事件数据
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    drop_nulls: 是否删除没有事件记录的日期数据
    
    返回:
    X: 特征矩阵
    y: 标签向量
    """
    
    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()  # 初始化特征和标签数据框
    
    for idx in tqdm(series_ids):  # 遍历每个series_id
        
        # 标准化样本特征
        sample = train_data.filter(pl.col('series_id') == idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32) for col in feature_cols if col != 'hour']
        )
        
        events = train_events.filter(pl.col('series_id') == idx)  # 获取当前series_id的事件数据
        
        if drop_nulls:
            # 移除没有事件记录的日期的数据点
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )
        
        # 添加特征数据
        X = X.vstack(sample[id_cols + feature_cols])  
        
        # 修复：使用is_not_null()检查空值
        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step').is_not_null()))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step').is_not_null()))['step'].to_list()
        
        # 修复：使用pl.sum_horizontal替代sum，并添加错误处理
        if onsets and wakeups and len(onsets) == len(wakeups):
            conditions = [(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in zip(onsets, wakeups)]
            y = y.vstack(sample.with_columns(
                pl.sum_horizontal(conditions).cast(pl.Boolean).alias('asleep')
            ).select('asleep'))
        else:
            # 如果没有有效的睡眠区间，创建全为False的列
            y = y.vstack(sample.with_columns(
                pl.lit(False).alias('asleep')
            ).select('asleep'))
    
    y = y.to_numpy().ravel()  # 将标签转换为一维数组
    
    return X, y

def get_events_original(series, classifier, feature_cols, id_cols, min_sleep_duration=12 * 30):
    """
    将分类器的预测结果转换为睡眠事件（onset和wakeup），并生成提交格式的数据框
    原始版本（已备份）
    
    参数:
    series: 时间序列数据
    classifier: 训练好的分类器模型
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    min_sleep_duration: 最小睡眠周期长度（步数），默认30分钟（12*30步）
    
    返回:
    events: 包含预测事件的DataFrame，格式符合提交要求
    """
    
    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    events = pl.DataFrame(schema={'series_id':str, 'step':int, 'event':str, 'score':float})  # 初始化事件数据框

    for idx in tqdm(series_ids) :  # 遍历每个series_id，显示进度条

        # 准备数据并标准化特征
        scale_cols = [col for col in feature_cols if (col != 'hour') & (series[col].std() !=0)]
        X = series.filter(pl.col('series_id') == idx).select(id_cols + feature_cols).with_columns(
            [(pl.col(col) / series[col].std()).cast(pl.Float32) for col in scale_cols]
        )

        # 使用分类器进行预测，获取类别和概率
        preds, probs = classifier.predict(X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]

        # 将预测结果添加到数据框
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'), 
            pl.lit(probs).alias('probability')
                        )
        
        # 检测睡眠开始和结束事件（通过预测值的变化）
        pred_onsets = X.filter(X['prediction'].diff() > 0)['step'].to_list()  # 从0变为1的点为onset
        pred_wakeups = X.filter(X['prediction'].diff() < 0)['step'].to_list()  # 从1变为0的点为wakeup
        
        if len(pred_onsets) > 0 : 
            
            # 确保所有预测的睡眠周期都有开始和结束
            if min(pred_wakeups) < min(pred_onsets) : 
                pred_wakeups = pred_wakeups[1:]  # 移除第一个wakeup（如果它在第一个onset之前）

            if max(pred_onsets) > max(pred_wakeups) :
                pred_onsets = pred_onsets[:-1]  # 移除最后一个onset（如果它在最后一个wakeup之后）

            # 只保留持续时间超过指定长度的睡眠周期
            sleep_periods = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if wakeup - onset >= min_sleep_duration]

            for onset, wakeup in sleep_periods :
                # 计算睡眠周期内的平均概率作为分数
                score = X.filter((pl.col('step') >= onset) & (pl.col('step') <= wakeup))['probability'].mean()

                # 将睡眠事件添加到数据框
                events = events.vstack(pl.DataFrame().with_columns(
                    pl.Series([idx, idx]).alias('series_id'), 
                    pl.Series([onset, wakeup]).alias('step'),
                    pl.Series(['onset', 'wakeup']).alias('event'),
                    pl.Series([score, score]).alias('score')
                ))

    # 添加行ID列
    events = events.to_pandas().reset_index().rename(columns={'index':'row_id'})

    return events

def get_events(series, classifier, feature_cols, id_cols, min_sleep_duration=12 * 30):
    """
    将分类器的预测结果转换为睡眠事件（onset和wakeup），并生成提交格式的数据框
    实时检测版本：移除事件配对逻辑，支持实时部署到STM32
    
    参数:
    series: 时间序列数据
    classifier: 训练好的分类器模型
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    min_sleep_duration: 最小睡眠周期长度（步数），默认30分钟（12*30步）
    
    返回:
    events: 包含预测事件的DataFrame，格式符合提交要求
    """
    
    # 优化1: 预先计算特征标准差，避免重复计算
    feature_stds = {}
    for col in feature_cols:
        if col != 'hour':
            std_val = series[col].std()
            if std_val != 0:
                feature_stds[col] = std_val
    
    # 优化2: 预生成 scale_cols，避免每次循环都重新计算
    scale_cols = list(feature_stds.keys())
    
    # 获取所有 series_id
    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    
    # 优化3: 使用列表收集事件数据，最后一次性创建 DataFrame，减少 vstack 操作
    event_data = []
    
    for idx in tqdm(series_ids):  # 遍历每个series_id，显示进度条
        # 优化4: 准备数据并标准化特征（使用预先计算的标准差）
        X = series.filter(pl.col('series_id') == idx).select(id_cols + feature_cols)
        
        if scale_cols:
            X = X.with_columns(
                [(pl.col(col) / feature_stds[col]).cast(pl.Float32) for col in scale_cols]
            )
        
        # 使用分类器进行预测，获取类别和概率
        preds, probs = classifier.predict(X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]
        
        # 将预测结果添加到数据框
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'), 
            pl.lit(probs).alias('probability')
        )
        
        # 优化5: 使用 NumPy 进行差异计算，提高事件检测效率
        preds_array = X['prediction'].to_numpy()
        steps_array = X['step'].to_numpy()
        probs_array = X['probability'].to_numpy()
        timestamps_array = X['timestamp'].to_numpy()
        
        # 计算差异
        diffs = np.diff(preds_array)
        
        # 检测睡眠开始和结束事件
        onset_indices = np.where(diffs > 0)[0]
        wakeup_indices = np.where(diffs < 0)[0]
        
        # 处理 onset 事件（实时检测）
        for i in onset_indices:
            # 获取事件的 step 和 timestamp
            step = steps_array[i]
            timestamp = timestamps_array[i]
            
            # 实时窗口验证：检查后续一段时间内是否持续为睡眠状态
            # 计算验证窗口的结束索引
            window_end = min(i + min_sleep_duration, len(preds_array) - 1)
            
            # 检查窗口内的预测值是否都为1（睡眠状态）
            if np.all(preds_array[i+1:window_end+1] == 1):
                # 计算窗口内的平均概率作为分数
                score = np.mean(probs_array[i:window_end+1])
                
                # 添加到列表
                event_data.append({
                    'series_id': idx, 
                    'step': step, 
                    'event': 'onset', 
                    'score': score,
                    'timestamp': timestamp
                })
        
        # 处理 wakeup 事件（实时检测）
        for i in wakeup_indices:
            # 获取事件的 step 和 timestamp
            step = steps_array[i]
            timestamp = timestamps_array[i]
            
            # 实时窗口验证：检查后续一段时间内是否持续为清醒状态
            # 计算验证窗口的结束索引
            window_end = min(i + min_sleep_duration, len(preds_array) - 1)
            
            # 检查窗口内的预测值是否都为0（清醒状态）
            if np.all(preds_array[i+1:window_end+1] == 0):
                # 计算窗口内的平均概率作为分数（使用1 - 概率表示清醒程度）
                score = np.mean(1 - probs_array[i:window_end+1])
                
                # 添加到列表
                event_data.append({
                    'series_id': idx, 
                    'step': step, 
                    'event': 'wakeup', 
                    'score': score,
                    'timestamp': timestamp
                })
    
    # 优化6: 最后一次性创建 DataFrame，而不是多次 vstack
    if event_data:
        events = pl.DataFrame(event_data)
    else:
        events = pl.DataFrame(schema={'series_id': str, 'step': int, 'event': str, 'score': float, 'timestamp': str})
    
    # 添加行ID列
    events = events.to_pandas().reset_index().rename(columns={'index': 'row_id'})
    
    return events

def analyze_predictions(rf_submission, val_data, val_ids, train_events, output_num_file='/home/zhuangzhuohan/sleep_data/rf/rf_num.csv', output_data_file='/home/zhuangzhuohan/sleep_data/rf/rf_data.csv', verbose=False):
    """
    统计预测结果并保存到CSV文件
    
    参数:
    rf_submission: 预测结果数据框
    val_data: 验证时间序列数据
    val_ids: 验证集的series_id列表
    train_events: 原始训练事件数据（用于获取night和timestamp信息）
    output_num_file: 系列统计结果保存的文件名，默认'rf_num.csv'
    output_data_file: 事件详细信息保存的文件名，默认'rf_data.csv'
    verbose: 是否显示详细输出，默认True
    
    返回:
    None，结果保存在CSV文件中
    """
    from tqdm import tqdm as tqdm_base
    from tqdm.notebook import tqdm as tqdm_notebook
    
    # 根据verbose参数选择是否显示进度条
    if verbose:
        tqdm = tqdm_notebook
    else:
        tqdm = lambda iterable, **kwargs: iterable
    
    # 1. 统计每个series_id的详细信息
    results = []
    
    # 预处理：将rf_submission按series_id分组，提取onset和wakeup
    series_events = {}
    for series_id in tqdm(val_ids, desc="处理每个series_id的预测事件"):
        # 获取当前series_id的事件
        events = rf_submission[rf_submission['series_id'] == series_id]
        
        # 提取onset和wakeup的step值并排序
        onset_steps = sorted(events[events['event'] == 'onset']['step'].tolist())
        wakeup_steps = sorted(events[events['event'] == 'wakeup']['step'].tolist())
        
        # 确保onset和wakeup成对
        sleep_periods = []
        min_len = min(len(onset_steps), len(wakeup_steps))
        for i in range(min_len):
            if onset_steps[i] < wakeup_steps[i]:
                sleep_periods.append((onset_steps[i], wakeup_steps[i]))
        
        series_events[series_id] = {
            'onset_count': len(onset_steps),
            'wakeup_count': len(wakeup_steps),
            'sleep_periods': sleep_periods
        }
    
    # 处理每个series_id
    for series_id in tqdm(val_ids, desc="统计每个series_id"):
        # 获取当前series_id的val_data数据点
        series_data = val_data.filter(pl.col('series_id') == series_id)
        total_points = len(series_data)
        
        if total_points > 0:
            # 获取预计算的事件信息
            event_info = series_events.get(series_id, {
                'onset_count': 0,
                'wakeup_count': 0,
                'sleep_periods': []
            })
            
            onset_count = event_info['onset_count']
            wakeup_count = event_info['wakeup_count']
            sleep_periods = event_info['sleep_periods']
            
            # 优化：批量处理数据点
            if sleep_periods:
                # 转换为pandas以提高处理速度
                series_data_pd = series_data.to_pandas()
                
                # 定义判断函数
                def is_asleep(step):
                    for onset, wakeup in sleep_periods:
                        if onset <= step <= wakeup:
                            return True
                    return False
                
                # 批量应用
                asleep_mask = series_data_pd['step'].apply(is_asleep)
                asleep_count = asleep_mask.sum()
                awake_count = total_points - asleep_count
            else:
                # 无睡眠周期，全部为清醒
                asleep_count = 0
                awake_count = total_points
            
            # 计算比例
            awake_ratio = (awake_count / total_points) * 100 if total_points > 0 else 0
            asleep_ratio = (asleep_count / total_points) * 100 if total_points > 0 else 0
            
            # 添加到结果列表
            results.append({
                'series_id': series_id,
                'total_points': total_points,
                'awake_count': awake_count,
                'asleep_count': asleep_count,
                'awake_ratio': awake_ratio,
                'asleep_ratio': asleep_ratio,
                'onset_count': onset_count,
                'wakeup_count': wakeup_count,
                'sleep_periods_count': len(sleep_periods)
            })
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_num_file, index=False)
    
    if verbose:
        print(f"\n系列统计完成，结果已保存到 {output_num_file}")
    
    # 2. 统计每个onset和wakeup事件的详细信息    
    # 从train_events中获取每个step对应的night和timestamp    
    # 构建step到night和timestamp的映射字典
    step_info_map = {}
    train_events_pd = train_events.to_pandas()
    for _, row in tqdm(train_events_pd.iterrows(), desc="构建step信息映射", total=len(train_events_pd)):
        key = (row['series_id'], row['step'])
        step_info_map[key] = {'night': row['night'], 'timestamp': row['timestamp']}
    
    # 为rf_submission添加night和timestamp列
    def get_step_info(row):
        key = (row['series_id'], row['step'])
        info = step_info_map.get(key, {'night': None, 'timestamp': None})
        return info['night'], info['timestamp']
    
    # 批量应用
    rf_events_with_info = rf_submission.copy()
    rf_events_with_info[['night', 'timestamp']] = rf_events_with_info.apply(get_step_info, axis=1, result_type='expand')
    
    # 选择需要的列
    rf_events_final = rf_events_with_info[['series_id', 'night', 'event', 'step', 'timestamp']]
    
    # 保存到CSV
    rf_events_final.to_csv(output_data_file, index=False)
    
    if verbose:
        print(f"事件详细信息统计完成，结果已保存到 {output_data_file}")
    
    # 3. 显示总体统计    
    if not results_df.empty:
        total_awake = results_df['awake_count'].sum()
        total_asleep = results_df['asleep_count'].sum()
        total_points = results_df['total_points'].sum()
        
        total_awake_ratio = (total_awake / total_points) * 100 if total_points > 0 else 0
        total_asleep_ratio = (total_asleep / total_points) * 100 if total_points > 0 else 0
        
        total_onset = results_df['onset_count'].sum()
        total_wakeup = results_df['wakeup_count'].sum()
        total_sleep_periods = results_df['sleep_periods_count'].sum()
        
        if verbose:
            print(f"验证集数据点总数: {total_points}")
            print(f"预测清醒: {total_awake} ({total_awake_ratio:.2f}%)")
            print(f"预测睡眠: {total_asleep} ({total_asleep_ratio:.2f}%)")
            print(f"预测onset事件总数: {total_onset}")
            print(f"预测wakeup事件总数: {total_wakeup}")
            print(f"预测睡眠周期总数: {total_sleep_periods}")
        
        # 事件详细信息统计
        event_count = len(rf_events_final)
        onset_event_count = len(rf_events_final[rf_events_final['event'] == 'onset'])
        wakeup_event_count = len(rf_events_final[rf_events_final['event'] == 'wakeup'])
        
        if verbose:
            print(f"\n预测事件详细信息统计：")
            print(f"预测事件总数: {event_count}")
            print(f"预测onset事件数: {onset_event_count}")
            print(f"预测wakeup事件数: {wakeup_event_count}")
    else:
        if verbose:
            print("没有数据可供统计")

def analyze_validation_data(val_data, val_ids, val_solution, train_events, output_num_file='/home/zhuangzhuohan/sleep_data/val/val_num.csv', output_data_file='/home/zhuangzhuohan/sleep_data/val/val_data.csv'):
    """
    统计验证数据并保存到CSV文件，对应tast_new.ipynb中的统计验证数据代码
    
    参数:
    val_data: 验证时间序列数据
    val_ids: 验证集的series_id列表
    val_solution: 验证集的事件数据（pandas DataFrame）
    train_events: 原始训练事件数据（用于获取完整的事件信息）
    output_num_file: 系列统计结果保存的文件名，默认'val_num.csv'
    output_data_file: 事件详细信息保存的文件名，默认'val_data.csv'
    
    返回:
    None，结果保存在CSV文件中
    """
    # 1. 统计每个series_id的详细信息
    results = []
    
    # 预处理：将val_solution按series_id分组，提取onset和wakeup
    series_events = {}
    for series_id in val_ids:
        # 获取当前series_id的事件
        events = val_solution[val_solution['series_id'] == series_id]
        
        # 提取onset和wakeup的step值并排序
        onset_steps = sorted(events[events['event'] == 'onset']['step'].tolist())
        wakeup_steps = sorted(events[events['event'] == 'wakeup']['step'].tolist())
        
        # 确保onset和wakeup成对
        sleep_periods = []
        min_len = min(len(onset_steps), len(wakeup_steps))
        for i in range(min_len):
            if onset_steps[i] < wakeup_steps[i]:
                sleep_periods.append((onset_steps[i], wakeup_steps[i]))
        
        series_events[series_id] = {
            'onset_count': len(onset_steps),
            'wakeup_count': len(wakeup_steps),
            'sleep_periods': sleep_periods
        }
    
    # 处理每个series_id
    for series_id in val_ids:
        # 获取当前series_id的val_data数据点
        series_data = val_data.filter(pl.col('series_id') == series_id)
        total_points = len(series_data)
        
        if total_points > 0:
            # 获取预计算的事件信息
            event_info = series_events.get(series_id, {
                'onset_count': 0,
                'wakeup_count': 0,
                'sleep_periods': []
            })
            
            onset_count = event_info['onset_count']
            wakeup_count = event_info['wakeup_count']
            sleep_periods = event_info['sleep_periods']
            
            # 优化：批量处理数据点
            if sleep_periods:
                # 转换为pandas以提高处理速度
                series_data_pd = series_data.to_pandas()
                
                # 定义判断函数
                def is_asleep(step):
                    for onset, wakeup in sleep_periods:
                        if onset <= step <= wakeup:
                            return True
                    return False
                
                # 批量应用
                asleep_mask = series_data_pd['step'].apply(is_asleep)
                asleep_count = asleep_mask.sum()
                awake_count = total_points - asleep_count
            else:
                # 无睡眠周期，全部为清醒
                asleep_count = 0
                awake_count = total_points
            
            # 计算比例
            awake_ratio = (awake_count / total_points) * 100 if total_points > 0 else 0
            asleep_ratio = (asleep_count / total_points) * 100 if total_points > 0 else 0
            
            # 添加到结果列表
            results.append({
                'series_id': series_id,
                'total_points': total_points,
                'awake_count': awake_count,
                'asleep_count': asleep_count,
                'awake_ratio': awake_ratio,
                'asleep_ratio': asleep_ratio,
                'onset_count': onset_count,
                'wakeup_count': wakeup_count,
                'sleep_periods_count': len(sleep_periods)
            })
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_num_file, index=False)
    
    print(f"系列统计完成，结果已保存到 {output_num_file}")
    
    # 2. 统计每个onset和wakeup事件的详细信息    
    # 获取val_ids对应的完整事件数据（包含night和timestamp）
    val_events_full = train_events.filter(pl.col('series_id').is_in(val_ids)).to_pandas()
    
    # 选择需要的列
    val_events_selected = val_events_full[['series_id', 'night', 'event', 'step', 'timestamp']]
    
    # 保存到CSV
    val_events_selected.to_csv(output_data_file, index=False)
    
    print(f"事件详细信息统计完成，结果已保存到 {output_data_file}")
    
    # 3. 显示总体统计    
    if not results_df.empty:
        total_awake = results_df['awake_count'].sum()
        total_asleep = results_df['asleep_count'].sum()
        total_points = results_df['total_points'].sum()
        
        total_awake_ratio = (total_awake / total_points) * 100 if total_points > 0 else 0
        total_asleep_ratio = (total_asleep / total_points) * 100 if total_points > 0 else 0
        
        total_onset = results_df['onset_count'].sum()
        total_wakeup = results_df['wakeup_count'].sum()
        total_sleep_periods = results_df['sleep_periods_count'].sum()
        
        print(f"验证集数据点总数: {total_points}")
        print(f"验证集清醒: {total_awake} ({total_awake_ratio:.2f}%)")
        print(f"验证集睡眠: {total_asleep} ({total_asleep_ratio:.2f}%)")
        print(f"验证集真实onset事件总数: {total_onset}")
        print(f"验证集真实wakeup事件总数: {total_wakeup}")
        print(f"验证集真实睡眠周期总数: {total_sleep_periods}")
        
        # 事件详细信息统计
        event_count = len(val_events_selected)
        onset_event_count = len(val_events_selected[val_events_selected['event'] == 'onset'])
        wakeup_event_count = len(val_events_selected[val_events_selected['event'] == 'wakeup'])
        
        print(f"验证事件详细信息统计：")
        print(f"验证事件总数: {event_count}")
        print(f"验证onset事件数: {onset_event_count}")
        print(f"验证wakeup事件数: {wakeup_event_count}")
    else:
        print("没有数据可供统计")

def count_days_in_data(data):
    """
    统计数据中的天数，按series_id分组统计
    
    参数:
    data: 包含timestamp列的Polars DataFrame
    
    返回:
    dict: 每个series_id对应的天数
    """
    days_by_series = {}
    series_ids = data['series_id'].unique().to_list()
    
    for series_id in series_ids:
        series_data = data.filter(pl.col('series_id') == series_id)
        unique_dates = series_data.select(
            pl.col('timestamp').dt.date()
        ).unique()
        days_by_series[series_id] = len(unique_dates)
    
    return days_by_series

def analyze_sleep_cycles_per_day(train_events):
    """
    分析每天的睡眠周期数量
    
    参数:
    train_events: 训练事件数据，包含series_id、night、event、step、timestamp列
    
    返回:
    dict: 每个series_id对应的睡眠周期数量
    """
    # 过滤掉空值
    valid_events = train_events.drop_nulls()
    
    # 按series_id和night分组，确保每个night都有onset和wakeup
    balanced_nights = valid_events.group_by(['series_id', 'night']).agg([
        ((pl.col('event') == 'onset').sum() == 1).alias('has_onset'),
        ((pl.col('event') == 'wakeup').sum() == 1).alias('has_wakeup'),
        pl.col('timestamp').first().alias('timestamp')
    ]).filter(pl.col('has_onset') & pl.col('has_wakeup'))
    
    # 按series_id分组，统计每个series_id的睡眠周期数量
    cycles_by_series = balanced_nights.group_by('series_id').agg(
        pl.count('night').alias('cycle_count')
    ).to_dict(as_series=False)
    
    # 转换为字典格式
    cycles_dict = {series_id: cycle_count 
                   for series_id, cycle_count in zip(cycles_by_series['series_id'], cycles_by_series['cycle_count'])}
    
    # 统计总体情况
    total_cycles = balanced_nights.height  # 每个night对应一个睡眠周期
    
    print(f"总睡眠周期数: {total_cycles}")
    
    return cycles_dict

def analyze_val_data_stats(val_data, train_events, output_file='/home/zhuangzhuohan/sleep_data/val/val_day_num.csv'):
    """
    分析验证数据的统计信息，包括天数和睡眠周期
    
    参数:
    val_data: 验证时间序列数据
    train_events: 训练事件数据（可以是polars或pandas DataFrame）
    output_file: 输出文件名，默认'val_day_num.csv'
    
    返回:
    None，结果保存在CSV文件中
    """
    # 获取每个series_id的天数
    days_by_series = count_days_in_data(val_data)
    
    # 确保train_events是polars DataFrame
    if isinstance(train_events, pd.DataFrame):
        train_events = pl.from_pandas(train_events)
    
    # 获取每个series_id的睡眠周期数量
    cycles_by_series = analyze_sleep_cycles_per_day(train_events.filter(pl.col('series_id').is_in(list(days_by_series.keys()))))
    
    # 计算总天数
    total_days = sum(days_by_series.values())
    print(f"总天数: {total_days}")
    
    # 创建结果DataFrame
    results = []
    for series_id in days_by_series.keys():
        results.append({
            'series_id': series_id,
            'days': days_by_series.get(series_id, 0),
            'cycles': cycles_by_series.get(series_id, 0)
        })
    
    # 保存到CSV文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"统计结果已保存到 {output_file}")

def analyze_model_performance(predictions, val_data_file='/home/zhuangzhuohan/sleep_data/val/val_data.csv', output_file='/home/zhuangzhuohan/sleep_data/val/model_performance.csv'):
    """
    分析模型性能，只考虑验证数据中有记录的事件
    
    参数:
    predictions: 模型预测结果（pandas DataFrame），包含series_id、event、step、score列
    val_data_file: 验证数据文件路径，默认'val_data.csv'
    output_file: 输出结果文件路径，默认'model_performance.csv'
    
    返回:
    dict: 包含模型性能统计信息
    """
    import numpy as np
    
    # 读取验证数据
    val_data = pd.read_csv(val_data_file)
    val_data = val_data[val_data['event'].isin(['onset', 'wakeup'])]  # 只保留onset和wakeup事件
    
    # 按series_id和event分组
    val_groups = val_data.groupby(['series_id', 'event'])
    pred_groups = predictions.groupby(['series_id', 'event'])
    
    # 存储匹配结果
    matches = []
    unmatched_val = []
    
    # 遍历每个series_id和event类型
    for (series_id, event), val_events in val_groups:
        # 排序验证事件
        val_events_sorted = val_events.sort_values('step').reset_index(drop=True)
        
        # 获取对应的预测事件
        try:
            pred_events = pred_groups.get_group((series_id, event))
            pred_events_sorted = pred_events.sort_values('step').reset_index(drop=True)
        except KeyError:
            # 没有对应的预测事件
            for _, val_event in val_events_sorted.iterrows():
                unmatched_val.append({
                    'series_id': series_id,
                    'event': event,
                    'val_step': val_event['step'],
                    'val_timestamp': val_event['timestamp'],
                    'matched': False
                })
            continue
        
        # 匹配验证事件和预测事件
        val_idx = 0
        pred_idx = 0
        
        while val_idx < len(val_events_sorted) and pred_idx < len(pred_events_sorted):
            val_step = val_events_sorted.loc[val_idx, 'step']
            pred_step = pred_events_sorted.loc[pred_idx, 'step']
            
            # 计算时间差异（一步是5秒）
            step_diff = abs(val_step - pred_step)
            time_diff_sec = step_diff * 5
            
            # 如果差异在合理范围内（例如30分钟内），视为匹配
            if step_diff <= 12 * 30:  # 30分钟 = 12步/分钟 * 30分钟
                matches.append({
                    'series_id': series_id,
                    'event': event,
                    'val_step': val_step,
                    'val_timestamp': val_events_sorted.loc[val_idx, 'timestamp'],
                    'pred_step': pred_step,
                    'step_diff': step_diff,
                    'time_diff_sec': time_diff_sec,
                    'matched': True
                })
                val_idx += 1
                pred_idx += 1
            elif val_step < pred_step:
                # 验证事件在预测事件之前，视为未匹配
                unmatched_val.append({
                    'series_id': series_id,
                    'event': event,
                    'val_step': val_step,
                    'val_timestamp': val_events_sorted.loc[val_idx, 'timestamp'],
                    'matched': False
                })
                val_idx += 1
            else:
                # 预测事件在验证事件之前，跳过该预测
                pred_idx += 1
        
        # 处理剩余的验证事件
        while val_idx < len(val_events_sorted):
            unmatched_val.append({
                'series_id': series_id,
                'event': event,
                'val_step': val_events_sorted.loc[val_idx, 'step'],
                'val_timestamp': val_events_sorted.loc[val_idx, 'timestamp'],
                'matched': False
            })
            val_idx += 1
    
    # 创建结果DataFrame
    matches_df = pd.DataFrame(matches)
    unmatched_val_df = pd.DataFrame(unmatched_val)
    
    # 计算统计指标
    stats = {}
    
    # 总验证事件数
    total_val_events = len(val_data)
    stats['total_val_events'] = total_val_events
    
    # 匹配事件数
    matched_events = len(matches_df)
    stats['matched_events'] = matched_events
    
    # 未匹配事件数
    unmatched_events = len(unmatched_val_df)
    stats['unmatched_events'] = unmatched_events
    
    # 匹配率
    match_rate = matched_events / total_val_events if total_val_events > 0 else 0
    stats['match_rate'] = match_rate
    
    # 计算时间差异统计
    if matched_events > 0:
        # 步数差异
        step_diffs = matches_df['step_diff'].values
        stats['avg_step_diff'] = np.mean(step_diffs)
        stats['std_step_diff'] = np.std(step_diffs)
        stats['min_step_diff'] = np.min(step_diffs)
        stats['max_step_diff'] = np.max(step_diffs)
        
        # 时间差异（秒）
        time_diffs = matches_df['time_diff_sec'].values
        stats['avg_time_diff_sec'] = np.mean(time_diffs)
        stats['std_time_diff_sec'] = np.std(time_diffs)
        stats['min_time_diff_sec'] = np.min(time_diffs)
        stats['max_time_diff_sec'] = np.max(time_diffs)
        
        # 按事件类型统计
        for event_type in ['onset', 'wakeup']:
            event_matches = matches_df[matches_df['event'] == event_type]
            if len(event_matches) > 0:
                stats[f'{event_type}_matched'] = len(event_matches)
                stats[f'{event_type}_avg_step_diff'] = np.mean(event_matches['step_diff'])
                stats[f'{event_type}_avg_time_diff_sec'] = np.mean(event_matches['time_diff_sec'])
            else:
                stats[f'{event_type}_matched'] = 0
                stats[f'{event_type}_avg_step_diff'] = 0
                stats[f'{event_type}_avg_time_diff_sec'] = 0
    else:
        # 没有匹配事件
        stats['avg_step_diff'] = 0
        stats['std_step_diff'] = 0
        stats['min_step_diff'] = 0
        stats['max_step_diff'] = 0
        stats['avg_time_diff_sec'] = 0
        stats['std_time_diff_sec'] = 0
        stats['min_time_diff_sec'] = 0
        stats['max_time_diff_sec'] = 0
        stats['onset_matched'] = 0
        stats['wakeup_matched'] = 0
        stats['onset_avg_step_diff'] = 0
        stats['wakeup_avg_step_diff'] = 0
        stats['onset_avg_time_diff_sec'] = 0
        stats['wakeup_avg_time_diff_sec'] = 0
    
    # 保存详细匹配结果
    all_results = pd.concat([matches_df, unmatched_val_df], ignore_index=True)
    all_results.to_csv(output_file, index=False)
    
    # 打印统计信息
    print(f"模型性能分析结果:")
    print(f"总验证事件数: {total_val_events}")
    print(f"匹配事件数: {matched_events}")
    print(f"未匹配事件数: {unmatched_events}")
    print(f"匹配率: {match_rate:.4f}")
    
    if matched_events > 0:
        print(f"\n时间差异统计:")
        print(f"平均步数差异: {stats['avg_step_diff']:.2f} 步")
        print(f"步数差异标准差: {stats['std_step_diff']:.2f} 步")
        print(f"平均时间差异: {stats['avg_time_diff_sec']:.2f} 秒")
        print(f"时间差异标准差: {stats['std_time_diff_sec']:.2f} 秒")
        
        print(f"\n按事件类型统计:")
        print(f"Onset事件匹配数: {stats['onset_matched']}")
        print(f"Onset平均步数差异: {stats['onset_avg_step_diff']:.2f} 步")
        print(f"Onset平均时间差异: {stats['onset_avg_time_diff_sec']:.2f} 秒")
        print(f"Wakeup事件匹配数: {stats['wakeup_matched']}")
        print(f"Wakeup平均步数差异: {stats['wakeup_avg_step_diff']:.2f} 步")
        print(f"Wakeup平均时间差异: {stats['wakeup_avg_time_diff_sec']:.2f} 秒")
    
    print(f"\n详细结果已保存到 {output_file}")
    
    return stats

def example_analyze_predictions():
    """
    analyze_predictions函数的使用示例
    """
    # 假设我们已经有了以下变量：
    # rf_submission: 模型预测结果
    # val_data: 验证时间序列数据
    # val_ids: 验证集的series_id列表
    # train_events: 原始训练事件数据
    
    # 1. 导入必要的库
    import pandas as pd
    import polars as pl
    from my_funs import get_events, analyze_predictions
    
    # 2. 加载模型预测结果（这里假设已经通过get_events函数生成）
    # rf_submission = get_events(test_series, classifier, feature_cols, id_cols)
    
    # 3. 调用analyze_predictions函数
    analyze_predictions(
        rf_submission=rf_submission,  # 模型预测结果
        val_data=val_data,  # 验证时间序列数据
        val_ids=val_ids,  # 验证集的series_id列表
        train_events=train_events,  # 原始训练事件数据
        output_num_file='rf_num.csv',  # 系列统计结果保存的文件名
        output_data_file='rf_data.csv'  # 事件详细信息保存的文件名
    )
    
    # 4. 查看输出结果
    print("\n预测结果分析完成！")
    print("系列统计结果已保存到 rf_num.csv")
    print("事件详细信息已保存到 rf_data.csv")
    
    # 5. 读取并查看结果
    rf_num = pd.read_csv('/home/zhuangzhuohan/sleep_data/rf/rf_num.csv')
    rf_data = pd.read_csv('/home/zhuangzhuohan/sleep_data/rf/rf_data.csv')
    
    print("\n系列统计结果预览:")
    print(rf_num.head())
    
    print("\n事件详细信息预览:")
    print(rf_data.head())
    
    return rf_num, rf_data

def integrated_analyze_predictions(rf_submission, val_data, val_ids, train_events, val_solution, output_num_file='/home/zhuangzhuohan/sleep_data/rf/rf_num.csv', output_data_file='/home/zhuangzhuohan/sleep_data/rf/rf_data.csv', output_avg_file='/home/zhuangzhuohan/sleep_data/rf/rf_average.csv'):
    """
    集成分析预测结果的函数，整合analyze_predictions功能，并按series_id独立计算事件差异统计
    
    参数:
    rf_submission: 模型预测结果（pandas DataFrame）
    val_data: 验证时间序列数据（Polars DataFrame）
    val_ids: 验证集的series_id列表
    train_events: 原始训练事件数据（Polars DataFrame）
    val_solution: 验证集的事件数据（pandas DataFrame）
    output_num_file: 系列统计结果保存的文件名，默认'rf_num.csv'
    output_data_file: 事件详细信息保存的文件名，默认'rf_data.csv'
    output_avg_file: 事件差异统计保存的文件名，默认'rf_average.csv'
    
    返回:
    dict: 包含模型性能统计信息
    """
    import numpy as np
    import pandas as pd
    import polars as pl
    
    # 1. 调用analyze_predictions函数统计预测结果
    analyze_predictions(
        rf_submission=rf_submission,
        val_data=val_data,
        val_ids=val_ids,
        train_events=train_events,
        output_num_file=output_num_file,
        output_data_file=output_data_file,
        verbose=False  # 不显示analyze_predictions的输出，避免重复
    )
    
    # 2. 计算验证数据天数
    days_by_series = count_days_in_data(val_data)
    total_val_days = sum(days_by_series.values())
    
    # 3. 计算预测天数（基于预测的睡眠周期）
    rf_submission_pd = rf_submission.copy()
    
    # 4. 计算预测事件总数
    total_predicted_onset = len(rf_submission_pd[rf_submission_pd['event'] == 'onset'])
    total_predicted_wakeup = len(rf_submission_pd[rf_submission_pd['event'] == 'wakeup'])
    
    # 5. 计算预测天数（基于onset事件数量，每个onset事件对应一天）
    # 每个onset事件对应一个睡眠周期，即一天
    total_predicted_days = total_predicted_onset
    
    # 6. 计算预测率
    prediction_rate = total_predicted_days / total_val_days if total_val_days > 0 else 0
    
    # 7. 按series_id独立计算事件差异统计
    # 准备验证数据和预测数据
    val_solution_sorted = val_solution.sort_values(['series_id', 'event', 'step'])
    rf_submission_sorted = rf_submission_pd.sort_values(['series_id', 'event', 'step'])
    
    # 存储每个series_id的差异统计
    series_diff_stats = {}
    total_onset_matches = 0
    total_wakeup_matches = 0
    
    # 遍历每个series_id
    for series_id in val_ids:
        # 获取当前series_id的验证事件
        val_onset_events = val_solution_sorted[(val_solution_sorted['series_id'] == series_id) & (val_solution_sorted['event'] == 'onset')]['step'].tolist()
        val_wakeup_events = val_solution_sorted[(val_solution_sorted['series_id'] == series_id) & (val_solution_sorted['event'] == 'wakeup')]['step'].tolist()
        
        # 获取当前series_id的预测事件
        pred_onset_events = rf_submission_sorted[(rf_submission_sorted['series_id'] == series_id) & (rf_submission_sorted['event'] == 'onset')]['step'].tolist()
        pred_wakeup_events = rf_submission_sorted[(rf_submission_sorted['series_id'] == series_id) & (rf_submission_sorted['event'] == 'wakeup')]['step'].tolist()
        
        # 匹配onset事件并计算差异
        onset_step_diffs = []
        val_idx, pred_idx = 0, 0
        while val_idx < len(val_onset_events) and pred_idx < len(pred_onset_events):
            val_step = val_onset_events[val_idx]
            pred_step = pred_onset_events[pred_idx]
            step_diff = abs(val_step - pred_step)
            
            # 如果差异在合理范围内（15分钟内），视为匹配
            if step_diff <= 12 * 15:
                onset_step_diffs.append(step_diff)
                val_idx += 1
                pred_idx += 1
            elif val_step < pred_step:
                val_idx += 1
            else:
                pred_idx += 1
        
        # 匹配wakeup事件并计算差异
        wakeup_step_diffs = []
        val_idx, pred_idx = 0, 0
        while val_idx < len(val_wakeup_events) and pred_idx < len(pred_wakeup_events):
            val_step = val_wakeup_events[val_idx]
            pred_step = pred_wakeup_events[pred_idx]
            step_diff = abs(val_step - pred_step)
            
            # 如果差异在合理范围内（15分钟内），视为匹配
            if step_diff <= 12 * 15:
                wakeup_step_diffs.append(step_diff)
                val_idx += 1
                pred_idx += 1
            elif val_step < pred_step:
                val_idx += 1
            else:
                pred_idx += 1
        
        # 计算当前series_id的差异统计
        onset_avg_step_diff = np.mean(onset_step_diffs) if onset_step_diffs else 0
        onset_std_step_diff = np.std(onset_step_diffs) if onset_step_diffs else 0
        onset_avg_time_diff = onset_avg_step_diff * 5 if onset_step_diffs else 0
        onset_std_time_diff = onset_std_step_diff * 5 if onset_step_diffs else 0
        
        wakeup_avg_step_diff = np.mean(wakeup_step_diffs) if wakeup_step_diffs else 0
        wakeup_std_step_diff = np.std(wakeup_step_diffs) if wakeup_step_diffs else 0
        wakeup_avg_time_diff = wakeup_avg_step_diff * 5 if wakeup_step_diffs else 0
        wakeup_std_time_diff = wakeup_std_step_diff * 5 if wakeup_step_diffs else 0
        
        # 计算当前series_id的匹配成功率
        onset_match_rate = len(onset_step_diffs) / len(val_onset_events) if len(val_onset_events) > 0 else 0
        wakeup_match_rate = len(wakeup_step_diffs) / len(val_wakeup_events) if len(val_wakeup_events) > 0 else 0
        
        # 存储当前series_id的差异统计
        series_diff_stats[series_id] = {
            'onset_avg_step_diff': onset_avg_step_diff,
            'onset_std_step_diff': onset_std_step_diff,
            'onset_avg_time_diff': onset_avg_time_diff,
            'onset_std_time_diff': onset_std_time_diff,
            'wakeup_avg_step_diff': wakeup_avg_step_diff,
            'wakeup_std_step_diff': wakeup_std_step_diff,
            'wakeup_avg_time_diff': wakeup_avg_time_diff,
            'wakeup_std_time_diff': wakeup_std_time_diff,
            'onset_matches': len(onset_step_diffs),
            'wakeup_matches': len(wakeup_step_diffs),
            'onset_match_rate': onset_match_rate,
            'wakeup_match_rate': wakeup_match_rate,
            'onset_step_diffs': onset_step_diffs,  # 保存原始差异值
            'wakeup_step_diffs': wakeup_step_diffs  # 保存原始差异值
        }
        
        # 累加到总匹配数
        total_onset_matches += len(onset_step_diffs)
        total_wakeup_matches += len(wakeup_step_diffs)
    
    # 8. 计算总体匹配率
    val_onset_total = len(val_solution[val_solution['event'] == 'onset'])
    val_wakeup_total = len(val_solution[val_solution['event'] == 'wakeup'])
    onset_match_rate = total_onset_matches / val_onset_total if val_onset_total > 0 else 0
    wakeup_match_rate = total_wakeup_matches / val_wakeup_total if val_wakeup_total > 0 else 0
    
    # 9. 创建并保存rf_average.csv文件，存储事件差异统计
    # 准备数据
    avg_data = []
    for series_id, stats in series_diff_stats.items():
        avg_data.append({
            'series_id': series_id,
            'onset_avg_step_diff': stats['onset_avg_step_diff'],
            'onset_std_step_diff': stats['onset_std_step_diff'],
            'onset_avg_time_diff': stats['onset_avg_time_diff'],
            'onset_std_time_diff': stats['onset_std_time_diff'],
            'wakeup_avg_step_diff': stats['wakeup_avg_step_diff'],
            'wakeup_std_step_diff': stats['wakeup_std_step_diff'],
            'wakeup_avg_time_diff': stats['wakeup_avg_time_diff'],
            'wakeup_std_time_diff': stats['wakeup_std_time_diff'],
            'onset_matches': stats['onset_matches'],
            'wakeup_matches': stats['wakeup_matches'],
            'onset_match_rate': stats['onset_match_rate'],
            'wakeup_match_rate': stats['wakeup_match_rate']
        })
    
    # 创建DataFrame并保存
    avg_df = pd.DataFrame(avg_data)
    avg_df.to_csv(output_avg_file, index=False)
    
    # 10. 计算总体差异统计（用于打印）
    all_onset_step_diffs = []
    all_wakeup_step_diffs = []
    for stats in series_diff_stats.values():
        # 使用每个series_id的原始差异值计算总体统计
        # 这样可以确保标准差计算准确
        if 'onset_step_diffs' in stats:
            all_onset_step_diffs.extend(stats['onset_step_diffs'])
        if 'wakeup_step_diffs' in stats:
            all_wakeup_step_diffs.extend(stats['wakeup_step_diffs'])
    
    # 如果原始差异值为空，使用匹配数和平均值作为备用
    if not all_onset_step_diffs:
        for stats in series_diff_stats.values():
            onset_matches = stats['onset_matches']
            if onset_matches > 0:
                avg_onset_diff = stats['onset_avg_step_diff']
                all_onset_step_diffs.extend([avg_onset_diff] * onset_matches)
    
    if not all_wakeup_step_diffs:
        for stats in series_diff_stats.values():
            wakeup_matches = stats['wakeup_matches']
            if wakeup_matches > 0:
                avg_wakeup_diff = stats['wakeup_avg_step_diff']
                all_wakeup_step_diffs.extend([avg_wakeup_diff] * wakeup_matches)
    
    overall_onset_avg_step_diff = np.mean(all_onset_step_diffs) if all_onset_step_diffs else 0
    overall_onset_std_step_diff = np.std(all_onset_step_diffs) if all_onset_step_diffs else 0
    overall_onset_avg_time_diff = overall_onset_avg_step_diff * 5 if all_onset_step_diffs else 0
    overall_onset_std_time_diff = overall_onset_std_step_diff * 5 if all_onset_step_diffs else 0
    
    overall_wakeup_avg_step_diff = np.mean(all_wakeup_step_diffs) if all_wakeup_step_diffs else 0
    overall_wakeup_std_step_diff = np.std(all_wakeup_step_diffs) if all_wakeup_step_diffs else 0
    overall_wakeup_avg_time_diff = overall_wakeup_avg_step_diff * 5 if all_wakeup_step_diffs else 0
    overall_wakeup_std_time_diff = overall_wakeup_std_step_diff * 5 if all_wakeup_step_diffs else 0
    
    # 11. 确保rf_data.csv包含timestamp和night字段
    # 重新读取rf_data.csv并检查字段
    rf_data_df = pd.read_csv(output_data_file)
    
    # 如果缺少timestamp或night字段，重新生成
    if 'timestamp' not in rf_data_df.columns or 'night' not in rf_data_df.columns:
        # 构建step到night和timestamp的映射
        step_info_map = {}
        train_events_pd = train_events.to_pandas()
        for _, row in train_events_pd.iterrows():
            key = (row['series_id'], row['step'])
            step_info_map[key] = {'night': row['night'], 'timestamp': row['timestamp']}
        
        def get_step_info_full(row):
            key = (row['series_id'], row['step'])
            info = step_info_map.get(key, {'night': None, 'timestamp': None})
            return info['night'], info['timestamp']
        
        # 为rf_submission添加night和timestamp列
        rf_submission_full = rf_submission.copy()
        rf_submission_full[['night', 'timestamp']] = rf_submission_full.apply(get_step_info_full, axis=1, result_type='expand')
        
        # 保存更新后的rf_data.csv
        rf_submission_full[['series_id', 'night', 'event', 'step', 'timestamp']].to_csv(output_data_file, index=False)
        #print(f"\nrf_data.csv已更新，补充了缺失的timestamp和night字段") 
    # 12. 打印所有统计信息
    print("\n=== 集成预测分析结果 ===")
    print(f"预测onset事件总数: {total_predicted_onset}")
    print(f"预测wakeup事件总数: {total_predicted_wakeup}")
    print(f"验证数据总天数: {total_val_days}")
    print(f"预测天数: {total_predicted_days}")
    print(f"预测率: {prediction_rate:.4f}")
    print(f"\nOnset事件匹配数: {total_onset_matches}")
    print(f"Onset事件匹配率: {onset_match_rate:.4f}")
    print(f"Wakeup事件匹配数: {total_wakeup_matches}")
    print(f"Wakeup事件匹配率: {wakeup_match_rate:.4f}")
    print(f"\nOnset事件平均步数差异: {overall_onset_avg_step_diff:.2f} 步")
    print(f"Onset事件步数差异标准差: {overall_onset_std_step_diff:.2f} 步")
    print(f"Onset事件平均时间差异: {overall_onset_avg_time_diff:.2f} 秒")
    print(f"Onset事件时间差异标准差: {overall_onset_std_time_diff:.2f} 秒")
    print(f"\nWakeup事件平均步数差异: {overall_wakeup_avg_step_diff:.2f} 步")
    print(f"Wakeup事件步数差异标准差: {overall_wakeup_std_step_diff:.2f} 步")
    print(f"Wakeup事件平均时间差异: {overall_wakeup_avg_time_diff:.2f} 秒")
    print(f"Wakeup事件时间差异标准差: {overall_wakeup_std_time_diff:.2f} 秒")
    print(f"\n事件差异统计已按series_id独立添加到 {output_avg_file}")
    # 13. 返回综合统计信息
    return {
        'total_predicted_onset': total_predicted_onset,
        'total_predicted_wakeup': total_predicted_wakeup,
        'total_val_days': total_val_days,
        'total_predicted_days': total_predicted_days,
        'prediction_rate': prediction_rate,
        'onset_matches': total_onset_matches,
        'onset_match_rate': onset_match_rate,
        'wakeup_matches': total_wakeup_matches,
        'wakeup_match_rate': wakeup_match_rate,
        'onset_diff_stats': {
            'avg_step_diff': overall_onset_avg_step_diff,
            'std_step_diff': overall_onset_std_step_diff,
            'avg_time_diff': overall_onset_avg_time_diff,
            'std_time_diff': overall_onset_std_time_diff
        },
        'wakeup_diff_stats': {
            'avg_step_diff': overall_wakeup_avg_step_diff,
            'std_step_diff': overall_wakeup_std_step_diff,
            'avg_time_diff': overall_wakeup_avg_time_diff,
            'std_time_diff': overall_wakeup_std_time_diff
        },
        'series_diff_stats': series_diff_stats
    }






def load_and_preprocess_data_new(train_series_path='/home/zhuangzhuohan/sleep_data/train_series/train_series.parquet', 
                           train_events_path='/home/zhuangzhuohan/sleep_data/train_events/train_events.csv',
                           test_series_path='/home/zhuangzhuohan/sleep_data/test_series/test_series.parquet'):
    """
    加载和预处理数据
    
    参数:
    train_series_path: 训练时间序列数据路径
    train_events_path: 训练事件数据路径
    test_series_path: 测试时间序列数据路径
    
    返回:
    train_series: 预处理后的训练时间序列数据
    train_events: 预处理后的训练事件数据
    test_series: 预处理后的测试时间序列数据
    """
    # 创建时间转换表达式
    timestamp_expr = pl.col('timestamp').str.to_datetime(format=TIME_FORMAT, time_zone='UTC')
    
    dt_transforms = [
        timestamp_expr.alias('timestamp'),  # 转换为UTC时区的datetime
        (timestamp_expr.dt.year() - 2000).cast(pl.UInt8).alias('year'),  # 提取年份（减去2000以节省空间）
        timestamp_expr.dt.month().cast(pl.UInt8).alias('month'),  # 提取月份
        timestamp_expr.dt.day().cast(pl.UInt8).alias('day'),  # 提取日期
        timestamp_expr.dt.hour().cast(pl.UInt8).alias('hour'),  # 提取小时
        timestamp_expr.dt.minute().cast(pl.UInt8).alias('minute') # 提取分钟
    ]
    
    data_transforms = [
        pl.col('anglez').cast(pl.Int16), # 将anglez转换为16位整数以节省空间
        (pl.col('enmo')*1000).cast(pl.UInt16), # 将enmo乘以1000并转换为16位无符号整数
    ]
    
    # 读取训练数据（使用lazy loading提高效率）
    train_series = pl.scan_parquet(train_series_path).with_columns(
        dt_transforms + data_transforms
        )
    
    # 读取训练事件数据
    train_events = pl.read_csv(train_events_path).with_columns(
        dt_transforms
        ).drop_nulls()  # 删除空值
    
    # 读取测试数据（使用lazy loading提高效率）
    test_series = pl.scan_parquet(test_series_path).with_columns(
        dt_transforms + data_transforms
        )
    
    # 移除事件数量不匹配的夜晚（确保每个onset对应一个wakeup）
    mismatches = train_events.drop_nulls().group_by(['series_id', 'night']).agg([
        ((pl.col('event') == 'onset').sum() == (pl.col('event') == 'wakeup').sum()).alias('balanced')
        ]).sort(by=['series_id', 'night']).filter(~pl.col('balanced'))
    
    # 移除不匹配的数据
    for mm in mismatches.to_numpy(): 
        train_events = train_events.filter(~((pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1])))
    
    # 获取唯一的series_id列表
    series_ids = train_events['series_id'].unique(maintain_order=True).to_list()
    
    # 更新train_series，只保留有事件数据的series_id
    train_series = train_series.filter(pl.col('series_id').is_in(series_ids))
    
    return train_series, train_events, test_series, series_ids

def load_and_preprocess_data_float(train_series_path='/home/zhuangzhuohan/sleep_data/train_series/train_series.parquet', 
                           train_events_path='/home/zhuangzhuohan/sleep_data/train_events/train_events.csv',
                           test_series_path='/home/zhuangzhuohan/sleep_data/test_series/test_series.parquet'):
    """
    加载和预处理数据
    
    参数:
    train_series_path: 训练时间序列数据路径
    train_events_path: 训练事件数据路径
    test_series_path: 测试时间序列数据路径
    
    返回:
    train_series: 预处理后的训练时间序列数据
    train_events: 预处理后的训练事件数据
    test_series: 预处理后的测试时间序列数据
    """
    # 创建时间转换表达式
    timestamp_expr = pl.col('timestamp').str.to_datetime(format=TIME_FORMAT, time_zone='UTC')
    
    dt_transforms = [
        timestamp_expr.alias('timestamp'),  # 转换为UTC时区的datetime
        (timestamp_expr.dt.year() - 2000).cast(pl.UInt8).alias('year'),  # 提取年份（减去2000以节省空间）
        timestamp_expr.dt.month().cast(pl.UInt8).alias('month'),  # 提取月份
        timestamp_expr.dt.day().cast(pl.UInt8).alias('day'),  # 提取日期
        timestamp_expr.dt.hour().cast(pl.UInt8).alias('hour'),  # 提取小时
        timestamp_expr.dt.minute().cast(pl.UInt8).alias('minute') # 提取分钟
    ]
    
    data_transforms = [
        pl.col('anglez').cast(pl.Float32), # 将anglez转换为32位浮点数
        pl.col('enmo').cast(pl.Float32), # 将enmo转换为32位浮点数
    ]
    
    # 读取训练数据（使用lazy loading提高效率）
    train_series = pl.scan_parquet(train_series_path).with_columns(
        dt_transforms + data_transforms
        )
    
    # 读取训练事件数据
    train_events = pl.read_csv(train_events_path).with_columns(
        dt_transforms
        ).drop_nulls()  # 删除空值
    
    # 读取测试数据（使用lazy loading提高效率）
    test_series = pl.scan_parquet(test_series_path).with_columns(
        dt_transforms + data_transforms
        )
    
    # 移除事件数量不匹配的夜晚（确保每个onset对应一个wakeup）
    mismatches = train_events.drop_nulls().group_by(['series_id', 'night']).agg([
        ((pl.col('event') == 'onset').sum() == (pl.col('event') == 'wakeup').sum()).alias('balanced')
        ]).sort(by=['series_id', 'night']).filter(~pl.col('balanced'))
    
    # 移除不匹配的数据
    for mm in mismatches.to_numpy(): 
        train_events = train_events.filter(~((pl.col('series_id') == mm[0]) & (pl.col('night') == mm[1])))
    
    # 获取唯一的series_id列表
    series_ids = train_events['series_id'].unique(maintain_order=True).to_list()
    
    # 更新train_series，只保留有事件数据的series_id
    train_series = train_series.filter(pl.col('series_id').is_in(series_ids))
    
    return train_series, train_events, test_series, series_ids

def create_features_new():
    """
    创建特征和特征列名列表
    
    返回:
    features: 特征变换列表
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    """
    # 初始化特征列表，先加入小时特征
    #features, feature_cols = [pl.col('hour')], ['hour']
    features, feature_cols = [pl.col('hour'), pl.col('minute')], ['hour', 'minute']  

    # 为不同时间窗口创建特征
    # 1分钟窗口特征
    # enmo - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('enmo').rolling_mean(12 * 1, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_1m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('enmo').rolling_max(12 * 1, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_1m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('enmo').rolling_std(12 * 1, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_1m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_1m_mean', 'enmo_1m_max', 'enmo_1m_std'
    ]
    # enmo - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('enmo').diff().abs().rolling_mean(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_1m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('enmo').diff().abs().rolling_max(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_1m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('enmo').diff().abs().rolling_std(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_1m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_1v_1m_mean', 'enmo_1v_1m_max', 'enmo_1v_1m_std'
    ]
    # anglez - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('anglez').rolling_mean(12 * 1, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_1m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('anglez').rolling_max(12 * 1, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_1m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('anglez').rolling_std(12 * 1, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_1m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_1m_mean', 'anglez_1m_max', 'anglez_1m_std'
    ]
    # anglez - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('anglez').diff().abs().rolling_mean(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_1m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('anglez').diff().abs().rolling_max(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_1m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('anglez').diff().abs().rolling_std(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_1m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_1v_1m_mean', 'anglez_1v_1m_max', 'anglez_1v_1m_std'
    ]
    # 5分钟窗口特征
    # enmo - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('enmo').rolling_mean(12 * 5, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_5m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('enmo').rolling_max(12 * 5, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_5m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('enmo').rolling_std(12 * 5, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_5m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_5m_mean', 'enmo_5m_max', 'enmo_5m_std'
    ]
    # enmo - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('enmo').diff().abs().rolling_mean(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_5m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('enmo').diff().abs().rolling_max(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_5m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('enmo').diff().abs().rolling_std(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_5m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_1v_5m_mean', 'enmo_1v_5m_max', 'enmo_1v_5m_std'
    ]
    # anglez - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('anglez').rolling_mean(12 * 5, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_5m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('anglez').rolling_max(12 * 5, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_5m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('anglez').rolling_std(12 * 5, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_5m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_5m_mean', 'anglez_5m_max', 'anglez_5m_std'
    ]
    # anglez - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('anglez').diff().abs().rolling_mean(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_5m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('anglez').diff().abs().rolling_max(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_5m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('anglez').diff().abs().rolling_std(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_5m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_1v_5m_mean', 'anglez_1v_5m_max', 'anglez_1v_5m_std'
    ]
    # 30分钟窗口特征
    # enmo - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('enmo').rolling_mean(12 * 30, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_30m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('enmo').rolling_max(12 * 30, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_30m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('enmo').rolling_std(12 * 30, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_30m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_30m_mean', 'enmo_30m_max', 'enmo_30m_std'
    ]
    # enmo - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('enmo').diff().abs().rolling_mean(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_30m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('enmo').diff().abs().rolling_max(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_30m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('enmo').diff().abs().rolling_std(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_30m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_1v_30m_mean', 'enmo_1v_30m_max', 'enmo_1v_30m_std'
    ]
    # anglez - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('anglez').rolling_mean(12 * 30, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_30m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('anglez').rolling_max(12 * 30, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_30m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('anglez').rolling_std(12 * 30, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_30m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_30m_mean', 'anglez_30m_max', 'anglez_30m_std'
    ]
    # anglez - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('anglez').diff().abs().rolling_mean(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_30m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('anglez').diff().abs().rolling_max(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_30m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('anglez').diff().abs().rolling_std(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_30m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_1v_30m_mean', 'anglez_1v_30m_max', 'anglez_1v_30m_std'
    ]
    # 60分钟窗口特征
    # enmo - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('enmo').rolling_mean(12 * 60, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_60m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('enmo').rolling_max(12 * 60, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_60m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('enmo').rolling_std(12 * 60, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_60m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_60m_mean', 'enmo_60m_max', 'enmo_60m_std'
    ]
    # enmo - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('enmo').diff().abs().rolling_mean(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_60m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('enmo').diff().abs().rolling_max(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_60m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('enmo').diff().abs().rolling_std(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_60m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_1v_60m_mean', 'enmo_1v_60m_max', 'enmo_1v_60m_std'
    ]
    # anglez - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('anglez').rolling_mean(12 * 60, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_60m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('anglez').rolling_max(12 * 60, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_60m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('anglez').rolling_std(12 * 60, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_60m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_60m_mean', 'anglez_60m_max', 'anglez_60m_std'
    ]
    # anglez - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('anglez').diff().abs().rolling_mean(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_60m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('anglez').diff().abs().rolling_max(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_60m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('anglez').diff().abs().rolling_std(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_60m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_1v_60m_mean', 'anglez_1v_60m_max', 'anglez_1v_60m_std'
    ]
    # 120分钟窗口特征
    # enmo - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('enmo').rolling_mean(12 * 120, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_120m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('enmo').rolling_max(12 * 120, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_120m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('enmo').rolling_std(12 * 120, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_120m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_120m_mean', 'enmo_120m_max', 'enmo_120m_std'
    ]
    # enmo - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('enmo').diff().abs().rolling_mean(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_120m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('enmo').diff().abs().rolling_max(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_120m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('enmo').diff().abs().rolling_std(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_120m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_1v_120m_mean', 'enmo_1v_120m_max', 'enmo_1v_120m_std'
    ]
    # anglez - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('anglez').rolling_mean(12 * 120, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_120m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('anglez').rolling_max(12 * 120, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_120m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('anglez').rolling_std(12 * 120, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_120m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_120m_mean', 'anglez_120m_max', 'anglez_120m_std'
    ]
    # anglez - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('anglez').diff().abs().rolling_mean(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_120m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('anglez').diff().abs().rolling_max(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_120m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('anglez').diff().abs().rolling_std(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_120m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_1v_120m_mean', 'anglez_1v_120m_max', 'anglez_1v_120m_std'
    ]
    # 240分钟窗口特征
    # enmo - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('enmo').rolling_mean(12 * 240, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_240m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('enmo').rolling_max(12 * 240, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_240m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('enmo').rolling_std(12 * 240, center=False, min_samples=1).abs().cast(pl.UInt16).alias('enmo_240m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_240m_mean', 'enmo_240m_max', 'enmo_240m_std'
    ]
    # enmo - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('enmo').diff().abs().rolling_mean(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_240m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('enmo').diff().abs().rolling_max(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_240m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('enmo').diff().abs().rolling_std(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('enmo_1v_240m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'enmo_1v_240m_mean', 'enmo_1v_240m_max', 'enmo_1v_240m_std'
    ]
    # anglez - 基础统计特征
    features += [
        # 计算滚动平均值（绝对值）
        pl.col('anglez').rolling_mean(12 * 240, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_240m_mean'),
        # 计算滚动最大值（绝对值）
        pl.col('anglez').rolling_max(12 * 240, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_240m_max'),
        # 计算滚动标准差（绝对值）
        pl.col('anglez').rolling_std(12 * 240, center=False, min_samples=1).abs().cast(pl.UInt16).alias('anglez_240m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_240m_mean', 'anglez_240m_max', 'anglez_240m_std'
    ]
    # anglez - 一阶差分特征
    features += [
        # 计算一阶差分的滚动平均值（绝对值）
        (pl.col('anglez').diff().abs().rolling_mean(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_240m_mean'),
        # 计算一阶差分的滚动最大值（绝对值）
        (pl.col('anglez').diff().abs().rolling_max(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_240m_max'),
        # 计算一阶差分的滚动标准差（绝对值）
        (pl.col('anglez').diff().abs().rolling_std(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.UInt32).alias('anglez_1v_240m_std')
    ]
    # 更新特征列名列表
    feature_cols += [ 
        'anglez_1v_240m_mean', 'anglez_1v_240m_max', 'anglez_1v_240m_std'
    ]

    id_cols = ['series_id', 'step', 'timestamp']  # 标识列
    
    return features, feature_cols, id_cols

def create_features_float():
    """
    创建特征和特征列名列表
    
    返回:
    features: 特征变换列表
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    """
    #(一阶差分(1v))滚动平均值(mean)/最大值(max)/标准差(std)
    # 初始化特征列表，先加入小时特征
    features, feature_cols = [pl.col('hour')], ['hour']
    #features, feature_cols = [pl.col('hour'), pl.col('minute')], ['hour', 'minute']  
    #features, feature_cols = [(pl.col('hour') * 60 + pl.col('minute')).cast(pl.UInt16).alias('time_data')], ['time_data']    # 为不同时间窗口创建特征
    # 1分钟窗口特征
    # enmo - 基础统计特征
    #features += [pl.col('enmo').rolling_mean(12 * 1, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_1m_mean')]
    #feature_cols += ['enmo_1m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 1, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_1m_max')]
    #feature_cols += ['enmo_1m_max']
    features += [pl.col('enmo').rolling_std(12 * 1, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_1m_std')]
    feature_cols += ['enmo_1m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_1m_mean')]
    feature_cols += ['enmo_1v_1m_mean']
    features += [(pl.col('enmo').diff().abs().rolling_max(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_1m_max')]
    feature_cols += ['enmo_1v_1m_max']
    features += [(pl.col('enmo').diff().abs().rolling_std(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_1m_std')]
    feature_cols += ['enmo_1v_1m_std']
    # anglez - 基础统计特征
    #features += [pl.col('anglez').rolling_mean(12 * 1, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_1m_mean')]
    #feature_cols += ['anglez_1m_mean']
    features += [pl.col('anglez').rolling_max(12 * 1, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_1m_max')]
    feature_cols += ['anglez_1m_max']
    features += [pl.col('anglez').rolling_std(12 * 1, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_1m_std')]
    feature_cols += ['anglez_1m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_1m_mean')]
    feature_cols += ['anglez_1v_1m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_1m_max')]
    feature_cols += ['anglez_1v_1m_max']
    features += [(pl.col('anglez').diff().abs().rolling_std(12 * 1, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_1m_std')]
    feature_cols += ['anglez_1v_1m_std']

    # 2分钟窗口特征
    # enmo - 基础统计特征
    #features += [pl.col('enmo').rolling_mean(12 * 2, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_2m_mean')]
    #feature_cols += ['enmo_2m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 2, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_2m_max')]
    #feature_cols += ['enmo_2m_max']
    #features += [pl.col('enmo').rolling_std(12 * 2, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_2m_std')]
    #feature_cols += ['enmo_2m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 2, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_2m_mean')]
    feature_cols += ['enmo_1v_2m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 2, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_2m_max')]
    #feature_cols += ['enmo_1v_2m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 2, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_2m_std')]
    #feature_cols += ['enmo_1v_2m_std']
    # anglez - 基础统计特征
    features += [pl.col('anglez').rolling_mean(12 * 2, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_2m_mean')]
    feature_cols += ['anglez_2m_mean'] 
    #features += [pl.col('anglez').rolling_max(12 * 2, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_2m_max')]
    #feature_cols += ['anglez_2m_max']
    features += [pl.col('anglez').rolling_std(12 * 2, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_2m_std')]
    feature_cols += ['anglez_2m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 2, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_2m_mean')]
    feature_cols += ['anglez_1v_2m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 2, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_2m_max')]
    feature_cols += ['anglez_1v_2m_max']
    #features += [(pl.col('anglez').diff().abs().rolling_std(12 * 2, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_2m_std')]
    #feature_cols += ['anglez_1v_2m_std']

    # 3分钟窗口特征
    # enmo - 基础统计特征
    #features += [pl.col('enmo').rolling_mean(12 * 3, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_3m_mean')]
    #feature_cols += ['enmo_3m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 3, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_3m_max')]
    #feature_cols += ['enmo_3m_max']
    #features += [pl.col('enmo').rolling_std(12 * 3, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_3m_std')]
    #feature_cols += ['enmo_3m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 3, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_3m_mean')]
    feature_cols += ['enmo_1v_3m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 3, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_3m_max')]
    #feature_cols += ['enmo_1v_3m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 3, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_3m_std')]
    #feature_cols += ['enmo_1v_3m_std']
    # anglez - 基础统计特征
    features += [pl.col('anglez').rolling_mean(12 * 3, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_3m_mean')]
    feature_cols += ['anglez_3m_mean'] 
    features += [pl.col('anglez').rolling_max(12 * 3, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_3m_max')]
    feature_cols += ['anglez_3m_max']
    features += [pl.col('anglez').rolling_std(12 * 3, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_3m_std')]
    feature_cols += ['anglez_3m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 3, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_3m_mean')]
    feature_cols += ['anglez_1v_3m_mean']
    #features += [(pl.col('anglez').diff().abs().rolling_max(12 * 3, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_3m_max')]
    #feature_cols += ['anglez_1v_3m_max']
    features += [(pl.col('anglez').diff().abs().rolling_std(12 * 3, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_3m_std')]
    feature_cols += ['anglez_1v_3m_std']

    # 4分钟窗口特征
    # enmo - 基础统计特征
    features += [pl.col('enmo').rolling_mean(12 * 4, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_4m_mean')]
    feature_cols += ['enmo_4m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 4, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_4m_max')]
    #feature_cols += ['enmo_4m_max']
    #features += [pl.col('enmo').rolling_std(12 * 4, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_4m_std')]
    #feature_cols += ['enmo_4m_std']
    # enmo - 一阶差分特征
    #features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 4, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_4m_mean')]
    #feature_cols += ['enmo_1v_4m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 4, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_4m_max')]
    #feature_cols += ['enmo_1v_4m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 4, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_4m_std')]
    #feature_cols += ['enmo_1v_4m_std']
    # anglez - 基础统计特征
    features += [pl.col('anglez').rolling_mean(12 * 4, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_4m_mean')]
    feature_cols += ['anglez_4m_mean'] 
    #features += [pl.col('anglez').rolling_max(12 * 4, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_4m_max')]
    #feature_cols += ['anglez_4m_max']
    #features += [pl.col('anglez').rolling_std(12 * 4, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_4m_std')]
    #feature_cols += ['anglez_4m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 4, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_4m_mean')]
    feature_cols += ['anglez_1v_4m_mean']
    #features += [(pl.col('anglez').diff().abs().rolling_max(12 * 4, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_4m_max')]
    #feature_cols += ['anglez_1v_4m_max']
    #features += [(pl.col('anglez').diff().abs().rolling_std(12 * 4, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_4m_std')]
    #feature_cols += ['anglez_1v_4m_std']

    # 5分钟窗口特征
    # enmo - 基础统计特征
    #features += [pl.col('enmo').rolling_mean(12 * 5, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_5m_mean')]
    #feature_cols += ['enmo_5m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 5, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_5m_max')]
    #feature_cols += ['enmo_5m_max']
    #features += [pl.col('enmo').rolling_std(12 * 5, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_5m_std')]
    #feature_cols += ['enmo_5m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_5m_mean')]
    feature_cols += ['enmo_1v_5m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_5m_max')]
    #feature_cols += ['enmo_1v_5m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_5m_std')]
    #feature_cols += ['enmo_1v_5m_std']
    # anglez - 基础统计特征
    #features += [pl.col('anglez').rolling_mean(12 * 5, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_5m_mean')]
    #feature_cols += ['anglez_5m_mean']
    #features += [pl.col('anglez').rolling_max(12 * 5, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_5m_max')]
    #feature_cols += ['anglez_5m_max']
    #features += [pl.col('anglez').rolling_std(12 * 5, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_5m_std')]
    #feature_cols += ['anglez_5m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_5m_mean')]
    feature_cols += ['anglez_1v_5m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_5m_max')]
    feature_cols += ['anglez_1v_5m_max']
    #features += [(pl.col('anglez').diff().abs().rolling_std(12 * 5, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_5m_std')]
    #feature_cols += ['anglez_1v_5m_std']

    # 10分钟窗口特征
    # enmo - 基础统计特征
    features += [pl.col('enmo').rolling_mean(12 * 10, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_10m_mean')]
    feature_cols += ['enmo_10m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 10, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_10m_max')]
    #feature_cols += ['enmo_10m_max']
    #features += [pl.col('enmo').rolling_std(12 * 10, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_10m_std')]
    #feature_cols += ['enmo_10m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 10, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_10m_mean')]
    feature_cols += ['enmo_1v_10m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 10, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_10m_max')]
    #feature_cols += ['enmo_1v_10m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 10, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_10m_std')]
    #feature_cols += ['enmo_1v_10m_std']
    # anglez - 基础统计特征
    features += [pl.col('anglez').rolling_mean(12 * 10, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_10m_mean')]
    feature_cols += ['anglez_10m_mean']
    #features += [pl.col('anglez').rolling_max(12 * 10, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_10m_max')]
    #feature_cols += ['anglez_10m_max']
    #features += [pl.col('anglez').rolling_std(12 * 10, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_10m_std')]
    #feature_cols += ['anglez_10m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 10, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_10m_mean')]
    feature_cols += ['anglez_1v_10m_mean']
    #features += [(pl.col('anglez').diff().abs().rolling_max(12 * 10, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_10m_max')]
    #feature_cols += ['anglez_1v_10m_max']
    features += [(pl.col('anglez').diff().abs().rolling_std(12 * 10, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_10m_std')]
    feature_cols += ['anglez_1v_10m_std']

    # 15分钟窗口特征
    # enmo - 基础统计特征
    features += [pl.col('enmo').rolling_mean(12 * 15, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_15m_mean')]
    feature_cols += ['enmo_15m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 15, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_15m_max')]
    #feature_cols += ['enmo_15m_max']
    #features += [pl.col('enmo').rolling_std(12 * 15, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_15m_std')]
    #feature_cols += ['enmo_15m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 15, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_15m_mean')]
    feature_cols += ['enmo_1v_15m_mean']
    features += [(pl.col('enmo').diff().abs().rolling_max(12 * 15, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_15m_max')]
    feature_cols += ['enmo_1v_15m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 15, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_15m_std')]
    #feature_cols += ['enmo_1v_15m_std']
    # anglez - 基础统计特征
    features += [pl.col('anglez').rolling_mean(12 * 15, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_15m_mean')]
    feature_cols += ['anglez_15m_mean']
    #features += [pl.col('anglez').rolling_max(12 * 15, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_15m_max')]
    #feature_cols += ['anglez_15m_max']
    #features += [pl.col('anglez').rolling_std(12 * 15, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_15m_std')]
    #feature_cols += ['anglez_15m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 15, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_15m_mean')]
    feature_cols += ['anglez_1v_15m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 15, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_15m_max')]
    feature_cols += ['anglez_1v_15m_max']
    #features += [(pl.col('anglez').diff().abs().rolling_std(12 * 15, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_15m_std')]
    #feature_cols += ['anglez_1v_15m_std']

    # 30分钟窗口特征
    # enmo - 基础统计特征
    features += [pl.col('enmo').rolling_mean(12 * 30, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_30m_mean')]
    feature_cols += ['enmo_30m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 30, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_30m_max')]
    #feature_cols += ['enmo_30m_max']
    features += [pl.col('enmo').rolling_std(12 * 30, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_30m_std')]
    feature_cols += ['enmo_30m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_30m_mean')]
    feature_cols += ['enmo_1v_30m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_30m_max')]
    #feature_cols += ['enmo_1v_30m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_30m_std')]
    #feature_cols += ['enmo_1v_30m_std']
    # anglez - 基础统计特征
    #features += [pl.col('anglez').rolling_mean(12 * 30, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_30m_mean')]
    #feature_cols += ['anglez_30m_mean']
    #features += [pl.col('anglez').rolling_max(12 * 30, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_30m_max')]
    #feature_cols += ['anglez_30m_max']
    #features += [pl.col('anglez').rolling_std(12 * 30, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_30m_std')]
    #feature_cols += ['anglez_30m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_30m_mean')]
    feature_cols += ['anglez_1v_30m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_30m_max')]
    feature_cols += ['anglez_1v_30m_max']
    features += [(pl.col('anglez').diff().abs().rolling_std(12 * 30, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_30m_std')]
    feature_cols += ['anglez_1v_30m_std']

    # 60分钟窗口特征
    # enmo - 基础统计特征
    features += [pl.col('enmo').rolling_mean(12 * 60, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_60m_mean')]
    feature_cols += ['enmo_60m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 60, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_60m_max')]
    #feature_cols += ['enmo_60m_max']
    features += [pl.col('enmo').rolling_std(12 * 60, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_60m_std')]
    feature_cols += ['enmo_60m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_60m_mean')]
    feature_cols += ['enmo_1v_60m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_60m_max')]
    #feature_cols += ['enmo_1v_60m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_60m_std')]
    #feature_cols += ['enmo_1v_60m_std']
    # anglez - 基础统计特征
    #features += [pl.col('anglez').rolling_mean(12 * 60, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_60m_mean')]
    #feature_cols += ['anglez_60m_mean']
    #features += [pl.col('anglez').rolling_max(12 * 60, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_60m_max')]
    #feature_cols += ['anglez_60m_max']
    features += [pl.col('anglez').rolling_std(12 * 60, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_60m_std')]
    feature_cols += ['anglez_60m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_60m_mean')]
    feature_cols += ['anglez_1v_60m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_60m_max')]
    feature_cols += ['anglez_1v_60m_max']
    features += [(pl.col('anglez').diff().abs().rolling_std(12 * 60, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_60m_std')]
    feature_cols += ['anglez_1v_60m_std']

    # 120分钟窗口特征
    # enmo - 基础统计特征
    features += [pl.col('enmo').rolling_mean(12 * 120, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_120m_mean')]
    feature_cols += ['enmo_120m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 120, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_120m_max')]
    #feature_cols += ['enmo_120m_max']
    features += [pl.col('enmo').rolling_std(12 * 120, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_120m_std')]
    feature_cols += ['enmo_120m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_120m_mean')]
    feature_cols += ['enmo_1v_120m_mean']
    features += [(pl.col('enmo').diff().abs().rolling_max(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_120m_max')]
    feature_cols += ['enmo_1v_120m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_120m_std')]
    #feature_cols += ['enmo_1v_120m_std']
    # anglez - 基础统计特征
    #features += [pl.col('anglez').rolling_mean(12 * 120, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_120m_mean')]
    #feature_cols += ['anglez_120m_mean']
    features += [pl.col('anglez').rolling_max(12 * 120, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_120m_max')]
    feature_cols += ['anglez_120m_max']
    features += [pl.col('anglez').rolling_std(12 * 120, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_120m_std')]
    feature_cols += ['anglez_120m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_120m_mean')]
    feature_cols += ['anglez_1v_120m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_120m_max')]
    feature_cols += ['anglez_1v_120m_max']
    features += [(pl.col('anglez').diff().abs().rolling_std(12 * 120, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_120m_std')]
    feature_cols += ['anglez_1v_120m_std']

    # 240分钟窗口特征
    # enmo - 基础统计特征
    #features += [pl.col('enmo').rolling_mean(12 * 240, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_240m_mean')]
    #feature_cols += ['enmo_240m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 240, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_240m_max')]
    #feature_cols += ['enmo_240m_max']
    #features += [pl.col('enmo').rolling_std(12 * 240, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_240m_std')]
    #feature_cols += ['enmo_240m_std']
    # enmo - 一阶差分特征
    features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_240m_mean')]
    feature_cols += ['enmo_1v_240m_mean']
    features += [(pl.col('enmo').diff().abs().rolling_max(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_240m_max')]
    feature_cols += ['enmo_1v_240m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_240m_std')]
    #feature_cols += ['enmo_1v_240m_std']
    # anglez - 基础统计特征
    #features += [pl.col('anglez').rolling_mean(12 * 240, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_240m_mean')]
    #feature_cols += ['anglez_240m_mean']
    features += [pl.col('anglez').rolling_max(12 * 240, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_240m_max')]
    feature_cols += ['anglez_240m_max']
    features += [pl.col('anglez').rolling_std(12 * 240, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_240m_std')]
    feature_cols += ['anglez_240m_std']
    # anglez - 一阶差分特征
    features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_240m_mean')]
    feature_cols += ['anglez_1v_240m_mean']
    features += [(pl.col('anglez').diff().abs().rolling_max(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_240m_max')]
    feature_cols += ['anglez_1v_240m_max']
    features += [(pl.col('anglez').diff().abs().rolling_std(12 * 240, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_240m_std')]
    feature_cols += ['anglez_1v_240m_std']

    # 360分钟窗口特征
    # enmo - 基础统计特征
    #features += [pl.col('enmo').rolling_mean(12 * 360, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_360m_mean')]
    #feature_cols += ['enmo_360m_mean']
    #features += [pl.col('enmo').rolling_max(12 * 360, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_360m_max')]
    #feature_cols += ['enmo_360m_max']
    #features += [pl.col('enmo').rolling_std(12 * 360, center=False, min_samples=1).abs().cast(pl.Float32).alias('enmo_360m_std')]
    #feature_cols += ['enmo_360m_std']
    # enmo - 一阶差分特征
    #features += [(pl.col('enmo').diff().abs().rolling_mean(12 * 360, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_360m_mean')]
    #feature_cols += ['enmo_1v_360m_mean']
    #features += [(pl.col('enmo').diff().abs().rolling_max(12 * 360, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_360m_max')]
    #feature_cols += ['enmo_1v_360m_max']
    #features += [(pl.col('enmo').diff().abs().rolling_std(12 * 360, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('enmo_1v_360m_std')]
    #feature_cols += ['enmo_1v_360m_std']
    # anglez - 基础统计特征
    #features += [pl.col('anglez').rolling_mean(12 * 360, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_360m_mean')]
    #feature_cols += ['anglez_360m_mean']
    #features += [pl.col('anglez').rolling_max(12 * 360, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_360m_max')]
    #feature_cols += ['anglez_360m_max']
    #features += [pl.col('anglez').rolling_std(12 * 360, center=False, min_samples=1).abs().cast(pl.Float32).alias('anglez_360m_std')]
    #feature_cols += ['anglez_360m_std']
    # anglez - 一阶差分特征
    #features += [(pl.col('anglez').diff().abs().rolling_mean(12 * 360, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_360m_mean')]
    #feature_cols += ['anglez_1v_360m_mean']
    #features += [(pl.col('anglez').diff().abs().rolling_max(12 * 360, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_360m_max')]
    #feature_cols += ['anglez_1v_360m_max']
    #features += [(pl.col('anglez').diff().abs().rolling_std(12 * 360, center=False, min_samples=1)*10).abs().cast(pl.Float32).alias('anglez_1v_360m_std')]
    #feature_cols += ['anglez_1v_360m_std']

    id_cols = ['series_id', 'step', 'timestamp']
    
    return features, feature_cols, id_cols

def make_train_dataset_new(train_data, train_events, feature_cols, id_cols, drop_nulls=False):
    """
    创建训练数据集的改进版本，修复了一些问题
    
    参数:
    train_data: 训练时间序列数据
    train_events: 训练事件数据
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    drop_nulls: 是否删除没有事件记录的日期数据
    
    返回:
    X: 特征矩阵
    y: 标签向量
    """
    
    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()  # 初始化特征和标签数据框
    
    for idx in tqdm(series_ids):  # 遍历每个series_id
        
        # 标准化样本特征（排除时间特征）
        time_features = ['hour', 'minute']
        sample = train_data.filter(pl.col('series_id') == idx).with_columns(
            [pl.col(col).cast(pl.Float32) for col in feature_cols if col not in time_features]
        )   
        
        events = train_events.filter(pl.col('series_id') == idx)  # 获取当前series_id的事件数据
        
        if drop_nulls:
            # 移除没有事件记录的日期的数据点
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )
        
        # 添加特征数据
        X = X.vstack(sample[id_cols + feature_cols])  
        
        # 修复：使用is_not_null()检查空值
        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step').is_not_null()))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step').is_not_null()))['step'].to_list()
        
        # 修复：使用pl.sum_horizontal替代sum，并添加错误处理
        if onsets and wakeups and len(onsets) == len(wakeups):
            conditions = [(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in zip(onsets, wakeups)]
            y = y.vstack(sample.with_columns(
                pl.sum_horizontal(conditions).cast(pl.Boolean).alias('asleep')
            ).select('asleep'))
        else:
            # 如果没有有效的睡眠区间，创建全为False的列
            y = y.vstack(sample.with_columns(
                pl.lit(False).alias('asleep')
            ).select('asleep'))
    
    y = y.to_numpy().ravel()  # 将标签转换为一维数组
    
    return X, y

def get_events_new(series, classifier, feature_cols, id_cols, min_sleep_duration=12 * 30):
    """
    将分类器的预测结果转换为睡眠事件（onset和wakeup），并生成提交格式的数据框
    实时检测版本：移除事件配对逻辑，支持实时部署到STM32
    
    参数:
    series: 时间序列数据
    classifier: 训练好的分类器模型
    feature_cols: 特征列名列表
    id_cols: 标识列列表
    min_sleep_duration: 最小睡眠周期长度（步数），默认30分钟（12*30步）
    
    返回:
    events: 包含预测事件的DataFrame，格式符合提交要求
    """
    # 只需要确定需要转换类型的列（排除时间特征）
    time_features = ['hour', 'minute']
    scale_cols = [col for col in feature_cols if col not in time_features]
    # 获取所有 series_id
    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    
    # 优化3: 使用列表收集事件数据，最后一次性创建 DataFrame，减少 vstack 操作
    event_data = []
    
    for idx in tqdm(series_ids):  # 遍历每个series_id，显示进度条
        # 优化4: 准备数据并标准化特征（使用预先计算的标准差）
        X = series.filter(pl.col('series_id') == idx).select(id_cols + feature_cols)
        
        if scale_cols:
            X = X.with_columns(
                [pl.col(col).cast(pl.Float32) for col in scale_cols]
            )
        
        # 使用分类器进行预测，获取类别和概率
        preds, probs = classifier.predict(X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]
        
        # 将预测结果添加到数据框
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'), 
            pl.lit(probs).alias('probability')
        )
        
        # 优化5: 使用 NumPy 进行差异计算，提高事件检测效率
        preds_array = X['prediction'].to_numpy()
        steps_array = X['step'].to_numpy()
        probs_array = X['probability'].to_numpy()
        timestamps_array = X['timestamp'].to_numpy()
        
        # 计算差异
        diffs = np.diff(preds_array)
        
        # 检测睡眠开始和结束事件
        onset_indices = np.where(diffs > 0)[0]
        wakeup_indices = np.where(diffs < 0)[0]
        
        # 处理 onset 事件（实时检测）
        for i in onset_indices:
            # 获取事件的 step 和 timestamp
            step = steps_array[i]
            timestamp = timestamps_array[i]
            
            # 实时窗口验证：检查后续一段时间内是否持续为睡眠状态
            # 计算验证窗口的结束索引
            window_end = min(i + min_sleep_duration, len(preds_array) - 1)
            
            # 检查窗口内的预测值是否都为1（睡眠状态）
            if np.all(preds_array[i+1:window_end+1] == 1):
                # 计算窗口内的平均概率作为分数
                score = np.mean(probs_array[i:window_end+1])
                
                # 添加到列表
                event_data.append({
                    'series_id': idx, 
                    'step': step, 
                    'event': 'onset', 
                    'score': score,
                    'timestamp': timestamp
                })
        
        # 处理 wakeup 事件（实时检测）
        for i in wakeup_indices:
            # 获取事件的 step 和 timestamp
            step = steps_array[i]
            timestamp = timestamps_array[i]
            
            # 实时窗口验证：检查后续一段时间内是否持续为清醒状态
            # 计算验证窗口的结束索引
            window_end = min(i + min_sleep_duration, len(preds_array) - 1)
            
            # 检查窗口内的预测值是否都为0（清醒状态）
            if np.all(preds_array[i+1:window_end+1] == 0):
                # 计算窗口内的平均概率作为分数（使用1 - 概率表示清醒程度）
                score = np.mean(1 - probs_array[i:window_end+1])
                
                # 添加到列表
                event_data.append({
                    'series_id': idx, 
                    'step': step, 
                    'event': 'wakeup', 
                    'score': score,
                    'timestamp': timestamp
                })
    
    # 优化6: 最后一次性创建 DataFrame，而不是多次 vstack
    if event_data:
        events = pl.DataFrame(event_data)
    else:
        events = pl.DataFrame(schema={'series_id': str, 'step': int, 'event': str, 'score': float, 'timestamp': str})
    
    # 添加行ID列
    events = events.to_pandas().reset_index().rename(columns={'index': 'row_id'})
    
    return events
