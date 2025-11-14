# -*- coding: utf-8 -*-
"""
流程：
1) 对每个模型先用 GridSearchCV (5-fold) 搜索最优超参数（refit='R2'）
2) 用该最优超参数构建模型，进行 10 次重复的 5 折 CV（每次 repeat 的划分不同）
3) 记录每折 R2/MSE、每次平均；选择平均 R2 最好的 repeat，保存该 repeat 的 train/val 预测（分开文件）
4) 为每个模型保存 summary txt（模型名、最优超参、每次的 5 折 R2/MSE 与每次平均值、总体均值±std）
不保存模型 pkl 文件。
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

# ========= 路径设置（按需修改） =========
input_file_path = r"C:\Users\xiaoping\Desktop\fushi\data\result\input_data\merge_features_54_0.csv"
target_file_path = r'C:\Users\xiaoping\Desktop\fushi\data\result\input_data\enzyme_selected.csv'
txt_dir = r'C:\Users\xiaoping\Desktop\fushi\data\result\model_output\POD\txt'
csv_dir = r'C:\Users\xiaoping\Desktop\fushi\data\result\model_output\POD\csv'

os.makedirs(txt_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# ========= 数据读取与对齐检查 =========
features_data = pd.read_csv(input_file_path)
target_data = pd.read_csv(target_file_path)

# 强烈建议检查并确保 features_data 与 target_data 行一一对应、顺序一致
assert len(features_data) == len(target_data), "ERROR: features_data 与 target_data 行数不一致，请先对齐（merge或排序）"

# 以 features 的索引作为 SampleID（更稳妥）
sample_ids = np.array(features_data.index)

# 特征与目标（按你的原始处理）
X = features_data.drop(['Time(h)', 'Position', 'BI'], axis=1)
y = target_data['POD']

# ========= 模型定义与参数 =========
models_and_params = {
    'KNN': (
        KNeighborsRegressor(),
        {'model__n_neighbors': [3, 5, 7, 9, 11]}
    ),
    'SVM': (
        SVR(),
        {'model__C': [0.1, 1, 10],
         'model__gamma': ['scale', 'auto'],
         'model__kernel': ['rbf', 'poly', 'sigmoid']}
    ),
    'ElasticNet': (
        ElasticNet(max_iter=5000),
        {'model__alpha': [0.001, 0.01, 0.1, 1],
         'model__l1_ratio': [0.1, 0.5, 0.9]}
    ),
    'DecisionTree': (
        DecisionTreeRegressor(random_state=42),
        {'model__max_depth': [None, 5, 10, 20],
         'model__min_samples_split': [2, 4, 6],
         'model__min_samples_leaf': [1, 2, 4]}
    ),
    'MLP': (
        MLPRegressor(max_iter=2000, random_state=42),
        {'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
         'model__alpha': [0.0001, 0.001, 0.01],
         'model__learning_rate_init': [0.001, 0.01]}
    )
}

# ========= 超参搜索的 CV 配置（用于 GridSearch） =========
# 这里用固定的 random_state 保证可复现的超参搜索
gs_kf = KFold(n_splits=5, shuffle=True, random_state=42)

# scoring：R2 与 MSE（MSE 需要 greater_is_better=False）
scoring = {'R2': 'r2', 'MSE': make_scorer(mean_squared_error, greater_is_better=False)}

# ========= 对每个模型：先 GridSearch 再 10 次重复 5 折 CV =========
for name, (base_model, param_grid) in models_and_params.items():
    print(f"\n===== 处理模型：{name} =====")
    txt_file = os.path.join(txt_dir, f'{name}_results.txt')

    # ========== 1) 网格搜索（使用 5-fold，refit='R2'） ==========
    pipe = Pipeline([('model', base_model)])
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=gs_kf,
        scoring=scoring,
        refit='R2',
        n_jobs=8,
        return_train_score=False
    )
    print(" 开始 GridSearchCV（5-fold）以确定最优超参数...（可复现 random_state=42）")
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    # best_estimator_ 已被 refit，在整个 X,y 上训练（GridSearchCV 的 refit 行为）
    # 但我们在后续 fold 训练时仍用 clone(best_estimator_) 避免覆盖其状态
    best_estimator = grid_search.best_estimator_

    print(f" 最优超参数 (GridSearchCV 得到): {best_params}")

    # ========== 2) 用最优超参进行 10 次重复的 5 折 CV ==========
    num_repeats = 10
    num_folds = 5

    # 用于记录每次 repeat 的每折指标，以及每次平均
    repeats_fold_metrics = []  # 列表元素为 dict：{'repeat':i, 'fold_r2':[...], 'fold_mse':[...], 'mean_r2':..., 'mean_mse':...}

    # 记录最佳 repeat 的 train/val 预测表
    best_repeat_mean_r2 = -np.inf
    best_repeat_train_df = None
    best_repeat_val_df = None
    best_repeat_index = None

    for repeat in range(num_repeats):
        print(f"  -> repeat {repeat+1}/{num_repeats}")
        # 每次 repeat 使用不同的随机种子，确保分割不同（可复现）
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat)

        fold_r2_list = []
        fold_mse_list = []

        # 为了分开保存 train/val，在当前 repeat 内构建两份记录表（累积每个 fold 的 train / val 预测）
        train_records = []
        val_records = []

        # 对每折：用 clone(best_estimator) 避免覆盖 best_estimator 的状态
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            id_train, id_val = sample_ids[train_idx], sample_ids[val_idx]

            model_clone = clone(best_estimator)  # 拷贝结构与超参
            model_clone.fit(X_train, y_train)    # 在当前 fold 的 train 上训练
            y_train_pred = model_clone.predict(X_train)
            y_val_pred = model_clone.predict(X_val)

            # 评估（对 val）
            r2 = r2_score(y_val, y_val_pred)
            mse = mean_squared_error(y_val, y_val_pred)
            fold_r2_list.append(r2)
            fold_mse_list.append(mse)

            # 保存 train / val 预测记录（后续用于保存最佳 repeat 的 csv）
            # 注意：Train 可能会在不同 fold 中重复出现（这是自然的）；我们保存的是“该 repeat 下每折的 train 预测”
            train_records.extend(zip(id_train,
                                     [fold_idx]*len(train_idx),
                                     ['Train']*len(train_idx),
                                     y_train.values,
                                     y_train_pred,
                                     [name]*len(train_idx)))
            val_records.extend(zip(id_val,
                                   [fold_idx]*len(val_idx),
                                   ['Val']*len(val_idx),
                                   y_val.values,
                                   y_val_pred,
                                   [name]*len(val_idx)))

        mean_r2 = float(np.mean(fold_r2_list))
        mean_mse = float(np.mean(fold_mse_list))

        repeats_fold_metrics.append({
            'repeat': repeat+1,
            'fold_r2': fold_r2_list,
            'fold_mse': fold_mse_list,
            'mean_r2': mean_r2,
            'mean_mse': mean_mse
        })

        # 若该 repeat 平均 R2 最好，则保存其 train/val 预测表（分开）
        if mean_r2 > best_repeat_mean_r2:
            best_repeat_mean_r2 = mean_r2
            best_repeat_index = repeat + 1
            best_repeat_train_df = pd.DataFrame(train_records, columns=['SampleID', 'Fold', 'DatasetType', 'True', 'Pred', 'ModelName'])
            best_repeat_val_df = pd.DataFrame(val_records, columns=['SampleID', 'Fold', 'DatasetType', 'True', 'Pred', 'ModelName'])

    # ========== 3) 将结果写入 TXT 与 CSV ==========
    # 构建 summary 文本
    lines = []
    lines.append(f"===== 模型：{name} =====")
    lines.append(f"最优超参数（GridSearchCV 得到，refit='R2'）：{best_params}")
    lines.append("")
    lines.append("以下为 10 次重复的五折交叉验证结果（每次显示每折的 R² 与 MSE 以及该次平均）：")
    for item in repeats_fold_metrics:
        r = item['repeat']
        lines.append(f"\n====== 第 {r} 次重复 ======")
        for fi, (fr2, fmse) in enumerate(zip(item['fold_r2'], item['fold_mse']), start=1):
            lines.append(f"Fold {fi}: R²={fr2:.6f}, MSE={fmse:.8f}")
        lines.append(f"平均 R²: {item['mean_r2']:.6f}, 平均 MSE: {item['mean_mse']:.8f}")

    # 总体统计
    mean_all_r2 = np.mean([it['mean_r2'] for it in repeats_fold_metrics])
    std_all_r2 = np.std([it['mean_r2'] for it in repeats_fold_metrics])
    mean_all_mse = np.mean([it['mean_mse'] for it in repeats_fold_metrics])
    std_all_mse = np.std([it['mean_mse'] for it in repeats_fold_metrics])

    lines.append("\n====== 10 次总体统计 ======")
    lines.append(f"平均 R²: {mean_all_r2:.6f} ± {std_all_r2:.6f}")
    lines.append(f"平均 MSE: {mean_all_mse:.8f} ± {std_all_mse:.8f}")
    lines.append("")
    lines.append(f"最佳重复（基于平均 R²）：第 {best_repeat_index} 次")
    lines.append(f"最佳重复 平均 R²: {best_repeat_mean_r2:.6f}")

    # 写 txt
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # 保存最佳 repeat 的 train / val CSV（若存在）
    if best_repeat_train_df is not None and best_repeat_val_df is not None:
        # 文件名示例：MLP_best_repeat_train.csv / MLP_best_repeat_val.csv
        best_repeat_train_df.to_csv(os.path.join(csv_dir, f"{name}_best_repeat_train.csv"), index=False)
        best_repeat_val_df.to_csv(os.path.join(csv_dir, f"{name}_best_repeat_val.csv"), index=False)
        print(f" 已保存最佳 repeat({best_repeat_index}) 的 train/val CSV 到：{csv_dir}")
    else:
        print(" 未产生最佳 repeat 的预测表（unexpected）")

    print(f" 关于 {name} 的 summary 已保存到：{txt_file}")
    print(f" 完成模型：{name}。\n")
