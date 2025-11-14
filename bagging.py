# -*- coding: utf-8 -*-
"""
Bagging 集成学习（KNN + SVM + ElasticNet + DecisionTree + MLP）：
1) 对每个子模型先用 GridSearchCV 搜索最优超参数（5-fold, refit='R2'）
2) 使用最优超参数进行 Bagging 集成：
   - 进行 10 次重复，每次 5-fold 交叉验证
   - 每折中对训练集进行 bootstrap（有放回采样）
   - 用 5 个子模型分别训练并预测，选出该折 R² 最高的模型为该折结果
3) 保存每次 repeat 的平均 R2/MSE，选择最佳 repeat 保存 train/val CSV
4) 生成 summary TXT
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import random
random.seed(42)
np.random.seed(42)

# ========= 路径设置 =========
txt_dir = r"C:\Users\xiaoping\Desktop\fushi\data\result\model_output\POD\txt"
csv_dir = r"C:\Users\xiaoping\Desktop\fushi\data\result\model_output\POD\csv"
os.makedirs(txt_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# ========= 数据读取 =========
features_data = pd.read_csv(r"C:\Users\xiaoping\Desktop\fushi\data\result\input_data\merge_features_54_0.csv")
target_data = pd.read_csv(r'C:\Users\xiaoping\Desktop\fushi\data\result\input_data\enzyme_selected.csv')

X = features_data.drop(['Time(h)', 'Position', 'BI'], axis=1)
y = target_data['POD'].astype(float)
sample_ids = np.array(features_data.index)

# ========= 子模型定义与超参数 =========
models_and_params = {
    'KNN': (KNeighborsRegressor(), {'model__n_neighbors':[3,5,7,9,11]}),
    'SVM': (SVR(), {'model__C':[0.1,1,10], 'model__gamma':['scale','auto'], 'model__kernel':['rbf','poly','sigmoid']}),
    'ElasticNet': (ElasticNet(max_iter=5000), {'model__alpha':[0.001,0.01,0.1,1], 'model__l1_ratio':[0.1,0.5,0.9]}),
    'DecisionTree': (DecisionTreeRegressor(random_state=42), {'model__max_depth':[None,5,10,20], 'model__min_samples_split':[2,4,6], 'model__min_samples_leaf':[1,2,4]}),
    'MLP': (MLPRegressor(max_iter=2000, random_state=42), {'model__hidden_layer_sizes':[(50,),(100,),(50,50)], 'model__alpha':[0.0001,0.001,0.01], 'model__learning_rate_init':[0.001,0.01]})
}

# scoring
scoring = {'R2':'r2', 'MSE': make_scorer(mean_squared_error, greater_is_better=False)}
n_jobs = 8
# ========= 1) 对每个子模型进行 GridSearchCV =========
best_estimators = {}
for name, (model, param_grid) in models_and_params.items():
    print(f"正在 GridSearchCV 子模型：{name}")
    pipe = Pipeline([('model', model)])
    gs = GridSearchCV(pipe, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                  scoring=scoring, refit='R2', n_jobs=4, return_train_score=False)
    gs.fit(X, y)
    best_estimators[name] = gs.best_estimator_
    print(f" 最优超参：{gs.best_params_}")

# ========= 2) Bagging + 10 次重复 5-fold CV =========
num_repeats = 10
num_folds = 5
repeats_metrics = []
best_repeat_r2 = -np.inf
best_repeat_idx = None
best_train_df = None
best_val_df = None
num_bootstrap = 10

for repeat in range(num_repeats):
    fold_best_models = []
    print(f"\n-> Repeat {repeat+1}/{num_repeats}")
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat)
    fold_r2_list, fold_mse_list = [], []
    train_records, val_records = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        id_train, id_val = sample_ids[train_idx], sample_ids[val_idx]

        best_fold_r2 = -np.inf
        best_fold_mse = np.inf
        best_fold_model_name = None
        best_y_val_pred = None
        best_y_train_pred = None

        # 对每个子模型进行：在当前折训练集上做 num_bootstrap 次 bootstrap，
        # 收集每次的 train/val 预测与指标，最后对同一模型的 num_bootstrap 次预测做加权平均
        for model_name, base_model in best_estimators.items():
            all_val_preds = []    # shape (num_bootstrap, len(val_idx))
            all_train_preds = []  # shape (num_bootstrap, len(train_idx))
            all_r2s = []
            all_mses = []

            for b in range(num_bootstrap):
                # bootstrap 采样（在训练集上）
                bag_idx = np.random.choice(len(X_train), len(X_train), replace=True)
                X_bag, y_bag = X_train.iloc[bag_idx], y_train.iloc[bag_idx]

                model = clone(base_model)
                model.fit(X_bag, y_bag)

                # 验证集和原训练集的预测都要收集
                y_val_pred = model.predict(X_val)
                y_train_pred = model.predict(X_train)

                r2 = r2_score(y_val, y_val_pred)
                mse = mean_squared_error(y_val, y_val_pred)

                all_val_preds.append(y_val_pred)
                all_train_preds.append(y_train_pred)
                all_r2s.append(r2)
                all_mses.append(mse)

            # 收集完成 num_bootstrap 次后，做加权平均（按 R2 作为权重）
            all_val_preds = np.array(all_val_preds)    # shape (B, n_val)
            all_train_preds = np.array(all_train_preds)  # shape (B, n_train)
            all_r2s = np.array(all_r2s)
            all_mses = np.array(all_mses)

            # 避免负权重或全为0情况：只保留 >=0 的 R2，若权重和为0则用等权平均
            weights = np.clip(all_r2s, a_min=0, a_max=None)
            if weights.sum() == 0:
                weights = None  # np.average 支持 None 表示等权平均
                y_val_pred_avg = np.mean(all_val_preds, axis=0)
                y_train_pred_avg = np.mean(all_train_preds, axis=0)
            else:
                y_val_pred_avg = np.average(all_val_preds, axis=0, weights=weights)
                y_train_pred_avg = np.average(all_train_preds, axis=0, weights=weights)

            # 计算该模型在该折的加权平均指标
            r2_avg = r2_score(y_val, y_val_pred_avg)
            mse_avg = mean_squared_error(y_val, y_val_pred_avg)

            # 如果该模型的加权平均 r2 比当前折最优高，则更新折最优
            if r2_avg > best_fold_r2:
                best_fold_r2 = r2_avg
                best_fold_mse = mse_avg
                best_fold_model_name = model_name
                best_y_val_pred = y_val_pred_avg
                best_y_train_pred = y_train_pred_avg

        # 记录该折的最优模型信息
        fold_best_models.append(best_fold_model_name)
        fold_r2_list.append(best_fold_r2)
        fold_mse_list.append(best_fold_mse)

        # 保存当前折的最优模型预测结果（对应训练集和验证集）
        # 注意：train_records 期望与原训练集每个样本一一对应，best_y_train_pred 长度为 len(train_idx)
        train_records.extend(zip(id_train, [fold_idx]*len(train_idx), ['Train']*len(train_idx),
                                 y_train.values, best_y_train_pred, [best_fold_model_name]*len(train_idx)))
        val_records.extend(zip(id_val, [fold_idx]*len(val_idx), ['Val']*len(val_idx),
                               y_val.values, best_y_val_pred, [best_fold_model_name]*len(val_idx)))

        print(f"Repeat {repeat+1} | Fold {fold_idx} 最优模型: {best_fold_model_name}, R²={best_fold_r2:.4f}, MSE={best_fold_mse:.6f}")

    # fold 循环结束，计算该 repeat 的平均指标
    mean_r2 = float(np.mean(fold_r2_list))
    mean_mse = float(np.mean(fold_mse_list))
    repeats_metrics.append({'repeat': repeat+1, 'fold_r2': fold_r2_list, 'fold_mse': fold_mse_list, 'fold_models': fold_best_models, 'mean_r2': mean_r2, 'mean_mse': mean_mse})

    # 如果是当前最优 repeat，则保存对应的 train/val 记录
    if mean_r2 > best_repeat_r2:
        best_repeat_r2 = mean_r2
        best_repeat_idx = repeat+1
        best_train_df = pd.DataFrame(train_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])
        best_val_df = pd.DataFrame(val_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])

# ========= 3) 写 TXT =========
txt_file = os.path.join(txt_dir, "Bagging_best_repeat_summary.txt")
lines = ["===== Bagging 集成学习（KNN+SVM+ElasticNet+DecisionTree+MLP） =====", ""]
for item in repeats_metrics:
    lines.append(f"\n====== 第 {item['repeat']} 次重复 ======")
    for fi, (fr2, fmse, fmodel) in enumerate(zip(item['fold_r2'], item['fold_mse'], item['fold_models']), start=1):
        lines.append(f"Fold {fi}: 最优模型={fmodel}, R²={fr2:.6f}, MSE={fmse:.8f}")
    lines.append(f"平均 R²: {item['mean_r2']:.6f}, 平均 MSE: {item['mean_mse']:.8f}")

mean_all_r2 = np.mean([it['mean_r2'] for it in repeats_metrics])
std_all_r2 = np.std([it['mean_r2'] for it in repeats_metrics])
mean_all_mse = np.mean([it['mean_mse'] for it in repeats_metrics])
std_all_mse = np.std([it['mean_mse'] for it in repeats_metrics])
lines.append("\n====== 10 次总体统计 ======")
lines.append(f"平均 R²: {mean_all_r2:.6f} ± {std_all_r2:.6f}")
lines.append(f"平均 MSE: {mean_all_mse:.8f} ± {std_all_mse:.8f}")
lines.append(f"最佳重复（平均 R² 最大）：第 {best_repeat_idx} 次, 平均 R²={best_repeat_r2:.6f}")

with open(txt_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"Summary TXT 已保存到 {txt_file}")

# ========= 4) 保存最佳 repeat 的 train/val CSV =========
if best_train_df is not None and best_val_df is not None:
    best_train_df.to_csv(os.path.join(csv_dir, "Bagging_best_repeat_train.csv"), index=False)
    best_val_df.to_csv(os.path.join(csv_dir, "Bagging_best_repeat_val.csv"), index=False)
    print(f"最佳 repeat 的 train/val CSV 已保存到 {csv_dir}")
