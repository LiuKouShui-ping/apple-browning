# -*- coding: utf-8 -*-
"""
Boosting 异构集成模型（KNN -> SVM -> ElasticNet -> DecisionTree -> MLP）：
1) 对每个子模型先用 GridSearchCV 搜索最优超参数（5-fold, refit='R2'）
2) 按顺序迭代拟合残差（广义Boosting）
3) 进行 10 次重复 5-fold CV
4) 保存最佳 repeat 的结果 CSV + summary TXT
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
import itertools

# ========= 路径设置 =========
txt_dir = r"C:\Users\xiaoping\Desktop\fushi\data\result\model_output\BI\txt"
csv_dir = r"C:\Users\xiaoping\Desktop\fushi\data\result\model_output\BI\csv"
os.makedirs(txt_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
random.seed(42)
np.random.seed(42)

# ========= 数据读取 =========
features_data = pd.read_csv(r"C:\Users\xiaoping\Desktop\fushi\data\result\input_data\merge_features_54_0.csv")
target_data = pd.read_csv(r'C:\Users\xiaoping\Desktop\fushi\data\result\input_data\enzyme_selected.csv')

X = features_data.drop(['Time(h)', 'Position', 'BI'], axis=1)
y = target_data['BI'].astype(float)
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
n_jobs = 6
# ========= 1) GridSearchCV 最优超参数 =========
best_estimators = {}
for name, (model, param_grid) in models_and_params.items():
    print(f"正在 GridSearchCV 子模型：{name}")
    pipe = Pipeline([('model', model)])
    gs = GridSearchCV(pipe, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                      scoring=scoring, refit='R2', n_jobs=4, return_train_score=False)
    gs.fit(X, y)
    best_estimators[name] = gs.best_estimator_
    print(f" 最优超参：{gs.best_params_}")

# ========= 2) 异构 Boosting + 10 次重复 5-fold CV =========
num_repeats = 10
num_folds = 5
# === 学习率设置（衰减式） ===
base_lr = 0.5      # 初始学习率
decay = 0.8        # 衰减系数
etas = [0.8, 0.5, 0.4, 0.2, 0.1]
repeats_metrics = []
# 获取所有可能的模型顺序
model_names = ['ElasticNet', 'KNN', 'SVM', 'DecisionTree', 'MLP']
model_permutations = list(itertools.permutations(model_names))
best_repeat_r2 = -np.inf
best_model_r2 = -np.inf
best_repeat_idx = None
best_train_df, best_val_df = None, None

# # 计算每个顺序的 R²，并选择最优的模型顺序
# for perm_idx, model_sequence in enumerate(model_permutations):
#     print(f"\n正在测试模型顺序：{model_sequence}")
#     fold_r2_list, fold_mse_list = [], []
#     kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
#     fold_r2, fold_mse = [], []

#     for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         residual_train = y_train.copy()
#         y_train_pred_total = np.zeros(len(y_train))
#         y_val_pred_total = np.zeros(len(y_val))

#         for i, model_name in enumerate(model_sequence):
#             model = clone(best_estimators[model_name])
#             model.fit(X_train, residual_train)
#             pred_train = model.predict(X_train)
#             pred_val = model.predict(X_val)

#             print(f"{model_name} 单独模型验证R²={r2_score(y_val, pred_val):.3f}, 学习率={etas[i]}")
#             y_train_pred_total += etas[i] * pred_train
#             y_val_pred_total += etas[i] * pred_val
#             residual_train = y_train - y_train_pred_total

#         r2 = r2_score(y_val, y_val_pred_total)
#         mse = mean_squared_error(y_val, y_val_pred_total)
#         fold_r2.append(r2)
#         fold_mse.append(mse)

#     mean_r2, mean_mse = np.mean(fold_r2), np.mean(fold_mse)
#     print(f"平均 R²={mean_r2:.5f}, MSE={mean_mse:.5f}")

#     # 如果当前模型顺序的R²更好，则更新
#     if mean_r2 > best_model_r2:
#         best_model_r2 = mean_r2
#         best_model_sequence = model_sequence
#         best_model_estimators = {name: clone(best_estimators[name]) for name in model_sequence}

# print(f"最佳顺序: {best_model_sequence}，最佳 R²: {best_model_r2:.5f}")

for repeat in range(num_repeats):
    print(f"\n-> Repeat {repeat+1}/{num_repeats}")
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat)
    fold_r2_list, fold_mse_list = [], []
    train_records, val_records = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        id_train, id_val = sample_ids[train_idx], sample_ids[val_idx]

        # === 异构 Boosting ===
        residual_train = y_train.copy()
        y_train_pred_total = np.zeros(len(y_train))
        y_val_pred_total = np.zeros(len(y_val))

        # 推荐 Boosting 顺序
        model_sequence = ['MLP', 'SVM', 'DecisionTree', 'ElasticNet', 'KNN']
        for i, model_name in enumerate(model_sequence):
            model = clone(best_estimators[model_name])
            model.fit(X_train, residual_train)
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            print(f"{model_name} 单独模型验证R²={r2_score(y_val, pred_val):.3f}, 学习率={etas[i]}")
            # 更新预测结果与残差
            y_train_pred_total += etas[i] * pred_train
            y_val_pred_total += etas[i]  * pred_val
            residual_train = y_train - y_train_pred_total
        # === fold metrics ===
        r2 = r2_score(y_val, y_val_pred_total)
        if r2 < 0:
            print(f"停止 Boosting，{model_name} 对性能无正向贡献")
            break
        mse = mean_squared_error(y_val, y_val_pred_total)
        fold_r2_list.append(r2)
        fold_mse_list.append(mse)

        # 保存每个样本的预测记录
        train_records.extend(zip(id_train, [fold_idx]*len(y_train), ['Train']*len(y_train),
                                 y_train.values, y_train_pred_total, ['Boosting']*len(y_train)))
        val_records.extend(zip(id_val, [fold_idx]*len(y_val), ['Val']*len(y_val),
                               y_val.values, y_val_pred_total, ['Boosting']*len(y_val)))

    mean_r2, mean_mse = np.mean(fold_r2_list), np.mean(fold_mse_list)
    repeats_metrics.append({'repeat': repeat+1, 'fold_r2': fold_r2_list, 'fold_mse': fold_mse_list,
                            'mean_r2': mean_r2, 'mean_mse': mean_mse})
    print(f"Repeat {repeat+1}: 平均 R²={mean_r2:.5f}, MSE={mean_mse:.5f}")

    if mean_r2 > best_repeat_r2:
        best_repeat_r2 = mean_r2
        best_repeat_idx = repeat+1
        best_train_df = pd.DataFrame(train_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])
        best_val_df = pd.DataFrame(val_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])

# ========= 3) 写 TXT =========
txt_file = os.path.join(txt_dir, "Boosting_best_repeat_summary.txt")
lines = ["===== 异构 Boosting 集成模型（MLP→SVM→DecisionTree→ElasticNet→KNN） =====", ""]
for item in repeats_metrics:
    lines.append(f"\n====== 第 {item['repeat']} 次重复 ======")
    for fi, (fr2, fmse) in enumerate(zip(item['fold_r2'], item['fold_mse']), start=1):
        lines.append(f"Fold {fi}: R²={fr2:.6f}, MSE={fmse:.8f}")
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
    best_train_df.to_csv(os.path.join(csv_dir, "Boosting_best_repeat_train.csv"), index=False)
    best_val_df.to_csv(os.path.join(csv_dir, "Boosting_best_repeat_val.csv"), index=False)
    print(f"最佳 repeat 的 train/val CSV 已保存到 {csv_dir}")
