# -*- coding: utf-8 -*-
"""
Voting 集成模型（KNN + SVM + ElasticNet + DecisionTree/MLP）：
1) 对每个子模型先用 GridSearchCV 搜索最优超参数（5-fold, refit='R2'）
2) 使用最优超参数训练子模型并进行 10 次重复 5-fold CV
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
from sklearn.ensemble import VotingRegressor
import random
random.seed(42)
np.random.seed(42)

# ========= 路径设置 =========
txt_dir = r"C:\Users\xiaoping\Desktop\fushi\data\result\model_output\PPO\txt"
csv_dir = r"C:\Users\xiaoping\Desktop\fushi\data\result\model_output\PPO\csv"
os.makedirs(txt_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# ========= 数据读取 =========
features_data = pd.read_csv(r"C:\Users\xiaoping\Desktop\fushi\data\result\input_data\merge_features_54_0.csv")
target_data = pd.read_csv(r'C:\Users\xiaoping\Desktop\fushi\data\result\input_data\enzyme_selected.csv')

X = features_data.drop(['Time(h)', 'Position', 'BI'], axis=1)
y = target_data['PPO'].astype(float)
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

# ========= 2) Voting + 10 次重复 5-fold CV =========
num_repeats = 10
num_folds = 5
repeats_metrics = []
best_repeat_r2 = -np.inf
best_train_df = None
best_val_df = None
best_repeat_idx = None

for repeat in range(num_repeats):
    print(f"\n-> Repeat {repeat+1}/{num_repeats}")
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat)
    fold_r2_list = []
    fold_mse_list = []
    train_records = []
    val_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        id_train, id_val = sample_ids[train_idx], sample_ids[val_idx]

        # 克隆每个子模型
        knn_model = clone(best_estimators['KNN']).fit(X_train, y_train)
        svm_model = clone(best_estimators['SVM']).fit(X_train, y_train)
        en_model = clone(best_estimators['ElasticNet']).fit(X_train, y_train)
        dt_model = clone(best_estimators['DecisionTree']).fit(X_train, y_train)
        mlp_model = clone(best_estimators['MLP']).fit(X_train, y_train)

        # Voting 集成
        voting_model = VotingRegressor([
            ('KNN', knn_model),
            ('SVM', svm_model),
            ('ElasticNet', en_model),
            ('DecisionTree', dt_model),
            ('MLP', mlp_model)
        ])
        voting_model.fit(X_train, y_train)

        
        # 预测训练集（仅用于拟合对比，不作为评估指标）
        y_train_pred = voting_model.predict(X_train)
        # 预测验证集（用于模型评估）
        y_val_pred = voting_model.predict(X_val)
        r2 = r2_score(y_val, y_val_pred)
        mse = mean_squared_error(y_val, y_val_pred)
        fold_r2_list.append(r2)
        fold_mse_list.append(mse)

        # 保存 train/val 数据
        train_records.extend(zip(id_train, [fold_idx]*len(train_idx), ['Train']*len(train_idx), y_train.values, y_train_pred, ['Voting']*len(train_idx)))
        val_records.extend(zip(id_val, [fold_idx]*len(val_idx), ['Val']*len(val_idx), y_val.values, y_val_pred, ['Voting']*len(val_idx)))

    mean_r2 = float(np.mean(fold_r2_list))
    mean_mse = float(np.mean(fold_mse_list))
    repeats_metrics.append({'repeat': repeat+1, 'fold_r2': fold_r2_list, 'fold_mse': fold_mse_list, 'mean_r2': mean_r2, 'mean_mse': mean_mse})

    if mean_r2 > best_repeat_r2:
        best_repeat_r2 = mean_r2
        best_repeat_idx = repeat+1
        best_train_df = pd.DataFrame(train_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])
        best_val_df = pd.DataFrame(val_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])

# ========= 3) 写 TXT =========
txt_file = os.path.join(txt_dir, "Voting_best_repeat_summary.txt")
lines = ["===== Voting 集成模型（KNN+SVM+ElasticNet+DecisionTree+MLP） =====", ""]
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
    best_train_df.to_csv(os.path.join(csv_dir, "Voting_best_repeat_train.csv"), index=False)
    best_val_df.to_csv(os.path.join(csv_dir, "Voting_best_repeat_val.csv"), index=False)
    print(f"最佳 repeat 的 train/val CSV 已保存到 {csv_dir}")
