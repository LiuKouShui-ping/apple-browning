# -*- coding: utf-8 -*-
"""
Stacking 完整流程（按用户要求）：
阶段1: base learners 在原始 X,y 上 GridSearchCV(5-fold, refit='R2') -> 最优 base estimator (Pipeline)
阶段2: 用最优 base 在 5-fold OOF 上生成 X_meta 并保存 X_meta.csv
阶段3: 在 X_meta 上对 meta learners（Ridge, ElasticNet, XGB, LGBM）做 GridSearchCV -> 最优 meta estimator
阶段4: 对每个最优 meta estimator 做 10 次重复的 5-fold CV（repeat=0..9），记录每折 R2/MSE，
         保存 stacking_evaluation.csv，保存 stacking_<meta>_results.txt，保存最佳 repeat 的 train/val CSV。
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

# ========== 路径配置（按需修改） ==========
input_file_path = r"C:\Users\xiaoping\Desktop\fushi\data\result\input_data\merge_features_54_0.csv"
target_file_path = r"C:\Users\xiaoping\Desktop\fushi\data\result\input_data\enzyme_selected.csv"
txt_dir = r'C:\Users\xiaoping\Desktop\fushi\data\result\model_output\POD\txt'
fitted_data_dir = r'C:\Users\xiaoping\Desktop\fushi\data\result\model_output\POD\csv'

os.makedirs(txt_dir, exist_ok=True)
os.makedirs(fitted_data_dir, exist_ok=True)

# ========== 读取数据 ==========
features_data = pd.read_csv(input_file_path)
target_data = pd.read_csv(target_file_path)
X = features_data.drop(['Time(h)', 'Position', 'BI'], axis=1).astype(float).reset_index(drop=True)
target_name = 'POD'   # 可改为 'BI' / 'PPO' 等
y = target_data[target_name].astype(float).reset_index(drop=True)
sample_ids = np.array(features_data.index)

# ========== 基学习器 & 超参数网格（使用你给定的 grids） ==========
base_models_and_grids = {
    'KNN': (KNeighborsRegressor(), {'model__n_neighbors': [3,5,7,9,11]}),
    'SVM': (SVR(), {'model__C': [0.1,1,10], 'model__gamma': ['scale','auto'], 'model__kernel': ['rbf','poly','sigmoid']}),
    'ElasticNet': (ElasticNet(max_iter=5000), {'model__alpha': [0.001,0.01,0.1,1], 'model__l1_ratio': [0.1,0.5,0.9]}),
    'DecisionTree': (DecisionTreeRegressor(random_state=42), {'model__max_depth': [None,5,10,20], 'model__min_samples_split': [2,4,6], 'model__min_samples_leaf': [1,2,4]}),
    'MLP': (MLPRegressor(max_iter=2000, random_state=42), {'model__hidden_layer_sizes': [(50,), (100,), (50,50)], 'model__alpha': [0.0001,0.001,0.01], 'model__learning_rate_init': [0.001,0.01]})
}

# ========== 元学习器与其 param grids（将在 X_meta 上搜索） ==========
meta_models_and_grids = {
    'Linear': (LinearRegression(), {}), 
    'Ridge': (Ridge(), {'model__alpha': [0.1, 1.0, 10.0]}),  # 用 Ridge + grid 统一处理
    'ElasticNet_meta': (ElasticNet(max_iter=5000), {'model__alpha': [0.001,0.01,0.1,1], 'model__l1_ratio': [0.1,0.5,0.9]}),
}

# ========== Search / CV 设置 ==========
gs_cv = KFold(n_splits=5, shuffle=True, random_state=42)   # 用于 GridSearch 的 CV（固定）
scoring = {'R2': 'r2', 'MSE': make_scorer(mean_squared_error, greater_is_better=False)}
# scoring = {'R2': 'r2', 'MSE': make_scorer(mean_squared_error, greater_is_better=True, greater_is_better=False)}
n_jobs = 8   # 可改为 -1

# -------------------------
# 阶段1：对每个 base learner 做 GridSearchCV（在原始 X,y 上）
# -------------------------
print("阶段1：对基学习器进行 GridSearchCV（5-fold, refit='R2'）")
best_base_pipes = {}   # name -> best Pipeline estimator (fitted by refit on whole X,y by GridSearchCV)
base_tuning_rows = []  # 保存每个模型的 summary，用于写 CSV

for name, (estimator, param_grid) in base_models_and_grids.items():
    print(f"-> 搜索基学习器：{name}")
    pipe = Pipeline([('model', estimator)])
    gs = GridSearchCV(pipe, param_grid, cv=gs_cv, scoring=scoring, refit='R2', n_jobs=n_jobs, return_train_score=False)
    gs.fit(X, y)
    best_base_pipes[name] = gs.best_estimator_   # pipeline fitted on full X,y by refit
    best_base_pipes[name].best_params = gs.best_params_   # ✅ 自定义属性保存最佳参数
    # 记录 summary 行
    row = {
        'Model': name,
        'BestParams': gs.best_params_,
        'CV_mean_R2': float(gs.cv_results_['mean_test_R2'][gs.best_index_]) if 'mean_test_R2' in gs.cv_results_ else np.nan,
        'CV_mean_MSE': float(gs.cv_results_['mean_test_MSE'][gs.best_index_]) if 'mean_test_MSE' in gs.cv_results_ else np.nan
    }
    base_tuning_rows.append(row)
    print(f"  完成 {name}，最优参数: {gs.best_params_}")

base_tuning_df = pd.DataFrame(base_tuning_rows)
# 可以选择保存 base_tuning_df，如果你想保存请取消下一行注释
# base_tuning_df.to_csv(os.path.join(fitted_data_dir, "base_tuning_results.csv"), index=False)

# -------------------------
# 阶段2：用最优参数做 5-fold OOF，生成 X_meta.csv
# -------------------------
print("\n阶段2：用最优基学习器生成 OOF meta 特征（5-fold）")
kf_oof = KFold(n_splits=5, shuffle=True, random_state=42)
n_samples = X.shape[0]
meta_feature_dict = {name: np.zeros(n_samples, dtype=float) for name in best_base_pipes.keys()}

for fold_idx, (train_idx, val_idx) in enumerate(kf_oof.split(X), start=1):
    print(f"  OOF fold {fold_idx}/5")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    for name, best_pipe in best_base_pipes.items():
        # 从 pipeline 里取出未拟合的 model 模板并 clone，再 fit 到 fold 的训练集，预测 val
        # 去掉 'model__' 前缀再传入
        best_params_clean = {k.replace('model__', ''): v for k, v in best_pipe.best_params.items()}
        best_model = clone(best_pipe.named_steps['model']).set_params(**best_params_clean)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_val)
        meta_feature_dict[name][val_idx] = preds

# 拼接 meta_X
meta_X = pd.DataFrame(meta_feature_dict)
meta_X.index = features_data.index
meta_X.to_csv(os.path.join(fitted_data_dir, "X_meta.csv"), index=True)   # 保存 X_meta.csv（包含 Sample index）
print("已保存 X_meta.csv 到：", os.path.join(fitted_data_dir, "X_meta.csv"))

# -------------------------
# 阶段3：在 meta_X 上对 meta learners 做 GridSearchCV（refit='R2'）
# -------------------------
print("\n阶段3：在 X_meta 上对元学习器做 GridSearchCV（5-fold, refit='R2'）")
best_meta_pipes = {}
meta_tuning_rows = []

for meta_name, (meta_estimator, meta_grid) in meta_models_and_grids.items():
    print(f"-> 搜索元学习器：{meta_name}")
    # 使用 Pipeline 统一接口
    meta_pipe = Pipeline([('model', meta_estimator)])
    gs_meta = GridSearchCV(meta_pipe, meta_grid, cv=gs_cv, scoring=scoring, refit='R2', n_jobs=n_jobs, return_train_score=False)
    gs_meta.fit(meta_X, y)
    best_meta_pipes[meta_name] = gs_meta.best_estimator_
    # 记录 summary
    row = {
        'MetaModel': meta_name,
        'BestParams': gs_meta.best_params_,
        'CV_mean_R2': float(gs_meta.cv_results_['mean_test_R2'][gs_meta.best_index_]) if 'mean_test_R2' in gs_meta.cv_results_ else np.nan,
        'CV_mean_MSE': float(gs_meta.cv_results_['mean_test_MSE'][gs_meta.best_index_]) if 'mean_test_MSE' in gs_meta.cv_results_ else np.nan
    }
    meta_tuning_rows.append(row)
    print(f"  完成 {meta_name}，最优参数: {gs_meta.best_params_}")

meta_tuning_df = pd.DataFrame(meta_tuning_rows)
# 可选保存 meta_tuning_df
# meta_tuning_df.to_csv(os.path.join(fitted_data_dir, "meta_tuning_results.csv"), index=False)

# -------------------------
# 阶段4：对每个最优 meta learner 做 10 次重复的 5-fold CV（记录并保存结果）
# -------------------------
print("\n阶段4：使用最优 base + 最优 meta，进行 10 次重复 5-fold CV，保存评估结果与最佳 repeat 的 train/val CSV")

num_repeats = 10
num_folds = 5

# 用于保存所有 meta 的总体评估表（stacking_evaluation.csv）
all_meta_evals = []

for meta_name, best_meta_pipe in best_meta_pipes.items():
    print(f"\n===== 处理元学习器: {meta_name} =====")
    txt_file = os.path.join(txt_dir, f"stacking_{meta_name}_results.txt")

    repeats_metrics = []   # 存每次 repeat 的各折指标和平均
    best_repeat_mean_r2 = -np.inf
    best_repeat_idx = None
    best_repeat_train_df = None
    best_repeat_val_df = None

    # 10 次重复
    for repeat in range(num_repeats):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=repeat)
        fold_r2_list = []
        fold_mse_list = []
        train_records = []
        val_records = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            id_train, id_val = sample_ids[train_idx], sample_ids[val_idx]

            # 构造 stacking 的 base estimators（使用阶段1 得到的最优超参）
            estimators_for_stack = []
            estimators_for_stack = []
            for name, best_pipe in best_base_pipes.items():
                best_model_template = clone(best_pipe.named_steps['model'])
                # 加入这一行，注入最优参数
                best_params_clean = {k.replace('model__', ''): v for k, v in best_pipe.best_params.items()}
                best_model_template.set_params(**best_params_clean)
                estimators_for_stack.append((name, best_model_template))

            # meta estimator：取出 pipeline 中的 model 模板并 clone
            meta_model_template = clone(best_meta_pipe.named_steps['model'])

            stacking_model = StackingRegressor(estimators=estimators_for_stack, final_estimator=meta_model_template, n_jobs=-1, passthrough=False)
            # fit on fold train
            stacking_model.fit(X_train, y_train)

            # 预测 train & val
            y_train_pred = stacking_model.predict(X_train)
            y_val_pred = stacking_model.predict(X_val)

            # 评估 val
            r2 = r2_score(y_val, y_val_pred)
            mse = mean_squared_error(y_val, y_val_pred)
            fold_r2_list.append(r2)
            fold_mse_list.append(mse)

            # 保存 train / val 记录
            train_records.extend(zip(id_train, [fold_idx]*len(train_idx), ['Train']*len(train_idx), y_train.values, y_train_pred, [f"stacking_{meta_name}"]*len(train_idx)))
            val_records.extend(zip(id_val, [fold_idx]*len(val_idx), ['Val']*len(val_idx), y_val.values, y_val_pred, [f"stacking_{meta_name}"]*len(val_idx)))

        mean_r2 = float(np.mean(fold_r2_list))
        mean_mse = float(np.mean(fold_mse_list))
        repeats_metrics.append({'repeat': repeat+1, 'fold_r2': fold_r2_list, 'fold_mse': fold_mse_list, 'mean_r2': mean_r2, 'mean_mse': mean_mse})

        # 保存最佳 repeat 的 train/val 表（基于平均 R2）
        if mean_r2 > best_repeat_mean_r2:
            best_repeat_mean_r2 = mean_r2
            best_repeat_idx = repeat+1
            best_repeat_train_df = pd.DataFrame(train_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])
            best_repeat_val_df = pd.DataFrame(val_records, columns=['SampleID','Fold','DatasetType','True','Pred','ModelName'])

        print(f" meta={meta_name} repeat {repeat+1}: mean R2={mean_r2:.6f}, mean MSE={mean_mse:.6f}")

    # 汇总 repeats_metrics 写入 stacking_<meta_name>_results.txt（格式与 Voting 代码一致）
    lines = []
    lines.append(f"===== Stacking meta: {meta_name} =====")
    lines.append(f"基学习器: {list(best_base_pipes.keys())}")
    lines.append(f"基学习器 最优超参数（阶段1 得到，Pipeline.best_params）：")
    for name, pipe in best_base_pipes.items():
        lines.append(f"  {name}: {pipe.best_params}")
    lines.append("")
    lines.append(f"元学习器: {meta_name}")
    lines.append(f"元学习器 最优超参数（阶段3 得到）： {best_meta_pipe.named_steps['model'].get_params()}")
    lines.append("")
    lines.append("以下为 10 次重复的 5 折交叉验证结果（每次显示每折的 R² 与 MSE 以及该次平均）：")
    for item in repeats_metrics:
        r = item['repeat']
        lines.append(f"\n====== 第 {r} 次重复 ======")
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
    lines.append("")
    lines.append(f"最佳重复（基于平均 R²）：第 {best_repeat_idx} 次")
    lines.append(f"最佳重复 平均 R²: {best_repeat_mean_r2:.6f}")

    # 写 txt
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Summary TXT 已保存到: {txt_file}")

    # 保存 stacking_evaluation 中该 meta 的 summary行（用于最终 stacking_evaluation.csv）
    # 我们把每个 repeat 的 mean_r2, mean_mse 以及 overall mean/std 写入一行（也保留 best_repeat_idx）
    eval_row = {
        'meta': meta_name,
        'best_meta_params': best_meta_pipe.get_params(),
        'mean_of_10_mean_R2': mean_all_r2,
        'std_of_10_mean_R2': std_all_r2,
        'mean_of_10_mean_MSE': mean_all_mse,
        'std_of_10_mean_MSE': std_all_mse,
        'best_repeat_idx': best_repeat_idx,
        'best_repeat_mean_R2': best_repeat_mean_r2
    }
    all_meta_evals.append(eval_row)

    # 保存最佳 repeat 的 train/val CSV（若存在）
    if best_repeat_train_df is not None and best_repeat_val_df is not None:
        best_repeat_train_df.to_csv(os.path.join(fitted_data_dir, f"stacking_{meta_name}_best_repeat_train.csv"), index=False)
        best_repeat_val_df.to_csv(os.path.join(fitted_data_dir, f"stacking_{meta_name}_best_repeat_val.csv"), index=False)
        print(f"已保存最佳 repeat({best_repeat_idx}) 的 train/val CSV 到：{fitted_data_dir}")

# 最终写 stacking_evaluation.csv（每个 meta 一行 summary）
stacking_eval_df = pd.DataFrame(all_meta_evals)
stacking_eval_df.to_csv(os.path.join(fitted_data_dir, "stacking_evaluation.csv"), index=False)
print("已保存 stacking_evaluation.csv 到：", os.path.join(fitted_data_dir, "stacking_evaluation.csv"))

print("===== 所有 Stacking 模型流程完成 =====")
