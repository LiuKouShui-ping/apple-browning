import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import pathlib

# ========== 文件路径 ==========
file_path_step1 = r'C:\Users\xiaoping\Desktop\fushi\data\radiomics_54.csv'
output_file_dir = r'C:\Users\xiaoping\Desktop\fushi\data\variance_filtered'
os.makedirs(output_file_dir, exist_ok=True)
output_file_path = os.path.join(output_file_dir,'merge_features_54.csv')
# ========== 第一步：去除零方差特征 ==========
data_step1 = pd.read_csv(file_path_step1)   #读取数据
X_step1 = data_step1.iloc[:, 1:]
X_numeric = X_step1.select_dtypes(include=['number'])  # 仅保留数值列

selector = VarianceThreshold(threshold=0)
X_zero_var = selector.fit_transform(X_numeric)
selected_columns_step1 = X_numeric.columns[selector.get_support()]  # 提取保留下来的特征名
print(f"去除零方差后剩余特征数: {len(selected_columns_step1)}")

# ========== 第二步：去除方差处于最低 5% 的特征 ==========
X_step2 = X_numeric[selected_columns_step1]

# ⚠️ 基于原始尺度计算方差
variances = X_step2.var()
threshold_value = np.percentile(variances, 5)  # 去掉方差最低 5%
selected_columns_step2 = variances[variances > threshold_value].index
print(f"进一步去掉方差最低 5% 后剩余特征数: {len(selected_columns_step2)}")

# ✅ 在筛选完后再做标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_step2[selected_columns_step2])
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_columns_step2)


data_filtered = pd.concat([data_step1[['Time(h)']], X_scaled_df[selected_columns_step2]], axis=1) # 拼接时间列 + 标准化后的特征
data_filtered.to_csv(output_file_path, index=False)
print(f"筛选后的数据已保存为: {output_file_path}")

# ========== 第三步：去掉两特征相关关系系数大于0.9的特征之一==========
file_path_step3 = r"C:\Users\xiaoping\Desktop\fushi\data\variance_filtered\merge_features_54.csv" # 输入输出路径
file_stem = pathlib.Path(file_path_step3).stem
output_dir = r"C:\Users\xiaoping\Desktop\fushi\data\features_selected_54"
os.makedirs(output_dir, exist_ok=True)

print("📂 正在读取数据……")
data_step3 = pd.read_csv(file_path_step3)

y_step3 = data_step3.iloc[:, 0].astype(float)  # 目标变量 (时间列第一列)
X_step3 = data_step3.iloc[:, 1:]  # 特征列

enet = ElasticNetCV(cv=5, random_state=42, l1_ratio=0.5) # 用 ElasticNet 计算特征权重（在原始数据上）
enet.fit(X_step3, y_step3)
coef = pd.Series(enet.coef_, index=X_step3.columns)

# 🔹【新增1】：保存 ElasticNet 各特征系数
coef_df = pd.DataFrame({
    "Feature": coef.index,
    "ElasticNet_Coefficient": coef.values
})
# coef_path = os.path.join(output_dir, f"{file_stem}_elasticnet_coefficients.csv")
# coef_df.to_csv(coef_path, index=False, encoding='utf-8-sig')
# print(f"✅ 已保存 ElasticNet 特征系数: {coef_path}")


#  1️⃣ 相关性筛选
final_selected_features = X_step3.columns.tolist()

while True:
    corr_df = X_step3[final_selected_features].corr()
    to_remove = set()

    for i in range(len(corr_df.columns)):
        for j in range(i + 1, len(corr_df.columns)):
            if abs(corr_df.iloc[i, j]) > 0.9:
                f1, f2 = corr_df.columns[i], corr_df.columns[j]
                # 保留权重绝对值更大的特征
                if abs(coef[f1]) >= abs(coef[f2]):
                    to_remove.add(f2)
                else:
                    to_remove.add(f1)

    if not to_remove:
        break

    final_selected_features = [f for f in final_selected_features if f not in to_remove]

# 🔹【新增2】：保存特征间的相关系数矩阵
corr_matrix = X_step3[final_selected_features].corr()
corr_matrix_path = os.path.join(output_dir, f"{file_stem}_feature_correlation.csv")
corr_matrix.to_csv(corr_matrix_path, encoding='utf-8-sig')

print(f"✅ 已保存特征间相关系数矩阵: {corr_matrix_path}")
print(f"相关性筛选后剩余特征数: {len(final_selected_features)}")
print("相关性筛选后特征：", final_selected_features)

corr_selected_data = data_step3[['Time(h)'] + final_selected_features] # 保存相关性0.9特征筛选结果
csv_path_corr = os.path.join(output_dir, f"{file_stem}_0.9.csv")
corr_selected_data.to_csv(csv_path_corr, index=False, encoding='utf-8-sig')
print(f"✅ 已保存相关性筛选结果: {csv_path_corr}")

#  2️⃣ 去掉权重 = 0 的特征 
final_selected_features_nonzero = [f for f in final_selected_features if coef[f] != 0]

print(f"最终筛选后剩余特征数: {len(final_selected_features_nonzero)}")
print("最终筛选后特征：", final_selected_features_nonzero)
# 🔹【新增1】：保存 ElasticNet 各特征系数
# 从 coef_df 中提取筛选特征对应的系数
selected_coefs = coef_df[coef_df["Feature"].isin(final_selected_features_nonzero)]
features = selected_coefs.iloc[:, 0]  # 获取第一列，即特征名
coefficients = selected_coefs.iloc[:, 1]  # 获取第二列，即系数值

# 创建 DataFrame 用于保存筛选后的特征及其对应系数
coef_df_selected = pd.DataFrame({
    "Feature": features,
    "ElasticNet_Coefficient": coefficients
})
# 打印以检查
print(coef_df_selected.head())
# 保存为 CSV 文件
coef_path = os.path.join(output_dir, f"{file_stem}_elasticnet_coefficients.csv")
coef_df_selected.to_csv(coef_path, index=False, encoding='utf-8-sig')
print(f"✅ 筛选后的特征系数已保存为: {coef_path}")

# 🔹【新增2】：保存特征间的相关系数矩阵
corr_matrix = X_step3[final_selected_features_nonzero].corr()
corr_matrix_path = os.path.join(output_dir, f"{file_stem}_feature_correlation.csv")
corr_matrix.to_csv(corr_matrix_path, encoding='utf-8-sig')

print(f"✅ 已保存特征间相关系数矩阵: {corr_matrix_path}")
print(f"相关性筛选后剩余特征数: {len(final_selected_features)}")
print("相关性筛选后特征：", final_selected_features)

final_selected_data = data_step3[['Time(h)'] + final_selected_features_nonzero] # 保存去掉权重 = 0 的特征结果
csv_path_final = os.path.join(output_dir, f"{file_stem}_0.csv")
final_selected_data.to_csv(csv_path_final, index=False, encoding='utf-8-sig')

print(f"🎯 已保存最终筛选后的数据为: {csv_path_final}")
print("✅ 第三步程序执行完毕。")

# ========== 第四步：绘制热力图 + 系数散点图 ==========
X_final = data_step3[final_selected_features_nonzero] # 取最终非零特征对应的数据

# 1️⃣ 热力图（Correlation Heatmap）
plt.figure(figsize=(8, 8))
sns.heatmap(X_final.corr(), cmap="coolwarm", annot=False, square=True, cbar=True)
plt.title("Correlation Heatmap of Selected Features", fontsize=12)
plt.tight_layout()

heatmap_path = os.path.join(output_dir, f"{file_stem}_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"📊 已保存相关性热力图: {heatmap_path}")

# 2️⃣ 系数散点图（Coefficient Scatter Plot）
coefs_nonzero = coef[final_selected_features_nonzero]

plt.figure(figsize=(8, 8))
plt.scatter(coefs_nonzero.index, coefs_nonzero.values, color="blue", alpha=0.7)
plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)

plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("ElasticNet Coefficient")
plt.title("ElasticNet Non-zero Feature Coefficients", fontsize=12)

plt.tight_layout()
scatter_path = os.path.join(output_dir, f"{file_stem}_coef_scatter.png")
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"📊 已保存系数散点图: {scatter_path}")




