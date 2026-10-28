"""
验收测试：modeling_toolkit.py 新加入的通用工具函数（第7章）
数据集: sklearn wine (13维特征, 3类)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from modeling_toolkit import (
    preprocess_pipeline,
    compare_models,
    sensitivity_analysis,
    df_to_latex,
    plot_correlation_heatmap,
)

PASS = 0
FAIL = 0

def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")

# ---- 加载 wine 数据 ----
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print("=" * 60)
print("数据集: wine (%d 样本, %d 特征, %d 类)" % (X.shape[0], X.shape[1], len(target_names)))
print("特征:", feature_names[:5], "...")
print("分类:", target_names)
print("=" * 60)

# ====== 1. 测试 preprocess_pipeline ======
print("\n----- 1. preprocess_pipeline (数据预处理) -----")
df_raw = pd.DataFrame(X, columns=feature_names)
df_raw['target'] = y

# 人为加点缺失值和分类列来测试鲁棒性
df_messy = df_raw.copy()
df_messy.iloc[:5, 0] = np.nan  # 在第一个特征加缺失
df_messy.iloc[10:12, 2] = np.nan
df_messy['cat_col'] = pd.Categorical(['A']*90 + ['B']*88)  # 人为加分类列

df_clean, scaler = preprocess_pipeline(df_messy, target_col='target')

check("返回 DataFrame", isinstance(df_clean, pd.DataFrame))
check("返回 scaler", hasattr(scaler, 'mean_'))
check("无缺失值", df_clean.isnull().sum().sum() == 0)
check("目标列保留", 'target' in df_clean.columns)
check("分类列被编码", 'cat_col_B' in df_clean.columns)
check("数值列已标准化", abs(df_clean.iloc[:, 0].mean()) < 1e-10)

# ====== 2. 测试 compare_models (分类) ======
print("\n----- 2. compare_models (多模型对比 - 分类) -----")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

df_cmp, best_name = compare_models(X_train, X_test, y_train, y_test, task='classification')

check("返回结果DataFrame", isinstance(df_cmp, pd.DataFrame))
check("至少3个模型", len(df_cmp) >= 3)
check("包含准确率列", '准确率' in df_cmp.columns)
check("有最优模型名", isinstance(best_name, str) and len(best_name) > 0)
check("最优模型准确率>0.5", df_cmp[df_cmp['模型']==best_name]['准确率'].values[0] > 0.5)
print("\n模型对比结果:")
print(df_cmp.to_string(index=False))

# ====== 3. 测试 compare_models (回归) ======
print("\n----- 3. compare_models (多模型对比 - 回归) -----")
# 用 wine 第1个特征做回归目标
X_reg = X[:, 1:]
y_reg = X[:, 0]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

df_reg, best_reg = compare_models(Xr_train, Xr_test, yr_train, yr_test, task='regression')

check("回归返回DataFrame", isinstance(df_reg, pd.DataFrame))
check("至少3个回归模型", len(df_reg) >= 3)
check("包含R2列", 'R2' in df_reg.columns)
check("最优模型R2>0.5", df_reg[df_reg['模型']==best_reg]['R2'].values[0] > 0.5)
print("\n回归对比结果:")
print(df_reg.to_string(index=False))

# ====== 4. 测试 sensitivity_analysis ======
print("\n----- 4. sensitivity_analysis (灵敏度分析) -----")
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

X_sens = pd.DataFrame(X_test, columns=feature_names)
param_cols = feature_names[:3]  # 只扰动前3个特征

df_sens, fig_sens = sensitivity_analysis(
    model_rf, X_sens, param_cols,
    param_range=(-0.2, 0.2), steps=5
)

check("返回灵敏度DataFrame", isinstance(df_sens, pd.DataFrame))
check("形状正确 (steps×param_cols)", df_sens.shape == (5, len(param_cols)))
check("返回fig对象", fig_sens is not None)

print("\n灵敏度分析结果 (输出变化%):")
print(df_sens.to_string())

# ====== 5. 测试 plot_correlation_heatmap ======
print("\n----- 5. plot_correlation_heatmap (相关性热力图) -----")
fig_heat = plot_correlation_heatmap(
    df_clean.drop(columns=['target']),
    figsize=(10, 8),
    save_path=os.path.join(os.path.dirname(__file__), '..', 'figures', 'test_corr_heatmap.png')
)

check("返回fig对象", fig_heat is not None)
check("图片文件已生成",
      os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'figures', 'test_corr_heatmap.png')))

# ====== 6. 测试 df_to_latex ======
print("\n----- 6. df_to_latex (LaTeX表格) -----")
latex_str = df_to_latex(df_cmp, caption='模型对比', label='tab:model_comp')
check("返回字符串", isinstance(latex_str, str))
check("包含caption", 'caption' in latex_str)
check("包含label", 'label' in latex_str)
check("包含booktabs", 'booktabs' in latex_str)

# ====== 汇总 ======
print("\n" + "=" * 60)
total = PASS + FAIL
print("验收汇总: %d / %d 通过, %d 失败" % (PASS, total, FAIL))
if FAIL == 0:
    print(">>> 全部通过! 所有工具函数正常运行。")
else:
    print(">>> 有 %d 个测试失败, 需要修复。" % FAIL)
print("=" * 60)
