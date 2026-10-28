# modeling-toolkit

数学建模竞赛常用方法工具箱：一份能直接跑的 Python 实现集合，覆盖评价、预测、分类、统计检验四类高频题型。出自 2026 年全国大学生数学建模竞赛（CUMCM）C 题的实战沉淀，所有方法都在真实赛题中使用并验证过。

## 功能

**评价类**（`modeling_toolkit.py`）
- `ahp(compare_matrix)` — 层次分析法 AHP 求权重
- `entropy_weight(X)` — 熵权法客观赋权
- `topsis(X, weights, cost_cols)` — TOPSIS 综合评价（支持成本型指标）

**预测类**
- `gm11(x0)` — 灰色预测 GM(1,1)

**机器学习流水线**（`modeling_toolkit.py`）
- `preprocess_pipeline(df, target_col)` — 缺失值/编码/标准化一条龙
- `compare_models(X_train, X_test, y_train, y_test, task)` — 一次对比多个模型并出表
- `sensitivity_analysis(model, X, param_cols)` — 参数敏感性分析

**统计与检验**（`toolkit_additions.py`）
- `knn_classify` — KNN 分类
- `agglo_cluster` — 层次聚类
- `chi2_test(table)` — 卡方检验
- `spearman_matrix(df)` — Spearman 相关矩阵

**输出工具**
- `df_to_latex(df, caption, label)` — DataFrame 一键转 LaTeX 三线表

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# Linux/macOS: .venv/bin/pip install -r requirements.txt

python demo.py    # 跑通全部功能演示（wine 数据集，约 10 秒）
```

## 依赖

见 `requirements.txt`（numpy / pandas / scikit-learn / xgboost / matplotlib / seaborn / jinja2）。

## 说明

- 本仓库方法在 2026 CUMCM C 题中实际使用，赛题完整解法（含论文与对账记录）另库发布。
- 欢迎提 issue 指出实现问题——每个方法的边界条件都在持续打磨。
