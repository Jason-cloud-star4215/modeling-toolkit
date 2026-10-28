"""
toolkit_additions.py — 2022 C 题演练工具箱缺口补充（波波补，供 Claude Code 调用）

配合 modeling_toolkit.py 使用。本文件提供四问建模中缺失的独立函数：
  - knn_classify : KNN 分类（Q3 未知鉴别）
  - agglo_cluster: 层次聚类（Q2 亚类划分）
  - chi2_test    : 卡方独立性检验（Q1 风化 vs 类型/纹饰/颜色）
  - spearman_matrix: Spearman 相关矩阵（Q4 关联关系）

AI 生成声明：本文件由 WorkBuddy(波波) 于赛前演练中生成，仅用于学习，严禁用于正式比赛提交。
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, spearmanr


def knn_classify(X_train, y_train, X_test, k=5, weights="distance"):
    """KNN 分类。返回 (预测标签, 最大类概率)。"""
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test).max(axis=1)
    return pred, proba


def agglo_cluster(X, n_clusters=2, linkage="ward"):
    """层次聚类（先标准化）。返回每样本簇标签。"""
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(Xs)


def chi2_test(table):
    """卡方独立性检验。table 为列联表(二维 array-like)。返回 dict(chi2,p,dof)。"""
    table = np.asarray(table, float)
    chi2, p, dof, _ = chi2_contingency(table)
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof)}


def spearman_matrix(df):
    """Spearman 相关矩阵。输入含数值列的 DataFrame，返回相关矩阵 DataFrame。"""
    cols = df.select_dtypes("number").columns
    mat = df[cols].corr(method="spearman")
    return mat


if __name__ == "__main__":
    print("toolkit_additions OK:",
          all(callable(f) for f in [knn_classify, agglo_cluster, chi2_test, spearman_matrix]))
