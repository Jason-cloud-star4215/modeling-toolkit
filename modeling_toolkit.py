"""
数模方法速成 · 可运行工具箱 (Python 3.11)
覆盖：评价类 / 预测类 / 优化类 / 分类与聚类 + 通用工具函数
所有函数均用合成数据演示，可直接运行、按需替换数据。
依赖: numpy scipy scikit-learn pandas statsmodels xgboost matplotlib seaborn
运行: python modeling_toolkit.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

rng = np.random.default_rng(42)
np.set_printoptions(suppress=True, precision=4)


# ============ 1. 评价类（给 m 个方案、n 个指标打分排序） ============
def ahp(compare_matrix):
    """层次分析法：输入判断矩阵(方阵)，返回权重 w 与一致性比率 CR。
    CR<0.1 视为通过一致性检验。"""
    n = compare_matrix.shape[0]
    eigvals, eigvecs = np.linalg.eig(compare_matrix)
    k = np.argmax(eigvals.real)
    w = eigvecs[:, k].real
    w = w / w.sum()
    lam = eigvals[k].real
    CI = (lam - n) / (n - 1)
    RI = {1:0,2:0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}.get(n, 1.5)
    CR = CI / RI if RI != 0 else 0.0
    return w, CR


def entropy_weight(X):
    """熵权法：X 为 m×n 样本-指标矩阵(已完成正向化)。
    返回 n 个指标权重，客观赋权。"""
    X = np.asarray(X, float)
    P = X / X.sum(axis=0)
    k = 1.0 / np.log(X.shape[0])
    e = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    g = 1 - e
    return g / g.sum()


def topsis(X, weights, cost_cols=None):
    """TOPSIS：X 为 m×n(已正向化)；weights 权重；cost_cols 为成本型指标列索引。
    返回每个方案的相对贴近度(0~1，越大越优)。"""
    X = np.asarray(X, float).copy()
    if cost_cols:
        for j in cost_cols:
            X[:, j] = X[:, j].max() - X[:, j]   # 成本型取反向(极大型化)
    Z = X / np.sqrt((X ** 2).sum(axis=0))
    V = Z * np.asarray(weights)
    best = V.max(axis=0)
    worst = V.min(axis=0)
    d_b = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_w = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return d_w / (d_b + d_w)


def demo_evaluation():
    print("\n===== 1. 评价类 =====")
    # 3 个候选方案, 3 个指标(均已正向化: 数值越大越好)
    X = np.array([
        [0.80, 0.60, 0.70],
        [0.50, 0.90, 0.40],
        [0.65, 0.55, 0.85],
    ])
    # AHP 判断矩阵(1~9标度, 对称且 a_ij=1/a_ji)
    A = np.array([[1,3,5],[1/3,1,2],[1/5,1/2,1]])
    w_ahp, cr = ahp(A)
    print(f"AHP 权重={w_ahp.round(4)}, CR={cr:.4f} -> {'通过' if cr<0.1 else '未通过,需调整判断矩阵'}")
    w_ent = entropy_weight(X)
    print(f"熵权法 权重={w_ent.round(4)}")
    score = topsis(X, w_ahp)          # 用 AHP 权重做 TOPSIS 综合打分
    print(f"TOPSIS 综合贴近度={score.round(4)} -> 排名 {np.argsort(-score)+1}")


# ============ 2. 预测类（已知历史, 外推未来） ============
def gm11(x0):
    """灰色预测 GM(1,1)：适用于小样本(>=4)。返回 (predict, x_hat)。
    predict(k) 给出累加序列第 k 步预测; x_hat 为还原后的原始序列拟合值。"""
    x0 = np.asarray(x0, float)
    n = len(x0)
    aggr = np.cumsum(x0)                       # 1-AGO 累加
    z = 0.5 * (aggr[:-1] + aggr[1:])           # 紧邻均值生成序列
    B = np.column_stack([-z, np.ones(n - 1)])
    Y = x0[1:]
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]
    def predict_aggr(k):
        """返回累加序列(1-AGO)第 k 步预测。"""
        return (x0[0] - b / a) * np.exp(-a * k) + b / a
    fit_aggr = np.array([predict_aggr(k) for k in range(n)])
    x_hat = np.empty(n)
    x_hat[0] = x0[0]
    x_hat[1:] = np.diff(fit_aggr)              # 累减还原为原始序列拟合值

    def forecast(steps=1):
        """返回未来 steps 期的原始序列预测(已累减还原)。"""
        out = []
        for s in range(1, steps + 1):
            k = n - 1 + s
            out.append(predict_aggr(k) - predict_aggr(k - 1))
        return np.array(out)
    return predict_aggr, x_hat, forecast


def demo_forecast():
    print("\n===== 2. 预测类 =====")
    # 2.1 多元线性回归
    m = 60
    X = rng.normal(size=(m, 3))
    y = 2.5 * X[:, 0] - 1.3 * X[:, 1] + 0.7 * X[:, 2] + rng.normal(0, 0.2, m)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"多元线性回归 系数={beta.round(3)}, R2={r2:.3f}")

    # 2.2 ARIMA 时间序列
    from statsmodels.tsa.arima.model import ARIMA
    t = np.arange(100)
    series = 0.05 * t + np.sin(t / 6) + rng.normal(0, 0.3, 100)
    model = ARIMA(series, order=(1, 1, 1)).fit()
    fc = model.forecast(steps=5)
    print(f"ARIMA(1,1,1) 后5步预测={np.round(fc, 3)}")

    # 2.3 灰色 GM(1,1)
    x0 = np.array([12.1, 13.2, 14.0, 15.1, 16.0, 17.4])
    _, x_hat, fc = gm11(x0)
    print(f"GM(1,1) 拟合={x_hat.round(3)}  第7期预测={fc(1)[0]:.3f}")

    # 2.4 轻量神经网络 MLP
    from sklearn.neural_network import MLPRegressor
    yr = rng.normal(size=(m, 3))
    yt = np.sin(yr[:, 0]) + yr[:, 1] ** 2 * 0.1
    mlp = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=2000).fit(yr, yt)
    print(f"MLP 回归 训练R2={mlp.score(yr, yt):.3f}")


# ============ 3. 优化类（在约束下求最大/最小） ============
def demo_optimization():
    print("\n===== 3. 优化类 =====")
    # 3.1 线性规划  max z=3x+5y  s.t. x+2y<=14, 3x-y<=9, x,y>=0
    # scipy.linprog 求最小值, 故取负
    c = [-3, -5]
    A = [[1, 2], [3, -1]]
    b = [14, 9]
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
    print(f"线性规划 最优值={-res.fun:.3f}, x={res.x.round(3)}")

    # 3.2 整数规划  同上但 x,y 必须为整数
    # 用 milp: 目标 min -3x-5y ; 约束写成 LinearConstraint
    c_int = np.array([-3.0, -5.0])
    constraints = LinearConstraint([[1, 2], [3, -1]], lb=-np.inf, ub=[14, 9])
    res_i = milp(c=c_int, constraints=constraints, bounds=Bounds(lb=0, ub=np.inf), integrality=1)
    print(f"整数规划 最优值={-res_i.fun:.3f}, x={res_i.x.round(3)}")

    # 3.3 遗传算法 GA (求解 min f(x)=sum(x^2), x∈[-5,5]^5)
    def ga(obj, dim, lb, ub, pop=60, gen=120):
        P = rng.uniform(lb, ub, (pop, dim))
        for _ in range(gen):
            fit = np.array([obj(ind) for ind in P])
            prob = 1 / (fit - fit.min() + 1e-9)
            prob /= prob.sum()
            idx = rng.choice(pop, size=pop, p=prob)
            C = P[idx]
            cross = rng.random((pop, dim)) < 0.8
            C[cross] = (C[cross] + C[np.roll(idx, 1)][cross]) / 2
            mut = rng.random((pop, dim)) < 0.1
            C[mut] = rng.uniform(lb, ub, C[mut].shape)
            P = np.clip(C, lb, ub)
        best = min(P, key=lambda ind: obj(ind))
        return best, obj(best)
    best, fval = ga(lambda x: np.sum(x ** 2), 5, -5, 5)
    print(f"遗传算法 GA 最优解={best.round(3)}, f={fval:.4f}")

    # 3.4 粒子群 PSO (同样 min sum x^2)
    def pso(obj, dim, lb, ub, n=40, iters=100):
        x = rng.uniform(lb, ub, (n, dim))
        v = rng.uniform(-1, 1, (n, dim))
        pbest = x.copy()
        pval = np.array([obj(ind) for ind in x])
        gbest = pbest[np.argmin(pval)]
        w, c1, c2 = 0.7, 1.5, 1.5
        for _ in range(iters):
            r1, r2 = rng.random((n, dim)), rng.random((n, dim))
            v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
            x = np.clip(x + v, lb, ub)
            f = np.array([obj(ind) for ind in x])
            upd = f < pval
            pbest[upd] = x[upd]
            pval[upd] = f[upd]
            gbest = pbest[np.argmin(pval)]
        return gbest, obj(gbest)
    gb, fv = pso(lambda x: np.sum(x ** 2), 5, -5, 5)
    print(f"粒子群 PSO 最优解={gb.round(3)}, f={fv:.4f}")


# ============ 4. 分类与聚类 ============
def demo_classification():
    print("\n===== 4. 分类与聚类 =====")
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 4.1 SVM 二分类
    X = np.r_[rng.normal(2, 1, (50, 2)), rng.normal(-2, 1, (50, 2))]
    y = np.r_[np.ones(50), np.zeros(50)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    svm = SVC(kernel='rbf').fit(Xtr, ytr)
    print(f"SVM 测试准确率={accuracy_score(yte, svm.predict(Xte)):.3f}")

    # 4.2 K-means 聚类
    blobs = np.r_[rng.normal(0, 1, (50, 2)), rng.normal(5, 1, (50, 2)), rng.normal([3, 6], 1, (50, 2))]
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(blobs)
    print(f"KMeans 簇中心=\n{km.cluster_centers_.round(2)}")

    # 4.3 决策树
    Xc = rng.normal(size=(120, 4))
    yc = (Xc[:, 0] + Xc[:, 1] > 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
    dt = DecisionTreeClassifier(max_depth=4).fit(Xtr, ytr)
    print(f"决策树 测试准确率={accuracy_score(yte, dt.predict(Xte)):.3f}")


# ============ 5. 通用工具函数（来自 C题操作手册第7章） ============
def preprocess_pipeline(df, target_col=None):
    """一键数据预处理。
    输入: raw DataFrame
    输出: 清洗+编码+标准化后的 DataFrame, scaler 对象"""
    from sklearn.preprocessing import StandardScaler

    df = df.copy()  # 修复：不修改原始df
    # 1. 分离数值列和分类列
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    # 2. 数值列：中位数填充 + 异常值截尾
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())  # 修复：inplace 在 pandas>=3 不可用
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # 3. 分类列：众数填充 + One-Hot编码
    for col in cat_cols:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else '未知'
        df[col] = df[col].fillna(mode_val)  # 修复：同上
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4. 标准化（排除目标列）
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler


def compare_models(X_train, X_test, y_train, y_test, task='regression'):
    """自动训练多个模型，输出对比表。
    task: 'regression' | 'classification'
    返回: 对比DataFrame + 最优模型名"""
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score

    results = []
    if task == 'regression':
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RF': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGB': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                '模型': name,
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),  # 修复：避免R²字符导致GBK控制台报错
            })
    else:  # classification
        models = {
            'Logistic': LogisticRegression(max_iter=1000),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGB': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            result = {
                '模型': name,
                '准确率': accuracy_score(y_test, y_pred),
                'F1-score': f1_score(y_test, y_pred, average='weighted'),
            }
            if y_prob is not None:
                try:
                    # 二分类用正类概率，多分类用 ovr
                    if y_prob.shape[1] == 2:
                        result['AUC'] = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        result['AUC'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
                except Exception:
                    pass  # 部分数据可能算不了AUC
            results.append(result)

    df_result = pd.DataFrame(results)  # 修复：移到 if-else 外面，regression 分支也能执行
    if task == 'regression':
        best_idx = df_result.iloc[:, 1].idxmin()  # MSE 最小
    else:
        best_idx = df_result.iloc[:, 1].idxmax()  # 准确率最大
    best_name = df_result.iloc[best_idx]['模型']
    return df_result.round(4), best_name


def sensitivity_analysis(model, X, param_cols, param_range=(-0.2, 0.2), steps=5):
    """通用灵敏度分析。
    对每个参数列做 ±range 扰动，记录模型输出变化
    返回: 灵敏度DataFrame + fig对象"""
    import matplotlib
    matplotlib.use('Agg')  # 修复：防止在无GUI环境报错
    import matplotlib.pyplot as plt

    perturbations = np.linspace(param_range[0], param_range[1], steps)
    results = {}
    baseline = np.mean(model.predict(X))

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in param_cols:
        outputs = []
        for p in perturbations:
            X_temp = X.copy()
            X_temp[col] = X[col] * (1 + p)
            pred = model.predict(X_temp)
            outputs.append(np.mean(pred))
        change_pct = [(o - baseline) / baseline * 100 for o in outputs]
        results[col] = change_pct
        ax.plot(perturbations, change_pct, marker='o', label=col)

    ax.set_xlabel('Parameter Perturbation')
    ax.set_ylabel('Output Change (%)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend()
    ax.set_title('Sensitivity Analysis', fontweight='bold')
    plt.tight_layout()

    return pd.DataFrame(results, index=[f'{p*100:+.0f}%' for p in perturbations]), fig


def df_to_latex(df, caption='', label=''):
    """DataFrame → LaTeX三线表代码"""
    # 修复：确保 tabulate 可用
    latex = df.to_latex(index=False, float_format=lambda x: f'{x:.4f}')
    header = '% 需在导言区: \\usepackage{booktabs}\n'
    cap = f'\\caption{{{caption}}}\n' if caption else ''
    lbl = f'\\label{{{label}}}\n' if label else ''
    return header + latex.replace('\\toprule', f'{cap}{lbl}\\toprule')


def df_to_markdown(df):
    """DataFrame → GitHub风格Markdown表格"""
    return df.to_markdown(index=False)


def plot_correlation_heatmap(df, figsize=(12, 10), save_path=None):
    """相关性热力图——EDA必备。
    返回 fig 对象"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    demo_evaluation()
    demo_forecast()
    demo_optimization()
    demo_classification()
    print("\n全部方法演示完成。把合成数据替换成你的真实数据即可。")
