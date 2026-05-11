# 10 - scikit-learn 实战：从调包侠到实战高手

## 引入：你不需要重新发明轮子

前面9篇介绍了各种算法原理。但在实际写代码时，你基本**不会自己手写梯度下降或信息增益**——已经有无数优秀工程师帮你写好了，封装在scikit-learn里。

**scikit-learn是Python机器学习的事实标准库**，提供了统一的API、丰富的算法实现和完善的工具链。本篇文章带你走一遍完整的机器学习实战流程。

---

## 一个完整的机器学习项目流程

```
原始数据 → 数据清洗 → 特征工程 → 模型训练 → 模型评估 → 调参优化 → 部署预测
```

每一步scikit-learn都有对应的工具。我们用一个真实案例串起来。

---

## 一、数据预处理：数据不会自己变干净

现实中的数据充满各种问题：缺失值、量纲不统一、文本需要编码……

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# 假设有一个混合类型的数据集
df = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 28],
    'income': [5000, 8000, 12000, 15000, 6000],
    'city': ['北京', '上海', '北京', '广州', '上海'],
    'purchased': [0, 1, 1, 0, 1]
})

# Pipeline：把多步处理串联起来
numeric_features = ['age', 'income']
categorical_features = ['city']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # 缺失值填中位数
    ('scaler', StandardScaler()),                    # 标准化
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first')),         # 独热编码
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

X_processed = preprocessor.fit_transform(df.drop('purchased', axis=1))
print(f"处理后特征维度: {X_processed.shape}")
```

要点：
- **StandardScaler**：让每个特征的均值为0、标准差为1。几乎所有对距离敏感的模型（SVM、神经网络、聚类）都需要它
- **Pipeline**：把多个步骤串成一个整体，避免数据泄露和代码重复
- **ColumnTransformer**：对不同列做不同处理，各管各的

---

## 二、线性回归实战：房价预测

### 问题描述
波士顿房价预测——根据房屋的多个属性预测房价中位数。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# 加载加州房价数据（波士顿房价已废弃，用加州数据替代）
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 对比三种线性模型
models = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
}

for name, model in models.items():
    # 配合标准化使用
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name:6s} | RMSE={rmse:.4f} | R²={r2:.4f}")

# 多项式特征——让线性模型能拟合曲线
poly_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=1.0))
])
poly_pipe.fit(X_train, y_train)
y_pred_poly = poly_pipe.predict(X_test)
print(f"二次多项式+Ridge | RMSE={np.sqrt(mean_squared_error(y_test, y_pred_poly)):.4f} "
      f"| R²={r2_score(y_test, y_pred_poly):.4f}")
```

---

## 三、逻辑回归实战：信用卡欺诈检测

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# 生成不平衡的模拟数据（欺诈交易很少）
np.random.seed(42)
n = 5000
X_fraud = np.column_stack([
    np.random.normal(5000, 2000, n//50),   # 欺诈交易金额异常高
    np.random.normal(4, 2, n//50)          # 历史违约次数多
])
X_normal = np.column_stack([
    np.random.normal(300, 150, n),
    np.random.normal(0.5, 1, n)
])
X = np.vstack([X_normal, X_fraud])
y = np.hstack([np.zeros(n), np.ones(n//50)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 处理样本不均衡：class_weight='balanced'自动给少数类更高权重
lr = LogisticRegression(class_weight='balanced', C=1.0, random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

print(f"准确率: {lr.score(X_test, y_test):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
print("\n混淆矩阵:", confusion_matrix(y_test, y_pred).ravel())
print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['正常', '欺诈'])}")
```

---

## 四、MLP神经网络实战

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

# 造一个线性不可分的"月亮"数据集
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLP——专门对付非线性问题
mlp = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
mlp.fit(X_train, y_train)
print(f"MLP 测试精度: {mlp.score(X_test, y_test):.3f}")
print(f"实际迭代: {mlp.n_iter_}")
```

---

## 五、集成学习实战：XGBoost 与 LightGBM

```python
# 注意：需要先 pip install xgboost lightgbm
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    # 使用红酒数据集
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # XGBoost——工业界的顶流
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                        use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    print(f"XGBoost 测试精度: {xgb.score(X_test, y_test):.3f}")

    # LightGBM——微软出品，比XGBoost更快
    lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4,
                         verbose=-1, random_state=42)
    lgb.fit(X_train, y_train)
    print(f"LightGBM 测试精度: {lgb.score(X_test, y_test):.3f}")

    # 对比特征重要性
    importances_xgb = xgb.feature_importances_
    importances_lgb = lgb.feature_importances_
    print(f"\nXGBoost Top3特征: {np.argsort(importances_xgb)[-3:][::-1]}")
    print(f"LightGBM Top3特征: {np.argsort(importances_lgb)[-3:][::-1]}")
except ImportError:
    print("XGBoost/LightGBM 未安装，跳过此部分")
```

---

## 六、模型选择与调参最佳实践

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

# GridSearchCV——暴力搜索最佳参数
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

print("=== 网格搜索结果 ===")
print(f"最佳参数: {grid.best_params_}")
print(f"最佳CV分数: {grid.best_score_:.4f}")
print(f"测试集分数: {grid.score(X_test, y_test):.4f}")

# 查看所有参数组合的前三名
results = pd.DataFrame(grid.cv_results_)
top3 = results.nlargest(3, 'mean_test_score')
print(f"\nTop3参数组合:")
for i, (_, row) in enumerate(top3.iterrows()):
    print(f"  {i+1}. 分数={row['mean_test_score']:.4f} 参数={row['params']}")
```

---

## 七、终极武器：LSTM时间序列预测

```python
# LSTM属于深度学习范畴，需要TensorFlow/PyTorch
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    # 生成模拟时间序列（正弦波 + 噪声）
    t = np.linspace(0, 100, 1000)
    series = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)

    # 构造监督学习样本：用前20个值预测第21个
    def create_sequences(data, seq_length=20):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X_seq, y_seq = create_sequences(series, 20)
    X_seq = X_seq.reshape(-1, 20, 1)

    split = int(0.8 * len(X_seq))
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

    # LSTM模型
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(20, 1)),
        Dropout(0.1),
        LSTM(16),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X_train_seq, y_train_seq, epochs=30, batch_size=32,
              validation_data=(X_test_seq, y_test_seq), verbose=0)

    y_pred_seq = model.predict(X_test_seq, verbose=0)
    mse = mean_squared_error(y_test_seq, y_pred_seq)
    print(f"\nLSTM时间序列预测 | 测试MSE: {mse:.6f}")
except ImportError:
    print("TensorFlow 未安装，跳过LSTM部分")
```

---

## 小结

> scikit-learn统一了机器学习的API：所有模型都是`fit()`训练、`predict()`预测、`score()`评估。Pipeline帮你避免数据泄露，GridSearchCV帮你自动调参。实战中遵循"先跑通baseline → 改进特征 → 调参优化 → 尝试更复杂模型"的节奏，比一开始就上最复杂的方案有效得多。
