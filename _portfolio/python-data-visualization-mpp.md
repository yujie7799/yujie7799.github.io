---
title: "Python-儿童肺炎支原体肺炎患者-数据可视化与模型评估实战"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/python-data-visualization-mpp
date: 2026-01-07
excerpt: "通过Matplotlib和Seaborn实现儿童肺炎支原体患者数据的探索性可视化与模型性能评估，为临床预后分析提供数据支撑"
header:
  teaser: /image/portfolio/python-data-visualization-mpp
tags:
- 数据可视化
- Matplotlib
- Seaborn
- 机器学习
- 临床数据分析
tech_stack:
- name: Python
- name: Matplotlib
- name: Seaborn
- name: Scikit-learn
---

## 项目背景
本项目针对肺炎支原体流行期间：2023年10-12月，在首都医科大学附属首都儿童医学中心被诊断为肺炎支原体肺炎(MPP）患者数据集，依据儿童肺炎支原体肺炎诊疗指南 (2023 年版) 中 MPP 轻重症诊断标准将患儿分为重症肺炎支原体肺炎患儿与轻症肺炎支原体肺炎患儿。通过Python数据可视化技术完成探索性数据分析（EDA）与模型性能评估，旨在挖掘患者年龄、实验室指标与肺炎轻重的关联规律，为临床预后预测提供数据支撑。项目核心目标包括掌握统计图表绘制方法、为不同分析场景选择合适可视化方案，以及提升代码复用效率。

## 核心实现

### 1. 探索性数据分析可视化
#### 年龄分布直方图
使用Seaborn绘制年龄、血常规、血生化、血免疫细胞计数等分布直方图并叠加核密度估计曲线，直观展示指标分布特征：
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.sans-serif"] = 'Arial Unicode MS'

# 绘制患儿实验室检查指标分布直方图
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.sans-serif"] = 'Arial Unicode MS'
colname = ['年龄', '最长发热时间', 'IgM', 'IgE', 'IgA', 'IgG', '白细胞数', '中性粒细胞比值', '淋巴细胞比值', '血红蛋白', '乳酸脱氢酶', 'ALT']
fig, axs = plt.subplots(int(len(colname)/3), 3, constrained_layout=False, figsize=(8, 6), dpi=150)
for i in range(len(colname)):
...     row = i // 3
...     col = i % 3
...     ax = axs[row, col]
        sns.histplot(x=colname[i], data=mpp_data, alpha=0.4, kde=True, ax=axs[row, col])
...     ax.set_title(colname[i], fontsize=6, pad=3)
...     ax.set_xlabel(colname[i], fontsize=5, labelpad=2)
...     ax.set_ylabel('Count', fontsize=5, labelpad=2)
...     ax.tick_params(axis='both', labelsize=4, pad=1)
plt.tight_layout(pad=0.3, h_pad=0.8)
plt.show()

#由于指标较多，因此分成两次绘图，除变更列名，其余代码不变。
#colname =['白蛋白', 'D二聚体', 'C反应蛋白', 'PCT', 'CD3CD4T细胞数', 'CD3CD8T细胞数', 'CD4_CD8T细胞比值', 'NK细胞比值', 'B淋巴细胞比值']
```

#### 不同结局下实验室指标分布箱线图
通过子图网格批量绘制多指标箱线图，对比重症肺炎支原体肺炎与轻症肺炎支原体肺炎患者的实验室指标分布差异：
```python
colname = ['年龄', '最长发热时间', 'IgM', 'IgE', 'IgA', 'IgG', '白细胞数', '中性粒细胞比值', '淋巴细胞比值', '血红蛋白', '乳酸脱氢酶', 'ALT']
fig, axs = plt.subplots(int(len(colname)/3), 3, constrained_layout=False, figsize=(8, 6), dpi=150)
for i in range(len(colname)):
...     row = i // 3
...     col = i % 3
...     ax = axs[row, col]
...     sns.boxplot(data=mpp_data, x="MPP分类", y=colname[i], ax=axs[row, col])
...     ax.set_title(colname[i], fontsize=6, pad=3)
...     ax.set_xlabel(colname[i], fontsize=5, labelpad=2)
...     ax.set_ylabel('Count', fontsize=5, labelpad=2)
...     ax.tick_params(axis='both', labelsize=4, pad=1)
plt.tight_layout(pad=0.3, h_pad=0.8)
plt.show()
```
### 2. 模型构建
#### 创建逻辑回归模型
```python
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings("ignore")
log_reg = LogisticRegression(penalty='l2', C=1.0, l1_ratio=None, solver='lbfgs', max_iter=1000, class_weight=None, fit_intercept=True, random_state=42, n_jobs=None, verbose=0, multi_class='auto')
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict_proba(X_test)[:, 1]
model_filename = 'mpp_log_reg.joblib'
joblib.dump(log_reg, model_filename)
print(f"模型已保存到: {model_filename}")
```

### 3. 模型评估可视化
#### 混淆矩阵热力图
自定义函数绘制混淆矩阵热力图，直观展示分类模型的预测结果分布：
```python
def confusion_matrix_plot(y_true, y_pred_lr, threshold=0.5, title='混淆矩阵'):
...     y_true: 真实标签数组
...     y_pred_lr: 模型预测的正类概率数组
...     threshold: 分类阈值
...     title: 图表标题
...     y_pred = (y_pred_lr > threshold).astype(int)
...     cm = confusion_matrix(y_true, y_pred)
...     fig, ax = plt.subplots(figsize=(5, 4))
...     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
...     ax.set_xlabel('预测标签')
...     ax.set_ylabel('真实标签')
...     ax.set_title(title)
...     ax.xaxis.set_ticklabels(['轻症肺炎(0)', '重症肺炎(1)'])
...     ax.yaxis.set_ticklabels(['轻症肺炎(0)', '重症肺炎(1)'])
...     plt.show()

#调用函数绘制混淆矩阵
y_true=y_test
y_pred_lr=y_pred_lr
threshold=0.5
confusion_matrix_plot(y_true=y_test, y_pred_lr=y_pred_lr, threshold=0.5)
```

#### ROC曲线
计算并绘制ROC曲线，评估模型的分类性能：
```python
from sklearn.metrics import roc_curve, roc_auc_score

# 计算ROC曲线参数
fpr, tpr, _ = roc_curve(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_pred_lr)

# 绘制ROC曲线
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (1 - Specificity)')
plt.ylabel('真阳性率 (Sensitivity)')
plt.title(f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.legend(loc="lower right")
plt.show()
```
#### shap摘要图
自定义函数绘制shap摘要图，展示全局的特征重要性，以及特征值对模型输出的影响:
```python
Import Shap
explainer = shap.Explainer(log_reg, X_train)
shape_values = explainer(X_train)
shap_values = explainer(X_train)

# 绘制shap蜂群图
shap.summary_plot(shap_values, X_train)

# 绘制shap条形图
shap.summary_plot(shap_values, X_train, plot_type="bar")
```

## 分析结果
### 1. 探索性数据分析结果
![实验室检查分布直方图-1](/image/portfolio/python-data-visualization-mpp/直方图-1.png)
![实验室检查分布直方图-2](/image/portfolio/python-data-visualization-mpp/直方图-2.png)

如直方图所示：
可见以下指标呈集中分布，左右基本对称：年龄、最长发热时间、IgM、IgG、IgA、白细胞数、血红蛋白、白蛋白、CD4/CD8 T 细胞比值、B 淋巴细胞比值；
以下指标呈偏态分布：IgE、中性粒细胞比值、淋巴细胞比值、乳酸脱氢酶、D二聚体、CD3+CD4+ T 细胞数、CD3+CD8+ T 细胞数、NK 细胞比值、C 反应蛋白、ALT、PCT。

![不同结局下各实验室指标分布箱线图-1](/image/portfolio/python-data-visualization-mpp/箱线图-1.png)
![不同结局下各实验室指标分布箱线图-2](/image/portfolio/python-data-visualization-mpp/箱线图-2.png)

通过箱线图可见，CD3CD4T细胞数、CD3CD8T细胞数在GMPP组中显著高于SMPP组，而最长发热时间、白细胞数、白蛋白、D - 二聚体、C 反应蛋白、PCT及IgE、乳酸脱氢酶、ALT等在SMPP组中呈现更高的分布水平与更多高值异常点，提示SMPP组患儿的炎症反应与凝血激活程度更显著且免疫细胞水平更低，可作为预后预测的潜在特征。相比之下，年龄、IgA、IgG、血红蛋白等指标在两组间的分布无明显差异，表明分组与这些基础临床特征无显著关联。

### 2. 模型评估结果
![混淆矩阵热力图](/image/portfolio/python-data-visualization-mpp/热力图.png)

**结论：**混淆矩阵显示模型对轻症肺炎支原体肺炎患者的识别准确率较高，但对重症患者的漏诊率需进一步优化，提示模型在少数类样本上的性能有待提升。

![ROC曲线](/image/portfolio/python-data-visualization-mpp/ROC曲线.png)

**结论：**ROC曲线的AUC值为0.8077，表明该模型对患者预后的区分能力处于中等偏上水平，具备一定的临床应用潜力。

### 3. 模型解释性分析
![shap蜂群图](/image/portfolio/python-data-visualization-mpp/shap蜂群图-1.png)
**结论：**shap蜂群图显示中性粒细胞比值、D 二聚体、C 反应蛋白升高，淋巴细胞比值的降低会增加预测为重症肺炎支原体肺炎的概率。

![shap条形图](/image/portfolio/python-data-visualization-mpp/shap条形图-1.png)
**结论：**shap条形显示中性粒细胞比值的平均贡献度最大，是对模型预测贡献最大的指标。其次为淋巴细胞比值、D 二聚体、C 反应蛋白，是模型的重要辅助预测因子。其余指标的平均贡 献度较小，对模型整体预测的贡献相对有限.

---
