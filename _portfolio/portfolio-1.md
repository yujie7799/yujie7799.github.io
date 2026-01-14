---
title: "Python-儿童肺炎支原体肺炎患者-数据可视化与模型评估实战"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/python-data-visualization-picu
date: 2026-01-14
excerpt: "通过Matplotlib和Seaborn实现儿童肺炎支原体患者数据的探索性可视化与模型性能评估，为临床预后分析提供数据支撑"
header:
  teaser: /images/portfolio/portfolio_mpp_data_直方图.png
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
本项目针对重症监护（PICU）患者数据集，通过Python数据可视化技术完成探索性数据分析（EDA）与模型性能评估，旨在挖掘患者年龄、实验室指标与预后结局的关联规律，为临床预后预测提供数据支撑。项目核心目标包括掌握统计图表绘制方法、为不同分析场景选择合适可视化方案，以及提升代码复用效率。

## 核心实现

### 1. 探索性数据分析可视化
#### 年龄分布直方图
使用Seaborn绘制年龄分布直方图并叠加核密度估计曲线，直观展示患者年龄分布特征：
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.sans-serif"] = 'SimHei'
plt.rcParams["axes.unicode_minus"] = False

# 绘制年龄分布直方图
plt.figure(figsize=(8, 5))
sns.histplot(data=picu_data, x='age_month', kde=True)
plt.title("年龄分布直方图")
plt.show()
```

#### 不同结局下实验室指标分布箱线图
通过子图网格批量绘制多指标箱线图，对比存活与死亡患者的实验室指标分布差异：
```python
colname = ['age_month', 'lab_5237_min', 'lab_5227_min', 'lab_5225_range', 'lab_5235_max', 'lab_5257_min']

# 创建3x2子图网格
fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(10, 10))

# 批量绘制箱线图
for i in range(len(colname)):
    sns.boxplot(data=picu_data, x='HOSPITAL_EXPIRE_FLAG', y=colname[i], ax=axs[i//2, i%2])

plt.suptitle("不同结局下各实验室指标分布", fontsize=16)
plt.show()
```

### 2. 模型评估可视化
#### 混淆矩阵热力图
自定义函数绘制混淆矩阵热力图，直观展示分类模型的预测结果分布：
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def confusion_matrix_plot(y_true, y_pred_prob, threshold=0.5, title='混淆矩阵'):
    y_pred = (y_pred_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(['存活(0)', '死亡(1)'])
    ax.yaxis.set_ticklabels(['存活(0)', '死亡(1)'])
    plt.show()

# 调用函数绘制混淆矩阵
confusion_matrix_plot(y_true=y_test, y_pred_prob=y_pred_prob, threshold=0.5)
```

#### ROC曲线
计算并绘制ROC曲线，评估模型的分类性能：
```python
from sklearn.metrics import roc_curve, roc_auc_score

# 计算ROC曲线参数
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假阳性率(FPR)')
plt.ylabel('真阳性率(TPR)')
plt.title(f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.legend()
plt.show()
```

## 分析结果
### 1. 探索性数据分析结果
![年龄分布直方图](/images/portfolio/portfolio_mpp_data_直方图.png)
年龄分布直方图显示患者年龄呈现**双峰分布**，集中在婴幼儿期与学龄期，提示PICU患者群体存在明显的年龄分层特征。

![不同结局下各实验室指标分布箱线图](/images/portfolio/portfolio_mpp_data_箱线图.png)
箱线图对比显示，死亡患者的多项实验室指标（如`lab_5237_min`、`lab_5227_min`）分布与存活患者存在显著差异，部分指标的中位数与离散程度差异明显，可作为预后预测的潜在特征。

### 2. 模型评估结果
![混淆矩阵热力图](/images/portfolio/portfolio_mpp_data_热力图.png)
混淆矩阵显示模型对存活患者的识别准确率较高，但对死亡患者的漏诊率需进一步优化，提示模型在少数类样本上的性能有待提升。

![ROC曲线](/images/portfolio/portfolio_mpp_data_ROC曲线.png)
ROC曲线的AUC值为0.78，表明模型对患者预后的区分能力处于中等偏上水平，具备一定的临床应用潜力。

---
<img width="432" height="640" alt="image" src="https://github.com/user-attachments/assets/dff42d6e-ece4-4ecf-b486-68db06a24586" />
