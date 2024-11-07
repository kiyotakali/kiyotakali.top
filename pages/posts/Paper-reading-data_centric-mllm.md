---
title: Paper Reading- Data-centric mllm
date: 2024-11-07
updated: 2024-11-07
categories: research
image: https://raw.githubusercontent.com/kiyotakali/kiyotakali.top/main/pic_back/elden2.webp
tags:
  - mllm
top: 1
---

## 《MAYBE ONLY 0.5% DATA IS NEEDED: A PRELIMINARY  EXPLORATION OF LOW TRAINING DATA INSTRUCTION TUNING》
![alt text](./image-17.png)
该文作者觉得现在缺乏与指令调整相关的研究来减少训练阶段使用的数据量来降低训练成本。

他们通过减少任务和指令的多样性来大量减少微调数据集的规模并仍然保证他在特定任务上的效果。

具体来说，他们的目标是从现有数据中识别中最有价值的核心样本，帮助模型获取下游任务所需要的知识，并仅用少量数据就能实现同等甚至更好的性能。

在实现方法上如上图所示，对文本数据使用预训练模型进行encoding然后进行均值池化和L2正则化，之后在一维向量 空间中使用K-means聚类来获得同一类别下游任务对应的数据并选择每个人物对应的中心点，最后使用KCentergreedy采样来获得一个缩小的但是接近完整数据集分布的小集合。