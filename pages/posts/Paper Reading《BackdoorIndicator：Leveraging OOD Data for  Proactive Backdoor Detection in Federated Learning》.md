---
title: Paper Reading-《BackdoorIndicator:Leveraging OOD Data for  Proactive Backdoor Detection in Federated Learning》
date: 2024-10-28
updated: 2024-10-28
categories: research
image: https://raw.githubusercontent.com/kiyotakali/kiyotakali.top/main/pic_back/ba1.webp
tags:
  - graph
  - ood
  - backdoor 
top: 1
---

## Title: 
BackdoorIndicator: Leveraging OOD Data for  Proactive Backdoor Detection in Federated Learning

## Paper link:
[Here](https://export.arxiv.org/abs/2405.20862)

## Code:
[Here](https://github.com/ybdai7/Backdoor-indicator-defense)

## Intro:
首先在摘要中提出了两个事实:
- 在目标标签一致的时候，后续后门的植入可以帮助保持先前种植后门的准确性
- 基于数据中毒的后门攻击中，后门样本相对于良性样本都是ood样本

然后总结了一下过去联邦学习中防御后门攻击的方法，主要有两种：
- Influence reduction: 这种方法的假设是后门的更新相比良性的更新要更少，所以可以将模型更新的范数约定到一个商定的边界，或者也可以向全局模型添加足够的噪声来限制后门更新对全局模型的影响，但是在攻击者持续参与训练过程或者控制足够多的客户端时候这样的防御并不总是有效的。
- Detection and filtering: 在影响减小方法的基础上，这种方法假设后门的引入会导致上传的模型与正常更新的模型在参数空间上不同。这种方法的重心是根据接收到的模型参数评估的某些距离指标来设计聚合算法更新的机制，当然也可以人为定义特征来检测后门更新。上传的模型中特征值异常的将会被标记为可疑的模型，减少他的权重或者直接从聚合过程中移除。

但是在不同的setting下，仅仅比较参数可能并不能很好地解决后门。比方说对于高度non-i.i.d的数据分布，与良性更新相比，来自某个学习率较小的后门攻击的中毒更新可能更加接近参数空间的良性更新。

在这个工作中，他们首先是研究了多个后门依次种植时的相互影响，结果显示：一旦消除了batch normalization(BN)统一偏移引起的误导，使用相同目标标签植入后续后门有助于保持先前保持先前植入后门的持久性



## Method:

### Threat Model

攻击者目标和能力: 
攻击者旨在通过破坏本地客户端来将后门注入联邦学习模型，使模型在遇到某些后门触发器时行为异常，而其他任务不受影响。

一旦攻击者成功破坏了客户端，它就可以完全控制客户端的训练和模型上传过程。攻击者可以从任何回合开始连续进行后门的植入以及自由选择不同类型的后门进行注入，同时页不限制攻击者破坏的客户端数量(即恶意客户端数量)

防御者目标和能力:
防御者打算通过一定的防御协议来检测嵌入在上传的本地模型中的后门，并从聚合中排除潜在的恶意模型。

防御者不能访问客户端原始数据但是可以白盒获得客户端上传的模型。
