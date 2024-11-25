---
title: Paper Reading- Text-to-Image Models
date: 2024-11-25
updated: 2024-11-25
categories: research
image: https://raw.githubusercontent.com/kiyotakali/kiyotakali.top/main/pic_back/ba12.webp
tags:
  - mllm
  - generative model
top: 1
---

## 《DART: DENOISING AUTOREGRESSIVE TRANSFORMER  FOR SCALABLE TEXT-TO-IMAGE GENERATION》

### Intro
首先介绍了**Diffusion Model**的原理以及他的缺陷
- 原理：逐步向图像添加高斯噪声并学习一系列的去噪步骤来逆向该过程。

- 缺陷：这些模型很难直接在高分辨率图像上进行训练

- 目前已有的解决方案：级联模型、multiscale方法、或者将图像预处理为较低的分辨率。

- 缺陷的可能原因：对马尔可夫假设的依赖，该假设限制了模型只能看到上一步的生成过程，导致训练和推理过程效率低下。

然后介绍了类似GPT-4的**自回归模型**，目前的自回归模型研究有将基于扩散的思想进行引入，但是仍然没有充分利用Diffusion Model的渐进式去噪优势，导致生成过程中局限的全局上下文和错误传递。

为了解决上述Diffusion Model和自回归模型各自的缺陷，作者提出了DART：一种将自回归模型与非马尔可夫Diffusion框架结合的模型，具体如下图所示。
![alt text](./image-25.png)

## 《Randomized Autoregressive Visual Generation》

文本数据是高度紧凑有序、含有语义意义，但是视觉数据往往更加low-level和冗余。过去关于自回归视觉模型的研究集中于将图像表示为**像素序列**，在受到最近的自回归语言模型的影响后将注意力转移到将图像建模为**离散值的tokens**,所以目标在于提高tokenization的质量以及使用现在的语言模型自回归模型架构进行训练。

然而这些工作都严格遵守处理像素或者tokens的raster-scan顺序，结果就是其实**单向的信息流对于视觉模型而言并不是最佳的**。这些传统的自回归模型依赖于因果注意力掩码，这强制了对token序列的单向依赖，但是在视觉模型中**双向注意力的效果是要明显优于因果注意力的**。

在本文中提出了一种新的视觉生成方法**RAR**(Randomized AutoRegressive)。具体来说，输入序列以概率$r$随机排列成不同的顺序，$r$从1开始在训练过程中线性衰减到0，r的衰减具体如下。

$$
r = 
\begin{cases}
    1.0, & \text{if } epoch < start, \\
    0.0, & \text{if } epoch > end, \\
    1.0 - \frac{epoch - start}{end - start}, & \text{otherwise},
\end{cases}
$$

其中，$epoch$ 是当前的训练轮数。

模型的优化目标如下：

$$
\begin{equation} \max_{\theta} p_{\theta}(\mathbf{x}) = \mathbb{E}_{\tau \sim S_T} \left[ \prod_{t=1}^T p_{\theta}(x_{\tau_t} | x_{\tau_{<t}}) \right], \tag{3} \end{equation}$$

其中 $ S_T $ 表示索引序列 $[1, 2, \cdots, T]$ 的所有可能排列的集合，$ \tau $ 表示从 $ S_T $ 中随机采样的排列。记号 $ \tau_t $ 指代排列后的序列中的第 $ t $-th 元素，而 $ \tau_{<t} $ 表示到 $ \tau_t $ 所有前面的位置。由于模型参数 $ \theta $ 在所有采样因子化顺序中共享，因此每个token $ x_t $ 都暴露于所有可能的上下文中，并且在训练期间与其他任何token $ x_i $ 学习relationship，其中 $ i \neq t $。这允许模型有效地捕获双向上下文同时保持自回归公式化的完整性。

具体的架构和形象化表示可以见下两张图：
![alt text](./image-26.png)
![alt text](./image-27.png)