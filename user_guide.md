# 椎间盘退变一体化分析系统

## 1\. 项目简介

本项目是一个基于Python的一体化椎间盘影像分析平台。

系统通过一个集成的 GUI，将三个核心功能模块无缝衔接：

1.  **特征提取模块**: 基于标准的 PyRadiomics 库和一系列源自前沿学术研究的经典特征提取算法，以及基于先进的深度学习模型，对椎间盘影像特征进行量化。
2.  **图像扰动模块**: 对原始图像和分割掩码应用一系列标准化的扰动，以模拟临床实践中的各种不确定性。
3.  **稳健性相关性分析模块**: 通过一系列的统计方法（ICC、分层聚类、相关性分析），从海量特征中筛选出稳健且信息不冗余的“黄金特征集”。

本项目支持对单个病例进行深度分析，也支持对大规模队列数据进行自动化批量处理。

## 2\. 系统架构

本系统采用模块化设计，通过统一的GUI界面调度三大核心功能，形成一个完整的数据分析工作流。

  * **特征提取**: 输入原始图像和分割掩码，输出包含海量定量特征的数值表格。
  * **图像扰动**: 输入原始图像和分割掩码，输出一系列包含受控扰动的图像和掩码文件，用于后续的稳健性分析。
  * **稳健性相关性分析**: 输入一个包含了“金标准”和多种“扰动后”特征值的数据表，输出最终筛选出的稳健特征列表和详细的分析报告。

## 3\. 主要功能

### I. 特征提取模块

  - **PyRadiomics 标准特征**:
      - 一阶统计特征
      - 形状特征
      - 纹理特征 (GLCM, GLRLM, GLSZM, GLDM, NGTDM等)
  - **经典特征** (PyRadiomics无法提取的特征):
      - **[DHI]** 椎间盘高度指数
      - **[ASI]** 峰值信号强度差
      - **[FD]** 分形维度
      - **[T2SI]** T2信号强度比率
      - **[Gabor]** Gabor纹理特征
      - **[Hu]** Hu不变矩
      - **[Texture]** 扩展纹理特征 (如LBP)
  - **深度学习特征**: 利用一个在大型的3D医学数据集上预训练好的深度学习模型: Radio-DINO，提取更深层次的特征。
  - **张量分解特征**: 通过张量分解来提取3D图像特征，捕捉多维数据间的相关性。

### II. 图像扰动模块

  - **掩膜扰动**: 膨胀, 腐蚀, 轮廓随机化
  - **几何变换**: 平移, 旋转
  - **强度变换**: 高斯噪声
  - **组合扰动**: 支持上述多种扰动的组合应用。

### III. 稳健性相关性分析模块

  - **稳健性量化**: 采用组内相关系数 (ICC) 精确评估每个特征在不同扰动下的稳定性。
  - **稳健特征筛选**: 借鉴前沿研究方法，通过分层聚类识别并筛选出整体表现优异的稳健特征群。
  - **特征冗余消除**: 在稳健特征群内部，通过 spearman 相关性分析剔除信息高度重叠的冗余特征。
  - **交互式可视化**: 提供ICC热图、聚类树状图和相关性矩阵的可视化工具，辅助分析和决策。

## 4\. 安装指南

### 环境要求

  - Python 3.9+
  - Windows / Linux / macOS

### 安装步骤

1.  **克隆项目**

    ```bash
    https://github.com/SgSc733/ivd_degeneration_analysis.git
    cd your-repo-name
    ```

2.  **创建并激活Conda虚拟环境**

    ```bash
    conda create -n ivd python=3.9
    conda activate ivd
    ```

3.  **安装依赖**

    ```bash
    pip install -r requirements.txt
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```
   *若无gpu，请安装对应的cpu版torch

   *如果遇到 `numpy` 版本问题，请手动安装指定版本：

    ```bash
    pip install numpy==1.26.4
    ```

## 5\. 使用方法

#### 启动GUI界面

```bash
python run_gui.py
```

系统启动后，会出现一个包含三个主功能选项卡（“特征提取”、“图像扰动”、“稳健性相关性分析”）的集成界面，可根据研究需要，按顺序或独立使用这些模块。


------

## 模块一：特征提取

### 1.1 经典特征计算方法

#### 1.1.1 椎间盘高度指数 (DHI)
采用基于面积的方法计算。首先分别计算椎间盘及其上、下两个相邻椎体的高度，然后通过以下公式进行标准化，以消除个体差异。
$$
DHI = \frac{2 \times \text{Height}_{\text{disc}}}{\text{Height}_{\text{upper}} + \text{Height}_{\text{lower}}}
$$
*   **椎间盘高度 ($\text{Height}_{\text{disc}}$)**: 首先根据椎间盘掩码的像素区域，计算其总像素数量并乘以图像的体素间距（$x$ 和 $y$ 方向）得到椎间盘的面积。同时，计算椎间盘掩码的最小外接矩形的宽度（即在椎间盘长轴方向上的最大尺寸）。椎间盘高度定义为椎间盘面积除以其宽度。为提高稳健性，在计算面积时，仅使用椎间盘中央 $80\%$ 的区域（由 `central_ratio` 参数定义）。
*   **椎体高度 ($\text{Height}_{\text{upper}}$, $\text{Height}_{\text{lower}}$)**: 对于椎间盘上、下方的椎体，程序通过识别其各自掩码的像素区域，并计算这些像素在图像垂直方向（通常是 $y$ 轴）上的最大范围。这个垂直范围乘以 $y$ 方向的体素间距，即得到椎体的高度。

#### 1.1.2 峰值信号强度差 (ASI)
通过对椎间盘感兴趣区域内的信号强度直方图拟合一个双峰高斯混合模型，分别代表纤维环和髓核的信号分布。ASI量化了这两个峰值之间的差异，并使用脑脊液信号进行标准化。
$$
ASI = \frac{|\text{SI}_{\text{peak\_NP}} - \text{SI}_{\text{peak\_AF}}|}{\text{SI}_{\text{mean\_CSF}}}
$$
*   **$\text{SI}_{\text{peak\_NP}}$ 和 $\text{SI}_{\text{peak\_AF}}$**:
    *   首先从图像的椎间盘感兴趣区域中提取所有像素的信号强度值。
    *   然后，利用高斯混合模型对这些信号强度值进行建模。高斯混合模型假设感兴趣区域内的像素信号强度由两个独立的、遵循高斯分布的组（即髓核和纤维环）混合而成。
    *   模型通过迭代优化，找到这两个高斯分布的**均值（峰值）**和**方差**。其中，信号强度较高的均值被识别为髓核峰值信号强度 ($\text{SI}_{\text{peak\_NP}}$)，而信号强度较低的均值被识别为**纤维环峰值信号强度** ($\text{SI}_{\text{peak\_AF}}$)。
    *   **峰值信号强度差** ($|\text{SI}_{\text{peak\_NP}} - \text{SI}_{\text{peak\_AF}}|$ ) 即为这两个峰值均值之间的绝对差值。
*   **$\text{SI}_{\text{mean\_CSF}}$ (脑脊液平均信号强度)**:
    *   使用统一的椎管内容物掩码（标签值在 `config.py` 中由 `DURAL_SAC_LABEL` 定义）作为信号强度参照区域。
    *   程序在计算平均值之前，会自动执行一个信号提纯步骤：它会移除该区域内信号强度最低的25%的像素（通常对应神经根等非CSF组织），仅使用剩余75%的较亮像素来计算平均信号强度。
*   **高斯混合模型算法**:
    高斯混合模型是一种概率模型，它假设数据集是由 $K$ 个不同的高斯（正态）分布混合而成的。对于椎间盘信号强度这类一维数据，其概率密度函数可以表示为：
    $$
    p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)
    $$
    其中：
    *   $K$ 是混合分量的数量（在本项目中，$K=2$，分别代表髓核和纤维环）。
    *   $\pi_k$ 是第 $k$ 个高斯分量的**混合权重**，满足 $\sum_{k=1}^{K} \pi_k = 1$ 且 $0 \le \pi_k \le 1$。它表示一个数据点来自第 $k$ 个分量的先验概率。
    *   $\mathcal{N}(x | \mu_k, \sigma_k^2)$ 是第 $k$ 个高斯分量的概率密度函数，其均值为 $\mu_k$ (即信号峰值)，方差为 $\sigma_k^2$。

    为了从数据中找出这些未知参数 ($\pi_k, \mu_k, \sigma_k^2$)，我们采用最大期望算法，通过迭代来最大化数据的对数似然函数。该算法包含以下两个核心步骤：

    1.  **期望步 (E-step)**: 在这一步，我们基于当前的参数估计，计算每个数据点 $x_n$ 由每个高斯分量 $k$ 生成的后验概率，也称为“责任”：
        $$
        \gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n | \mu_j, \sigma_j^2)}
        $$
        $\gamma(z_{nk})$ 可以理解为数据点 $x_n$ “属于” 第 $k$ 个分量的概率。

    2.  **最大化步 (M-step)**: 在这一步，我们使用上一步计算出的“责任”来更新模型参数，以最大化期望的对数似然。更新公式如下：
        *   更新均值 $\mu_k$：
            $$
            \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) x_n
            $$
        *   更新方差 $\sigma_k^2$：
            $$
            (\sigma_k^2)^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (x_n - \mu_k^{\text{new}})^2
            $$
        *   更新混合权重 $\pi_k$：
            $$
            \pi_k^{\text{new}} = \frac{N_k}{N}
            $$
        其中，$N$ 是数据点的总数，$N_k = \sum_{n=1}^{N} \gamma(z_{nk})$ 是分配给分量 $k$ 的有效数据点数。

    算法通过反复交替执行 E 步和 M 步，直到参数收敛或达到最大迭代次数。最终，得到的两个均值 $\mu_1$ 和 $\mu_2$ 即被认为是髓核和纤维环的峰值信号强度。

#### 1.1.3 分形维度 (FD)
分形维度是衡量一个复杂形状如何填充空间的指标，它量化了图像的“不规则性”或“纹理复杂度”。在医学影像中，它常被用来描述病变区域内部结构的异质性。本项目采用的盒计数法来计算FD。

1.  **预处理**: 原始图像感兴趣区域经过一系列标准化处理（重采样、8位转换、窗位窗宽调整、二值化、边缘检测），最终生成一个代表椎间盘内部结构的二值化边缘图像。

2.  **盒计数原理**:
    *   **盒子尺寸序列**:首先根据边缘图像的尺寸，自动生成一系列递减的盒子边长 `s`。这些边长通常是2的幂次方（例如，`1, 2, 4, 8, ...` 像素），覆盖从最小单个像素到图像最小维度的范围。
    *   **计数盒子**: 对于每一个给定的盒子边长 `s`将这个二值化边缘图像覆盖在一个由边长为 `s` 的正方形组成的网格上。然后，统计所有至少包含一个前景像素（边缘点）的盒子的数量，记为 `N(s)`。
    *   **对数线性关系**: 分形对象的 `N(s)` 和 `s` 之间存在一个幂律关系：
        $$
        N(s) \propto s^{-D}
        $$
        其中 `D` 就是分形维度。为了求解 `D`，我们对上式两边取对数：
        $$
        \log(N(s)) = -D \cdot \log(s) + C
        $$
        这是一个线性方程。因此，通过在 log-log 坐标系中绘制 `log(N(s))` 相对于 `log(s)` 的关系图，并对这些数据点进行线性回归拟合，所得直线的斜率的相反数就是该图像的分形维度 `D`。
        一个更复杂的、更不规则的边缘轮廓会占据更多的盒子，导致 `D` 值更高。
3.  **盒计数法算法**:
    盒计数法是一种广泛使用的分形维度估计方法，其算法步骤如下：
    1.  **二值化处理**: 将感兴趣区域的图像转换为二值图像，其中前景像素（如边缘）为1，背景为0。
    2.  **定义盒子网格**: 选择一系列逐渐增大的盒子尺寸 $s_i$（例如，1, 2, 4, ...）。
    3.  **遍历计数**: 对于每个 $s_i$，将图像分割成 $s_i \times s_i$ 的网格。遍历所有网格单元，如果某个网格单元中包含任何一个前景像素，则将其计入有效盒子数量 $N(s_i)$。
    4.  **对数变换与拟合**: 收集所有 $(s_i, N(s_i))$ 对。然后计算 $\log(s_i)$ 和 $\log(N(s_i))$。
    5.  **线性回归**: 对数据点 $(\log(s_i), \log(N(s_i)))$ 执行线性回归，得到一条最佳拟合直线。
    6.  **计算维度**: 这条直线的斜率的负值即为该图像的分形维度D。

#### T2信号强度比率 (T2SI)
根据`TARGET-ROI`策略，在髓核中最亮的区域和椎管内的脑脊液中分别定义ROI。T2SI是这两个区域平均信号强度的直接比率，用于客观评估髓核的水合状态。
$$
ASI = \frac{\text{SI}_{\text{mean\_NP}}}{\text{SI}_{\text{mean\_CSF}}}
$$
*   **$\text{SI}_{\text{mean\_NP}}$ (髓核平均信号强度)**:
    *   首先从提供的椎间盘掩码中提取所有像素的信号强度值。
    *   然后，根据`brightness_percentile`参数（默认为75%），计算这些像素强度的第75百分位数作为阈值。
    *   所有信号强度高于此阈值且位于掩码内的连通区域被识别为“最亮髓核代理区域”。
    *   如果存在多个连通区域，则选择其中最稳定（变异系数最低）或最大的一个。
    *   最终，髓核平均信号强度 ($\text{SI}_{\text{mean\_NP}}$) 是该“最亮髓核代理区域”内所有像素的平均信号强度。
*   **$\text{SI}_{\text{mean\_CSF}}$ (脑脊液平均信号强度)**:
    *   使用统一的椎管内容物掩码（标签值在 `config.py` 中由 `DURAL_SAC_LABEL` 定义）作为信号强度参照区域。
    *   程序在计算平均值之前，会自动执行一个信号提纯步骤：它会移除该区域内信号强度最低的25%的像素（通常对应神经根等非CSF组织），仅使用剩余75%的较亮像素来计算平均信号强度。

#### 1.1.4 Gabor纹理特征
Gabor特征是一种强大的纹理分析工具，因为它能模拟人类视觉系统对不同频率和方向的感知。其计算过程如下：

1.  **构建Gabor滤波器组**: Gabor滤波器本质上是一个由高斯函数调制的复正弦函数。一个2D的Gabor滤波器可以通过以下公式定义：
    $$
    g(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi \frac{x'}{\lambda} + \psi\right)
    $$
    其中：
    *   $x' = x \cos\theta + y \sin\theta$ 和 $y' = -x \sin\theta + y \cos\theta$ 是旋转后的坐标。
    *   $\lambda$ (波长): 正弦波的波长，决定了要捕捉的纹理尺度。程序根据预设的波长列表（`wavelengths`参数）进行选择。
    *   $\theta$ (方向): 滤波器的方向。程序根据预设的方向列表（`orientations`参数）生成多个方向的滤波器。
    *   $\psi$ (相位偏移): 相位偏移。
    *   $\sigma$ (高斯标准差): 高斯包络的标准差，决定了滤波器的有效范围。
    *   $\gamma$ (空间纵横比): 决定了Gabor形状的椭圆度。
    通过组合不同的`wavelengths`和`orientations`，系统构建出一个“滤波器组”，每个滤波器对特定尺度和方向的纹理最敏感。

2.  **滤波与响应图**: 将椎间盘感兴趣区域的原始图像（或预处理后的图像）与滤波器组中的每一个Gabor滤波器进行卷积操作，会生成一系列“Gabor响应图”。每个响应图都突显了原始图像中与该滤波器参数匹配的纹理区域。

3.  **特征提取**: 最后，在每个Gabor响应图的椎间盘感兴趣区域内，计算一系列统计指标：
    *   **均值**: 响应图在该感兴趣区域内的平均像素强度。
    *   **标准差**: 响应图在该感兴趣区域内的像素强度标准差。
    *   **能量**: 响应图在该感兴趣区域内所有像素强度平方和，即 $\sum I_{ROI}^2$。它量化了纹理的“活跃度”或“强度”。
    *   **熵**: 响应图在该感兴趣区域内的像素强度直方图的香农熵，量化了纹理的随机性或无序程度。
    这些来自所有响应图的统计指标共同构成了一个高维的Gabor特征向量，全面地描述了椎间盘的纹理信息。

#### 1.1.5 Hu不变矩
Hu不变矩是七个用来描述物体形状的特征量。它们的核心优势在于其对平移、旋转和缩放等几何变换保持不变，因此非常适合进行鲁棒的形状分析。其计算基于图像的矩：

1.  **预处理**: 椎间盘掩码首先被处理成二值图像，其中椎间盘区域为前景（像素值为1），其他区域为背景（像素值为0）。
2.  **几何矩 ($M_{pq}$)**: 对于这个二值图像，其 $(p+q)$ 阶几何矩 $M_{pq}$ 定义为图像内所有像素位置 $(x, y)$ 及其对应像素值 $I(x, y)$ 的加权和：
    $$
    M_{pq} = \sum_{x}\sum_{y} x^p y^q I(x, y)
    $$
    其中，$I(x, y)$ 是像素 $(x, y)$ 的值（二值图像中为0或1）。
3.  **中心矩 ($\mu_{pq}$)**: 为了消除平移对特征的影响，引入中心矩。它是在图像的质心 $(\bar{x}, \bar{y})$ 处计算的：
    $$
    \mu_{pq} = \sum_{x}\sum_{y} (x-\bar{x})^p (y-\bar{y})^q I(x, y) \quad \text{其中 } \bar{x} = \frac{M_{10}}{M_{00}}, \bar{y} = \frac{M_{01}}{M_{00}}
    $$
    其中，$M_{00}$ 是图像的总像素数或总质量。
4.  **归一化中心矩 ($\eta_{pq}$)**: 为了进一步消除缩放对特征的影响，中心矩需要进行归一化：
    $$
    \eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^{(1 + \frac{p+q}{2})}}
    $$
5.  **Hu不变矩 ($h_1, \dots, h_7$)**: Hu通过对这些归一化的中心矩进行七种特定的非线性组合，推导出了七个对平移、旋转和缩放都保持不变的矩不变量。这七个公式列举如下：
    $$
    h_1 = \eta_{20} + \eta_{02}
    $$
    $$
    h_2 = (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2
    $$
    $$
    h_3 = (\eta_{30} - 3\eta_{12})^2 + (3\eta_{21} - \eta_{03})^2
    $$
    $$
    h_4 = (\eta_{30} + \eta_{12})^2 + (\eta_{21} + \eta_{03})^2
    $$
    $$
    h_5 = (\eta_{30} - 3\eta_{12})(\eta_{30} + \eta_{12})[(\eta_{30} + \eta_{12})^2 - 3(\eta_{21} + \eta_{03})^2] + (3\eta_{21} - \eta_{03})(\eta_{21} + \eta_{03})[3(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2]
    $$
    $$
    h_6 = (\eta_{20} - \eta_{02})[(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2] + 4\eta_{11}(\eta_{30} + \eta_{12})(\eta_{21} + \eta_{03})
    $$
    $$
    h_7 = (3\eta_{21} - \eta_{03})(\eta_{30} + \eta_{12})[(\eta_{30} + \eta_{12})^2 - 3(\eta_{21} + \eta_{03})^2] - (\eta_{30} - 3\eta_{12})(\eta_{21} + \eta_{03})[3(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2]
    $$

#### 1.1.6 扩展纹理特征

##### 1.1.6.1 局部二值模式 (LBP)
LBP是一种高效的纹理描述符，通过比较一个像素与其邻域像素的灰度值来工作。

1.  **LBP算子**: 对图像中的每一个中心像素 $(x_c, y_c)$，考虑其周围 $P$ 个邻域像素（在半径为 $R$ 的圆上采样）。将中心像素的灰度值 $g_c$ 作为阈值，邻域像素 $g_p$ 灰度值大于或等于中心的记为1，否则为0，形成一个$P$位的二进制数。
2.  **LBP值**: 将这个二进制数转换为十进制数，即为中心像素的LBP值。本项目采用`uniform`模式，它将某些不常见的LBP模式映射到特定的值，并为所有非`uniform`模式分配一个公共值。
    $$
    LBP_{P,R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p \quad \text{其中 } s(z) = \begin{cases} 1 & z \geq 0 \\ 0 & z < 0 \end{cases}
    $$
3.  **LBP特征**:
    *   **直方图分箱值**: 对椎间盘感兴趣区域内的所有LBP值构建直方图，反映了微观纹理模式的分布。程序直接使用该直方图的每个分箱的值作为特征。
    *   **熵**: LBP直方图的香农熵，量化LBP模式的随机性。
    *   **能量**: LBP直方图的平方和，量化LBP模式的强度或均匀性。
    *   **均值**和**标准差**: LBP图像本身在该感兴趣区域内的均值和标准差。

##### 1.1.6.2 基于梯度的特征
这些特征量化了感兴趣区域内信号强度的变化速率和方向。

1.  **梯度计算**: 使用 Sobel 算子计算图像在x和y方向上的一阶偏导 $G_x$ 和 $G_y$。
2.  **梯度幅值**: 描述了像素点上信号强度变化的大小或“陡峭”程度。
    $$
    \text{Mag}(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}
    $$
    计算椎间盘感兴趣区域内所有梯度幅值的一系列统计量，包括均值、标准差、最大值、偏度和峰度，以全面描述纹理的粗糙程度和对比度。
3.  **梯度方向**: 描述了信号强度变化最快的方向。
    $$
    \text{Dir}(x, y) = \arctan2(G_y(x, y), G_x(x, y))
    $$
    计算椎间盘感兴趣区域内所有梯度方向的圆周熵和平均合矢量长度。
    *   **圆周熵**: 通过构建梯度方向的直方图，并计算其香农熵来衡量方向分布的随机性。熵值高表示纹理方向混乱无序。
    *   **平均合矢量长度**: 将所有梯度方向视为单位矢量，计算它们的矢量和的长度再除以矢量数量。该值接近1表示纹理具有高度一致的方向性（例如，纤维环的层状结构），接近0表示方向随机。

##### 1.1.6.3 基于形态学的特征
这些特征基于掩码的几何形状，通过距离变换和骨架分析来提供PyRadiomics标准形状特征之外的信息。

1.  **距离变换**
    *   **算法原理**: 距离变换是一种计算二值图像中每个前景像素到最近背景像素距离的算法。本项目采用欧几里得距离变换，对于掩码前景区域 `F` 中的任意一个像素 `p`，其距离变换值定义为：
        $$
        EDT(p) = \min_{q \in B} \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}
        $$
        其中 `B` 是背景像素的集合。最终生成一个“距离图”，图中每个像素的强度值等于其EDT值。这个图谱直观地反映了形状的“厚度”分布。
    *   **提取的特征**:
        *   **平均距离**: 整个ROI内所有前景像素距离变换值的平均值。它反映了形状的平均“厚度”或“饱满度”。
            $$
            \mu_{dist} = \frac{1}{|F|} \sum_{p \in F} EDT(p)
            $$
            其中 $|F|$ 是前景像素的总数。
        *   **最大距离**: ROI内所有前景像素距离变换值的最大值。它对应形状最“厚”或最中心点的位置。
            $$
            max_{dist} = \max_{p \in F} EDT(p)
            $$

2.  **骨架分析**
    *   **算法原理**: 骨架分析（或称中轴变换）是一种将二维形状细化为一条宽度仅为一像素的“骨架”线的过程，同时保持原始形状的拓扑结构（如连通性和孔洞）。本项目采用基于迭代细化的形态学算法，通过反复剥离对象边界像素，直到无法再移除任何像素而不改变其拓扑连通性为止，最终得到形状的骨架 `S`。
    *   **提取的特征**:
        通过对生成的骨架 `S` 进行邻域分析来提取以下特征。对于骨架上的任意像素 `p`，我们统计其3x3邻域内的骨架像素数量 $C(p)$：
        $$
        C(p) = \left( \sum_{i=-1}^{1} \sum_{j=-1}^{1} S(p_x+i, p_y+j) \right) - 1
        $$
        *   **骨架像素数**: 构成骨架的总像素数量，反映了形状的整体“长度”或“范围”。
            $$
            N_{skel} = \sum_{p \in S} 1
            $$
        *   **分支点数**: 骨架上一个点连接了三条或更多路径的点，其数量反映了形状的复杂分支结构。一个像素 `p` 是分支点，当且仅当 $C(p) > 2$。
            $$
            N_{branch} = \sum_{p \in S} \mathbb{I}(C(p) > 2) \quad \text{其中 } \mathbb{I}(\cdot) \text{ 是指示函数}
            $$
        *   **端点数**: 骨架上只连接了一条路径的点，反映了形状末梢的数量。一个像素 `p` 是端点，当且仅当 $C(p) = 1$。
            $$
            N_{end} = \sum_{p \in S} \mathbb{I}(C(p) = 1)
            $$

#### 1.1.7 椎管狭窄量化 (DSCR)
该特征旨在客观量化由椎间盘突出或黄韧带肥厚等原因造成的椎管狭窄程度，其目的是比较实际椎管轮廓和理想椎管轮廓。

1.  **构建理想椎管后缘曲线**:
    1.  **地标点提取**: 首先从提供的掩码中识别出代表椎体后缘的解剖学地标点。
    2.  **B样条拟合**: 利用这些地标点的坐标，使用B样条插值算法拟合出一条平滑的曲线。这条曲线代表了在没有椎间盘退变影响下的理想椎管后缘。

2.  **获取实际硬脊膜囊前缘**:
    从提供的硬脊膜囊分割掩码中（即统一的“椎管内容物”掩码），提取在椎间盘水平上，硬脊膜囊的最前缘（靠近椎间盘的一侧）的轮廓。


3.  **计算实际椎管前后径 ($d$)**:
    在待测椎间盘的垂直水平上，找到硬脊膜囊掩码在水平方向（通常是 $x$ 轴）上的最小 $x$ 坐标（硬脊膜囊前缘）和最大 $x$ 坐标（硬脊膜囊后缘）。这两个坐标之间的距离即为实际椎管前后径。

4.  **计算理想椎管前后径 ($m$)**:
    在相同的椎间盘垂直水平上，计算从B样条曲线定义的“理想椎管后缘”到实际硬脊膜囊最大 $x$ 坐标（硬脊膜囊后缘）的水平距离。这个距离即为理想椎管前后径。

5.  **DSCR计算**:
    最后，椎管狭窄率（DSCR）通过比较实际和理想的椎管前后径来计算：
    $$
    \text{DSCR} = (1 - \frac{d}{m}) \times 100
    $$
    DSCR值越高，表示椎管狭窄的百分比越严重。该方法通过引入“理想轮廓”作为基准，可以更准确地量化由局部病变（如椎间盘突出）引起的椎管侵占。
*   **B样条插值算法**:
    B样条是一种强大的分段多项式曲线表示方法，它通过一组控制点、一个节点向量和多项式的次数来定义。一条 $p$ 次的 B 样条曲线 $C(t)$ 的数学表达式为：
    $$
    C(t) = \sum_{i=0}^{n} P_i N_{i,p}(t)
    $$
    其中：
    *   $P_i$ 是 $n+1$ 个控制点，它们构成一个控制多边形，曲线的形状由这些点的位置决定。
    *   $N_{i,p}(t)$ 是第 $i$ 个 $p$ 次的 B 样条基函数，它在参数 $t$ 处定义了控制点 $P_i$ 对曲线的“权重”或“影响”。

    基函数 $N_{i,p}(t)$ 是通过 Cox-de Boor 递归公式定义的：
    *   **基本情况 (0次)**:
        $$
        N_{i,0}(t) = \begin{cases} 1 & \text{if } u_i \le t < u_{i+1} \\ 0 & \text{otherwise} \end{cases}
        $$
    *   **递归步骤 ($p>0$)**:
        $$
        N_{i,p}(t) = \frac{t - u_i}{u_{i+p} - u_i} N_{i,p-1}(t) + \frac{u_{i+p+1} - t}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(t)
        $$
    这里的 $u_i$ 是节点向量 $U = \{u_0, u_1, ..., u_m\}$ 中的元素。节点向量是一个非递减的参数值序列，它决定了多项式段之间的连接点和曲线的平滑度。

    在 DSCR 的计算中，我们的目标是找到一条穿过所有给定地标点 $Q_j$ 的平滑曲线。这是一个插值问题：我们需要求解未知的控制点 $P_i$。通过为每个地标点 $Q_j$ 分配一个参数值 $t_j$，我们可以建立一个线性方程组：
    $$
    Q_j = C(t_j) = \sum_{i=0}^{n} P_i N_{i,p}(t_j) \quad \text{for } j=0, \dots, n
    $$
    这个方程组可以写成矩阵形式 $[Q] = [M][P]$，其中 $[M]$ 是一个由基函数在各个 $t_j$ 处的值构成的矩阵。通过求解这个线性方程组，就可以得到控制点 $[P]$，从而唯一确定了穿过所有地标点的理想椎管后缘曲线。B样条算法能够生成平滑且灵活的曲线，这对于模拟解剖结构（如椎体后缘）的连续性至关重要。

    **注**：有可能遇到硬脊膜囊掩码缺失 / 太窄的情况，此时 dscr 无法计算，输出结果是空值，属于正常现象。

### 1.2 PyRadiomics 标准特征提取原理简介

本系统在3维图像输入下提取了超过1600个PyRadiomics标准特征（2维输入可提取1000+特征）。由于数量庞大，此处不一一列举，而是介绍其核心的构建原理。这些特征主要分为三大类：

#### 1.2.1. 一阶统计特征
一阶特征描述了感兴趣区域内体素强度的分布情况，它们不考虑体素间的空间关系，仅基于强度直方图计算。

*   **核心思想**: 将感兴趣区域内的所有体素强度值视为一个集合，分析其统计分布特性。
*   **关键特征**:
    *   **能量**: $\text{Energy} = \sum_{i=1}^{N_g} p(i)^2$ (衡量图像强度分布的均匀性)
    *   **熵**: $\text{Entropy} = -\sum_{i=1}^{N_g} p(i) \log_2(p(i))$ (衡量图像强度的随机性或复杂性)
    *   **均值**, **标准差**, **偏度**, **峰度** 等标准统计量。
    其中，$p(i)$ 是灰度值为 $i$ 的体素出现的概率，$N_g$ 是离散化后的灰度级总数。

#### 1.2.2. 形状特征
形状特征描述了感兴趣区域的几何形态，这些特征与体素强度无关，仅在二值化的掩码上计算。

*   **核心思想**: 量化感兴趣区域的三维（或二维）大小和形状。
*   **关键特征**: 体积, 表面积, 球度, 致密度, 伸长度 , 扁平度 。

#### 1.2.3. 纹理特征
纹理特征通过分析体素间的空间关系来量化图像的异质性，这是放射组学中最丰富、最强大的特征类别。它们主要通过构建不同的共现矩阵来实现：

*   **灰度共生矩阵**: 描述在指定方向和距离上，成对体素的灰度值出现频率，反映图像的方向性、对比度、均匀性。
*   **灰度游程矩阵**: 量化在特定方向上具有相同灰度值的连续体素的“游程”长度，用于捕捉图像中的线性结构。
*   **灰度尺寸区域矩阵**: 量化由空间连通的、具有相同灰度值的体素组成的“区域”的大小，用于捕捉斑块状纹理。
*   **灰度依赖矩阵**: 量化一个体素与其邻域体素在灰度值上的依赖性，描述体素与其周围环境的相似度。
*   **邻域灰度差分矩阵**: 通过计算每个体素与其邻域平均灰度值之差来量化图像的“空间变化率”，描述纹理的复杂性和变化剧烈程度。


### 1.3 深度学习特征
为捕捉传统方法难以量化的信息，本系统集成了一个基于 **Radio-DINO** 模型的深度学习特征提取器，可为每个椎间盘提取4000+特征。Radio-DINO 是一个在大型放射学图像数据集（RadImageNet）上通过自监督学习预训练的视觉Transformer (ViT) 模型。源项目地址如下：https://github.com/Snarci/Radio-DINO 。

1.  **模型与预训练**:
    *   **模型架构**: 采用 ViT 架构，它将2D图像切片分割成一系列不重叠的图像块，并将这些块作为序列输入到Transformer编码器中。
    *   **自监督学习**: 使用 DINO 框架进行预训练。DINO 让模型在没有任何人工标注的情况下，通过对比同一图像不同增强版本（视图）的输出来学习图像的内在表示。这使得模型能够捕捉到图像的语义信息，例如区分不同的组织结构。
    *   **预训练数据集**: 在包含135万张医学图像的 **RadImageNet** 数据集上进行预训练，确保模型学习到的特征具有跨模态、跨解剖区域的泛化能力。

2.  **特征提取原理**:
    系统采用一种基于感兴趣区域的特征提取策略，以确保深度特征严格对应于特定的解剖结构（如单个椎间盘）。
    1.  **三视图切片提取**: 对于每个3D椎间盘掩码，首先计算其质心，并从该质心位置提取矢状面、冠状面、轴状面三个正交的2D图像切片及其对应的2D掩码。
    2.  **预处理**: 每个2D切片都经过一系列标准化处理：裁剪、添加边距、缩放至224x224、三通道复制、ImageNet标准化，以符合ViT模型的输入要求。
    3.  **获取Patch令牌**: 预处理后的2D图像张量输入到Radio-DINO模型。我们提取模型最后一个Transformer Block的输出，该输出是一个包含了所有图像块的高维表示的序列，称为Patch令牌。
        $$
        \text{Tokens}_{\text{patch}} = \text{ViT}_{\text{final\_layer}}(\text{Image}_{\text{preprocessed}})
        $$
    4.  **构建Patch级掩码**: 将224x224的2D掩码下采样到与ViT的Patch网格相匹配的尺寸（例如14x14）。这创建了一个二值的“Patch掩码”，指明了哪些Patch块落在了感兴趣区域内部。
    5.  **掩码聚合**: 使用Patch掩码从所有Patch令牌中筛选出那些完全位于感兴趣区域内的令牌。然后，根据用户选择的聚合策略对这些筛选出的令牌进行池化操作：
        *   **Mean Pooling**: 计算所有选中令牌的平均值，捕捉区域的整体、平均特征。
        *   **Max Pooling**: 计算所有选中令牌的最大值，捕捉区域内最显著、最突出的特征。
        *   **Both**: 同时计算平均值和最大值，并将两者拼接，形成一个维度加倍的、更全面的特征向量。
    6.  **特征拼接**: 最后，将从三个视图（矢状、冠状、轴状）中提取的特征向量进行拼接，形成代表该椎间盘的最终高维深度特征向量。


### 1.4 张量分解特征

#### 1.4.1 张量分解简介

**记号**

- 标量：小写字母（例如 $x$）。
- 向量：粗体小写字母（例如 $\mathbf{x}$），$\mathbf{x}\in\mathbb{R}^I$。
- 矩阵：粗体大写字母（例如 $\mathbf{X}$），$\mathbf{X}\in\mathbb{R}^{I\times J}$。
- 张量：花体大写字母（例如 $\mathcal{X}$），$N$ 阶张量为
  $$
  \mathcal{X}\in\mathbb{R}^{I_1\times I_2\times\cdots\times I_N}.
  $$
  向量是 1 阶张量，矩阵是 2 阶张量。

**模‑$n$ 展开**

张量的模-$n$ 展开是将高维张量重新排列为矩阵的过程。对于 $N$ 阶张量 $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$，其模-$n$ 展开记为 $\mathbf{X}_{(n)}$。

定义：$\mathbf{X}_{(n)} \in \mathbb{R}^{I_n \times (I_1 \cdots I_{n-1} I_{n+1} \cdots I_N)}$ 的列向量由张量的模-$n$ fibers 组成。具体地，张量元素 $(i_1, i_2, \dots, i_N)$ 映射到矩阵元素 $(i_n, j)$，其中
$$
j = 1 + \sum_{k=1, k \neq n}^{N} (i_k - 1) J_k, \quad \text{with } J_k = \prod_{m=1, m \neq n}^{k-1} I_m.
$$
直观上，$\mathbf{X}_{(1)}$ 是将张量的“列”作为列向量排列，$\mathbf{X}_{(2)}$ 是将张量的“行”作为列向量排列，$\mathbf{X}_{(3)}$ 是将张量的“管” 作为列向量排列。

**模‑$n$ 张量–矩阵乘积**

$N$ 阶张量 $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ 与矩阵 $\mathbf{U} \in \mathbb{R}^{J \times I_n}$ 的模-$n$ 乘积记为 $\mathcal{X} \times_n \mathbf{U}$，结果为一个 $N$ 阶张量，其尺寸为 $I_1 \times \cdots \times I_{n-1} \times J \times I_{n+1} \times \cdots \times I_N$。

定义
$$
(\mathcal{X} \times_n \mathbf{U})_{i_1 \dots i_{n-1} j i_{n+1} \dots i_N} = \sum_{i_n=1}^{I_n} x_{i_1 \dots i_n \dots i_N} u_{j i_n}
$$

这等价于
$$
\mathcal{Y} = \mathcal{X} \times_n \mathbf{U} \iff \mathbf{Y}_{(n)} = \mathbf{U} \mathbf{X}_{(n)}
$$

**Frobenius 范数**

对张量 $\mathcal{X}$，Frobenius 范数定义为
$$
\|\mathcal{X}\|_F
=\sqrt{\sum_{i_1,\dots,i_N} x_{i_1\cdots i_N}^2},
$$
等价于把所有元素当作一个长向量后的 $\ell^2$ 范数。

**核范数**

对矩阵 $M$，其奇异值为 $\{\sigma_k\}$ 时，核范数为
$$
\|M\|_* = \sum_k \sigma_k,
$$

**Tucker 分解**

Tucker 分解可被视为高阶的主成分分析（PCA）。对于三阶张量 $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times I_3}$，其分解形式为：
$$
\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)}
$$
其中：
- $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$ 为 Core Tensor，描述了不同模态因子之间的相互作用水平，其元素值反映了对应特征组合的能量或重要性。
- $\mathbf{U}^{(n)} \in \mathbb{R}^{I_n \times R_n}$ 为正交的因子矩阵，通常包含该模态下的主成分基向量。
- $(R_1, R_2, R_3)$ 为多线性秩，决定了模型的复杂度。

**CP 分解（CANDECOMP/PARAFAC）**

CP 分解将张量分解为有限个秩-1 张量的和，可视作 Tucker 分解的特例（核心张量为超对角阵）。对三阶张量
$$
\mathcal{X}\in\mathbb{R}^{I_1\times I_2\times I_3},
$$
CP 分解写作
$$
\hat{\mathcal{X}}
=\sum_{r=1}^R \lambda_r\, a_r\circ b_r\circ c_r,
$$
其中
- $\lambda_r\in\mathbb{R},\ a_r\in\mathbb{R}^{I_1},\ b_r\in\mathbb{R}^{I_2},\ c_r\in\mathbb{R}^{I_3}$；
- $\circ$ 为外积运算；
- $R$ 为 CP rank。

若把列向量堆叠成矩阵
$$
A=[a_1,\dots,a_R],\quad
B=[b_1,\dots,b_R],\quad
C=[c_1,\dots,c_R],
$$
则 $\hat{\mathcal{X}}$ 完全由 $(A,B,C,\lambda_1,\dots,\lambda_R)$ 决定。

**软阈值算子**

对于标量 $x \in \mathbb{R}$ 和阈值 $\tau > 0$，软阈值算子定义为：
$$
S_\tau(x) = \operatorname{sgn}(x) \max(|x| - \tau, 0)
$$

#### 1.4.2 数据预处理

1. 对每个椎间盘层面，基于分割结果取覆盖终板上下少量椎体的 3D 包围盒，裁剪出大小统一的 ROI，例如（？）：
   $$
   I_x\times I_y\times I_z = 64\times 64\times 32.
   $$
   如原始体素尺寸不均一，先重采样到 $(1~\text{mm})^3$。

   注：“张量特征参数”里的 `ROI尺寸 (Z,Y,X)` 提供“自动”选项：系统会基于当前输入的图像和掩码，统计所有椎间盘掩码的物理包围盒尺寸，按“98% 分位 + 安全边界”的策略自动给出一个统一的张量 ROI 尺寸。

2. 对每个 ROI 做强度归一化：
   - 先按分位数（如 $[1\%,99\%]$）或 IQR 去除极端值；
   - 再做 $z$‑score 标准化（均值 0，方差 1），每个 ROI 单独标准化。

#### 1.4.3 第一类：全局 Tucker 特征

##### 1.4.3.1 目标

对单个椎间盘 ROI $\mathcal{X}\in\mathbb{R}^{I_x\times I_y\times I_z}$ 做 HOSVD / Tucker 分解。该过程旨在将高维 MRI 信号投影到由数据驱动的正交基底上，捕获其在三个空间方向上的“主变形模式”和整体结构复杂度：

- 类似 3D PCA，但不把体素拉平成一维向量，而是保留 $(x,y,z)$ 三个模之间的结构；
- HOSVD 的奇异值和  core tensor  刻画了椎间盘在不同方向上的能量分布和耦合关系；
- 重构误差可以理解为“用少数主模态能否近似这个椎间盘”的程度。

##### 1.4.3.2 HOSVD 算法

对 $\mathcal{X}\in\mathbb{R}^{I_x\times I_y\times I_z}$，$\mathcal{X}$ 可以被视为“信号（结构信息）”与“噪声”的叠加。HOSVD 通过保留核心张量中能量较大的部分，重构出一个低秩近似张量 $\hat{\mathcal{X}}$。$\hat{\mathcal{X}}$ 是 $\mathcal{X}$ 在低维子空间上的投影。残差 $\mathcal{E} = \mathcal{X} - \hat{\mathcal{X}}$ 包含了高频噪声或非典型的微细纹理。我们将 $\hat{\mathcal{X}}$ 的核心能量特征（代表主要解剖结构）与残差 $\mathcal{E}$ 的能量特征（代表纹理复杂度和噪声水平）同时作为诊断特征。HOSVD 算法步骤如下：

1. 分别按 3 个模展开并做 SVD：
   $$
   X_{(1)} = U^{(1)}\Sigma^{(1)}V^{(1)\top},\quad
   X_{(2)} = U^{(2)}\Sigma^{(2)}V^{(2)\top},\quad
   X_{(3)} = U^{(3)}\Sigma^{(3)}V^{(3)\top},
   $$
   其中 $U^{(n)}$ 为左奇异向量矩阵，$\Sigma^{(n)}$ 为奇异值对角矩阵。
2. 得到 HOSVD 形式：
   $$
   \mathcal{X}
   =\mathcal{S}\times_1 U^{(1)}\times_2 U^{(2)}\times_3 U^{(3)}.
   $$
3. 为获得低秩近似，需选定多线性截断秩 $(R_1,R_2,R_3)$。对每个模 $n=1,2,3$，令奇异值为
   $\{\sigma^{(n)}_k\}_{k\ge 1}$，选最小 $R_n$ 使得
   $$
   \frac{\sum_{k=1}^{R_n}\bigl(\sigma^{(n)}_k\bigr)^2}
        {\sum_k \bigl(\sigma^{(n)}_k\bigr)^2}
   \ge \eta,\qquad \eta\approx 0.95.
   $$
4. 构造截断因子矩阵
   $$
   \tilde{U}^{(n)} = U^{(n)}(:,1:R_n),
   $$
   并得到截断  core tensor  ：
   $$
   \tilde{\mathcal{S}}
   = \mathcal{X}
   \times_1 \tilde{U}^{(1)\top}
   \times_2 \tilde{U}^{(2)\top}
   \times_3 \tilde{U}^{(3)\top}.
   $$
5. 重构张量：
   $$
   \hat{\mathcal{X}}
   = \tilde{\mathcal{S}}
   \times_1 \tilde{U}^{(1)}
   \times_2 \tilde{U}^{(2)}
   \times_3 \tilde{U}^{(3)}.
   $$

#### 1.4.3.3 特征构造

1. 各模主奇异值向量

   取每个模的前 $K_n$ 个奇异值，$K_n\approx 10\sim 15$：
   $$
   f^{(n)}_{\sigma}
   =\bigl[\sigma^{(n)}_1,\dots,\sigma^{(n)}_{K_n}\bigr].
   $$
   这些值反映该模整体能量及其衰减速度。

2. 各模能量分布

   定义
   $$
   e^{(n)}_k
   =\frac{\bigl(\sigma^{(n)}_k\bigr)^2}
          {\sum_j \bigl(\sigma^{(n)}_j\bigr)^2},
   \quad k=1,\dots,K_n,
   $$
   刻画能量是否集中在少数主方向上。

3. core tensor 能量与方向“重要性”分布

   - 核心总能量：
     $$
     E_{\text{core}}=\|\tilde{\mathcal{S}}\|_F^2.
     $$
   - 沿第 1 模的能量分布：
     $$
     p_1(i)
     =\frac{\sum_{j,k}\tilde{\mathcal{S}}_{ijk}^2}
            {\|\tilde{\mathcal{S}}\|_F^2},\quad i=1,\dots,R_1,
     $$
     类似可定义 $p_2(j),p_3(k)$。
   - 直观解释：$p_1,p_2,p_3$ 描述  core tensor  在各主方向上的能量集中程度，可以反映椎间盘的各向异性模式。
   - 注意：由于每个病例的 $R_n$ 不一定相同，从而对应的特征数不同，因此输出结果中 tucker 特征部分大概率会出现空值，属于正常现象。

4. 重构误差比例

   定义
   $$
   r
   =\frac{\|\mathcal{X}-\hat{\mathcal{X}}\|_F^2}
          {\|\mathcal{X}\|_F^2}.
   $$
   - 若 $r$ 很小，说明少数模式就能很好近似原图，结构较规则；
   - 若 $r$ 较大，说明结构复杂或噪声较多。

#### 1.4.4 第二类：非局部低秩 patch 张量特征

##### 1.4.4.1 目标

基于Wang, Z., et al. (2022)提出的 NLRTA-LSR（Nonlocal Low-Rank Tensor Approximation with Logarithmic-Sum Regularization） 框架，构建非局部相似 patch 组成的四阶张量，并采用对数和（Logarithmic-Sum）范数作为低秩正则项。与传统的核范数不同，对数和正则化能自适应地对奇异值进行不同程度的收缩，从而更好地区分图像的解剖结构（大奇异值）和噪声（小奇异值）。本模块可提取：
- 在对数和正则化约束下表现出的非凸低秩特性；
- 自适应奇异值权重分布（反映局部结构的稀疏性）；
- 经过 NLRTA-LSR 优化后的重构误差与各向异性指标。

#### 1.4.4.2 patch 组张量构造

1. 方差稳定变换 (VST)

由于 MRI 的 Rician 噪声具有信号依赖性，首先对原始椎间盘 ROI 数据 $\mathcal{Y}_{raw}$ 进行前向 VST 变换，将其转化为近似的高斯白噪声分布数据 $\mathcal{Y}$：
$$
\mathcal{Y} = f_{VST}(\mathcal{Y}_{raw}, \sigma_n) = \sqrt{\mathcal{Y}_{raw}^2 + \sigma_{n}^2}
$$
其中 $\sigma_n$ 为预估的背景噪声标准差。

2. Method Noise 外部迭代

为了找回在去噪过程中可能丢失的Method Noise，算法引入外部反馈机制。设定外部迭代次数 $Iter$（通常取 2），在第 $k$ 次迭代中：
$$
\mathcal{Y}^{(k)} = \mathcal{X}^{(k-1)} + \alpha (\mathcal{Y} - \mathcal{X}^{(k-1)})
$$
其中 $\mathcal{Y}$ 是初始 VST 数据，$\mathcal{X}^{(k-1)}$ 是上一轮的恢复结果（初始 $\mathcal{X}^{(0)} = \mathcal{Y}$），$\alpha$ 为反馈系数。

3. Patch 组张量构造
   
对当前的输入 $\mathcal{Y}^{(k)}$，利用非局部自相似性构建张量：
- 分块：以步长 $m-1$ 滑动，提取尺寸为 $m \times m \times m$ 的参考块。
- 搜索与匹配：在 $s \times s \times s$ 的搜索窗口内，寻找与参考块欧氏距离最小的 $n$ 个相似块。
- 堆叠：将这 $n$ 个相似块沿第 4 维堆叠，形成观测张量 $\mathcal{G}_{\mathcal{Y}} \in \mathbb{R}^{m \times m \times m \times n}$。

##### 1.4.4.3 对数和正则化模型 (NLRTA-LSR)

对每个 patch 组张量 $\mathcal{G}_{\mathcal{Y}}$，旨在寻找低秩近似张量 $\mathcal{G}_{\mathcal{X}}$，优化目标函数定义为：
$$
\min_{\mathcal{G}_{\mathcal{X}}} \sum_{i=1}^{4} \theta_i \bigl\|\mathcal{G}_{\mathcal{X},(i)}\bigr\|_{log} \quad \text{s.t.} \quad \|\mathcal{G}_{\mathcal{Y}} - \mathcal{G}_{\mathcal{X}}\|_F^2 \le \varepsilon
$$
转化为无约束拉格朗日形式：
$$
\min_{\mathcal{G}_{\mathcal{X}}} \sum_{i=1}^{4} \theta_i \bigl\|\mathcal{G}_{\mathcal{X},(i)}\bigr\|_{log} + \frac{\mu}{2} \|\mathcal{G}_{\mathcal{Y}} - \mathcal{G}_{\mathcal{X}}\|_F^2
$$

其中：
- $\mathcal{G}_{\mathcal{X},(i)}$ 是张量沿第 $i$ 模的展开矩阵；
- $\theta_i$ 为权重，满足 $\sum \theta_i = 1$且 $\theta_i \ge 0$；
- $\|\mathbf{A}\|_{log}$ 为矩阵的**对数和范数**，定义为：
  $$
  \|\mathbf{A}\|_{log} = \sum_{j=1}^{r} \log(\delta_j(\mathbf{A}) + \varepsilon)
  $$
  这里 $\delta_j(\mathbf{A})$ 是矩阵 $\mathbf{A}$ 的第 $j$ 个奇异值，$\varepsilon$ 是一个微小的常数用于保持数值稳定性。
该模型是非凸的，能比核范数更强地诱导低秩性，同时对大奇异值（重要特征）的惩罚更小。

##### 1.4.4.4 数值求解：ADMM 与加权奇异值阈值

采用交替方向乘子法（ADMM）将上述问题分解为子问题求解。引入辅助变量 $\mathcal{O}_i = \mathcal{X}_{group}$，将原问题转化为约束优化，并构造增广拉格朗日函数。ADMM 迭代核心在于求解辅助变量 $\mathcal{O}_i$ 的更新。

1. 更新辅助变量 $\mathcal{O}_i$ (转化为 WNNM 问题)：
   固定其他变量，求解 $\mathcal{O}_i$ 等价于求解加权核范数最小化（WNNM）问题：
   $$
   \mathcal{O}_i^{t+1} = \arg\min_{\mathcal{O}_i} \frac{\theta_i}{\mu^t} \|\mathcal{O}_{i,(i)}\|_{log} + \frac{1}{2} \|\mathcal{O}_i - \mathcal{T}_i^t\|_F^2
   $$
   其中:
   - $\mathcal{T}_i^t = \mathcal{G}_{\mathcal{X}}^t - \frac{\mathcal{Q}_i^t}{\mu^t}$ ($\mathcal{Q}_i$ 为拉格朗日乘子)。
   - $\mathcal{O}_i^{t+1}$：本轮迭代中，基于第 $i$ 模态展开矩阵奇异值收缩后折叠回来的张量。
   - $\mathcal{Q}_i^t$：第 $i$ 个约束对应的拉格朗日乘子张量（记录了上一轮的残差信息）。
   - $\mu^t$：当前的惩罚参数。
   
   根据 Wang, Z., et al. (2022) Lemma 1，该问题有闭式解，通过加权奇异值阈值算子(WSVT)求解。设 $\mathcal{T}_{i,(i)}$ 的奇异值分解为 $\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$，则更新后的奇异值 $\pi_j$ 为：
   $$
   \pi_j(\mathcal{O}_{i,(i)}) = \max\left(0, \frac{c_1 + \sqrt{c_2}}{2}\right) \quad \text{if } c_2 \ge 0, \text{ else } 0
   $$
   其中：
   - $c_1 = \delta_j(\mathcal{T}_{i,(i)}) - \varepsilon$
   - $c_2 = (\delta_j(\mathcal{T}_{i,(i)}) + \varepsilon)^2 - \frac{4\theta_i}{\mu^t}$
   
   **注意**：此处产生的自适应权重隐含在阈值操作中，其等效权重 $\omega_j$ 与奇异值大小成反比，即 $\omega_j = \frac{1}{\delta_j + \varepsilon}$。

2. 更新全局张量 $\mathcal{G}_{\mathcal{X}}$：
   $$
   \mathcal{G}_{\mathcal{X}}^{t+1} = \frac{1}{4} \sum_{i=1}^{4} \left( \mathcal{O}_i^{t+1} + \frac{\mathcal{Q}_i^t}{\mu^t} \right)
   $$

3. 更新乘子 $\mathcal{Q}_i$ 与惩罚参数 $\mu$：
   $$
   \mathcal{Q}_i^{t+1} = \mathcal{Q}_i^t + \mu^t (\mathcal{O}_i^{t+1} - \mathcal{G}_{\mathcal{X}}^{t+1})
   $$
   $$
   \mu^{t+1} = \rho \mu^t
   $$
   其中 $\rho$ 是一个大于 1 的常数（取 $\rho \approx 1.05 \sim 1.1$），保证算法在后期强制收敛。

4. 收敛性检查: 
   计算相对误差，若小于阈值（如 $\epsilon = 10^{-4}$）则停止迭代：
   $$
   \frac{\|\mathcal{G}_{\mathcal{X}}^{t+1} - \mathcal{G}_{\mathcal{X}}^{t}\|_F^2}{\|\mathcal{G}_{\mathcal{X}}^{t}\|_F^2} \le \epsilon
   $$

##### 1.4.4.5 特征构造

利用上述算法对每个 patch 组优化后的低秩张量 $\hat{\mathcal{X}}_{group}$ 及其过程参数提取特征：

1. 各模态的奇异值

   对最终恢复的低秩张量 $\hat{\mathcal{X}}_{group}$ 进行四个方向的模态展开，得到矩阵 $\hat{\mathcal{X}}_{(n)}$ ($n=1,2,3,4$)。对每个矩阵进行 SVD 分解，获取奇异值向量：
   $$
   \boldsymbol{\delta}^{(n)} = [\delta_1^{(n)}, \delta_2^{(n)}, \dots, \delta_{r}^{(n)}]
   $$

2. 自适应权重向量

   利用公式计算对应的权重向量 $\boldsymbol{\omega}^{(n)}$：
   $$
   \omega_j^{(n)} = \frac{1}{\delta_j^{(n)} + \varepsilon}
   $$
   同时计算均值和标准差。其中 $\varepsilon$ 是一个极小正常数（如 $10^{-16}$）。

3. 对数和正则化强度

   计算每个模态的对数和范数值，这反映了在该模态下数据的稀疏程度：
   $$
   L^{(i)} = \sum_{j} \log(\delta_j^{(i)} + \varepsilon)
   $$
   同时计算$L^{(1)}, \dots, L^{(4)}$ 的均值。


4. 非局部自相似残差
   
   $$
   e_{NL} = \frac{\|\mathcal{G}_{\mathcal{Y}} - \hat{\mathcal{G}}_{\mathcal{X}}\|_F^2}{\|\mathcal{G}_{\mathcal{Y}}\|_F^2}
   $$
   同时计算所有 patch 组 $e_{NL}$ 的均值和标准差，反映该区域是否符合“非局部低秩”假设，病变区域往往具有异常的结构复杂性。

5. 对数和有效秩

   对最终低秩张量 $\hat{\mathcal{G}}_{\mathcal{X}}$ 的 4 个模态展开矩阵，分别计算：
   $$
   R_{log}^{(i)} = \sum_{j} \log\left(\delta_j(\hat{\mathcal{G}}_{\mathcal{X},(i)}) + \varepsilon\right), \quad i=1,2,3,4
   $$

   计算 ROI 内所有 Patch 组的 $R_{log}^{(i)}$ 的均值，构成 4 维特征向量：
   $$
   \mathbf{f}_{rank} = \left[ \text{mean}(R_{log}^{(1)}), \text{mean}(R_{log}^{(2)}), \text{mean}(R_{log}^{(3)}), \text{mean}(R_{log}^{(4)}) \right]
   $$

#### 1.4.5 第三类：单病例 CP 分解张量特征

##### 1.4.5.1 目标

对单个椎间盘三阶张量直接做 CP 分解（CANDECOMP/PARAFAC），将其表示为若干个秩‑1 张量之和，并用这些模式的权重及空间分布统计量刻画椎间盘内部信号的组织结构复杂度，从这些秩‑1 成分的权重和因子向量中构造描述“多向协同变化模式”的特征。

##### 1.4.5.2 CP 模型

对单个已标准化的椎间盘 ROI（三阶张量）
$$
\mathcal{X}\in\mathbb{R}^{I_x\times I_y\times I_z},
$$
考虑秩为 $R$ 的 CP 分解：
$$
\hat{\mathcal{X}}
=\sum_{r=1}^{R}\lambda_r\cdot\mathbf{a}_r\circ\mathbf{b}_r\circ\mathbf{c}_r,
$$
其中：

- $\lambda_r\in\mathbb{R}$ 为第 $r$ 个秩‑1 成分的权重；
- $\mathbf{a}_r\in\mathbb{R}^{I_x}$、$\mathbf{b}_r\in\mathbb{R}^{I_y}$、$\mathbf{c}_r\in\mathbb{R}^{I_z}$ 分别为三条空间方向上的因子向量；
- $\circ$ 为向量外积。

将因子列向量堆叠成矩阵：
$$
\mathbf{A}=[\mathbf{a}_1,\dots,\mathbf{a}_R]\in\mathbb{R}^{I_x\times R},\quad
\mathbf{B}=[\mathbf{b}_1,\dots,\mathbf{b}_R]\in\mathbb{R}^{I_y\times R},\quad
\mathbf{C}=[\mathbf{c}_1,\dots,\mathbf{c}_R]\in\mathbb{R}^{I_z\times R}.
$$

约定在分解的任何时刻都保持
$$
\|\mathbf{a}_r\|_2=\|\mathbf{b}_r\|_2=\|\mathbf{c}_r\|_2=1,
$$
所有尺度信息集中到 $\lambda_r$ 中。此时若忽略成分之间的非正交性，可以把 $\lambda_r^2$ 近似理解为第 $r$ 个模式所携带的“能量大小”。

##### 1.4.5.3 单 ROI 预处理

1. 尺寸重采样
    - 依据模块二中统一设定，将 ROI 裁剪到相同尺寸，例如：
      $$
      I_x\times I_y\times I_z = 64\times 64\times 32.
      $$
2. 强度裁剪与标准化
    - 取本 ROI 体素强度的 $[1\%,99\%]$ 分位数做裁剪，去除极端值；
    - 在 ROI 内做 $z$‑score 标准化：
      $$
      x'_{ijk}
      =\frac{x_{ijk}-\mu_{\mathcal{X}}}{\sigma_{\mathcal{X}}},
      $$
      其中 $\mu_{\mathcal{X}},\sigma_{\mathcal{X}}$ 均为本 ROI 体素的均值和标准差。

后续 CP 分解都在标准化后的张量上进行。为简化记号，下文仍记为 $\mathcal{X}$。

##### 1.4.5.4 CP 分解的 ALS 求解

对单个三阶张量 $\mathcal{X}$，采用交替最小二乘（ALS）求解 CP 分解，目标函数为最小化重构误差：
$$
\min_{\mathbf{A},\mathbf{B},\mathbf{C},\boldsymbol{\lambda}}
\left\|
\mathcal{X}
-\sum_{r=1}^R \lambda_r\cdot \mathbf{a}_r\circ \mathbf{b}_r\circ \mathbf{c}_r
\right\|_F^2.
$$

1. 初始化（直接用 HOSVD 结果初始化）
    - 对 $\mathcal{X}$ 做三模展开：
      $$
      X_{(1)},X_{(2)},X_{(3)}
      $$
      并各自做 SVD：
      $$
      X_{(1)}=U^{(1)}\Sigma^{(1)}V^{(1)\top},\
      X_{(2)}=U^{(2)}\Sigma^{(2)}V^{(2)\top},\
      X_{(3)}=U^{(3)}\Sigma^{(3)}V^{(3)\top}.
      $$
    - 取前 $R$ 列作为初始因子矩阵：
      $$
      \mathbf{A}^{(0)}=U^{(1)}(:,1:R),\
      \mathbf{B}^{(0)}=U^{(2)}(:,1:R),\
      \mathbf{C}^{(0)}=U^{(3)}(:,1:R).
      $$
    - 初始权重：
      $$
      \lambda_r^{(0)}
      =\langle\mathcal{X},\mathbf{a}_r^{(0)}\circ\mathbf{b}_r^{(0)}\circ\mathbf{c}_r^{(0)}\rangle
      =\sum_{i,j,k}x_{ijk}\mathbf{a}^{(0)}_{r,i}\mathbf{b}^{(0)}_{r,j}\mathbf{c}^{(0)}_{r,k}.
      $$
2. ALS 迭代更新因子矩阵

   定义 Khatri–Rao 积（列对应 Kronecker 积）：
   $$
   \mathbf{C}\ast\mathbf{B}
   =[\mathbf{c}_1\otimes\mathbf{b}_1,\dots,\mathbf{c}_R\otimes\mathbf{b}_R]
   \in\mathbb{R}^{(I_yI_z)\times R},
   $$
   记 $\odot$ 为 Hadamard 乘积。

   在第 $t$ 轮迭代中：
    - 更新 $\mathbf{A}$：
      $$
      \mathbf{A}^{(t+1)}
      = X_{(1)}\bigl(\mathbf{C}^{(t)}\ast\mathbf{B}^{(t)}\bigr)
      \left(
      \bigl(\mathbf{C}^{(t)\top}\mathbf{C}^{(t)}\bigr)\odot
      \bigl(\mathbf{B}^{(t)\top}\mathbf{B}^{(t)}\bigr)
      \right)^{\dagger},
      $$
    - 更新 $\mathbf{B}$：
      $$
      \mathbf{B}^{(t+1)}
      = X_{(2)}\bigl(\mathbf{C}^{(t)}\ast\mathbf{A}^{(t+1)}\bigr)
      \left(
      \bigl(\mathbf{C}^{(t)\top}\mathbf{C}^{(t)}\bigr)\odot
      \bigl(\mathbf{A}^{(t+1)\top}\mathbf{A}^{(t+1)}\bigr)
      \right)^{\dagger},
      $$
    - 更新 $\mathbf{C}$：
      $$
      \mathbf{C}^{(t+1)}
      = X_{(3)}\bigl(\mathbf{B}^{(t+1)}\ast\mathbf{A}^{(t+1)}\bigr)
      \left(
      \bigl(\mathbf{B}^{(t+1)\top}\mathbf{B}^{(t+1)}\bigr)\odot
      \bigl(\mathbf{A}^{(t+1)\top}\mathbf{A}^{(t+1)}\bigr)
      \right)^{\dagger}.
      $$
3. 列归一化与权重更新

   为保持稳定性和可解释性，在每轮迭代末，对每个成分 $r$：
    - 计算各模向量的范数：
      $$
      s_a=\|\mathbf{a}_r^{(t+1)}\|_2,\quad
      s_b=\|\mathbf{b}_r^{(t+1)}\|_2,\quad
      s_c=\|\mathbf{c}_r^{(t+1)}\|_2;
      $$
    - 归一化列向量：
      $$
      \mathbf{a}_r^{(t+1)}\leftarrow\frac{\mathbf{a}_r^{(t+1)}}{s_a},\quad
      \mathbf{b}_r^{(t+1)}\leftarrow\frac{\mathbf{b}_r^{(t+1)}}{s_b},\quad
      \mathbf{c}_r^{(t+1)}\leftarrow\frac{\mathbf{c}_r^{(t+1)}}{s_c};
      $$
    - 将缩放系数累积到权重：
      $$
      \lambda_r^{(t+1)}\leftarrow \lambda_r^{(t)}\cdot s_a s_b s_c.
      $$
4. 收敛条件

   $$
   \hat{\mathcal{X}}^{(t)}
   =\sum_{r=1}^R\lambda_r^{(t)}\mathbf{a}_r^{(t)}\circ\mathbf{b}_r^{(t)}\circ\mathbf{c}_r^{(t)},\quad
   \mathcal{E}^{(t)}=\mathcal{X}-\hat{\mathcal{X}}^{(t)}.
   $$

   若
   $$
   \frac{\|\mathcal{E}^{(t)}\|_F^2-\|\mathcal{E}^{(t-1)}\|_F^2}
   {\|\mathcal{X}\|_F^2}
   \le \varepsilon_{\text{ALS}}
   $$
   如果 $\varepsilon_{\text{ALS}}=10^{-4}$，或迭代次数超过 $T{\max}$（如 1000），则停止。

##### 1.4.5.5 秩 $R$ 的选取

- 经验上，CP 秩通常取一个较小到中等的整数以避免病态问题，医学图像应用中多取 $R\in[3,10]$；
- 默认：$R=10$

可通过以下方式检验：对每个 ROI 计算重构误差比例
$$
r_{\text{CP}}
=\frac{\|\mathcal{X}-\hat{\mathcal{X}}\|_F^2}{\|\mathcal{X}\|_F^2},
$$
若在绝大多数 ROI 上 $r_{\text{CP}}<0.05\sim 0.1$，说明 $R$ 足以表达主要结构。

##### 1.4.5.6 特征构造

在得到单 ROI 的 CP 分解 $(\boldsymbol{\lambda},\mathbf{A},\mathbf{B},\mathbf{C})$ 后，从三个层面构造特征：
1）权重谱与能量衰减；2）因子“集中程度”（空间分布）；3）整体重构质量。

1. 权重与能量谱特征

   首先按权重绝对值排序：

   - 找到排列 $\pi$ 使得
     $$
     |\lambda_{\pi(1)}|\ge|\lambda_{\pi(2)}|\ge\cdots\ge|\lambda_{\pi(R)}|;
     $$
   - 定义重排序后的权重和因子：
     $$
     \tilde{\lambda}_r=\lambda_{\pi(r)},\quad
     \tilde{\mathbf{a}}_r=\mathbf{a}_{\pi(r)},\
     \tilde{\mathbf{b}}_r=\mathbf{b}_{\pi(r)},\
     \tilde{\mathbf{c}}_r=\mathbf{c}_{\pi(r)}.
     $$

   定义：

   - 主权重向量：
      $$
      f_{\lambda}
      =[\tilde{\lambda}_1,\dots,\tilde{\lambda}_R]^T.
      $$
   - 相对能量比例：
      $$
      E_{\lambda}
      =\sum_{r=1}^R \tilde{\lambda}_r^2,\qquad
      e_r
      =\frac{\tilde{\lambda}r^2}{E_{\lambda}+\varepsilon},\ r=1,\dots,R,
      $$
      其中 $\varepsilon$ 防止除零。向量 $f_e=[e_1,\dots,e_R]^T$ 描述能量是否集中在少数几个成分。
   - 有效 CP 秩：
      $$
      R_{\text{eff}}
      =\frac{\left(\sum_{r=1}^R \tilde{\lambda}_r^2\right)^2}
      {\sum_{r=1}^R \tilde{\lambda}_r^4+\varepsilon}.
      $$

2. 因子向量的“集中度”与熵

   对每个成分 $r$ 和每个模（以 $\tilde{\mathbf{a}}_r$ 为例），把平方归一化为一维分布：
   $$
   p^{(a)}_{r,i}
   =\frac{\bigl(\tilde{\mathbf{a}}_{r,i}\bigr)^2}{\sum_{k=1}^{I_x}\bigl(\tilde{\mathbf{a}}_{r,k}\bigr)^2},
   \quad i=1,\dots,I_x,
   $$
   类似可得 $p^{(b)}_{r,j},p^{(c)}_{r,k}$。

   对每个模定义：

   - 熵：
      $$
      H^{(a)}_r
      =-\sum_{i=1}^{I_x}p^{(a)}_{r,i}\log(p^{(a)}_{r,i}+\varepsilon),
      $$
      $$
      H^{(b)}_r,\ H^{(c)}_r\ \text{类似定义}.
      $$
      熵小表示该方向分布集中（局灶模式），熵大表示分布均匀（弥散模式）。
   - 集中度（类似 Gini 指标）：
      $$
      G^{(a)}_r
      =\sum_{i=1}^{I_x}\bigl(p^{(a)}_{r,i}\bigr)^2,
      $$
      $$
      G^{(b)}_r,\ G^{(c)}_r\ \text{类似定义}.
      $$
      $G$ 越接近 1，越“尖锐集中”；越接近 $1/I_x$，越平滑分散。

   为控制维度，只对前 $K$ 个最重要成分统计这些量，取 $K=\min(3,R)$；特征集合为：
     $$
     \{H^{(a)}_r,H^{(b)}_r,H^{(c)}_r,G^{(a)}_r,G^{(b)}_r,G^{(c)}_r\}_{r=1}^K.
     $$

3. 整体强度

   定义两个整体强度指标：
   $$
   S_{\lambda}
   =\sum_{r=1}^R|\tilde{\lambda}_r|,\qquad
   \|\boldsymbol{\tilde{\lambda}}\|_2
   =\left(\sum_{r=1}^R\tilde{\lambda}_r^2\right)^{1/2}.
   $$

4. 特征向量汇总

   综合上述，单个椎间盘 ROI 的第三类张量特征向量定义为：
   $$
   F^{(3)}_{tensor}
   =\bigl(
   f_{\lambda},\
   f_e,\
   R_{\text{eff}},\
   S_{\lambda},\
   \|\boldsymbol{\tilde{\lambda}}\|_2,\
   \{H^{(\cdot)}_r,G^{(\cdot)}_r\}_{r=1}^K
   \bigr).
   $$


### 1.5 详细参数设置说明

所有核心参数均在 `config.py` 文件中进行统一管理，方便用户根据具体研究需求进行调整。下面对主要参数进行详细说明。

#### I. 核心设置

| 参数 | 示例值 | 意义 |
| :--- | :--- | :--- |
| `DISC_LABELS` | `{'L1-L2': {'disc': 3, ...}}` | 定义掩码文件中每个椎间盘及其相邻椎体的标签值。 |
| `NUM_SLICES` | `3` | 指定从3D图像中提取用于2D分析的中间切片数量。 |
| `SLICE_AXIS` | `0` | 指定切片方向 (0: 矢状位, 1: 冠状位, 2: 轴位)。 |

#### II. 经典特征的预处理参数 (`PREPROCESSING_PARAMS`)

| 类别 | 参数 | 示例值 | 意义 |
| :--- | :--- | :--- | :--- |
| **通用** | `target_size` | `[512, 512]` | 空间重采样的目标尺寸，确保所有分析都在统一的空间分辨率下进行。 |
| **纹理** | `bin_width` | `16` | 强度离散化的组宽度，用于计算纹理矩阵前减少噪声影响。 |
| **纹理** | `normalize` | `True` | 是否对ROI内强度进行Z-score标准化，提升特征在不同设备下的鲁棒性。 |
| **纹理** | `robust` | `False` | Z-score标准化时是否使用中位数/IQR替代均值/标准差，以抵抗离群值。 |
| **分形** | `window_center` | `128` | 窗位窗宽调整的窗位。 |
| **分形**| `window_width` | `255` | 窗位窗宽调整的窗宽。此组合用于标准化8位图像对比度。 |
| **分形**| `threshold_percentile` | `65` | 二值化的灰度阈值百分比，用于从灰度图中分离前景结构。 |
| **分形**| `edge_method` | `'canny'` | 边缘检测算法，可选 `'canny'`, `'sobel'`等。 |
| **信号**| `interpolation` | `'linear'` | 重采样时用于信号强度图像的插值方法。 |

#### III.经典特征计算器参数

| 特征 | 参数 | 示例值 | 意义与参考文献 |
| :--- | :--- | :--- | :--- |
| **DHI** | `central_ratio` | `0.8` | 计算椎间盘和椎体高度时，所使用的中心区域比例，以避免边缘效应。 |
| **DHI** | `calculate_dwr` | `True` | 是否额外计算椎间盘-椎体宽度比 (Disc-to-Vertebra Width Ratio)。 |
| **ASI** | `n_components` | `2` | 拟合信号强度直方图的高斯混合模型(GMM)的组分数（髓核NP+纤维环AF）。 |
| **ASI** | `scale_factor` | `255.0` | 信号强度值的缩放因子。 |
| **FD** | `threshold_percent`| `0.65` | 二值化阈值，与预处理参数联动。 |
| **FD** | `min_box_size` | `1` | 盒计数法的最小盒子边长（像素）。 |
| **T2SI** | `roi_method` | `'TARGET'` | 定义髓核(NP)的ROI策略。'TARGET'模式旨在勾画最亮区域。 |
| **T2SI** | `brightness_percentile`| `75` | 在'TARGET'模式下生效，用于定义“最亮”区域的信号强度百分位阈值。 |
| **T2SI** | `min_roi_size` | `20` | 'TARGET'模式下，生成的ROI所需的最小像素数。 |
| **Gabor**| `wavelengths` | `[2, 4, ...]` | Gabor滤波器组的波长列表，用于捕捉不同尺度的纹理。 |
| **Gabor**| `orientations` | `None` | Gabor滤波器组的方向列表。`None`表示使用默认的多角度。 |
| **Gabor**| `frequency`, `sigma`, `gamma`, `psi` | ... | Gabor滤波器的其他标准数学参数。 |
| **扩展纹理**| `lbp_radius` | `1` | LBP算法的邻域半径。 |
| **扩展纹理**| `lbp_n_points` | `8` | LBP算法的邻域采样点数。 |

#### IV. PyRadiomics滤波器参数 (`FILTER_PARAMS`)

| 滤波器 | 参数 | 示例值 | 意义 |
| :--- | :--- | :--- | :--- |
| **LoG** | `sigma_list` | `[1, 3, 5]` | 高斯拉普拉斯(LoG)滤波器的Sigma值列表，用于增强特定尺寸的斑点状结构。 |
| **Wavelet** | `wavelet`, `level`| `'db1'`, `1` | 小波变换的类型和分解层级，用于在不同频率子带提取特征。 |

更多参数请查阅PyRadiomics官方使用文档：https://pyradiomics.readthedocs.io/

#### V. 深度学习特征参数

| 参数 | 可选项 | 意义 |
| :--- | :--- | :--- |
| **模型版本** | `base`, `small` | 选择使用的Radio-DINO预训练模型的大小。`base`模型更大、特征维度更高（768维），理论上性能更强但计算开销也更大；`small`模型更轻量，特征维度较低（384维），计算速度更快。 |
| **Patch聚合策略**| `mean`, `max`, `both` | 定义如何将ROI内部的多个图像块的深度特征聚合成一个单一向量。`mean`捕捉平均特征，`max`捕捉最显著特征，`both`将两者拼接以获得更丰富的信息。 |
| **安全边距**| `0.2` | 在根据掩码裁剪ROI时，向外扩展的边距比例。例如，`0.2`表示在ROI的每个方向上增加其自身尺寸20%的边距，以包含更多的上下文信息。 |

#### VI. 张量分解特征参数

| 参数 | 示例值 | 意义 |
| :--- | :--- | :--- |
| `roi_size` | `[72, 40, 64]` | 在重采样到各向同性体素后，张量 ROI 的统一尺寸，顺序为 (Z, Y, X)。例如 `[72,40,64]` 表示在切片方向 Z 取 72 个体素、在前后方向 Y 取 40 个体素、在左右方向 X 取 64 个体素。程序以椎间盘掩码的质心为中心构造这个固定大小的立方体 ROI。若原始图像某一维度不足，会自动做零填充，不会缩小 ROI。 |
| `target_spacing_mm` | `1.0` | 重采样到张量 ROI 前，所有 3D 图像统一到的各向同性物理间距 (mm)。推荐保持 `1.0`，即 1mm³ 体素，方便将 `roi_size` 直接解释为 mm。 |
| `q_low` | `1` | ROI 内强度分布的下分位数 (百分位)，用于强度裁剪（去除极端低值，例如噪声和伪影）。只影响强度，不改变 ROI 几何范围。 |
| `q_high` | `99` | ROI 内强度分布的上分位数 (百分位)，用于强度裁剪（去除极端高值，例如极亮噪声点）。裁剪后再对掩码内的像素做 z‑score 标准化。 |
| `energy_threshold` | `0.95` | 能量阈值 $\eta$。对每个模的奇异值序列，选择最小的多线性秩 $R_n$，使前 $R_n$ 个奇异值的能量比例 $\sum_{k=1}^{R_n}\sigma_k^2 / \sum_j \sigma_j^2 \ge \eta$。该值越接近 1，保留的主模态越多，重构误差越小，但特征维度增加。 |
| `k_singular_values` | `10` | 每个模中用于构造特征的主奇异值个数 $K_n$。例如取 10 表示对每个模的前 10 个奇异值及其能量比例进行编码，形成 “奇异值谱” 特征。 |
| `patch_size` | `4` | Patch 尺寸 $m$，构造 $m\times m\times m$ 的 3D 小块。 |
| `similar_patches` | `64` | 相似块数量 $n$。对每个参考 patch，在非局部搜索窗口中取最相似的 $n$ 个块堆叠成四阶张量 $\mathcal{G}_{\mathcal{Y}}$。 |
| `search_window` | `15` | 搜索窗口边长 $s$（体素数）。决定在多大空间范围内寻找自相似 patch。 |
| `internal_iterations` | `50` | ADMM 内部迭代次数 $T$。|
| `epsilon` | `1e-16` | 对数和范数中的平滑常数 $\varepsilon$，防止 $\log(0)$ 并稳定权重计算。 |
| `alpha_feedback` | `0.1` | Method Noise 外部迭代的反馈系数 $\alpha$，控制噪声残差注入的强度。 |
| `beta_noise` | `0.3` | 噪声估计参数 $\beta$，用于在 ROI 中估计 Rician 噪声标准差 $\sigma_n$。 |
| `max_patch_groups` | `64` | 每个椎间盘 ROI 中最多处理的 patch 组个数。过大时会显著增加计算时间和内存消耗。 |
| `max_singular_values` | `10` | 在构造 patch 级奇异值特征时，每个模态保留的奇异值个数 $K$。 |
| `rank` | `8` | 单 ROI CP 分解的秩 $R$。表示用多少个秩‑1 张量来近似 3D 椎间盘张量。$R$ 越大，潜在空间模式越多、表达能力越强，但计算复杂度和过拟合风险也会增加，可通过重构误差 $r_{\text{CP}}$ 检查是否足够。 |
| `max_iter` | `1000` | CP‑ALS 最大迭代次数 $T_{\max}$。若在 `max_iter` 内达到 `tol` 指定的收敛标准则提前停止。 |
| `tol` | `1e-4` | CP‑ALS 收敛阈值 $\varepsilon_{\text{ALS}}$。两次迭代的目标函数变化比例小于该值时认为收敛。 |
| `epsilon_cp` | `1e-6` | CP 特征构造中的平滑常数 $\varepsilon$，用于计算权重能量比例、有效秩以及因子向量熵/集中度时防止除零或 $\log(0)$ 引起的数值不稳定。通常保持默认即可。 |
| `top_components` | `3` | 参与计算因子向量熵 $H$ 和集中度 $G$ 的前 $K$ 个主成分个数。实际使用时取 $\min(K, R)$，方案默认 `K=3`。 |
| `random_state` | `0` | 随机种子，用于控制 CP 分解中的随机性（如内部初始化），以保证结果可复现。 |

#### VI. 系统行为与性能参数

| 参数 | 示例值 | 意义 |
| :--- | :--- | :--- |
| `OUTPUT_FORMATS` | `['excel', 'json', 'csv']` | **输出格式**。定义最终特征结果要保存为哪些文件格式。 |
| `FEATURE_SETS` | `{'texture': ['gabor', ...], ...}` | **特征集定义**。允许用户通过命令行或GUI方便地选择计算预定义的特征子集。 |
| `PARALLEL_CONFIG` | `{'enabled': True, 'max_workers': None}` | **全局并行处理配置**。控制批量分析时是否启用多进程，以及使用的CPU核心数。 |
| `MEMORY_CONFIG` | `{'max_memory_gb': 8, 'cache_enabled': True}` | **内存管理配置**。用于控制内存使用上限，并启用缓存以提高效率。 |
| `CALCULATOR_PARALLEL`| `{'gabor': {'enabled': True, ...}}`| **计算器并行配置**。允许对单个计算密集型特征（如Gabor）的内部计算启用或禁用并行化。 |

### 1.6 输入数据要求

#### 文件格式

*   **推荐格式**: 建议将数据转换为 **NIfTI (.nii.gz)** 格式。这是一个标准的医学图像格式，可以将一个完整的3D扫描序列（包含数十个2D切片）及其空间信息打包在一个单独的文件中，管理起来非常方便。可以使用 ITK-SNAP 或 3D Slicer 等软件将DICOM序列转换为`.nii.gz`文件。
*   **文件命名**: 掩码图像名在原始图像名基础上加"_mask"，例如：
    *   原始图像: `Case01.nii.gz`
    *   掩码图像: `Case01_mask.nii.gz`
*   **要求**: 原始图像和掩码图像必须在空间上严格对齐，并且具有完全相同的维度（长、宽、切片数）和体素间距。原始图像名字不要带"_"。 

#### **配置 `config.py` 中的标签表**

**标签表示例：**

| 解剖结构类别 | 具体名称 | 分配的标签值 | 在 `config.py` 中的对应 | 
| :--- | :--- | :--- | :--- | 
| **椎体** | L1 椎体 | `2` | `DISC_LABELS` | 
| | L2 椎体 | `4` | `DISC_LABELS` | 
| | L3 椎体 | `6` | `DISC_LABELS` | 
| | L4 椎体 | `8` | `DISC_LABELS` | 
| | L5 椎体 | `10` | `DISC_LABELS` | 
| | S1 椎体 (骶骨) | `12` | `DISC_LABELS` | 
| **椎间盘** | L1-L2 椎间盘 | `3` | `DISC_LABELS` | 
| | L2-L3 椎间盘 | `5` | `DISC_LABELS` | 
| | L3-L4 椎间盘 | `7` | `DISC_LABELS` | 
| | L4-L5 椎间盘 | `9` | `DISC_LABELS` | 
| | L5-S1 椎间盘 | `11` | `DISC_LABELS` | 
| **椎管/CSF** | 椎管内容物 | `20` | `DURAL_SAC_LABEL` |


**标签表在config.py中的对应示例：**

    DISC_LABELS = {
        'L1-L2': {'disc': 3, 'upper': 2, 'lower': 4},
        'L2-L3': {'disc': 5, 'upper': 4, 'lower': 6},
        'L3-L4': {'disc': 7, 'upper': 6, 'lower': 8},
        'L4-L5': {'disc': 9, 'upper': 8, 'lower': 10},
        'L5-S1': {'disc': 11, 'upper': 10, 'lower': 12}
    }


    DURAL_SAC_LABEL = 20  

---
### **如何进行2D特征提取 (针对单切片DICOM或2D图像)**

系统在默认情况下，即使输入的是单张2D图像，PyRadiomics也会将其当作一个仅包含单层切片的“极扁平”三维对象来处理。如果目的是进行2D分析，必须手动启用2D提取模式。

#### 操作步骤:

1.  在特征提取模块的 GUI 界面中，找到并切换到 “PyRadiomics特征” 的参数设置区域。

2.  找到 **“2D设置”** 区域，并**勾选 “强制2D提取”** 复选框。

------


## 模块二：图像扰动

本模块旨在通过对原始图像和分割掩码应用一系列标准化的、参数可控的扰动，系统性地评估放射组学特征的稳健性。其核心目的是模拟临床实践中可能出现的各种不确定性，例如患者的微小移动、设备噪声以及分割边界的主观差异，从而筛选出那些在各种干扰下依然保持稳定、可重复的“黄金特征”。

### 2.1 扰动类型和原理

#### 2.1.1 膨胀

*   **目的**: 模拟分割时对目标区域（椎间盘）边界的轻微高估。
*   **原理**:
    膨胀是一种基本的形态学操作。对于一个二值化的掩码图像 `A` 和一个被称为结构元素的小模板 `B`，`A` 被 `B` 膨胀的结果，记为 $A \oplus B$，是所有点 `z` 的集合，使得 `B` 平移 `z` 后与 `A` 的交集不为空。
    $$
    A \oplus B = \{z | (\hat{B})_z \cap A \neq \emptyset \}
    $$
    其中 $\hat{B}$ 是 `B` 的映像。在实际计算中，该操作通过以下算法实现：

    1.  **识别目标**: 程序首先遍历预设的椎间盘标签列表（如3, 5, 7, 9, 11），对每个标签对应的区域独立进行处理。
    2.  **构建结构元素**: 根据用户在界面中设置的形态学核大小（`kernel_size`）参数，程序构建一个尺寸为 (`kernel_size`, `kernel_size`) 的椭圆形结构元素 `B`。
    3.  **执行膨胀**: 将结构元素 `B` 的中心逐一滑过掩码图像中的每一个像素。在每个位置，目标像素的值被替换为其邻域（由结构元素 `B` 定义）内所有像素值的最大值。
    4.  **迭代应用**: 此过程将根据用户设置的迭代次数（`iterations`）参数重复执行。
    5.  **合并结果**: 对所有椎间盘标签处理完成后，将膨胀后的区域以其原始标签值合并回最终的掩码中。

    该算法的效果是使掩码的前景区域（高像素值部分）向外扩张，从而实现边界的高估。

#### 2.1.2 腐蚀

*   **目的**: 模拟分割时对目标区域边界的轻微低估。
*   **原理**:
    腐蚀是膨胀的对偶操作。`A` 被 `B` 腐蚀的结果，记为 $A \ominus B$，是所有点 `z` 的集合，使得 `B` 平移 `z` 后完全包含于 `A` 中。
    $$
    A \ominus B = \{z | (B)_z \subseteq A \}
    $$
    其计算算法与膨胀类似，核心区别在于第三步，并增加了一个安全检查：

    1.  **识别目标**和**构建结构元素**步骤与膨胀操作完全相同，同样使用用户定义的形态学核大小和椭圆形结构元素 `B`。
    2.  **执行腐蚀**: 将结构元素 `B` 的中心滑过图像，目标像素的值被替换为其邻域内所有像素值的最小值。
    3.  **迭代应用**: 根据用户设置的迭代次数重复此过程。
    4.  **安全检查**: 迭代完成后，程序会计算腐蚀后区域的总像素数。只有当该像素数大于一个预设的最小阈值（`MIN_PIXEL_THRESHOLD`，代码中为20）时，腐蚀结果才会被接受。如果腐蚀导致区域过小或消失，则放弃本次更改，保留原始掩码区域。
    5.  **合并结果**: 将通过安全检查的腐蚀后区域合并回最终掩码。

    该算法通过收缩掩码的前景区域来实现边界的低估，同时通过安全检查避免了因过度腐蚀导致有效分割区域完全消失的问题。

#### 2.1.3 轮廓随机化

*   **目的**: 模拟分割边界的随机、非系统性的不确定性，比单一的膨胀或腐蚀更接近真实世界的分割误差。
*   **原理**:

    1.  **随机选择操作类型**: 程序首先生成一个 `[0, 1]` 区间内的随机数 `p`。如果 `p > 0.5`，则选择膨胀操作；否则，选择腐蚀操作。
    2.  **随机化参数**:
        *   **随机核大小**: 以用户设定的形态学核大小为基准，在一个预定义的范围内随机生成一个新的整数作为本次操作的核大小。
        *   **随机迭代次数**: 以用户设定的迭代次数为基准，在一个预定义的范围内随机生成一个新的整数作为本次操作的迭代次数。
    3.  **执行随机形态学操作**: 使用上一步随机生成的参数，对掩码执行选定的膨胀或腐蚀操作。
        $$
        \text{Mask}_{\text{perturbed}} = \begin{cases} \text{Mask}_{\text{original}} \oplus B_{\text{rand}} & \text{if } p > 0.5 \\ \text{Mask}_{\text{original}} \ominus B_{\text{rand}} & \text{if } p \le 0.5 \end{cases}
        $$
    其中，结构元素 $B_{\text{rand}}$ 的尺寸和应用的迭代次数都是随机生成的。这个过程确保了每次生成的扰动掩码都具有独特的、不可预测的边界变化。

#### 2.1.4 平移

*   **目的**: 模拟患者在扫描过程中的轻微位置移动。
*   **原理**:
    平移是一种仿射变换，它将图像上的每个点 `(x, y)` 移动到一个新的位置 `(x', y')`。该变换通过一个2x3的矩阵 `M` 来实现。

    1.  **生成随机平移量**:
        *   根据用户设置的平移范*参数（`range_val`），程序在 `[-range_val, range_val]` 的整数区间内，独立地随机抽取一个x轴平移量 $t_x$ 和一个y轴平移量 $t_y$。
    2.  **构建变换矩阵**:
        使用随机生成的平移量构建如下的仿射变换矩阵 `M`：
        $$
        M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}
        $$
    3.  **应用变换**:
        将此矩阵 `M` 应用于原始图像和掩码，计算出每个像素的新位置：
        $$
        \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} x + t_x \\ y + t_y \end{bmatrix}
        $$
    对于移出图像边界的像素，其值被填充为0（黑色）。

#### 2.1.5 旋转

*   **目的**: 模拟患者在扫描床上的轻微旋转运动。
*   **原理与算法**:
    旋转同样是一种仿射变换，但其变换矩阵 `M` 的构建更为复杂，因为它涉及到围绕一个特定中心点的旋转。

    1.  **定义旋转中心**: 程序的旋转中心被固定为图像的几何中心 $(c_x, c_y) = (w/2, h/2)$，其中 `w` 和 `h` 分别是图像的宽度和高度。
    2.  **生成随机旋转角**: 根据用户设置的旋转范围参数（`max_angle`），程序在 `[-max_angle, max_angle]` 的浮点数区间内随机抽取一个旋转角度 $\theta$。
    3.  **构建变换矩阵**: 程序调用一个标准函数（如OpenCV的`getRotationMatrix2D`）来生成最终的2x3仿射变换矩阵 `M`。该函数的内部算法相当于执行了以下三步操作的矩阵乘积：
        *   将坐标系平移，使旋转中心移动到原点。
        *   执行标准的二维旋转，其矩阵为：
            $$
            R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
            $$
        *   将坐标系平移回原来的位置。
    4.  **应用变换**: 将生成的矩阵 `M` 应用于原始图像和掩码以完成旋转。移出边界的区域同样填充为0。

#### 2.1.6 高斯噪声

*   **目的**: 模拟MRI信号采集过程中产生的电子噪声，测试特征对图像信噪比变化的敏感度。
*   **原理与算法**:
    此操作通过向原始图像的每个像素强度值 $I(x, y)$ 上加性地叠加一个从高斯（正态）分布中随机抽取的噪声值 $N(x, y)$ 来实现。

    1.  **生成噪声矩阵**:
        程序首先创建一个与原始图像尺寸完全相同的噪声矩阵 `N`。该矩阵中的每一个元素 $N(x,y)$ 都是从一个高斯分布中独立随机抽取的样本。
    2.  **定义高斯分布**:
        该高斯分布的概率密度函数由以下公式定义：
        $$
        P(z) = \frac{1}{\sigma \sqrt{2\pi}} e^{ - \frac{(z - \mu)^2}{2\sigma^2} }
        $$
        根据代码实现，该分布的参数被设定为：
        *   **均值 $\mu$**: 固定为 0。
        *   **标准差 $\sigma$**: 由用户在界面中设置的噪声标准差（`noise_std`）参数决定。
    3.  **应用噪声**:
        将原始图像矩阵 `I` 与生成的噪声矩阵 `N` 进行逐像素相加，得到含噪图像 `I'`：
        $$
        I'(x, y) = I(x, y) + N(x, y)
        $$
    最终的图像 `I'` 即为添加了高斯噪声的结果。这个过程只作用于图像，掩码保持不变。


### 2.2 参数设置说明

| 参数 | 对应扰动 | 说明与影响 |
| :--- | :--- | :--- |
| **平移范围** | 平移、组合扰动 | 定义在x和y轴上随机平移的最大像素数。程序将从 `[-该值, +该值]` 的整数范围内随机选择一个平移量。值越大，模拟的患者位置变动幅度越大。 |
| **旋转范围** | 旋转、组合扰动 | 定义了随机旋转的最大角度。程序将从 `[-该值, +该值]` 的浮点数范围内随机选择一个旋转角度。值越大，模拟的患者旋转运动越剧烈。 |
| **噪声标准差** | 高斯噪声、组合扰动 | 定义了所添加高斯噪声分布的标准差 $\sigma$（均值固定为0）。值越大，添加到图像中的噪声强度越高，图像信噪比越低。 |
| **形态学核大小** | 膨胀、腐蚀、轮廓随机化 | 定义了用于形态学操作的椭圆形结构元素的基础尺寸。对于轮廓随机化，实际尺寸会在此基础值附近随机波动。值越大，单次操作对掩码边界的改变越显著。 |
| **迭代次数** | 膨胀、腐蚀、轮廓随机化 | 定义了形态学操作的重复次数。对于轮廓随机化，实际迭代次数会在此基础值附近随机波动。值越大，形态学效果（膨胀或腐蚀）越强烈。 |

### 2.3 输出结构

程序会在指定的输出文件夹内，根据原始文件的相对路径创建相同的子文件夹结构。输出文件名将包含原始文件名和所应用的扰动类型。

*   **示例**:
    *   原始文件: `Case01/slice_10.nii.gz`
    *   输出路径: `D:/output/`
    *   应用“膨胀+平移+旋转”扰动后，输出文件为:
        *   `D:/output/image/Case01/slice_10_dilation_trans_rot_image.nii.gz`
        *   `D:/output/mask/Case01/slice_10_dilation_trans_rot_mask.nii.gz`

------

## 模块三：特征稳健性相关性分析

### 3.1 分析工作流

**ICC计算 ->分层聚类筛选 ->相关性冗余消除**

### 3.2 输入文件要求

**注**：可通过项目目录下的`ivd_csv_converter.py`进行模块一到模块三的csv格式转换。具体地说，先用模块一提取原始图像的特征，然后用模块二对图像和掩码进行扰动，再用模块一提取扰动后的特征，最后将两个特征csv文件用`ivd_csv_converter.py`转换成模块三要求的输入格式。。

输入文件必须是CSV格式，包含以下结构：

    - **行 (Rows)**: 每行代表一个椎间盘病例（如 `Case01_L1L2`）
    - **列 (Columns)**: 每列代表一个特征在特定条件下的值
    - **列名规范**: 必须采用 `特征名_条件` 的格式 (e.g., `glcm_Contrast_gold`, `glcm_Contrast_noise`)。

### 列名示例：
```
glcm_Contrast_gold      # 金标准
glcm_Contrast_noise     # 噪声扰动
glcm_Contrast_dilate    # 膨胀扰动
glcm_Contrast_erode     # 腐蚀扰动
glcm_Contrast_geom      # 几何扰动
```
### 必要条件：
- 必须包含 `_gold` 后缀的金标准数据
- 至少包含一种扰动条件

### 3.3 计算原理

#### I. 组内相关系数

  * **目的**: 量化特征在不同扰动条件下的稳定性和一致性。

  * **计算原理**:
    本系统采用双向混合效应模型下的绝对一致性组内相关系数，即`ICC(3,k)`。其计算严格遵循双因素方差分析的框架，具体步骤如下：

    1.  **构建数据表**: 对于每一个特征，将其在 $n$ 个病例和 $k$ 个条件（金标准及各种扰动）下的值排列成一个 $n \times k$ 的矩阵。设 $x_{ij}$ 为第 $i$ 个病例在第 $j$ 个条件下的特征值。

    2.  **计算基本统计量**:
        *   总均值 $\bar{x}$: 所有 $x_{ij}$ 的平均值。
        *   行均值 $\bar{x}_{i.}$: 第 $i$ 个病例在所有条件下的平均值。
        *   列均值 $\bar{x}_{.j}$: 第 $j$ 个条件在所有病例下的平均值。

    3.  **计算离差平方和**:
        *   **总离差平方和**: 反映数据的总变异。
            $$
            SS_{\text{总}} = \sum_{i=1}^{n} \sum_{j=1}^{k} (x_{ij} - \bar{x})^2
            $$
        *   **行间离差平方和 (病例间)**: 反映由不同病例引起的变异。
            $$
            SS_{\text{行}} = k \sum_{i=1}^{n} (\bar{x}_{i.} - \bar{x})^2
            $$
        *   **列间离差平方和 (条件间)**: 反映由不同扰动条件引起的变异。
            $$
            SS_{\text{列}} = n \sum_{j=1}^{k} (\bar{x}_{.j} - \bar{x})^2
            $$
        *   **误差离差平方和**: 反映除了病例和条件之外的随机误差。
            $$
            SS_{\text{误差}} = SS_{\text{总}} - SS_{\text{行}} - SS_{\text{列}}
            $$

    4.  **计算均方**:
        将各离差平方和除以其对应的自由度，得到均方。
        *   病例间均方: $MS_{\text{行}} = \frac{SS_{\text{行}}}{n-1}$
        *   误差均方: $MS_{\text{误差}} = \frac{SS_{\text{误差}}}{(n-1)(k-1)}$

    5.  **计算组内相关系数值**:
        最终，`ICC(3,k)`的值由病例间均方和误差均方计算得出，它量化了病例间的真实变异占总变异（真实变异+误差变异）的比例。
        $$
        \text{ICC}(3,k) = \frac{MS_{\text{行}} - MS_{\text{误差}}}{MS_{\text{行}}}
        $$

#### II. 分层聚类

  * **目的**: 识别并筛选出稳健性表现模式相似的特征群组。

  * **原理与算法**:
    本系统采用自底向上的聚合式分层聚类，其核心是Ward最小方差链接算法。

    1.  **特征向量化**: 将每个特征的稳健性表现抽象为一个向量 $\mathbf{f}$，向量的维度等于扰动条件的数量，每个分量是该特征在对应条件下的组内相关系数值。
        $$
        \mathbf{f}_{\text{特征A}} = [\text{ICC}_{\text{条件1}}, \text{ICC}_{\text{条件2}}, \dots, \text{ICC}_{\text{条件k}}]
        $$

    2.  **距离矩阵计算**: 使用欧几里得距离计算每两个特征向量之间的距离，形成一个距离矩阵。向量 $\mathbf{a}$ 和 $\mathbf{b}$ 之间的距离定义为：
        $$
        d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{k} (a_i - b_i)^2}
        $$

    3.  **Ward链接算法**:
        Ward算法是一种特殊的链接策略，其目标是使合并后的簇内方差增加得最小。这是一个迭代过程：
        *   **初始化**: 将每一个特征都视为一个独立的簇。
        *   **迭代合并**: 在每一步中，算法会计算所有可能的簇对合并后，所导致的总簇内误差平方和的增量。
            *   单个簇 $C_m$ 的误差平方和定义为簇内所有特征向量到该簇质心（均值向量）$\mathbf{\mu}_m$ 的平方距离之和：
                $$
                E_m = \sum_{\mathbf{f}_i \in C_m} ||\mathbf{f}_i - \mathbf{\mu}_m||^2
                $$
            *   总误差平方和是所有簇的误差平方和之和: $E_{\text{总}} = \sum_{m} E_m$。
            *   算法会选择那一对簇进行合并，该合并能使 $E_{\text{总}}$ 的增量最小。
        *   **终止**: 重复迭代合并过程，直到所有特征都合并成一个簇为止。这个过程的完整层级结构可以用一个树状图来表示。

#### III. Spearman相关性分析

  * **目的**: 在筛选出的稳健特征群内部，剔除信息高度重叠的冗余特征。

  * **原理与算法**:
    Spearman相关性分析的核心是计算 Pearson 相关系数在秩次数据上的应用。

    1.  **秩次转换**:
        对于待分析的两个特征（例如特征X和特征Y），首先不使用它们的原始值，而是将每个特征内部的数据进行排序，并用它们的秩次（排名）来替换原始值。
        例如，对于特征X的数值 `[10, 50, 20]`，其对应的秩次为 `[1, 3, 2]`。
        如果存在相同的值（结），则取它们秩次的平均值。例如，`[10, 20, 20]` 的秩次为 `[1, 2.5, 2.5]`。
        经过转换后，我们得到两个新的秩次变量 $rg_X$ 和 $rg_Y$。

    1.  **计算相关系数**:
        接下来，对这两个秩次变量计算标准的 Pearson 积矩相关系数。其计算公式为：
        $$
        \rho = \frac{\sum_{i=1}^{n}(rg_{X_i} - \overline{rg_X})(rg_{Y_i} - \overline{rg_Y})}{\sqrt{\sum_{i=1}^{n}(rg_{X_i} - \overline{rg_X})^2 \sum_{i=1}^{n}(rg_{Y_i} - \overline{rg_Y})^2}}
        $$
        其中，$rg_{X_i}$ 和 $rg_{Y_i}$ 是第 $i$ 个病例在特征X和Y上的秩次，$\overline{rg_X}$ 和 $\overline{rg_Y}$ 分别是两个秩次变量的平均值，$n$ 是病例总数。

        这个值 $\rho$ 就是Spearman相关系数，它衡量了两个原始变量之间单调关系的强度，而对具体数值分布和异常值不敏感。

    2.  **冗余剔除**:
        *   设定一个极高的相关性阈值（如0.99）。
        *   遍历所有特征对，若其Spearman相关系数的绝对值大于该阈值，则认为它们高度冗余。
        *   在每一对冗余特征中，计算它们原始值的方差，并移除方差较低的那个特征。

### 3.4 核心参数设置

#### I. ICC计算设置

| 参数 | 推荐设置 | 原理与依据 |
| :--- | :--- | :--- |
| **ICC类型** | `ICC(3,k)` | **双向混合模型**。在我们的研究设计中，研究目标（椎间盘病例）可被视为从总体中随机抽样的样本（随机效应），而测量条件（`gold`, `noise`等）是我们特意施加的、固定的、感兴趣的效应（固定效应）。`ICC(3,k)`正是为此场景设计的标准模型。 |
| **ICC置信水平** | `0.95` | **95%置信区间**是医学统计研究中广泛接受的标准。 |
| **启用预筛选** | `True` (勾选) | **推荐**。这是一个高效的“去噪”步骤，可以显著改善后续分析的质量。 |
| **最小ICC阈值** | `0.25 ~ 0.5` | **“木桶效应”原则**。该值是特征在所有扰动条件下的最低表现。设置一个大于0的值（例如`0.25`）可以确保没有任何一个特征存在明显的“短板”，即在某一种扰动下完全崩溃。对于初步探索，可以设为`0.0`以保留更多特征；对于严格筛选，建议设为`0.5`。 |
| **平均ICC阈值** | `0.5 ~ 0.75` | **“综合表现”原则**。该值是特征在所有扰动下的平均表现。设置一个较高的值（例如`0.75`）可以确保最终入选的特征不仅没有短板，而且其整体的稳健性达到了“良好”或以上的水平。这是筛选“黄金特征”的关键参数。 |


*   **ICC值的含义**:
    *   **ICC < 0.5**: 稳健性差。特征值受扰动影响极大，几乎不可信。
    *   **0.5 ≤ ICC < 0.75**: 稳健性中等。特征表现出一定的稳定性，但仍有明显波动。
    *   **0.75 ≤ ICC < 0.9**: 稳健性良好。特征在大多数扰动下表现稳定，是理想的候选特征。
    *   **ICC ≥ 0.9**: 稳健性极好。特征几乎不受扰动影响。


#### II. 聚类分析设置

| 参数 | 推荐设置 | 原理与依据 |
| :--- | :--- | :--- |
| **链接方法** | `ward` | 旨在最小化簇内方差之和。与其他方法（如single-linkage可能产生链状簇）相比，`ward`倾向于产生大小更均匀、结构更紧凑的球状簇，这在识别功能相似的特征组时通常是理想的。 |
| **距离度量** | `euclidean` | 欧几里得距离是与`ward`链接方法配套使用的标准度量，因为它直接衡量空间中的直线距离，而`ward`方法正是基于最小化与簇质心（均值）的平方欧氏距离之和，两者在数学上是兼容的。 |
| **簇选择方式** | `min_icc` | "最大化-最小"策略。该策略旨在选择一个“下限”最高的簇。具体来说，它会先找出每个簇中平均ICC (`mean_icc`) 最低的特征，然后比较这些“最低分”，并选择那个“最低分”最高的簇。这确保了所选簇中的所有成员至少都有一个相对较高的平均稳健性基线。 |
| **聚类数量** | `manual` | 建议首先使用**“显示聚类树状图”**功能，通过观察树状图的结构（例如，寻找合并距离突然增大的“肘部”）来做出专业的、有依据的判断，然后手动输入簇数量。程序提供的“自动建议k值”可作为初步参考。 |

#### III. 相关性分析设置

| 参数 | 推荐设置 | 原理与依据 |
| :--- | :--- | :--- |
| **相关性阈值** | `0.99` | 一个非常严格的阈值。使用如此高的值是为了确保只移除那些信息几乎完全重叠、可以被认为是“同义词”的特征，从而在不损失过多信息的前提下最大化地减少冗余。 |
| **方差准则** | `移除方差较低的特征` | 当两个特征高度相关时，它们携带的信息高度重叠。此时保留方差较高的那个，意味着保留了在数据集中动态范围更大、可能更有能力区分不同病例状态的特征。 |


### 3.5 输出文件说明

  - **`final_robust_features.csv`**: 最终筛选的稳健特征列表。
  - **`analysis_report.txt`**: 详细可读的文本分析报告。

------

##  参考文献

[1] McSweeney T, Tiulpin A, Kowlagi N, Määttä J, Karppinen J, Saarakkala S. Robust Radiomic Signatures of Intervertebral Disc Degeneration from MRI. Spine (Phila Pa 1976). 2025 Jun 20.

[2] Ma J, Wang R, Yu Y, Xu X, Duan H, Yu N. Is fractal dimension a reliable imaging biomarker for the quantitative classification of an intervertebral disk? Eur Spine J. 2020 May;29(5):1175-1180.

[3] Murto N, Luoma K, Lund T, Kerttula L. Reliability of T2-weighted signal intensity-based quantitative measurements and visual grading of lumbar disc degeneration on MRI. Acta Radiol. 2023 Jun;64(6):2145-2151.

[4] Ruiz-España S, Arana E, Moratal D. Semiautomatic computer-aided classification of degenerative lumbar spine disease in magnetic resonance imaging. Comput Biol Med. 2015 Jul;62:196-205.

[5] Beulah, A., Sharmila, T.S. & Pramod, V.K. Degenerative disc disease diagnosis from lumbar MR images using hybrid features. Vis Comput 38, 2771–2783 (2022).

[6] Michopoulou S, Costaridou L, Vlychou M, Speller R, Todd-Pokropek A. Texture-based quantification of lumbar intervertebral disc degeneration from conventional T2-weighted MRI. Acta Radiol. 2011 Feb 1;52(1):91-8.

[7] Zheng, HD., Sun, YL., Kong, DW. et al. Deep learning-based high-accuracy quantitation for lumbar intervertebral disc degeneration from MRI. Nat Commun 13, 841 (2022).

[8] Waldenberg C, Hebelka H, Brisby H, Lagerstrand KM. MRI histogram analysis enables objective and continuous classification of intervertebral disc degeneration. Eur Spine J. 2018 May;27(5):1042-1048.

[9] van Griethuysen, J. J. M., et al. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107.

[10] Lin A, et al. Radiomics based on MRI to predict recurrent L4-5 disc herniation after percutaneous endoscopic lumbar discectomy. BMC Med Imaging. 2024 Oct 10;24(1):273.

[11]Luca Zedda, Andrea Loddo, Cecilia Di Ruberto,Radio DINO: A foundation model for advanced radiomics and AI-driven medical imaging analysis,Computers in Biology and Medicine,Volume 195,2025,110583,ISSN 0010-4825

[12]Mei X, Liu Z, Robson PM, Marinelli B, Huang M, Doshi A, Jacobi A, Cao C, Link KE, Yang T, Wang Y, Greenspan H, Deyer T, Fayad ZA, Yang Y. RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning. Radiol Artif Intell. 2022 Jul 27;4(5):e210315. 

[13] Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. SIAM Review.

[14] Burch, M., et al. (2024). Towards Quantum Tensor Decomposition in Biomedical Applications.

[15] Zhang, X., et al. (2017). Denoise diffusion‑weighted images using higher‑order singular value decomposition. NeuroImage.

[16] Wang, L., et al. (2020). A modified higher‑order singular value decomposition framework with adaptive multilinear tensor rank approximation for 3D MRI Rician noise removal. Frontiers in Oncology.

[17] Wang, Z., et al. (2022). A nonlocal enhanced low‑rank tensor approximation framework for 3D magnetic resonance image denoising. Biomedical sgnal Processing and Control.

[18] Amin, M. R., et al. (2023). Tensor decomposition‑based feature extraction and classification to detect natural selection from genomic data (T‑REx).

[19] Le, T. N., et al. (2023). Tensor‑based color feature image extraction for blood cell recognition.

[20]孔德兴, 孙剑, 何炳生, 沈纯理. 医学影像精准分析的数学理论与算法. 科学出版社, 2025:74-108
