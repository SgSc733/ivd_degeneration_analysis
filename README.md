# 椎间盘退变一体化分析系统 (Integrated IVD Degeneration Analysis System)

## 1\. 项目简介

本项目是一个基于Python的**一体化**椎间盘影像分析平台。

系统通过一个集成的图形用户界面（GUI），将三个核心功能模块无缝衔接：

1.  **特征提取模块**: 基于标准的 **PyRadiomics** 库和一系列源自前沿学术研究的**自定义特征提取算法**，对椎间盘影像特征进行量化。
2.  **图像扰动模块**: 对原始图像和分割掩码应用一系列标准化的扰动，以模拟临床实践中的各种不确定性。
3.  **稳健性相关性分析模块**: 通过一系列的统计方法（ICC、分层聚类、相关性分析），从海量特征中筛选出稳健且信息不冗余的“黄金特征集”。

本项目支持对单个病例进行深度分析，也支持对大规模队列数据进行自动化批量处理。

## 2\. 系统架构与核心模块

本系统采用模块化设计，通过统一的GUI界面调度三大核心功能，形成一个完整的数据分析工作流。

  * **特征提取**: 输入原始图像和分割掩码，输出包含海量定量特征的数值表格。
  * **图像扰动**: 输入原始图像和分割掩码，输出一系列包含受控扰动的图像和掩码文件，用于后续的稳健性分析。
  * **稳健性相关性分析**: 输入一个包含了“金标准”和多种“扰动后”特征值的数据表，输出最终筛选出的稳健特征列表和详细的分析报告。

## 3\. 主要功能

### I. 特征提取模块

  - **PyRadiomics 标准特征**:
      - 一阶统计特征 (First Order)
      - 形状特征 (Shape 2D/3D)
      - 纹理特征 (GLCM, GLRLM, GLSZM, GLDM, NGTDM)
  - **自定义特征** (PyRadiomics无法提取的特征):
      - **[DHI]** 椎间盘高度指数 (Disc Height Index)
      - **[ASI]** 峰值信号强度差 (Peak Signal Intensity Difference)
      - **[FD]** 分形维度 (Fractal Dimension)
      - **[T2SI]** T2信号强度比率 (T2 Signal Intensity Ratio)
      - **[Gabor]** Gabor纹理特征
      - **[Hu]** Hu不变矩 (Hu Invariant Moments)
      - **[Texture]** 扩展纹理特征 (如LBP)

### II. 图像扰动模块

  - **掩膜扰动**: 膨胀 (Dilation), 腐蚀 (Erosion), 轮廓随机化 (Contour Randomization)
  - **几何变换**: 平移 (Translation), 旋转 (Rotation)
  - **强度变换**: 高斯噪声 (Gaussian Noise)
  - **组合扰动**: 支持上述多种扰动的组合应用，以模拟更复杂的真实世界变量。

### III. 稳健性相关性分析模块

  - **稳健性量化**: 采用组内相关系数 (ICC) 精确评估每个特征在不同扰动下的稳定性。
  - **稳健特征筛选**: 借鉴前沿研究方法，通过**分层聚类**识别并筛选出整体表现优异的稳健特征群。
  - **特征冗余消除**: 在稳健特征群内部，通过**斯皮尔曼相关性分析**剔除信息高度重叠的冗余特征。
  - **交互式可视化**: 提供ICC热图、聚类树状图和相关性矩阵的可视化工具，辅助分析和决策。

## 4\. 安装指南

### 环境要求

  - Python 3.9+
  - Windows / Linux / macOS
  - **重要**: `numpy<2.0` (PyRadiomics的兼容性要求)

### 安装步骤

1.  **克隆项目**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
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
    ```

    *如果遇到 `numpy` 版本问题，请手动安装指定版本：*

    ```bash
    pip install "numpy<2.0"
    pip install -r requirements.txt
    ```

## 5\. 使用方法

#### 启动GUI界面

```bash
python run_gui.py
```

系统启动后，您会看到一个包含三个主功能选项卡（“特征提取”、“图像扰动”、“稳健性相关性分析”）的集成界面。您可以根据研究需要，按顺序或独立使用这些模块。

-----

## 模块一：特征提取

### 1.1 自定义特征计算方法

#### 椎间盘高度指数 (DHI)
采用基于面积的可靠方法计算。首先分别计算椎间盘（Disc）及其上（Upper）、下（Lower）两个相邻椎体（Vertebral Body）的高度，然后通过以下公式进行标准化，以消除个体差异。
$$
DHI = \frac{2 \times \text{Height}_{\text{disc}}}{\text{Height}_{\text{upper}} + \text{Height}_{\text{lower}}}
$$

#### 峰值信号强度差 (ASI)
通过对椎间盘ROI内的信号强度直方图拟合一个双峰高斯混合模型（GMM），分别代表纤维环（AF）和髓核（NP）的信号分布。ASI量化了这两个峰值之间的差异，并使用脑脊液（CSF）信号进行标准化。
$$
ASI = \frac{|\text{SI}_{\text{peak_NP}} - \text{SI}_{\text{peak_AF}}|}{\text{SI}_{\text{mean_CSF}}}
$$

#### 分形维度 (FD)
该特征通过一系列图像处理步骤（分割、8位转换、窗位窗宽调整、二值化、边缘检测）生成椎间盘内部结构的边缘图像。最后在生成的边缘图像上应用**盒计数法 (Box-Counting Algorithm)**，通过计算log-log图的斜率来量化其复杂性。

#### T2信号强度比率 (T2SI)
根据`TARGET-ROI`策略，在髓核（NP）中最亮的区域和椎管内的脑脊液（CSF）中分别定义ROI。T2SI是这两个区域平均信号强度的直接比率，用于客观评估髓核的水合状态。
$$
T2SI = \frac{\text{SI}_{\text{mean_NP}}}{\text{SI}_{\text{mean_CSF}}}
$$

#### Gabor纹理特征
首先构建一个包含不同尺度（波长）和方向的Gabor滤波器组，并将它们应用于原始图像，生成一系列“响应图”。然后从每个响应图的椎间盘ROI内提取均值、标准差等一阶统计特征，共同构成高维的Gabor特征集。

#### Hu不变矩
基于椎间盘的二值分割掩码，通过对其几何矩（原始矩、中心矩）进行一系列非线性组合，计算出七个对平移、旋转和缩放不敏感的矩不变量，用于鲁棒地描述椎间盘的整体形状。

### 1.2 预处理功能

  - **空间处理**: 图像重采样至统一体素间距。
  - **强度处理**: Z-score 标准化, 窗位窗宽调整 (Windowing), 灰度离散化 (Discretization / Binning)
  - **图像变换**: 边缘检测、高斯拉普拉斯(LoG)滤波、小波(Wavelet)变换等。

### 1.3 详细参数设置说明

所有核心参数均在 `config.py` 文件中进行统一管理，方便用户根据具体研究需求进行调整。下面对主要参数进行详细说明。

### I. 核心设置 (Core Settings)

这些参数定义了系统如何识别和处理分割图像中的不同区域。

| 参数 | 示例值 | 意义 |
| :--- | :--- | :--- |
| `DISC_LABELS` | `{'L1-L2': {'disc': 3, ...}}` | 定义了掩码文件中每个椎间盘及其相邻椎体的标签值。**这是所有计算的基础**。 |
| `CSF_LABEL` | `1` | 定义了脑脊液(CSF)的标签值，用于T2SI和ASI的信号强度标准化。 |
| `NUM_SLICES` | `3` | 指定从3D图像中提取用于2D分析的中间切片数量。 |
| `SLICE_AXIS` | `0` | 指定切片方向 (0: 矢状位, 1: 冠状位, 2: 轴位)。 |

### II. 预处理参数 (`PREPROCESSING_PARAMS`)

这些参数控制着特征提取前对图像进行的一系列标准化和变换操作，对保证特征的鲁棒性和可比性至关重要。

| 类别 | 参数 | 示例值 | 意义与参考文献 |
| :--- | :--- | :--- | :--- |
| **通用** | `target_size` | `[512, 512]` | **空间重采样**的目标尺寸，确保所有分析都在统一的空间分辨率下进行。这是保证特征（尤其是形状和纹理）可比性的关键步骤。 |
| **纹理特征** | `bin_width` | `16` | **强度离散化**的组宽度。这是计算纹理矩阵前的标准步骤，旨在减少噪声影响并标准化强度范围。
| **纹理特征** | `normalize` | `True` | 是否对ROI内强度进行**Z-score标准化**。该步骤能显著提升纹理特征在不同设备和扫描参数下的鲁棒性。
| **分形维度** | `window_center` | `128` | **窗位窗宽调整**的窗位。 |
| **分形维度**| `window_width` | `255` | **窗位窗宽调整**的窗宽。此参数组合用于标准化8位图像的对比度，为后续的阈值分割做准备。 
| **分形维度**| `threshold_percentile` | `65` | **二值化**的灰度阈值百分比，用于从灰度图中分离出代表结构异质性的前景。 

### III. 自定义特征计算器参数

这些参数分别控制各个自定义特征的计算细节。

| 特征 | 参数 | 示例值 | 意义与参考文献 |
| :--- | :--- | :--- | :--- |
| **DHI** | `central_ratio` | `0.8` | 计算椎间盘和椎体高度时，所使用的中心区域比例，以避免边缘效应。 |
| **ASI** | `n_components` | `2` | 用于拟合髓核(NP)和纤维环(AF)信号强度直方图的高斯混合模型(GMM)的组分数。 
| **T2SI** | `roi_method` | `'TARGET'` | 定义髓核(NP)的分割策略。'TARGET'模式旨在只勾画最亮、最均匀的NP区域，以精确量化水合状态。
| **Gabor**| `wavelengths` | `[2, 4, ...]` | Gabor滤波器组的波长列表，用于捕捉不同尺度的纹理信息。
| **Hu矩** | (无参数) | - | Hu不变矩对平移、旋转和缩放具有不变性，是鲁棒的形状描述符。 

### IV. PyRadiomics滤波器参数 (`FILTER_PARAMS`)

这些参数控制在提取标准纹理特征之前，对图像应用的额外滤波操作，以生成衍生特征。

| 滤波器 | 参数 | 示例值 | 意义与参考文献 |
| :--- | :--- | :--- | :--- |
| **LoG** | `sigma_list` | `[1, 3, 5]` | 高斯拉普拉斯(LoG)滤波器的Sigma值列表。LoG滤波器可以增强特定尺寸的斑点状结构，不同的Sigma值对应不同的结构尺度。
| **Wavelet** | `wavelet`, `level`| `'db1'`, `1` | 小波变换的类型和分解层级。小波变换能将图像分解到不同的频率和方向子带，从中提取的特征能更细致地描述纹理。

### 1.4 输入数据要求与标注指南

### 输入数据要求与标注指南

为了准确提取本项目支持的所有特征，您需要根据目标特征准备相应的分割掩码文件。不同的特征依赖于不同的解剖结构标注，以下是详细的标注指南：

| 需要分割的掩码/标注 | 对应的特征类别 |
| :--- | :--- |
| **1. 整个椎间盘掩码** | • **标准Pyradiomics特征** (纹理、形状等)<br>• **分形维度 (FD)**<br>• **椎间盘突出量化** (自定义形状特征) |
| **2. 髓核掩码** | • **T2信号强度比率 (T2SI)**<br>• **峰值信号强度差 (ASI)** |
| **3. 脑脊液掩码** | • **T2SI** 和 **ASI** 的信号强度标准化参考值 |
| **4. 椎管/硬脊膜囊掩码** | • **椎管狭窄量化 (DSCR)** (自定义形状特征) |
| **5. 椎体解剖学地标** | • **DSCR** 的基线参考 |

#### 1. 整个椎间盘掩码 (Whole Disc Mask)

* **这是什么？**
    代表完整椎间盘解剖区域的掩码，包含了**髓核 (Nucleus Pulposus, NP)** 和**纤维环 (Annulus Fibrosus, AF)** 。

* **如何标注？**
    沿着椎间盘的最外层边界精确勾画轮廓。**必须仔细排除**上、下方的椎体终板（骨性边界）、前纵韧带和后纵韧带，否则会严重影响形状和纹理特征的准确性。

* **在哪个视图中标注？**
    * **轴位 (Axial View)**: 用于计算**分形维度 (FD)**和**椎间盘突出量化** 。
    * **矢状位 (Sagittal View)**: 用于计算**标准的Pyradiomics特征**和**椎间盘高度指数 (DHI)**。

#### 2. 髓核掩码 (Nucleus Pulposus Mask)

* **这是什么？**
    仅代表椎间盘中央的髓核区域，用于精确量化髓核的信号变化。

* **如何标注？**
    用手绘方式**只勾画髓核中信号最亮、最均匀的区域**，并主动**排除**低信号的髓核内裂隙(INC)和任何其他暗区。

* **在哪个视图中标注？**
    * **矢状位 (Sagittal View)**。
 

#### 3. 脑脊液掩码 (Cerebrospinal Fluid Mask)

* **这是什么？**
    椎管内**脑脊液 (CSF)** 的一个样本区域，用作信号强度标准化的稳定参照物。

* **如何标注？**
    在与待测椎间盘相同的矢状位图像上，于椎间盘后方的椎管内选择一个清晰、无流动伪影的纯净CSF区域，放置一个大小适中的圆形或椭圆形ROI。

* **在哪个视图中标注？**
    * **矢状位 (Sagittal View)**。


#### 4. 椎管/硬脊膜囊掩码 (Spinal Canal / Dural Sac Mask)

* **这是什么？**
    椎管内包裹着神经和脑脊液的**硬脊膜囊 (Dural Sac)** 的区域，其前后径用于评估椎管狭窄。

* **如何标注？**
    在矢状位图像上，分割出椎管内信号明亮的CSF区域，重点是精确勾画出硬脊膜囊的**前缘**（靠近椎间盘的一侧）。

* **在哪个视图中标注？**
    * **矢状位 (Sagittal View)**。


#### 5. 椎体解剖学地标 (Vertebral Anatomical Landmarks)

* **这是什么？**
    这不是一个区域掩码，而是一系列用于定义“理想”椎管后缘的**关键点 (Landmarks)**。

* **如何标注？**
    在矢状位图像上，为每个腰椎椎体，在其后方的**椎弓根 (Pedicle)** 的垂直中点水平线上，找到椎体的**最后缘**，并在此处放置一个标记点。

* **在哪个视图中标注？**
    * **矢状位 (Sagittal View)**。

### 其他注意事项
- **标签一致**: 掩码文件中的标签值必须与 `config.py` 中的 `DISC_LABELS` 和 `CSF_LABEL` 定义严格一致。
- **图像质量**: 为保证特征的可靠性，建议使用高分辨率、信噪比良好的T2加权MRI图像。
- **故障排除**:
  - **PyRadiomics导入失败**: 确认 `numpy` 版本低于2.0，运行 `pip install "numpy<2.0" pyradiomics`。
  - **DICOM文件读取错误**: 确保安装了 `pydicom`，运行 `pip install pydicom`。

-----

## 模块二：图像扰动

### 2.1 扰动类型说明

  * **膨胀 (Dilation):** 将掩膜边界向外扩张，模拟分割时对目标区域的轻微**高估**。
  * **腐蚀 (Erosion):** 将掩膜边界向内收缩，模拟分割时对目标区域的轻微**低估**。
  * **轮廓随机化 (Contour Randomization):** 对整个掩膜进行一次随机的膨胀或腐蚀，模拟分割边界的随机不确定性。
  * **平移 (Translation):** 将图像和掩膜在x和y轴上随机移动，模拟患者在扫描过程中的轻微**位置移动**。
  * **旋转 (Rotation):** 将图像和掩膜围绕中心随机旋转，模拟患者的轻微**旋转运动**。
  * **高斯噪声 (Gaussian Noise):** 在原始图像的像素值上添加符合高斯分布的随机噪声，模拟MRI信号采集过程中产生的**电子噪声**。

### 2.2 参数设置

#### 1. 平移范围 (Translation Range)

* **参数说明:** 该值定义了在x和y轴上随机平移的最大像素数。程序将从 `[-该值, +该值]` 的整数范围内随机选择一个平移量。
* **效果:** 值越大，图像和掩膜的移动距离就越远，模拟的患者位置变动幅度也越大。

#### 2. 旋转范围 (Rotation Range)

* **参数说明:** 该值定义了随机旋转的最大角度。程序将从 `[0, 该值]` 的范围内随机选择一个旋转角度。
* **效果:** 值越大，图像和掩膜可能发生的旋转角度就越大，模拟的患者旋转运动也越剧烈。

#### 3. 噪声标准差 (Noise Standard Deviation)

* **参数说明:** 该值定义了所添加高斯噪声分布的标准差（均值固定为0）。
* **效果:** 值越大，添加到图像中的噪声强度就越高，图像看起来会更“嘈杂”或“有颗粒感”，信噪比（SNR）更低。

### 2.3 输出结构

程序会在您指定的输出文件夹内，根据原始文件的相对路径创建相同的子文件夹结构。输出文件名将包含原始文件名和所应用的扰动类型。

* **示例:**
    * 原始文件: `slice_10.nii.gz`
    * 应用“膨胀+平移+旋转”扰动后，输出文件为:
        * `slice_10_膨胀_平移_旋转_image.nii.gz`
        * `slice_10_膨胀_平移_旋转_mask.nii.gz`

-----

## 模块三：特征稳健性相关性分析

本模块用于对一个预先计算好的特征数据表进行稳健性分析，其核心是执行一个**三步精炼流程**来筛选特征。

### 3.1 分析工作流

**ICC计算 -\> 分层聚类筛选 -\> 相关性冗余消除**

### 3.2 输入文件要求

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

#### I. 组内相关系数 (Intraclass Correlation Coefficient, ICC)

  * **目的**: ICC是用于量化特征稳健性的核心指标。它的目标是评估不同测量条件（例如，“金标准” vs “加噪图像”）下，由同一组目标（各个椎间盘）产生的定量测量值的一致性或可重复性。

  * **原理**: ICC的计算基于方差分析 (ANOVA)，它将数据的总变异分解为来自不同研究目标（椎间盘）的真实变异和来自测量误差（包括不同条件和随机误差）的变异。一个高的ICC值意味着误差变异远小于真实变异，表明该特征在不同条件下非常稳定。与仅衡量趋势的Pearson或Spearman相关系数不同，ICC评估的是**绝对一致性**，这对于稳健性——即抵抗任何数值变化的能力——的定义至关重要。

#### II. 分层聚类 (Hierarchical Clustering)

  * **目的**: 在计算出所有特征的ICC值后，此步骤的目标是从众多特征中筛选出**一群（a cluster）** 整体表现稳健的特征。

  * **原理**:

    1.  **特征向量化**: 首先，每个特征的“稳健性表现”被视为一个向量（即该特征在所有不同扰动条件下的ICC值组成的数组）。
    2.  **距离计算**: 采用**欧几里得 (Euclidean) 距离**来计算每两个特征向量之间的“不相似度”。距离越近，说明这两个特征在面对各种扰动时的稳定性表现模式越相似。
    3.  **层次构建**: 采用**Ward链接方法 (Ward's linkage)**，这是一个自底向上的聚合过程。它会迭代地合并簇，每一步都选择能使所有簇内方差总和增加最小的合并方案。这倾向于产生大小均等、结构紧凑的球状簇。整个过程的结果可以用一个**树状图 (Dendrogram)** 来可视化。
    4.  **簇的选择**: 在形成的多个特征簇中，根据预设的客观标准（如\*\*`min_icc`\*\*），选择出那个整体表现最好、最可靠的特征群组。

#### III. Spearman相关性分析 (Spearman Correlation Analysis)

  * **目的**: 经过聚类筛选后，我们得到了一组稳健的特征，但它们之间可能存在信息冗余。此步骤旨在剔除这些高度相关的特征，确保最终的特征集既稳健又高效。

  * **原理**:

    1.**采用Spearman相关系数**: 我们选用Spearman相关系数，因为它是一种**非参数**的秩相关方法。它不要求数据呈正态分布，并且对异常值不敏感，非常适合用于评估变量间单调关系的强度，这在放射组学特征分析中是标准做法。
    2.**高阈值筛选**: 设置一个极高的相关性阈值（如**0.99**），以确保只移除那些信息几乎完全重叠的特征。
    3.**方差准则剔除**: 对于每一对高度相关的特征，我们保留方差较高的那一个。这是因为方差更大的特征在数据集中表现出更大的动态范围，可能携带了更多的信息和更强的区分能力。

### 3.4 核心参数设置

#### I. ICC计算设置

| 参数 | 推荐设置 | 原理与依据 |
| :--- | :--- | :--- |
| **ICC类型** | `ICC(3,k)` | **双向混合模型 (Two-Way Mixed-Effects)**。研究目标（椎间盘）是随机样本，而测量条件（`gold`, `noise`等）是我们设定的**固定效应**。 |
| **ICC置信水平** | `0.95` | **95%置信区间**是医学统计研究的黄金标准。 |

#### II. 聚类分析设置

| 参数 | 推荐设置 | 原理与依据 |
| :--- | :--- | :--- |
| **链接方法** | `ward` | 旨在最小化簇内方差之和，倾向于产生大小均等、结构紧凑的球状簇。 |
| **距离度量** | `euclidean` | 欧几里得距离是与`ward`链接方法配套使用的标准度量，因为它基于方差计算，两者在数学上是兼容的。 |
| **簇选择方式** | `min_icc` | 确保了最终入选的特征集中，**每一个成员**在**最差情况**下的表现都有保障（即“木桶的短板最长”）。 |
| **聚类数量** | `manual` | 建议首先使用\*\*“显示聚类树状图”\*\*功能，通过观察树状图的结构来做出专业的、有依据的判断，然后手动输入簇数量。程序提供的“自动建议k值”可作为参考。 |

#### III. 相关性分析设置

| 参数 | 推荐设置 | 原理与依据 |
| :--- | :--- | :--- |
| **相关性阈值** | `0.99` | 旨在只移除那些信息几乎完全重叠、可以被认为是“同义词”的特征。 |
| **方差准则** | `移除方差较低的特征` | 当两个特征高度相关时，保留方差较高的那个，意味着保留了在数据集中动态范围更大、可能携带更多信息的特征。 |

### 3.4 输出文件说明

  - **`robustness_summary_matrix.csv`**: ICC矩阵文件，包含每个特征在每种扰动下的ICC值、平均ICC、最小ICC以及所属的簇标签。
  - **`final_robust_features.csv`**: 最终筛选的稳健特征列表。
  - **`analysis_parameters.json`**: 记录本次分析所有参数的JSON文件，用于保证结果的可复现性。
  - **`analysis_report.txt`**: 详细可读的文本分析报告。

## 7\. 参考文献

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
