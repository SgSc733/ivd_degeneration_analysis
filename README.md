# MRI-T2 椎间盘退变特征提取系统

## 项目简介

本项目是一个基于Python的MRI-T2椎间盘影像特征提取系统。它集成了标准的 **PyRadiomics** 库和一系列源自前沿学术研究的**自定义特征提取算法**，旨在为椎间盘退变的定量化、可重复性研究提供一套完整的解决方案。

系统通过图形用户界面（GUI）和命令行两种方式，支持对单个病例进行分析，也支持对大规模队列数据进行批量处理。

## 主要功能

### 特征提取
- **PyRadiomics 标准特征**:
  - 一阶统计特征 (First Order)
  - 形状特征 (Shape 2D/3D)
  - 纹理特征 (GLCM, GLRLM, GLSZM, GLDM, NGTDM)
- **自定义特征**(PyRadiomics无法提取的特征):
  - **[DHI]** 椎间盘高度指数 (Disc Height Index)
  - **[ASI]** 峰值信号强度差 (Peak Signal Intensity Difference)
  - **[FD]** 分形维度 (Fractal Dimension)
  - **[T2SI]** T2信号强度比率 (T2 Signal Intensity Ratio)
  - **[Gabor]** Gabor纹理特征
  - **[Hu]** Hu不变矩 (Hu Invariant Moments)
  - **[Texture]** 扩展纹理特征 (如LBP)

### 自定义特征计算方法简介

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

### 预处理功能
- **空间处理**: 图像重采样至统一体素间距。
- **强度处理**:
  - Z-score 标准化
  - 窗位窗宽调整 (Windowing)
  - 灰度离散化 (Discretization / Binning)
- **图像变换**: 边缘检测、高斯拉普拉斯(LoG)滤波、小波(Wavelet)变换等。

### 验证工具
项目内含一个独立的预处理效果验证工具 (`preprocessing_validator`)，可以通过可视化界面，直观地检查和验证每一步预处理操作的效果是否符合预期。

## 安装指南

### 环境要求
- Python 3.9+
- Windows / Linux / macOS
- **重要**: `numpy<2.0` (PyRadiomics的兼容性要求)

### 安装步骤

1.  **克隆项目**
    ```bash
    git clone https://github.com/SgSc733/ivd_degeneration_analysis.git
    cd ivd-feature-extraction
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

## 使用方法

#### 1. 启动GUI界面
```bash
python run_gui.py
```

#### 2. 命令行使用

- **单个病例分析**:
  ```bash
  python main.py single --image path/to/image.nii --seg path/to/segmentation.nii --output-dir output/
  ```

- **批量处理**:
  ```bash
  python main.py batch --input-csv batch_list.csv --output-dir output/
  ```

- **批量处理CSV格式**: `batch_list.csv` 文件应包含以下列:
  ```csv
  case_id,image_path,segmentation_path
  case001,/path/to/image1.nii,/path/to/seg1.nii
  case002,/path/to/image2.nii,/path/to/seg2.nii
  ```

#### 3. 预处理验证工具
```bash
python preprocessing_validator.py
```

## 详细参数设置说明

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

## 项目结构
```
ivd-feature-extraction/
├── calculator/          # 特征计算器模块
├── gui/                # 图形用户界面 (GUI)
├── utils/              # 工具类 (图像IO, 预处理，并行计算)
├── visualization/      # 可视化模块
├── preprocessing_validator/  # 预处理验证工具
├── config.py           # 统一配置文件
├── main.py             # 命令行主程序
├── run_gui.py          # GUI启动脚本
└── requirements.txt    # 依赖包列表
```

## 输出说明
系统会为每个病例或整个批次生成以下输出：
- **Excel (.xlsx) / CSV (.csv) 文件**: 包含所有提取的特征值，每个椎间盘为一行。
- **可视化图像 (.png)**: (可选) 保存关键特征的计算过程和结果的可视化图像，便于直观检查。
- **日志文件 (.log)**: 记录详细的运行过程、警告和错误信息。


## 注意事项

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


### 参考文献
[1]McSweeney T, Tiulpin A, Kowlagi N, Määttä J, Karppinen J, Saarakkala S. Robust Radiomic Signatures of Intervertebral Disc Degeneration from MRI. Spine (Phila Pa 1976). 2025 Jun 20.

[2]Ma J, Wang R, Yu Y, Xu X, Duan H, Yu N. Is fractal dimension a reliable imaging biomarker for the quantitative classification of an intervertebral disk? Eur Spine J. 2020 May;29(5):1175-1180. 

[3]Murto N, Luoma K, Lund T, Kerttula L. Reliability of T2-weighted signal intensity-based quantitative measurements and visual grading of lumbar disc degeneration on MRI. Acta Radiol. 2023 Jun;64(6):2145-2151.

[4]Ruiz-España S, Arana E, Moratal D. Semiautomatic computer-aided classification of degenerative lumbar spine disease in magnetic resonance imaging. Comput Biol Med. 2015 Jul;62:196-205. 

[5]Beulah, A., Sharmila, T.S. & Pramod, V.K. Degenerative disc disease diagnosis from lumbar MR images using hybrid features. Vis Comput 38, 2771–2783 (2022).

[6]Michopoulou S, Costaridou L, Vlychou M, Speller R, Todd-Pokropek A. Texture-based quantification of lumbar intervertebral disc degeneration from conventional T2-weighted MRI. Acta Radiol. 2011 Feb 1;52(1):91-8. 

[7]Zheng, HD., Sun, YL., Kong, DW. et al. Deep learning-based high-accuracy quantitation for lumbar intervertebral disc degeneration from MRI. Nat Commun 13, 841 (2022).

[8]Waldenberg C, Hebelka H, Brisby H, Lagerstrand KM. MRI histogram analysis enables objective and continuous classification of intervertebral disc degeneration. Eur Spine J. 2018 May;27(5):1042-1048.

[9]van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. 

[10]Lin A, Zhang H, Wang Y, Cui Q, Zhu K, Zhou D, Han S, Meng S, Han J, Li L, Zhou C, Ma X. Radiomics based on MRI to predict recurrent L4-5 disc herniation after percutaneous endoscopic lumbar discectomy. BMC Med Imaging. 2024 Oct 10;24(1):273.
