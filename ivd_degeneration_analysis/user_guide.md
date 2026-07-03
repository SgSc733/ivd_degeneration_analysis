# 椎间盘退变一体化分析系统

## 1\. 项目简介

本项目是一个基于Python的一体化椎间盘影像分析平台。

系统通过一个集成的 GUI，将三个核心功能模块无缝衔接：

1.  **特征提取模块**: 基于标准的 PyRadiomics 库和一系列源自前沿学术研究的经典特征提取算法，以及基于先进的深度学习模型，对椎间盘影像特征进行量化。
2.  **图像扰动模块**: 对原始图像和分割掩码应用一系列标准化的扰动，以模拟临床实践中的各种不确定性。
3.  **稳健性相关性分析模块**: 按“深度/张量特征 PCA 预降维 → ICC(2,1) 鲁棒性筛选 → Pfirrmann Spearman + BH-FDR → Spearman 去冗余 → 病人级 bootstrap + ElasticNet 稳定选择（λ 由按病人分组 CV 选择）”的流程，从海量特征中筛选出既相关、又稳健、且信息不冗余的“黄金特征集”。

本项目支持对单个病例进行深度分析，也支持对大规模队列数据进行自动化批量处理。

## 2\. 安装指南

### 环境要求

  - Python 3.9+

### 安装步骤

1.  **克隆项目**

    ```bash
    https://github.com/SgSc733/ivd_degeneration_analysis.git
    cd your-repo-name
    ```

2.  **创建并激活Conda环境**

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

## 3\. 使用方法

#### 启动GUI界面

```bash
python run_gui.py
```

系统启动后，会出现一个包含三个主功能选项卡（“特征提取”、“图像扰动”、“稳健性相关性分析”）的集成界面，可根据研究需要，按顺序或独立使用这些模块。


## 4\. 参数设置说明

### 1. 特征提取参数说明与输入格式

所有核心参数均在 `config.py` 文件中进行统一管理。

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
| **DHI** | `central_ratio` | `0.8` | 计算椎间盘高度时用于裁剪中央区域的比例，以减少边缘膨出/边缘效应。 |
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
| **模型版本** | `base`, `small` | 选择使用的Radio-DINO预训练模型的大小。 |
| **Patch聚合策略**| `mean`, `max`, `both` | 定义如何将ROI内部的多个图像块的深度特征聚合成一个单一向量。`mean`捕捉平均特征，`max`捕捉最显著特征，`both`将两者拼接以获得更丰富的信息。 |
| **安全边距**| `0.2` | 在根据掩码裁剪ROI时，向外扩展的边距比例。 |

#### VI. 张量分解特征参数

| 参数 | 示例值 | 意义 |
| :--- | :--- | :--- |
| `roi_size` | `[72, 40, 64]` | 在重采样到各向同性体素后，张量 ROI 的统一尺寸，顺序为 (Z, Y, X)。 |
| `target_spacing_mm` | `1.0` | 重采样到张量 ROI 前，所有 3D 图像统一到的各向同性物理间距 (mm)。 |
| `q_low` | `1` | ROI 内强度分布的下分位数 (百分位)，用于强度裁剪。 |
| `q_high` | `99` | ROI 内强度分布的上分位数 (百分位)，用于强度裁剪。 |
| `energy_threshold` | `0.95` | 能量阈值 $\eta$。对每个模的奇异值序列，选择最小的多线性秩 $R_n$，使前 $R_n$ 个奇异值的能量比例 $\sum_{k=1}^{R_n}\sigma_k^2 / \sum_j \sigma_j^2 \ge \eta$。 |
| `k_singular_values` | `10` | 每个模中用于构造特征的主奇异值个数 $K_n$。 |
| `patch_size` | `4` | Patch 尺寸 $m$，构造 $m\times m\times m$ 的 3D 小块。 |
| `similar_patches` | `64` | 相似块数量 $n$。对每个参考 patch，在非局部搜索窗口中取最相似的 $n$ 个块堆叠成四阶张量 $\mathcal{G}_{\mathcal{Y}}$。 |
| `search_window` | `15` | 搜索窗口边长 $s$（体素数）。决定在多大空间范围内寻找自相似 patch。 |
| `internal_iterations` | `50` | ADMM 内部迭代次数 $T$。|
| `epsilon` | `1e-16` | 对数和范数中的平滑常数 $\varepsilon$。 |
| `alpha_feedback` | `0.1` | Method Noise 外部迭代的反馈系数 $\alpha$，控制噪声残差注入的强度。 |
| `beta_noise` | `0.3` | 噪声估计参数 $\beta$，用于在 ROI 中估计 Rician 噪声标准差 $\sigma_n$。 |
| `max_patch_groups` | `64` | 每个椎间盘 ROI 中最多处理的 patch 组个数。 |
| `max_singular_values` | `10` | 在构造 patch 级奇异值特征时，每个模态保留的奇异值个数 $K$。 |
| `rank` | `8` | 单 ROI CP 分解的秩 $R$。表示用多少个秩‑1 张量来近似 3D 椎间盘张量。 |
| `max_iter` | `1000` | CP‑ALS 最大迭代次数 $T_{\max}$。 |
| `tol` | `1e-4` | CP‑ALS 收敛阈值 $\varepsilon_{\text{ALS}}$。 |
| `epsilon_cp` | `1e-6` | CP 特征构造中的平滑常数 $\varepsilon$，用于计算权重能量比例、有效秩以及因子向量熵/集中度时防止除零或 $\log(0)$ 引起的数值不稳定。 |
| `top_components` | `3` | 参与计算因子向量熵 $H$ 和集中度 $G$ 的前 $K$ 个主成分个数。 |
| `random_state` | `0` | 随机种子，用于控制 CP 分解中的随机性（如内部初始化）。 |

#### VI. 系统行为与性能参数

| 参数 | 示例值 | 意义 |
| :--- | :--- | :--- |
| `OUTPUT_FORMATS` | `['excel', 'json', 'csv']` | **输出格式**。 |
| `FEATURE_SETS` | `{'texture': ['gabor', ...], ...}` | **特征集定义**。 |
| `PARALLEL_CONFIG` | `{'enabled': True, 'max_workers': None}` | **全局并行处理配置**。 |
| `MEMORY_CONFIG` | `{'max_memory_gb': 8, 'cache_enabled': True}` | **内存管理配置**。 |
| `CALCULATOR_PARALLEL`| `{'gabor': {'enabled': True, ...}}`| **计算器并行配置**。 |

#### VII. 输入数据要求

##### 文件格式

*   **支持格式**:  `.nii/.nii.gz`、`.mha/.mhd`、DICOM 等常见医学影像格式。
*   **文件命名**: 掩码图像名在原始图像名基础上加"_mask"，例如：
    *   原始图像: `Case01.nii.gz`
    *   掩码图像: `Case01_mask.nii.gz`
*   **要求**: 原始图像和掩码图像必须在空间上严格对齐，并且具有完全相同的维度（长、宽、切片数）和体素间距。原始图像名字不要带"_"。

##### **配置 `config.py` 中的标签表**

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



### 图像扰动参数设置说明

| 参数 | 对应扰动 | 说明与影响 |
| :--- | :--- | :--- |
| **平移范围** | 平移、组合扰动 | 定义在x和y轴上随机平移的最大像素数。 |
| **旋转范围** | 旋转、组合扰动 | 定义了随机旋转的最大角度。 |
| **噪声标准差** | 高斯噪声、组合扰动 | 定义了所添加高斯噪声分布的标准差 $\sigma$（均值固定为0）。 |
| **形态学核大小** | 膨胀、腐蚀 | 定义了用于形态学操作的椭圆形结构元素的基础尺寸。 |
| **迭代次数** | 膨胀、腐蚀 | 定义了形态学操作的重复次数。 |

#### 输出结构

程序会在指定的输出文件夹内，根据原始文件的相对路径创建相同的子文件夹结构。输出文件名将包含原始文件名和所应用的扰动类型。

*   **示例**:
    *   原始文件: `Case01/slice_10.nii.gz`
    *   输出路径: `D:/output/`
    *   应用“膨胀+平移+旋转”扰动后，输出文件为:
        *   `D:/output/image/Case01/slice_10_dilation_trans_rot_image.nii.gz`
        *   `D:/output/mask/Case01/slice_10_dilation_trans_rot_mask.nii.gz`


### 特征筛选参数设置说明

| 参数 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| `feature_types` | `classic,pyradiomics,deep,tensor` | 特征处理对象选择 |
| `enable_pca` | `True` | 是否启用步骤1 PCA（仅对 deep/tensor 生效） |
| `eta_pca` | `0.95` | PCA 解释方差阈值（深度/张量特征） |
| `m_cap` | `50` | PCA 维度上限 |
| `T_ICC` | `0.80` | ICC 鲁棒阈值（保留 `ICC(2,1) >= T_ICC` 的特征） |
| `alpha_FDR` | `0.05` | Spearman `p` 值做 BH-FDR 校正后的阈值（保留 `q < alpha_FDR` 的特征） |
| `rho_min` | `0.20` | Spearman 效应量门槛（保留 `|rho| >= rho_min` 的特征） |
| `T_dup` | `0.95` | 去冗余阈值；若 `|rho| > T_dup`，移除综合得分更小的特征 |
| `enable_step5` | `True` | 是否启用步骤5稳定选择 |
| `enet_l1_ratio` | `0.80` | ElasticNet 混合系数（越大越偏 L1） |
| `lambda(alpha)` | `0.01` | 手动 λ（仅在关闭“自动选择 λ”时生效） |
| `K_lambda` | `5` | 自动 λ 的分组 K 折数（按病人分组） |
| `L` | `30` | λ 路径长度（候选 λ 个数） |
| `epsilon` | `0.01` | λ 路径比例（最小 λ = λ_max * epsilon） |
| `1-SE` | `False` | 1-SE 规则选 λ（更稀疏、更稳） |
| `enable_lambda_size_tuning` | `False` | 可选：在 `auto_cv` 下沿 λ 路径做“尺寸约束微调”，当 `|S_4.1(λ)|` 过大/过小时尝试向稀疏/稠密端移动 |
| `bootstrap_B` | `50` | 病人级 bootstrap 次数 |
| `delta` | `1e-4` | 系数阈值（判定是否“被选中”） |
| `tau` | `0.30` | 稳定频次阈值（保留 `pi >= tau`） |
| `K_max` | `120` | 最终特征数上限（可设为 `0` 表示不限制） |

#### 输出文件说明

- `pca_deep_info.csv`: 深度特征 PCA 解释方差表（启用 PCA 且存在 Deep_ 特征时输出）。
- `pca_tensor_info.csv`: 张量特征 PCA 解释方差表（启用 PCA 且存在 Tucker_/Tensor_ 特征时输出）。
- `icc_values.csv`: ICC 计算用表。
- `robust_features_by_icc.csv`: 通过 ICC 阈值的特征。
- `spearman_with_pfirrmann.csv`: Spearman+BH-FDR 统计量表。
- `dedup_pairs.csv`: 去冗余过程中触发删除的高相关特征对对照表。
- `lambda_cv.csv`: 自动 λ 模式下的 CV 误差表。
- `lambda_size_tuning.csv`: 启用“尺寸约束微调”时记录沿 λ 路径尝试的 `alpha` 与对应稳定集合大小。
- `stability_selection.csv`: 进入稳定选择阶段（去冗余后）的特征表。
- `final_robust_features.csv`: 最终筛选特征（含全部中间量列；第一列为 `feature`。
- `最终模型输入.csv`: 基于 gold（未扰动）且已清洗）的最终特征输入表。
- `analysis_report.txt`: 文本摘要报告。
- `icc_hist_kde_pyradiomics.png` / `icc_hist_kde_all.png`: ICC(2,1) 分布的直方图 + KDE。
- `icc_retained_vs_threshold_pyradiomics.png` / `icc_retained_vs_threshold_all.png`: “保留特征数 vs ICC 阈值”的曲线。
- `corr_heatmap_pre_dedup.png`: “相关性去冗余”前（`S_rel`）的 Spearman 相关矩阵热图。
- `corr_heatmap_post_dedup.png`: “相关性去冗余”后（`S_nr`）的 Spearman 相关矩阵热图。
- （可选，提供 statistics.csv 时）`cohort_pfirrmann_distribution.png`: Pfirrmann 等级分布图。
- （可选，提供 statistics.csv 时）`cohort_scanner_batch_distribution.png`: manufacturer / batch_scanner 分布图。


输出特征表（上述所有 `*.csv`）中的关键审计字段：

| 字段 | 含义 |
| :--- | :--- |
| `pass_icc` | 是否通过 ICC 阈值筛选（`icc_21 >= T_ICC`） |
| `pass_spearman` | 是否通过 Spearman+FDR 预筛（属于 `S_rel`） |
| `pass_dedup` | 是否通过步骤4去冗余（在鲁棒特征内部去冗余后仍被保留） |
| `pass_topk` | 是否在 `K_max` 约束下进入最终特征集 |
| `dedup_removed_by` | 若 `pass_dedup=False`，记录将其替代并保留的特征名 |
| `dedup_removed_corr` | `feature` 与 `dedup_removed_by` 的 Spearman 相关系数 ρ（gold 条件） |
| `dedup_removed_abs_corr` | `|dedup_removed_corr|` |
| `dedup_removed_order` | 被剔除的顺序（1=最先被剔除；通常对应更强的相关对） |
| `dedup_max_abs_corr` | `feature` 与其余鲁棒特征的最大 `|rho|`（衡量冗余程度） |
| `dedup_n_corr_over_threshold` | `feature` 与其余鲁棒特征中满足 `|rho| > T_dup` 的个数 |
| `pi` | 病人级 bootstrap 稳定频次（被选中的比例） |
| `mean_abs_beta` | bootstrap 中 `|beta|` 的平均值（辅助排序） |
| `rank_score` | `r_j = pi * mean_abs_beta`（用于 `K_max` 截断排序） |
| `pass_stability` | 是否通过稳定频次阈值（`pi >= tau`） |
| `final_selected` | 是否进入最终特征集（考虑 `K_max` 后） |