import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import tkinter as tk
from gui import IntegratedFeatureExtractorGUI
from calculator import (
    DHICalculator, ASICalculator, FractalDimensionCalculator,
    GaborCalculator, HuMomentsCalculator, TextureFeaturesCalculator,
    T2SignalIntensityCalculator
)

from config import Config
from utils import ImageIO, Preprocessor
from visualization import Visualizer


class IVDAnalysisSystem:

    def __init__(self, config: Optional[Config] = None):

        self.config = config or Config()

        self.image_io = ImageIO()
        self.preprocessor = Preprocessor()
        self.visualizer = Visualizer(**self.config.VIS_PARAMS)

        self.dhi_calculator = DHICalculator(**self.config.DHI_PARAMS)
        self.asi_calculator = ASICalculator(**self.config.ASI_PARAMS)
        self.fd_calculator = FractalDimensionCalculator(**self.config.FD_PARAMS)
        self.t2si_calculator = T2SignalIntensityCalculator(**self.config.T2SI_PARAMS) 
        self.gabor_calculator = GaborCalculator(**self.config.GABOR_PARAMS)
        self.hu_moments_calculator = HuMomentsCalculator(**self.config.HU_MOMENTS_PARAMS)
        self.texture_calculator = TextureFeaturesCalculator(**self.config.TEXTURE_PARAMS)

        self._setup_logging()
        
    def _setup_logging(self):

        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.config.OUTPUT_DIR / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_single_case(self, image_path: str, segmentation_path: str,
                        case_id: Optional[str] = None,
                        feature_set: str = 'all',
                        spacing: Optional[List[float]] = None) -> pd.DataFrame:

        if case_id is None:
            case_id = Path(image_path).stem
            
        self.logger.info(f"开始分析病例: {case_id}")

        output_dir = self.config.OUTPUT_DIR / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            image, segmentation = self.image_io.load_image_and_mask(
                image_path, segmentation_path
            )

            if spacing is None:
                spacing = list(image.GetSpacing())[::-1]  
                self.logger.info(f"从图像读取间距: {spacing}")

            image_array = self.image_io.sitk_to_numpy(image)
            seg_array = self.image_io.sitk_to_numpy(segmentation)

            image_slices = self._extract_middle_slices(image_array, self.config.NUM_SLICES, self.config.SLICE_AXIS)
            seg_slices = self._extract_middle_slices(seg_array, self.config.NUM_SLICES, self.config.SLICE_AXIS)

            selected_features = self.config.FEATURE_SETS.get(feature_set, self.config.FEATURE_SETS['all'])

            results = []
            for level_name, labels in self.config.DISC_LABELS.items():
                self.logger.info(f"处理{level_name}层级")
                
                level_result = {
                    'case_id': case_id,
                    'level': level_name
                }
                
                try:
                    disc_masks = []
                    upper_masks = []
                    lower_masks = []
                    csf_masks = []
                    
                    for seg_slice in seg_slices:
                        disc_mask = (seg_slice == labels['disc']).astype(np.uint8)
                        upper_mask = (seg_slice == labels['upper']).astype(np.uint8)
                        lower_mask = (seg_slice == labels['lower']).astype(np.uint8)
                        csf_mask = (seg_slice == self.config.CSF_LABEL).astype(np.uint8)
                        
                        disc_masks.append(disc_mask)
                        upper_masks.append(upper_mask)
                        lower_masks.append(lower_mask)
                        csf_masks.append(csf_mask)

                    if not any(np.any(mask) for mask in disc_masks):
                        self.logger.warning(f"{level_name}层级没有找到椎间盘区域")
                        level_result['status'] = 'no_disc_found'
                        results.append(level_result)
                        continue

                    processed_data = self._preprocess_for_features(
                        image_slices, disc_masks, csf_masks, spacing, selected_features
                    )

                    if 'dhi' in selected_features:
                        dhi_result = self.dhi_calculator.process_multi_slice(
                            upper_masks, disc_masks, lower_masks,
                            is_l5_s1=(level_name == 'L5-S1')
                        )
                        level_result.update({f'dhi_{k}': v for k, v in dhi_result.items()})

                    if 'asi' in selected_features:
                        asi_slices = processed_data.get('signal_intensity', image_slices)
                        asi_result = self.asi_calculator.process_multi_slice(
                            asi_slices, disc_masks, csf_masks
                        )
                        level_result.update({f'asi_{k}': v for k, v in asi_result.items()})

                    if 't2si' in selected_features:
                        t2si_slices = processed_data.get('signal_intensity', image_slices)
                        t2si_result = self.t2si_calculator.process_multi_slice(
                            t2si_slices, disc_masks, csf_masks
                        )
                        level_result.update({f't2si_{k}': v for k, v in t2si_result.items()})

                    if 'fd' in selected_features:
                        fd_slices = processed_data.get('fractal', [])
                        if fd_slices:
                            fd_result = self.fd_calculator.process_multi_slice(
                                fd_slices, disc_masks
                            )
                            level_result.update({f'fd_{k}': v for k, v in fd_result.items()})

                    if 'texture_features' in selected_features:
                        texture_slices = processed_data.get('texture', image_slices)
                        texture_result = self.texture_calculator.process_multi_slice(
                            texture_slices, disc_masks
                        )
                        level_result.update({f'texture_{k}': v for k, v in texture_result.items()})

                    if 'gabor' in selected_features:
                        gabor_slices = processed_data.get('texture', image_slices)
                        gabor_result = self.gabor_calculator.process_multi_slice(
                            gabor_slices, disc_masks
                        )
                        level_result.update({f'gabor_{k}': v for k, v in gabor_result.items()})

                    if 'hu_moments' in selected_features:
                        hu_masks = processed_data.get('shape', disc_masks)
                        hu_result = self.hu_moments_calculator.process_multi_slice(
                            image_slices, hu_masks 
                        )
                        level_result.update({f'hu_{k}': v for k, v in hu_result.items()})
                    
                    level_result['status'] = 'success'

                    if self.config.VIS_PARAMS['save_intermediate']:
                        self._save_visualizations(
                            output_dir, level_name, image_slices[0], 
                            disc_masks[0], csf_masks[0], upper_masks[0], lower_masks[0],
                            level_result
                        )
                    
                except Exception as e:
                    self.logger.error(f"{level_name}分析失败: {str(e)}")
                    level_result['status'] = 'failed'
                    level_result['error'] = str(e)
                
                results.append(level_result)

            df = pd.DataFrame(results)

            self._save_results(df, output_dir, case_id)
            
            self.logger.info(f"病例{case_id}分析完成")
            
            return df
            
        except Exception as e:
            self.logger.error(f"病例{case_id}分析失败: {str(e)}")
            raise
    
    def _extract_middle_slices(self, array: np.ndarray, num_slices: int, axis: int) -> List[np.ndarray]:

        size = array.shape[axis]
        middle = size // 2
        half_num = num_slices // 2
        
        start_idx = max(0, middle - half_num)
        end_idx = min(size, start_idx + num_slices)
        
        slices = []
        for i in range(start_idx, end_idx):
            if axis == 0:
                slices.append(array[i, :, :])
            elif axis == 1:
                slices.append(array[:, i, :])
            else:
                slices.append(array[:, :, i])
        
        return slices
    
    def _preprocess_for_features(self, image_slices: List[np.ndarray], 
                               disc_masks: List[np.ndarray],
                               csf_masks: List[np.ndarray],
                               spacing: List[float],
                               selected_features: List[str]) -> Dict[str, List[np.ndarray]]:

        processed_data = {}

        if any(feat in selected_features for feat in ['texture_features', 'gabor']):
            texture_slices = []
            for img, mask in zip(image_slices, disc_masks):
                processed, _ = self.preprocessor.preprocess_for_texture(
                    img, mask, spacing[:2] + [1.0],  
                    target_spacing=[1.0, 1.0, 1.0]
                )
                texture_slices.append(processed)
            processed_data['texture'] = texture_slices

        if any(feat in selected_features for feat in ['asi', 't2si']):
            si_slices = []
            for img, mask in zip(image_slices, disc_masks):
                processed, _ = self.preprocessor.preprocess_for_signal_intensity(
                    img, mask, spacing[:2] + [1.0],
                    target_spacing=[1.0, 1.0, 1.0]
                )
                si_slices.append(processed)
            processed_data['signal_intensity'] = si_slices

        if 'fd' in selected_features:
            fd_slices = []
            for img, mask in zip(image_slices, disc_masks):
                edges, _ = self.preprocessor.preprocess_for_fractal(
                    img, mask, spacing[:2] + [1.0],
                    target_spacing=[1.0, 1.0, 1.0]
                )
                fd_slices.append(edges)
            processed_data['fractal'] = fd_slices

        if 'hu_moments' in selected_features:
            shape_masks = []
            for mask in disc_masks:
                binary_mask = self.preprocessor.preprocess_for_shape(
                    mask, spacing[:2] + [1.0],
                    target_spacing=[1.0, 1.0, 1.0]
                )
                shape_masks.append(binary_mask)
            processed_data['shape'] = shape_masks
        
        return processed_data
    
    def _save_visualizations(self, output_dir: Path, level_name: str,
                           image: np.ndarray, disc_mask: np.ndarray,
                           csf_mask: np.ndarray, upper_mask: np.ndarray,
                           lower_mask: np.ndarray, results: Dict):

        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        dhi_result = {k.replace('dhi_', ''): v for k, v in results.items() if k.startswith('dhi_')}
        asi_result = {k.replace('asi_', ''): v for k, v in results.items() if k.startswith('asi_')}
        fd_result = {k.replace('fd_', ''): v for k, v in results.items() if k.startswith('fd_')}
        t2si_result = {k.replace('t2si_', ''): v for k, v in results.items() if k.startswith('t2si_')}

        if dhi_result and 'dhi' in dhi_result:
            self.visualizer.visualize_dhi_result(
                image, upper_mask, disc_mask, lower_mask,
                dhi_result, str(vis_dir / f'{level_name}_dhi.png')
            )

        if asi_result and 'asi' in asi_result:
            self.visualizer.visualize_asi_result(
                image, disc_mask, csf_mask,
                asi_result, str(vis_dir / f'{level_name}_asi.png')
            )

        if t2si_result and 'si_ratio' in t2si_result:
            t2si_result['roi_mask'] = disc_mask
            t2si_result['roi_method'] = self.config.T2SI_PARAMS['roi_method']
            self.visualizer.visualize_t2si_result(
                image, disc_mask, csf_mask,
                t2si_result, str(vis_dir / f'{level_name}_t2si.png')
            )

        if dhi_result and asi_result:
            self.visualizer.visualize_combined_results(
                image, dhi_result, asi_result, fd_result,
                str(vis_dir / f'{level_name}_combined.png')
            )
    
    def batch_analysis(self, input_csv: str, output_dir: Optional[str] = None,
                      feature_set: str = 'all') -> pd.DataFrame:

        if output_dir:
            self.config.OUTPUT_DIR = Path(output_dir)

        input_df = pd.read_csv(input_csv)
        required_cols = ['case_id', 'image_path', 'segmentation_path']
        
        if not all(col in input_df.columns for col in required_cols):
            raise ValueError(f"CSV文件必须包含以下列: {required_cols}")
        
        self.logger.info(f"找到{len(input_df)}个病例")
        
        all_results = []
        success_count = 0
        
        for idx, row in input_df.iterrows():
            try:
                self.logger.info(f"处理进度: {idx+1}/{len(input_df)}")

                spacing = None
                if 'spacing' in row and pd.notna(row['spacing']):
                    try:
                        spacing = [float(x) for x in row['spacing'].split(',')]
                    except:
                        self.logger.warning(f"无法解析间距信息: {row['spacing']}")
                
                df = self.analyze_single_case(
                    row['image_path'], 
                    row['segmentation_path'],
                    row['case_id'],
                    feature_set,
                    spacing
                )
                all_results.append(df)
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"处理{row['case_id']}失败: {str(e)}")

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)

            self._save_batch_results(final_df)
            
            self.logger.info(f"批量分析完成: 成功{success_count}/{len(input_df)}")
            
            return final_df
        else:
            self.logger.warning("没有成功处理的病例")
            return pd.DataFrame()
        
    def batch_analysis_parallel(self, input_csv: str, output_dir: Optional[str] = None,
                            feature_set: str = 'all', 
                            parallel: bool = True) -> pd.DataFrame:

        if output_dir:
            self.config.OUTPUT_DIR = Path(output_dir)

        input_df = pd.read_csv(input_csv)
        
        if parallel and self.config.PARALLEL_CONFIG['enabled']:
            self.logger.info(f"使用并行处理，工作进程数: {self.config.PARALLEL_CONFIG['max_workers']}")

            from utils.parallel_processor import ParallelProcessor, parallel_feature_extraction
            
            processor = ParallelProcessor(
                max_workers=self.config.PARALLEL_CONFIG['max_workers'],
                backend=self.config.PARALLEL_CONFIG['backend']
            )

            tasks = []
            for _, row in input_df.iterrows():
                case_info = {
                    'case_id': row['case_id'],
                    'image_path': row['image_path'],
                    'mask_path': row['segmentation_path']
                }
                tasks.append((case_info, feature_set, self.config))

            results = processor.process_batch_with_memory_limit(
                parallel_feature_extraction,
                tasks,
                max_memory_gb=self.config.MEMORY_CONFIG['max_memory_gb'],
                desc="提取特征"
            )
            
            final_df = pd.DataFrame(results)
            
        else:
            self.logger.info("使用串行处理")
            final_df = self.batch_analysis(input_csv, output_dir, feature_set)
        
        return final_df
    
    def _save_results(self, df: pd.DataFrame, output_dir: Path, case_id: str):

        if 'excel' in self.config.OUTPUT_FORMATS:
            excel_path = output_dir / f"{case_id}_results.xlsx"
            df.to_excel(excel_path, index=False)
            self.logger.info(f"结果已保存到: {excel_path}")

        if 'json' in self.config.OUTPUT_FORMATS:
            json_path = output_dir / f"{case_id}_results.json"
            df.to_json(json_path, orient='records', indent=2)

        if 'csv' in self.config.OUTPUT_FORMATS:
            csv_path = output_dir / f"{case_id}_results.csv"
            df.to_csv(csv_path, index=False)

        summary_path = output_dir / f"{case_id}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"椎间盘退变分析报告\n")
            f.write(f"================\n\n")
            f.write(f"病例ID: {case_id}\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for _, row in df.iterrows():
                if row.get('status') == 'success':
                    f.write(f"\n{row['level']}层级:\n")
                    f.write(f"  DHI: {row.get('dhi_dhi', 'N/A'):.3f}\n")
                    f.write(f"  ASI: {row.get('asi_asi', 'N/A'):.2f}\n")
                    f.write(f"  FD: {row.get('fd_fd', 'N/A'):.3f}\n")
                    f.write(f"  T2SI: {row.get('t2si_si_ratio', 'N/A'):.3f}\n")
    
    def _save_batch_results(self, df: pd.DataFrame):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if 'excel' in self.config.OUTPUT_FORMATS:
            summary_excel = self.config.OUTPUT_DIR / f"batch_results_{timestamp}.xlsx"
            with pd.ExcelWriter(summary_excel, engine='openpyxl') as writer:

                df.to_excel(writer, sheet_name='原始数据', index=False)

                summary_stats = self._calculate_summary_statistics(df)
                summary_stats.to_excel(writer, sheet_name='统计摘要')

        report_path = self.config.OUTPUT_DIR / f"batch_report_{timestamp}.txt"
        self._generate_batch_report(df, report_path)
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:

        success_df = df[df['status'] == 'success']
        
        if success_df.empty:
            return pd.DataFrame()
        
        stats = []
        for level in self.config.DISC_LABELS.keys():
            level_df = success_df[success_df['level'] == level]
            if not level_df.empty:
                stats_dict = {
                    '层级': level,
                    '样本数': len(level_df)
                }

                feature_prefixes = ['dhi_', 'asi_', 'fd_', 't2si_']
                for prefix in feature_prefixes:
                    main_col = f'{prefix}{prefix[:-1]}' 
                    if main_col in level_df.columns:
                        stats_dict[f'{prefix[:-1].upper()}均值'] = level_df[main_col].mean()
                        stats_dict[f'{prefix[:-1].upper()}标准差'] = level_df[main_col].std()
                
                stats.append(stats_dict)
        
        return pd.DataFrame(stats)
    
    def _generate_batch_report(self, df: pd.DataFrame, report_path: Path):

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("椎间盘退变批量分析报告\n")
            f.write("=====================\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总病例数: {df['case_id'].nunique()}\n")
            f.write(f"总椎间盘数: {len(df)}\n")
            f.write(f"成功率: {(df['status'] == 'success').mean():.1%}\n\n")

            summary_stats = self._calculate_summary_statistics(df)
            if not summary_stats.empty:
                f.write("各层级统计摘要:\n")
                f.write(summary_stats.to_string(index=False))


def main():

    parser = argparse.ArgumentParser(
        description='椎间盘退变特征分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--gui', action='store_true', help='启动GUI界面')

    subparsers = parser.add_subparsers(dest='command', help='分析命令')

    single_parser = subparsers.add_parser('single', help='分析单个病例')
    single_parser.add_argument('--image', type=str, required=True, help='图像文件路径')
    single_parser.add_argument('--seg', type=str, required=True, help='分割文件路径')
    single_parser.add_argument('--output-dir', type=str, help='输出目录')
    single_parser.add_argument('--case-id', type=str, help='病例ID')
    single_parser.add_argument('--spacing', type=str, help='体素间距，格式：x,y,z')

    batch_parser = subparsers.add_parser('batch', help='批量分析')
    batch_parser.add_argument('--input-csv', type=str, required=True, help='输入CSV文件')
    batch_parser.add_argument('--output-dir', type=str, help='输出目录')

    parser.add_argument('--config', type=str, help='配置文件路径（JSON格式）')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--feature-set', type=str, default='all',
                   choices=['conventional', 'shape', 'texture', 'fractal', 'signal', 'all'],
                   help='选择要计算的特征集')
    
    args = parser.parse_args()

    if args.gui:
        root = tk.Tk()
        app = IntegratedFeatureExtractorGUI(root)
        root.mainloop()
        return

    if not args.command:
        parser.print_help()
        return

    config = Config()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    if hasattr(args, 'output_dir') and args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)

    system = IVDAnalysisSystem(config)

    if args.command == 'single':
        spacing = None
        if args.spacing:
            spacing = [float(x) for x in args.spacing.split(',')]
        
        system.analyze_single_case(
            args.image, args.seg, args.case_id, 
            args.feature_set, spacing
        )
    elif args.command == 'batch':
        system.batch_analysis(
            args.input_csv,
            args.output_dir,
            args.feature_set
        )


if __name__ == "__main__":
    main()