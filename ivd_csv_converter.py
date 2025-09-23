import os
import pandas as pd
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import warnings
warnings.filterwarnings('ignore')


class FeatureDataConverter:
    
    def __init__(self):
        self.gold_standard_df = None
        self.perturbation_dfs = {}
        self.merged_df = None
        
    def extract_base_case_id(self, case_id):

        if pd.isna(case_id):
            return None, None
            
        case_id_str = str(case_id).strip()
        
        first_underscore = case_id_str.find('_')
        
        if first_underscore == -1:
            return case_id_str, 'gold'
        else:
            base_id = case_id_str[:first_underscore]
            perturbation_type = case_id_str[first_underscore + 1:]
            
            for suffix in ['_image', '_mask']:
                if perturbation_type.endswith(suffix):
                    perturbation_type = perturbation_type[:-len(suffix)]
                    break
            
            return base_id, perturbation_type
    
    def load_gold_standard(self, filepath):

        print(f"加载金标准文件: {filepath}")
        self.gold_standard_df = pd.read_csv(filepath)
        
        if 'case_id' not in self.gold_standard_df.columns:
            raise ValueError("金标准文件缺少'case_id'列")
        
        self.gold_standard_df['base_case_id'] = self.gold_standard_df['case_id'].apply(
            lambda x: self.extract_base_case_id(x)[0]
        )
        
        print(f"金标准文件包含 {len(self.gold_standard_df)} 个样本")
        print(f"基础case_id列表: {self.gold_standard_df['base_case_id'].unique()}")
        
    def load_perturbation_file(self, filepath):

        print(f"\n加载扰动文件: {filepath}")
        df = pd.read_csv(filepath)
        
        if 'case_id' not in df.columns:
            raise ValueError(f"扰动文件 {filepath} 缺少'case_id'列")
        
        df[['base_case_id', 'perturbation_type']] = df['case_id'].apply(
            lambda x: pd.Series(self.extract_base_case_id(x))
        )
        
        perturbation_types = df['perturbation_type'].unique()
        perturbation_types = [pt for pt in perturbation_types if pt != 'gold']
        
        print(f"文件包含 {len(perturbation_types)} 种扰动类型")
        
        for ptype in perturbation_types:
            ptype_df = df[df['perturbation_type'] == ptype].copy()
            
            if ptype in self.perturbation_dfs:
                self.perturbation_dfs[ptype] = pd.concat([self.perturbation_dfs[ptype], ptype_df], 
                                                         ignore_index=True)
                print(f"  扰动类型 '{ptype}': 追加 {len(ptype_df)} 个样本")
            else:
                self.perturbation_dfs[ptype] = ptype_df
                print(f"  扰动类型 '{ptype}': {len(ptype_df)} 个样本")
        
        print(f"基础case_id列表: {sorted(df['base_case_id'].unique())}")
    
    def merge_data(self):

        print("\n" + "="*60)
        print("开始合并数据...")
        print("="*60)
        
        if self.gold_standard_df is None:
            raise ValueError("未加载金标准数据")
        
        all_base_ids = set(self.gold_standard_df['base_case_id'].unique())
        for df in self.perturbation_dfs.values():
            all_base_ids.update(df['base_case_id'].unique())
        
        all_base_ids = sorted(list(all_base_ids))
        print(f"\n总共找到 {len(all_base_ids)} 个唯一的基础case_id: {all_base_ids}")
        
        print(f"\n发现 {len(self.perturbation_dfs)} 种扰动类型:")
        for i, ptype in enumerate(sorted(self.perturbation_dfs.keys()), 1):
            print(f"  {i}. {ptype} ({len(self.perturbation_dfs[ptype])} 个样本)")
        
        feature_columns = [col for col in self.gold_standard_df.columns 
                          if col not in ['case_id', 'base_case_id', 'perturbation_type']]
        print(f"\n找到 {len(feature_columns)} 个特征列")
        if len(feature_columns) <= 10:
            print(f"特征列: {feature_columns}")
        else:
            print(f"特征列示例: {feature_columns[:5]} ... (共{len(feature_columns)}个)")
        
        result_data = []
        
        for base_id in all_base_ids:
            row_data = {'case_id': base_id}
            
            gold_rows = self.gold_standard_df[self.gold_standard_df['base_case_id'] == base_id]
            if not gold_rows.empty:
                gold_row = gold_rows.iloc[0]
                for col in feature_columns:
                    if col in gold_row.index:
                        new_col_name = f"{col}_gold"
                        row_data[new_col_name] = gold_row[col]
            else:
                for col in feature_columns:
                    new_col_name = f"{col}_gold"
                    row_data[new_col_name] = np.nan
            
            for perturbation_type, df in self.perturbation_dfs.items():
                perturb_rows = df[df['base_case_id'] == base_id]
                if not perturb_rows.empty:
                    perturb_row = perturb_rows.iloc[0]
                    for col in feature_columns:
                        if col in perturb_row.index:
                            new_col_name = f"{col}_{perturbation_type}"
                            row_data[new_col_name] = perturb_row[col]
                else:
                    for col in feature_columns:
                        new_col_name = f"{col}_{perturbation_type}"
                        row_data[new_col_name] = np.nan
            
            result_data.append(row_data)
        
        self.merged_df = pd.DataFrame(result_data)
        
        self.reorder_columns()
        
        print("\n" + "="*60)
        print("合并完成！")
        print("="*60)
        print(f"最终数据维度: {self.merged_df.shape}")
        print(f"行数（基础case_id数）: {len(self.merged_df)}")
        print(f"列数（特征×条件）: {len(self.merged_df.columns)}")
        
        conditions_per_feature = (len(self.merged_df.columns) - 1) / len(feature_columns)
        print(f"平均每个特征有 {conditions_per_feature:.1f} 个条件（1个gold + {conditions_per_feature-1:.0f}个扰动）")
        
    def reorder_columns(self):

        if self.merged_df is None:
            return
        
        ordered_columns = ['case_id']
        
        feature_columns = [col for col in self.merged_df.columns if col != 'case_id']
        
        feature_groups = {}
        
        for col in feature_columns:
            
            if col.endswith('_gold'):
                base_name = col[:-5]
                condition = 'gold'
            else:
                
                perturbation_types = list(self.perturbation_dfs.keys())
                
                base_name = None
                condition = None
                
                for ptype in perturbation_types:
                    suffix = f"_{ptype}"
                    if col.endswith(suffix):
                        base_name = col[:-len(suffix)]
                        condition = ptype
                        break
                
                if base_name is None:
                    last_underscore = col.rfind('_')
                    if last_underscore > 0:
                        base_name = col[:last_underscore]
                        condition = col[last_underscore + 1:]
                    else:
                        base_name = col
                        condition = 'unknown'
            
            if base_name not in feature_groups:
                feature_groups[base_name] = {}
            feature_groups[base_name][condition] = col
        
        for base_name in sorted(feature_groups.keys()):
            conditions = feature_groups[base_name]
            
            if 'gold' in conditions:
                ordered_columns.append(conditions['gold'])
            
            other_conditions = sorted([c for c in conditions.keys() if c != 'gold'])
            for condition in other_conditions:
                ordered_columns.append(conditions[condition])
        
        self.merged_df = self.merged_df[ordered_columns]
        
    def save_merged_data(self, output_path):

        if self.merged_df is None:
            raise ValueError("没有可保存的合并数据")
        
        self.merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存至: {output_path}")
        
        print("\n数据摘要:")
        print(f"- 总行数: {len(self.merged_df)}")
        print(f"- 总列数: {len(self.merged_df.columns)}")
        print(f"- 基础case_id列表: {self.merged_df['case_id'].tolist()}")
        
        condition_counts = {}
        known_conditions = ['gold'] + list(self.perturbation_dfs.keys())
        
        sorted_conditions = sorted(known_conditions, key=len, reverse=True)
        
        processed_cols = set()
        
        for col in self.merged_df.columns:
            if col == 'case_id' or col in processed_cols:
                continue
                
            for condition in sorted_conditions:
                suffix = f"_{condition}"
                if col.endswith(suffix):
                    feature_part = col[:-len(suffix)]
                    if len(feature_part) > 0 and not feature_part.endswith('_'):
                        condition_counts[condition] = condition_counts.get(condition, 0) + 1
                        processed_cols.add(col)
                        break
        
        print("\n每种条件的特征数量:")
        if 'gold' in condition_counts:
            print(f"  - gold: {condition_counts['gold']} 个特征")
        
        for condition in sorted([c for c in condition_counts.keys() if c != 'gold']):
            print(f"  - {condition}: {condition_counts[condition]} 个特征")
        
        feature_counts = list(condition_counts.values())
        if feature_counts and not all(count == feature_counts[0] for count in feature_counts):
            print("\n⚠️ 警告：不同条件的特征数量不一致，请检查数据完整性")


class ConverterGUI:
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("椎间盘特征数据格式转换工具")
        self.root.geometry("800x600")
        
        self.converter = FeatureDataConverter()
        self.gold_file_path = None
        self.perturb_file_paths = []
        self.output_path = None
        
        self.setup_ui()
        
    def setup_ui(self):

        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="椎间盘特征数据格式转换工具", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        info_text = "将多个特征提取CSV文件合并转换为符合模块三输入格式要求的单个CSV文件"
        info_label = ttk.Label(main_frame, text=info_text, font=('Arial', 10))
        info_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        separator1 = ttk.Separator(main_frame, orient='horizontal')
        separator1.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(main_frame, text="1. 选择金标准CSV文件:", font=('Arial', 11, 'bold')).grid(
            row=3, column=0, sticky=tk.W, pady=5)
        
        self.gold_label = ttk.Label(main_frame, text="未选择文件", foreground='gray')
        self.gold_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=20)
        
        ttk.Button(main_frame, text="选择金标准文件", 
                  command=self.select_gold_file).grid(row=4, column=2, pady=5)
        
        ttk.Label(main_frame, text="2. 选择扰动CSV文件 (可多选):", 
                 font=('Arial', 11, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=5)
        
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.perturb_listbox = tk.Listbox(list_frame, height=8, 
                                          yscrollcommand=scrollbar.set)
        self.perturb_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.perturb_listbox.yview)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=2, sticky=tk.N, pady=20)
        
        ttk.Button(button_frame, text="添加扰动文件", 
                  command=self.add_perturb_files).pack(pady=2)
        ttk.Button(button_frame, text="清空列表", 
                  command=self.clear_perturb_files).pack(pady=2)
        
        ttk.Label(main_frame, text="3. 选择输出文件夹:", 
                 font=('Arial', 11, 'bold')).grid(row=7, column=0, sticky=tk.W, pady=5)
        
        self.output_label = ttk.Label(main_frame, text="未选择路径", foreground='gray')
        self.output_label.grid(row=8, column=0, columnspan=2, sticky=tk.W, padx=20)
        
        ttk.Button(main_frame, text="选择输出文件夹", 
                  command=self.select_output_path).grid(row=8, column=2, pady=5)
        
        separator2 = ttk.Separator(main_frame, orient='horizontal')
        separator2.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.convert_button = ttk.Button(main_frame, text="开始转换", 
                                        command=self.run_conversion,
                                        state='disabled')
        self.convert_button.grid(row=10, column=1, pady=20)
        
        self.status_label = ttk.Label(main_frame, text="请按步骤选择文件", 
                                     foreground='blue')
        self.status_label.grid(row=11, column=0, columnspan=3, pady=5)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
    def select_gold_file(self):

        file_path = filedialog.askopenfilename(
            title="选择金标准CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.gold_file_path = file_path
            self.gold_label.config(text=os.path.basename(file_path), 
                                  foreground='black')
            self.check_ready()
            self.status_label.config(text="已选择金标准文件")
            
    def add_perturb_files(self):

        file_paths = filedialog.askopenfilenames(
            title="选择扰动CSV文件（可多选）",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        for file_path in file_paths:
            if file_path not in self.perturb_file_paths:
                self.perturb_file_paths.append(file_path)
                self.perturb_listbox.insert(tk.END, os.path.basename(file_path))
        
        if file_paths:
            self.check_ready()
            self.status_label.config(text=f"已添加 {len(file_paths)} 个扰动文件")
            
    def clear_perturb_files(self):

        self.perturb_file_paths = []
        self.perturb_listbox.delete(0, tk.END)
        self.check_ready()
        self.status_label.config(text="已清空扰动文件列表")
        
    def select_output_path(self):

        folder_path = filedialog.askdirectory(
            title="选择输出文件夹"
        )
        
        if folder_path:
            self.output_path = os.path.join(folder_path, "features_for_robustness.csv")
            self.output_label.config(text=f"features_for_robustness.csv", 
                                   foreground='black')
            self.check_ready()
            self.status_label.config(text=f"输出路径: {folder_path}")
            
            if os.path.exists(self.output_path):
                response = messagebox.askyesno("文件已存在", 
                                              f"文件 features_for_robustness.csv 已存在。\n是否覆盖？")
                if not response:
                    self.output_path = None
                    self.output_label.config(text="未选择路径", foreground='gray')
                    self.check_ready()
                    self.status_label.config(text="已取消输出路径选择")
            
    def check_ready(self):

        if (self.gold_file_path and 
            len(self.perturb_file_paths) > 0 and 
            self.output_path):
            self.convert_button.config(state='normal')
        else:
            self.convert_button.config(state='disabled')
            
    def run_conversion(self):

        try:
            self.status_label.config(text="正在转换数据...", foreground='orange')
            self.convert_button.config(state='disabled')
            self.root.update()
            
            self.converter = FeatureDataConverter()
            
            self.converter.load_gold_standard(self.gold_file_path)
            
            for perturb_path in self.perturb_file_paths:
                self.converter.load_perturbation_file(perturb_path)
            
            self.converter.merge_data()
            
            self.converter.save_merged_data(self.output_path)
            
            num_perturbations = len(self.converter.perturbation_dfs)
            perturbation_list = ", ".join(sorted(self.converter.perturbation_dfs.keys())[:5])
            if num_perturbations > 5:
                perturbation_list += f"... (共{num_perturbations}种)"
            
            self.status_label.config(text=f"转换成功！识别到{num_perturbations}种扰动类型", 
                                   foreground='green')
            messagebox.showinfo("成功", 
                              f"数据转换成功！\n\n"
                              f"输出文件: {self.output_path}\n"
                              f"数据维度: {self.converter.merged_df.shape}\n"
                              f"识别到的扰动类型: {perturbation_list}")
            
        except Exception as e:
            self.status_label.config(text="转换失败", foreground='red')
            messagebox.showerror("错误", f"转换过程中发生错误:\n{str(e)}")
            import traceback
            print(traceback.format_exc())
            
        finally:
            self.convert_button.config(state='normal')
            
    def run(self):

        self.root.mainloop()


def main():

    gui = ConverterGUI()
    gui.run()


if __name__ == "__main__":
    main()