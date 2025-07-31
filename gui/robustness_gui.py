import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path
import json
import warnings
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

TEXT_DICT = {
    'cn': {
        'title': '椎间盘特征稳健性分析系统',
        'file_selection': '📁 文件选择',
        'input_file': '输入文件:',
        'select': '选择',
        'output_path': '输出路径:',
        'select_input_file': '选择特征表CSV文件',
        'select_output_path': '选择输出文件夹',
        'analysis_settings': '🔧 分析设置',
        'icc_settings': 'ICC计算设置',
        'icc_type': 'ICC类型:',
        'icc_confidence': 'ICC置信水平:',
        'clustering_settings': '聚类分析设置',
        'clustering_method': '聚类方法:',
        'linkage_method': '链接方法:',
        'distance_metric': '距离度量:',
        'cluster_selection': '簇选择方式:',
        'n_clusters': '聚类数量:',
        'correlation_settings': '相关性分析设置',
        'correlation_threshold': '相关性阈值:',
        'variance_criterion': '方差准则:',
        'remove_lower': '移除方差较低的特征',
        'remove_higher': '移除方差较高的特征',
        'execution_control': '执行控制',
        'start_analysis': '🚀 开始分析',
        'stop': '⏹ 停止',
        'run_log': '📝 运行日志',
        'visualization': '📊 可视化',
        'show_icc_heatmap': '显示ICC热图',
        'show_dendrogram': '显示聚类树状图',
        'show_correlation': '显示相关性矩阵',
        'error': '错误',
        'warning': '警告',
        'info': '提示',
        'success': '成功',
        'remove_lower': '移除方差较低的特征',
        'remove_higher': '移除方差较高的特征',
        'manual_only': '(仅手动选择时使用)',
        'welcome_msg': """
🎯 椎间盘特征稳健性分析系统已就绪！

📋 支持ICC计算、聚类分析和相关性筛选
💡 提示：请先选择包含所有条件特征的CSV文件！
    """,
        'file_not_selected': '请先选择输入文件',
        'output_not_selected': '请选择输出路径',
        'loading_data': '加载数据中...',
        'data_loaded': '数据加载完成',
        'calculating_icc': '计算ICC矩阵...',
        'icc_complete': 'ICC计算完成',
        'performing_clustering': '执行聚类分析...',
        'clustering_complete': '聚类分析完成',
        'analyzing_correlation': '分析特征相关性...',
        'correlation_complete': '相关性分析完成',
        'saving_results': '保存结果...',
        'analysis_complete': '分析完成！',
        'results_saved': '结果已保存到：',
        'feature_relevance_filter': '特征相关性过滤',
        'min_correlation_with_grade': '与分级最小相关性:',
        'remove_irrelevant': '移除无关特征(p>0.05)',
        'cluster_cut_height': '聚类切割高度:',
        'auto_cut_height': '自动确定切割高度'
    },
    'en': {
        'title': 'IVD Feature Robustness Analysis System',
        'file_selection': '📁 File Selection',
        'input_file': 'Input File:',
        'select': 'Select',
        'output_path': 'Output Path:',
        'select_input_file': 'Select Feature CSV File',
        'select_output_path': 'Select Output Folder',
        'analysis_settings': '🔧 Analysis Settings',
        'icc_settings': 'ICC Calculation Settings',
        'icc_type': 'ICC Type:',
        'icc_confidence': 'ICC Confidence Level:',
        'clustering_settings': 'Clustering Settings',
        'clustering_method': 'Clustering Method:',
        'linkage_method': 'Linkage Method:',
        'distance_metric': 'Distance Metric:',
        'cluster_selection': 'Cluster Selection:',
        'n_clusters': 'Number of Clusters:',
        'correlation_settings': 'Correlation Analysis Settings',
        'correlation_threshold': 'Correlation Threshold:',
        'variance_criterion': 'Variance Criterion:',
        'remove_lower': 'Remove Lower Variance Features',
        'remove_higher': 'Remove Higher Variance Features',
        'execution_control': 'Execution Control',
        'start_analysis': '🚀 Start Analysis',
        'stop': '⏹ Stop',
        'run_log': '📝 Run Log',
        'visualization': '📊 Visualization',
        'show_icc_heatmap': 'Show ICC Heatmap',
        'show_dendrogram': 'Show Dendrogram',
        'show_correlation': 'Show Correlation Matrix',
        'error': 'Error',
        'warning': 'Warning',
        'info': 'Info',
        'success': 'Success',
        'remove_lower': 'Remove Lower Variance Features',
        'remove_higher': 'Remove Higher Variance Features',
        'manual_only': '(For manual selection only)',
        'welcome_msg': """
🎯 IVD Feature Robustness Analysis System is ready!

📋 Supports ICC calculation, clustering analysis and correlation filtering
💡 Tip: Please select CSV file containing all condition features first!
    """,
        'file_not_selected': 'Please select input file',
        'output_not_selected': 'Please select output path',
        'loading_data': 'Loading data...',
        'data_loaded': 'Data loaded',
        'calculating_icc': 'Calculating ICC matrix...',
        'icc_complete': 'ICC calculation complete',
        'performing_clustering': 'Performing clustering analysis...',
        'clustering_complete': 'Clustering analysis complete',
        'analyzing_correlation': 'Analyzing feature correlation...',
        'correlation_complete': 'Correlation analysis complete',
        'saving_results': 'Saving results...',
        'analysis_complete': 'Analysis complete!',
        'results_saved': 'Results saved to:',
        'feature_relevance_filter': 'Feature Relevance Filter',
        'min_correlation_with_grade': 'Min Correlation with Grade:',
        'remove_irrelevant': 'Remove Irrelevant Features (p>0.05)',
        'cluster_cut_height': 'Cluster Cut Height:',
        'auto_cut_height': 'Auto Determine Cut Height'
    }
}

class RobustnessGUI:
    def __init__(self, parent):
        self.parent = parent
        
        self.current_lang = tk.StringVar(value="cn")
        self.input_file = tk.StringVar()
        self.output_path = tk.StringVar()
        
        self.icc_type = tk.StringVar(value="ICC(3,k)")
        self.icc_confidence = tk.DoubleVar(value=0.95)
        
        self.clustering_method = tk.StringVar(value="hierarchical")
        self.linkage_method = tk.StringVar(value="ward")
        self.distance_metric = tk.StringVar(value="euclidean")
        self.cluster_selection = tk.StringVar(value="mean_icc")
        self.n_clusters = tk.IntVar(value=4)
        

        self.correlation_threshold = tk.DoubleVar(value=0.99)
        self.variance_criterion = tk.StringVar(value="lower")

        self.remove_irrelevant = tk.BooleanVar(value=True)
        self.min_correlation = tk.DoubleVar(value=0.0)
        self.auto_cut_height = tk.BooleanVar(value=True)
        self.cut_height = tk.DoubleVar(value=0.0)
        
        self.feature_data = None
        self.icc_matrix = None
        self.selected_features = None
        self.final_features = None
        
        self.widgets = {}
        
        self.setup_gui()
        
        self.setup_logging()
        
    def setup_logging(self):

        self.logger = logging.getLogger('FeatureRobustness')
        self.logger.setLevel(logging.INFO)
        
    def get_text(self, key):

        lang = self.current_lang.get()
        return TEXT_DICT[lang].get(key, key)
    
    def update_language(self):

        for widget_name, widget in self.widgets.items():
            if widget_name.endswith('_frame'):
                widget.config(text=self.get_text(widget_name.replace('_frame', '')))
            elif widget_name.endswith('_label'):
                widget.config(text=self.get_text(widget_name.replace('_label', '')))
            elif widget_name.endswith('_btn'):
                widget.config(text=self.get_text(widget_name.replace('_btn', '')))
            elif widget_name.endswith('_cb'):
                widget.config(text=self.get_text(widget_name.replace('_cb', '')))
        
        if hasattr(self, 'log_text'):
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, self.get_text('welcome_msg').strip())
        
        if 'n_clusters_note' in self.widgets:
            self.widgets['n_clusters_note'].config(text=self.get_text('manual_only'))

        if 'remove_lower_radio' in self.widgets:
            self.widgets['remove_lower_radio'].config(text=self.get_text('remove_lower'))
        if 'remove_higher_radio' in self.widgets:
            self.widgets['remove_higher_radio'].config(text=self.get_text('remove_higher'))
    
    def setup_gui(self):

        main_frame = ttk.Frame(self.parent, padding="15")
        main_frame.pack(fill="both", expand=True)
        
        self._setup_file_selection(main_frame)

        self._setup_analysis_settings(main_frame)

        self._setup_controls(main_frame)

        self._setup_log_display(main_frame)
        
    def _setup_file_selection(self, parent):

        file_frame = ttk.LabelFrame(parent, text=self.get_text('file_selection'), padding="10")
        file_frame.pack(fill="x", pady=5)
        self.widgets['file_selection_frame'] = file_frame

        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill="x", pady=2)
        
        input_label = ttk.Label(input_frame, text=self.get_text('input_file'), width=10)
        input_label.pack(side="left")
        self.widgets['input_file_label'] = input_label
        
        input_entry = ttk.Entry(input_frame, textvariable=self.input_file)
        input_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        input_btn = ttk.Button(input_frame, text="📂 " + self.get_text('select'), 
                              command=self.select_input_file)
        input_btn.pack(side="left")
        self.widgets['select_input_btn'] = input_btn

        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill="x", pady=2)
        
        output_label = ttk.Label(output_frame, text=self.get_text('output_path'), width=10)
        output_label.pack(side="left")
        self.widgets['output_path_label'] = output_label
        
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path)
        output_entry.pack(side="left", fill="x", expand=True, padx=5)
        
        output_btn = ttk.Button(output_frame, text="💾 " + self.get_text('select'), 
                               command=self.select_output_path)
        output_btn.pack(side="left")
        self.widgets['select_output_btn'] = output_btn
        
    def _setup_analysis_settings(self, parent):

        settings_frame = ttk.LabelFrame(parent, text=self.get_text('analysis_settings'), padding="10")
        settings_frame.pack(fill="x", pady=5)
        self.widgets['analysis_settings_frame'] = settings_frame
        
        left_frame = ttk.Frame(settings_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        middle_frame = ttk.Frame(settings_frame)
        middle_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        right_frame = ttk.Frame(settings_frame)
        right_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))

        icc_group = ttk.LabelFrame(left_frame, text=self.get_text('icc_settings'), padding="10")
        icc_group.pack(fill="x", pady=5)
        self.widgets['icc_settings_frame'] = icc_group

        icc_type_frame = ttk.Frame(icc_group)
        icc_type_frame.pack(fill="x", pady=2)
        icc_type_label = ttk.Label(icc_type_frame, text=self.get_text('icc_type'))
        icc_type_label.pack(side="left")
        self.widgets['icc_type_label'] = icc_type_label
        
        icc_type_combo = ttk.Combobox(icc_type_frame, textvariable=self.icc_type,
                                     values=["ICC(2,k)", "ICC(3,k)"],
                                     width=15, state="readonly")
        icc_type_combo.pack(side="left", padx=5)
        self.widgets['icc_type_combo'] = icc_type_combo

        icc_conf_frame = ttk.Frame(icc_group)
        icc_conf_frame.pack(fill="x", pady=2)
        icc_conf_label = ttk.Label(icc_conf_frame, text=self.get_text('icc_confidence'))
        icc_conf_label.pack(side="left")
        self.widgets['icc_confidence_label'] = icc_conf_label
        
        icc_conf_spinbox = ttk.Spinbox(icc_conf_frame, from_=0.90, to=0.99, 
                                      increment=0.01, textvariable=self.icc_confidence,
                                      width=10)
        icc_conf_spinbox.pack(side="left", padx=5)

        cluster_group = ttk.LabelFrame(middle_frame, text=self.get_text('clustering_settings'), padding="10")
        cluster_group.pack(fill="x", pady=5)
        self.widgets['clustering_settings_frame'] = cluster_group

        filter_frame = ttk.Frame(cluster_group)
        filter_frame.pack(fill="x", pady=(10, 2))
        filter_cb = ttk.Checkbutton(filter_frame, text=self.get_text('remove_irrelevant'),
                                    variable=self.remove_irrelevant)
        filter_cb.pack(anchor="w")
        self.widgets['remove_irrelevant_cb'] = filter_cb

        link_frame = ttk.Frame(cluster_group)
        link_frame.pack(fill="x", pady=2)
        link_label = ttk.Label(link_frame, text=self.get_text('linkage_method'))
        link_label.pack(side="left")
        self.widgets['linkage_method_label'] = link_label
        
        link_combo = ttk.Combobox(link_frame, textvariable=self.linkage_method,
                                 values=["ward", "complete", "average", "single"],
                                 width=15, state="readonly")
        link_combo.pack(side="left", padx=5)
        self.widgets['linkage_combo'] = link_combo

        dist_frame = ttk.Frame(cluster_group)
        dist_frame.pack(fill="x", pady=2)
        dist_label = ttk.Label(dist_frame, text=self.get_text('distance_metric'))
        dist_label.pack(side="left")
        self.widgets['distance_metric_label'] = dist_label
        
        dist_combo = ttk.Combobox(dist_frame, textvariable=self.distance_metric,
                                 values=["correlation", "euclidean"],
                                 width=15, state="readonly")
        dist_combo.pack(side="left", padx=5)
        self.widgets['distance_combo'] = dist_combo

        select_frame = ttk.Frame(cluster_group)
        select_frame.pack(fill="x", pady=2)
        select_label = ttk.Label(select_frame, text=self.get_text('cluster_selection'))
        select_label.pack(side="left")
        self.widgets['cluster_selection_label'] = select_label

        select_combo = ttk.Combobox(select_frame, textvariable=self.cluster_selection,
                                values=["mean_icc", "min_icc", "manual"],
                                width=15, state="readonly")
        select_combo.pack(side="left", padx=5)
        self.widgets['cluster_select_combo'] = select_combo

        n_clusters_frame = ttk.Frame(cluster_group)
        n_clusters_frame.pack(fill="x", pady=2)
        n_clusters_label = ttk.Label(n_clusters_frame, text=self.get_text('n_clusters'))
        n_clusters_label.pack(side="left")
        self.widgets['n_clusters_label'] = n_clusters_label

        n_clusters_spinbox = ttk.Spinbox(n_clusters_frame, from_=2, to=10,
                                        textvariable=self.n_clusters, width=10)
        n_clusters_spinbox.pack(side="left", padx=5)
        n_clusters_note = ttk.Label(n_clusters_frame, text=self.get_text('manual_only'), 
                                font=('Segoe UI', 8))
        n_clusters_note.pack(side="left", padx=5)
        self.widgets['n_clusters_note'] = n_clusters_note 

        def update_cluster_display(*args):
            if self.cluster_selection.get() == "mean_icc":
                select_combo.set("mean_icc")
            elif self.cluster_selection.get() == "min_icc":
                select_combo.set("min_icc")
            elif self.cluster_selection.get() == "manual":
                select_combo.set("manual")

        self.cluster_selection.trace('w', update_cluster_display)
        update_cluster_display()  

        def on_cluster_select_change(event):
            current = select_combo.get()
            if "平均ICC" in current:
                self.cluster_selection.set("mean_icc")
            elif "最小ICC" in current:
                self.cluster_selection.set("min_icc")
            elif "手动" in current:
                self.cluster_selection.set("manual")
                
        select_combo.bind("<<ComboboxSelected>>", on_cluster_select_change)

        select_combo['values'] = ["选择平均ICC最高的簇", "选择最小ICC最高的簇", "手动指定簇数量"]

        corr_group = ttk.LabelFrame(right_frame, text=self.get_text('correlation_settings'), padding="10")
        corr_group.pack(fill="x", pady=5)
        self.widgets['correlation_settings_frame'] = corr_group

        thresh_frame = ttk.Frame(corr_group)
        thresh_frame.pack(fill="x", pady=2)
        thresh_label = ttk.Label(thresh_frame, text=self.get_text('correlation_threshold'))
        thresh_label.pack(side="left")
        self.widgets['correlation_threshold_label'] = thresh_label
        
        thresh_spinbox = ttk.Spinbox(thresh_frame, from_=0.80, to=1.00,
                                    increment=0.01, textvariable=self.correlation_threshold,
                                    width=10)
        thresh_spinbox.pack(side="left", padx=5)

        var_label = ttk.Label(corr_group, text=self.get_text('variance_criterion'))
        var_label.pack(anchor="w", pady=(5, 2))
        self.widgets['variance_criterion_label'] = var_label
        
        var_lower = ttk.Radiobutton(corr_group, text=self.get_text('remove_lower'),
                                   variable=self.variance_criterion, value="lower")
        var_lower.pack(anchor="w", pady=1)
        self.widgets['remove_lower_radio'] = var_lower
        
        var_higher = ttk.Radiobutton(corr_group, text=self.get_text('remove_higher'),
                                    variable=self.variance_criterion, value="higher")
        var_higher.pack(anchor="w", pady=1)
        self.widgets['remove_higher_radio'] = var_higher
        
    def _setup_controls(self, parent):

        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=10)

        exec_frame = ttk.LabelFrame(control_frame, text=self.get_text('execution_control'), padding="5")
        exec_frame.pack(side="left", padx=5)
        self.widgets['execution_control_frame'] = exec_frame
        
        start_btn = ttk.Button(exec_frame, text=self.get_text('start_analysis'), 
                              command=self.start_analysis)
        start_btn.pack(side="left", padx=2)
        self.widgets['start_analysis_btn'] = start_btn
        
        stop_btn = ttk.Button(exec_frame, text=self.get_text('stop'), 
                             command=self.stop_analysis)
        stop_btn.pack(side="left", padx=2)
        self.widgets['stop_btn'] = stop_btn

        viz_frame = ttk.LabelFrame(control_frame, text=self.get_text('visualization'), padding="5")
        viz_frame.pack(side="left", padx=5)
        self.widgets['visualization_frame'] = viz_frame
        
        icc_btn = ttk.Button(viz_frame, text=self.get_text('show_icc_heatmap'), 
                            command=self.show_icc_heatmap)
        icc_btn.pack(side="left", padx=2)
        self.widgets['show_icc_heatmap_btn'] = icc_btn
        
        dendro_btn = ttk.Button(viz_frame, text=self.get_text('show_dendrogram'), 
                               command=self.show_dendrogram)
        dendro_btn.pack(side="left", padx=2)
        self.widgets['show_dendrogram_btn'] = dendro_btn
        
        corr_btn = ttk.Button(viz_frame, text=self.get_text('show_correlation'), 
                             command=self.show_correlation_matrix)
        corr_btn.pack(side="left", padx=2)
        self.widgets['show_correlation_btn'] = corr_btn
        
    def _setup_log_display(self, parent):

        log_frame = ttk.LabelFrame(parent, text=self.get_text('run_log'), padding="5")
        log_frame.pack(fill="both", expand=True, pady=5)
        self.widgets['run_log_frame'] = log_frame
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=('Consolas', 9), wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)

        self.log_text.insert(tk.END, self.get_text('welcome_msg').strip())
                
    def log_message(self, message):

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"\n[{timestamp}] {message}")
        self.log_text.see(tk.END)
        self.parent.update()
        
    def select_input_file(self):

        filename = filedialog.askopenfilename(
            title=self.get_text('select_input_file'),
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            
    def select_output_path(self):

        path = filedialog.askdirectory(title=self.get_text('select_output_path'))
        if path:
            self.output_path.set(path)
            
    def start_analysis(self):

        if not self.input_file.get():
            messagebox.showerror(self.get_text('error'), self.get_text('file_not_selected'))
            return
            
        if not self.output_path.get():
            messagebox.showerror(self.get_text('error'), self.get_text('output_not_selected'))
            return
            
        try:
            self.log_message(self.get_text('loading_data'))
            self.load_data()
            self.log_message(self.get_text('data_loaded'))

            self.log_message(self.get_text('calculating_icc'))
            self.calculate_icc()
            self.log_message(self.get_text('icc_complete'))

            self.log_message(self.get_text('performing_clustering'))
            self.perform_clustering()
            self.log_message(self.get_text('clustering_complete'))

            self.log_message(self.get_text('analyzing_correlation'))
            self.analyze_correlation()
            self.log_message(self.get_text('correlation_complete'))

            self.log_message(self.get_text('saving_results'))
            self.save_results()
            
            self.log_message(self.get_text('analysis_complete'))
            messagebox.showinfo(self.get_text('success'), self.get_text('analysis_complete'))
            
        except Exception as e:
            self.log_message(f"❌ {self.get_text('error')}: {str(e)}")
            messagebox.showerror(self.get_text('error'), str(e))
            
    def load_data(self):

        self.feature_data = pd.read_csv(self.input_file.get(), index_col=0)

        self.parse_features_and_conditions()
        
        self.log_message(f"加载了 {len(self.cases)} 个病例，{len(self.features)} 个特征，{len(self.conditions)} 个条件")
        
    def parse_features_and_conditions(self):

        columns = self.feature_data.columns.tolist()

        self.features = []
        self.conditions = []
        feature_condition_map = {}
        
        for col in columns:
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                feature_name, condition = parts
                if feature_name not in self.features:
                    self.features.append(feature_name)
                if condition not in self.conditions:
                    self.conditions.append(condition)
                feature_condition_map[col] = (feature_name, condition)
        
        self.feature_condition_map = feature_condition_map
        self.cases = self.feature_data.index.tolist()

        if 'gold' in self.conditions:
            self.conditions.remove('gold')
            self.conditions.insert(0, 'gold')
            
    def calculate_icc(self):

        num_features = len(self.features)
        num_conditions = len(self.conditions) - 1
        self.icc_matrix = pd.DataFrame(
            index=self.features,
            columns=[c for c in self.conditions if c != 'gold']
        )

        for feature in self.features:
            for condition in self.conditions:
                if condition == 'gold':
                    continue

                gold_col = f"{feature}_gold"
                condition_col = f"{feature}_{condition}"
                
                if gold_col in self.feature_data.columns and condition_col in self.feature_data.columns:
                    data_for_icc = []
                    for case in self.cases:
                        gold_value = self.feature_data.loc[case, gold_col]
                        condition_value = self.feature_data.loc[case, condition_col]

                        data_for_icc.append({
                            'case': case,
                            'rater': 'gold',
                            'value': gold_value
                        })
                        data_for_icc.append({
                            'case': case,
                            'rater': condition,
                            'value': condition_value
                        })

                    icc_df = pd.DataFrame(data_for_icc)

                    try:
                        icc_result = pg.intraclass_corr(
                            data=icc_df,
                            targets='case',
                            raters='rater',
                            ratings='value'
                        )

                        if self.icc_type.get() == "ICC(2,k)":
                            icc_value = icc_result[icc_result['Type'] == 'ICC2k']['ICC'].values[0]
                        else: 
                            icc_value = icc_result[icc_result['Type'] == 'ICC3k']['ICC'].values[0]
                        
                        self.icc_matrix.loc[feature, condition] = icc_value
                        
                    except Exception as e:
                        self.log_message(f"计算 {feature} vs {condition} 的ICC时出错: {str(e)}")
                        self.icc_matrix.loc[feature, condition] = np.nan

        self.icc_matrix = self.icc_matrix.astype(float)

        self.icc_matrix['mean_icc'] = self.icc_matrix.mean(axis=1)
        self.icc_matrix['min_icc'] = self.icc_matrix.min(axis=1)
        
        self.log_message(f"ICC计算完成，平均ICC范围: {self.icc_matrix['mean_icc'].min():.3f} - {self.icc_matrix['mean_icc'].max():.3f}")
        
    def perform_clustering(self):

        cluster_data = self.icc_matrix.drop(['mean_icc', 'min_icc'], axis=1)
        
        if self.distance_metric.get() == 'correlation':
            dist_matrix = pdist(cluster_data, metric='correlation')
        else:
            dist_matrix = pdist(cluster_data, metric='euclidean')

        self.linkage_matrix = hierarchy.linkage(dist_matrix, method=self.linkage_method.get())

        if self.cluster_selection.get() == 'manual':
            n_clusters = self.n_clusters.get()
            self.log_message(f"使用手动指定的簇数量: {n_clusters}")
        else:
            n_clusters = self._find_optimal_clusters()

        cluster_labels = hierarchy.fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')

        self.icc_matrix['cluster'] = cluster_labels

        best_cluster = self._select_best_cluster()

        self.selected_features = self.icc_matrix[self.icc_matrix['cluster'] == best_cluster].index.tolist()
        
        self.log_message(f"聚类完成，选择了簇 {best_cluster}，包含 {len(self.selected_features)} 个特征")
        
    def _find_optimal_clusters(self):

        max_clusters = min(10, len(self.features) // 2)
        inertias = []
        
        for k in range(2, max_clusters + 1):
            cluster_labels = hierarchy.fcluster(self.linkage_matrix, k, criterion='maxclust')

            cluster_icc_means = []
            for i in range(1, k + 1):
                cluster_features = self.icc_matrix[cluster_labels == i]['mean_icc']
                if len(cluster_features) > 0:
                    cluster_icc_means.append(cluster_features.mean())
            
            inertias.append(np.var(cluster_icc_means))

        if len(inertias) > 1:
            diffs = np.diff(inertias)
            optimal_k = np.argmin(diffs) + 3 
        else:
            optimal_k = 3
        
        return optimal_k
        
    def _select_best_cluster(self):

        cluster_stats = self.icc_matrix.groupby('cluster').agg({
            'mean_icc': ['mean', 'min', 'count']
        })
        
        if self.cluster_selection.get() == 'mean_icc':
            best_cluster = cluster_stats['mean_icc']['mean'].idxmax()
        elif self.cluster_selection.get() == 'min_icc':
            best_cluster = cluster_stats['mean_icc']['min'].idxmax()
        else:
            best_cluster = 1
            
        return best_cluster
        
    def analyze_correlation(self):

        selected_data = pd.DataFrame()
        
        for feature in self.selected_features:
            gold_col = f"{feature}_gold"
            if gold_col in self.feature_data.columns:
                selected_data[feature] = self.feature_data[gold_col]

        self.correlation_matrix = selected_data.corr(method='spearman')

        threshold = self.correlation_threshold.get()
        high_corr_pairs = []
        
        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                if abs(self.correlation_matrix.iloc[i, j]) > threshold:
                    feature1 = self.correlation_matrix.index[i]
                    feature2 = self.correlation_matrix.columns[j]
                    corr_value = self.correlation_matrix.iloc[i, j]

                    var1 = selected_data[feature1].var()
                    var2 = selected_data[feature2].var()
                    
                    high_corr_pairs.append({
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': corr_value,
                        'var1': var1,
                        'var2': var2
                    })

        features_to_remove = set()
        
        for pair in high_corr_pairs:
            if self.variance_criterion.get() == 'lower':
                if pair['var1'] < pair['var2']:
                    features_to_remove.add(pair['feature1'])
                else:
                    features_to_remove.add(pair['feature2'])
            else:
                if pair['var1'] > pair['var2']:
                    features_to_remove.add(pair['feature1'])
                else:
                    features_to_remove.add(pair['feature2'])

        self.final_features = [f for f in self.selected_features if f not in features_to_remove]
        
        self.log_message(f"相关性分析完成，移除了 {len(features_to_remove)} 个高度相关的特征")
        self.log_message(f"最终保留 {len(self.final_features)} 个特征")
        
    def save_results(self):

        output_dir = self.output_path.get()

        icc_file = os.path.join(output_dir, 'robustness_summary_matrix.csv')
        self.icc_matrix.to_csv(icc_file)
        self.log_message(f"{self.get_text('results_saved')} {icc_file}")

        final_features_df = pd.DataFrame({
            'feature': self.final_features,
            'mean_icc': [self.icc_matrix.loc[f, 'mean_icc'] for f in self.final_features],
            'min_icc': [self.icc_matrix.loc[f, 'min_icc'] for f in self.final_features]
        })
        final_file = os.path.join(output_dir, 'final_robust_features.csv')
        final_features_df.to_csv(final_file, index=False)
        self.log_message(f"{self.get_text('results_saved')} {final_file}")

        params = {
            'input_file': self.input_file.get(),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'icc_type': self.icc_type.get(),
            'icc_confidence': self.icc_confidence.get(),
            'linkage_method': self.linkage_method.get(),
            'distance_metric': self.distance_metric.get(),
            'cluster_selection': self.cluster_selection.get(),
            'correlation_threshold': self.correlation_threshold.get(),
            'variance_criterion': self.variance_criterion.get(),
            'num_initial_features': len(self.features),
            'num_selected_features': len(self.selected_features),
            'num_final_features': len(self.final_features)
        }
        
        params_file = os.path.join(output_dir, 'analysis_parameters.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        self.log_message(f"{self.get_text('results_saved')} {params_file}")

        self.save_detailed_report(output_dir)
        
    def save_detailed_report(self, output_dir):

        report_file = os.path.join(output_dir, 'analysis_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("椎间盘特征稳健性分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {self.input_file.get()}\n\n")
            
            f.write("分析参数:\n")
            f.write(f"  ICC类型: {self.icc_type.get()}\n")
            f.write(f"  聚类方法: {self.linkage_method.get()}\n")
            f.write(f"  距离度量: {self.distance_metric.get()}\n")
            f.write(f"  相关性阈值: {self.correlation_threshold.get()}\n\n")
            
            f.write("分析结果:\n")
            f.write(f"  初始特征数: {len(self.features)}\n")
            f.write(f"  聚类后特征数: {len(self.selected_features)}\n")
            f.write(f"  最终特征数: {len(self.final_features)}\n\n")
            
            f.write("最终稳健特征列表:\n")
            for i, feature in enumerate(self.final_features, 1):
                mean_icc = self.icc_matrix.loc[feature, 'mean_icc']
                min_icc = self.icc_matrix.loc[feature, 'min_icc']
                f.write(f"  {i}. {feature} (平均ICC={mean_icc:.3f}, 最小ICC={min_icc:.3f})\n")
                
        self.log_message(f"{self.get_text('results_saved')} {report_file}")
        
    def stop_analysis(self):

        self.log_message("⏹ 分析已停止")
        messagebox.showinfo(self.get_text('info'), "当前版本不支持中途停止")
        
    def show_icc_heatmap(self):

        if self.icc_matrix is None:
            messagebox.showwarning(self.get_text('warning'), "请先运行分析")
            return

        heatmap_window = tk.Toplevel(self.parent)
        heatmap_window.title("ICC热图")
        heatmap_window.geometry("800x600")

        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        plot_data = self.icc_matrix.drop(['mean_icc', 'min_icc', 'cluster'], axis=1, errors='ignore')

        sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'ICC值'})
        
        ax.set_title('特征稳健性ICC热图', fontsize=16)
        ax.set_xlabel('扰动条件', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)

        canvas = FigureCanvasTkAgg(fig, master=heatmap_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_dendrogram(self):

        if self.linkage_matrix is None:
            messagebox.showwarning(self.get_text('warning'), "请先运行分析")
            return

        dendro_window = tk.Toplevel(self.parent)
        dendro_window.title("聚类树状图")
        dendro_window.geometry("800x600")

        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        dendro = hierarchy.dendrogram(self.linkage_matrix, labels=self.features,
                                     ax=ax, orientation='right')
        
        ax.set_title('特征聚类树状图', fontsize=16)
        ax.set_xlabel('距离', fontsize=12)

        canvas = FigureCanvasTkAgg(fig, master=dendro_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_correlation_matrix(self):

        if self.correlation_matrix is None:
            messagebox.showwarning(self.get_text('warning'), "请先运行分析")
            return

        corr_window = tk.Toplevel(self.parent)
        corr_window.title("特征相关性矩阵")
        corr_window.geometry("800x600")

        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        mask = np.triu(np.ones_like(self.correlation_matrix), k=1)
        sns.heatmap(self.correlation_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax,
                   cbar_kws={'label': '斯皮尔曼相关系数'})
        
        ax.set_title('特征相关性矩阵', fontsize=16)

        threshold = self.correlation_threshold.get()
        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                if abs(self.correlation_matrix.iloc[i, j]) > threshold:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                             edgecolor='red', lw=3))

        canvas = FigureCanvasTkAgg(fig, master=corr_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

__all__ = ['RobustnessGUI']