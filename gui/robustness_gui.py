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
import threading
from queue import Queue

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
        'suggest_clusters': '自动建议k值',
        'suggested_clusters': '建议的最佳聚类数量为: ',
        'adjust_value_hint': '\n您可以根据需要调整此值。',
        'calculate_icc_first': '请先计算ICC矩阵',
        'cannot_calculate_suggestion': '无法计算建议值: ',
        'icc_filter_group': '稳健性预筛选',
        'enable_icc_filter': '启用预筛选',
        'min_icc_threshold': '最小ICC>=',
        'mean_icc_threshold': '平均ICC>=',
        'welcome_msg': """
🎯 椎间盘特征稳健性分析系统已就绪！
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
        self.cluster_selection = tk.StringVar(value="min_icc")
        self.n_clusters = tk.IntVar(value=4)
        

        self.correlation_threshold = tk.DoubleVar(value=0.99)
        self.variance_criterion = tk.StringVar(value="lower")

        self.remove_irrelevant = tk.BooleanVar(value=True)
        self.min_correlation = tk.DoubleVar(value=0.0)
        self.auto_cut_height = tk.BooleanVar(value=True)
        self.cut_height = tk.DoubleVar(value=0.0)

        self.enable_icc_filter = tk.BooleanVar(value=True)
        self.min_icc_threshold = tk.DoubleVar(value=0.25)
        self.mean_icc_threshold = tk.DoubleVar(value=0.5)
        
        self.feature_data = None
        self.icc_matrix = None
        self.selected_features = None
        self.final_features = None
        
        self.widgets = {}

        self.queue = Queue() 
        
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
                if widget_name == 'select_input_btn':
                    widget.config(text="📂 " + self.get_text('select'))
                elif widget_name == 'select_output_btn':
                    widget.config(text="💾 " + self.get_text('select'))
                else:
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

        if 'suggest_clusters_btn' in self.widgets:
            self.widgets['suggest_clusters_btn'].config(text=self.get_text('suggest_clusters'))
    
    def setup_gui(self):
        self.canvas = tk.Canvas(self.parent, bg='#f0f0f0', highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollable_frame = ttk.Frame(self.canvas)
        canvas_window = self.canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def configure_scroll_region(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            canvas_width = self.canvas.winfo_width()
            if canvas_width > 0:
                self.canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(canvas_window, width=e.width))

        main_frame = ttk.Frame(scrollable_frame, padding="15")
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

        icc_group = ttk.LabelFrame(middle_frame, text=self.get_text('icc_settings'), padding="10")
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
                                values=["min_icc", "mean_icc", "manual"],
                                width=15, state="readonly")
        select_combo.pack(side="left", padx=5)
        select_combo.set("min_icc")
        self.widgets['cluster_select_combo'] = select_combo

        n_clusters_frame = ttk.Frame(cluster_group)
        n_clusters_frame.pack(fill="x", pady=2)
        n_clusters_label = ttk.Label(n_clusters_frame, text=self.get_text('n_clusters'))
        n_clusters_label.pack(side="left")
        self.widgets['n_clusters_label'] = n_clusters_label

        n_clusters_spinbox = ttk.Spinbox(n_clusters_frame, from_=2, to=10,
                                        textvariable=self.n_clusters, width=10)
        n_clusters_spinbox.pack(side="left", padx=5)

        suggest_btn = ttk.Button(n_clusters_frame, text=self.get_text('suggest_clusters'), 
                                command=self.suggest_optimal_clusters)
        suggest_btn.pack(side="left", padx=5)
        self.widgets['suggest_clusters_btn'] = suggest_btn

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

        icc_filter_group = ttk.LabelFrame(left_frame, text=self.get_text('icc_filter_group'), padding="10")
        icc_filter_group.pack(fill="x", pady=5, side="top")
        self.widgets['icc_filter_group_frame'] = icc_filter_group

        filter_enable_cb = ttk.Checkbutton(icc_filter_group, text=self.get_text('enable_icc_filter'), variable=self.enable_icc_filter)
        filter_enable_cb.pack(anchor="w")
        self.widgets['enable_icc_filter_cb'] = filter_enable_cb
        
        min_icc_frame = ttk.Frame(icc_filter_group)
        min_icc_frame.pack(fill="x", pady=2)
        min_icc_label = ttk.Label(min_icc_frame, text=self.get_text('min_icc_threshold'))
        min_icc_label.pack(side="left")
        self.widgets['min_icc_threshold_label'] = min_icc_label
        min_icc_spinbox = ttk.Spinbox(min_icc_frame, from_=0.0, to=1.0, increment=0.05,
                                      textvariable=self.min_icc_threshold, width=10)
        min_icc_spinbox.pack(side="left", padx=5)

        mean_icc_frame = ttk.Frame(icc_filter_group)
        mean_icc_frame.pack(fill="x", pady=2)
        mean_icc_label = ttk.Label(mean_icc_frame, text=self.get_text('mean_icc_threshold'))
        mean_icc_label.pack(side="left")
        self.widgets['mean_icc_threshold_label'] = mean_icc_label
        mean_icc_spinbox = ttk.Spinbox(mean_icc_frame, from_=0.0, to=1.0, increment=0.05,
                                       textvariable=self.mean_icc_threshold, width=10)
        mean_icc_spinbox.pack(side="left", padx=5)
                
        select_combo.bind("<<ComboboxSelected>>", on_cluster_select_change)

        select_combo['values'] = ["min_icc", "mean_icc", "manual"]

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

        start_button_container = ttk.Frame(control_frame)
        start_button_container.pack(fill="x", expand=True)

        start_btn = ttk.Button(start_button_container, text="🚀 开始分析", 
                              command=self.start_analysis)
        start_btn.pack(anchor="center")
        self.widgets['start_analysis_btn'] = start_btn
        
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
            
        self.widgets['start_analysis_btn'].config(state="disabled")
        
        self.analysis_thread = threading.Thread(target=self._run_analysis_thread)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        self._poll_analysis_queue()

    def _run_analysis_thread(self):
        try:
            self.queue.put(('log', self.get_text('loading_data')))
            self.load_data()
            self.queue.put(('log', self.get_text('data_loaded')))
            self.queue.put(('log', f"加载了 {len(self.cases)} 个病例，{len(self.features)} 个特征，{len(self.conditions)} 个条件"))

            self.queue.put(('log', self.get_text('calculating_icc')))
            self.calculate_icc()
            self.queue.put(('log', self.get_text('icc_complete')))

            self._filter_by_icc_threshold()

            if len(self.features) < self.n_clusters.get():
                self.log_message(f"错误: 预筛选后特征数量({len(self.features)})少于聚类数量({self.n_clusters.get()})，无法继续。")
                self.log_message("建议降低ICC筛选阈值或减少聚类数量。")
                raise ValueError("预筛选后特征不足，无法进行聚类。")

            self.queue.put(('log', self.get_text('performing_clustering')))
            self.perform_clustering()
            self.queue.put(('log', self.get_text('clustering_complete')))

            self.queue.put(('log', self.get_text('analyzing_correlation')))
            self.analyze_correlation()
            self.queue.put(('log', self.get_text('correlation_complete')))

            self.queue.put(('log', self.get_text('saving_results')))
            self.save_results()
            
            self.queue.put(('log', self.get_text('analysis_complete')))
            self.queue.put(('success', self.get_text('analysis_complete')))
            
        except Exception as e:
            error_msg = f"❌ {self.get_text('error')}: {str(e)}"
            self.queue.put(('log', error_msg))
            self.queue.put(('error', str(e)))
        finally:
            self.queue.put(('finished', None))

    def _poll_analysis_queue(self):
        try:
            message_type, data = self.queue.get_nowait()
            
            if message_type == 'log':
                self.log_message(data)
            elif message_type == 'success':
                messagebox.showinfo(self.get_text('success'), data)
            elif message_type == 'error':
                messagebox.showerror(self.get_text('error'), data)
            elif message_type == 'finished':
                self.widgets['start_analysis_btn'].config(state="normal")
                return

        except:
            pass

        self.parent.after(100, self._poll_analysis_queue)
            
    def load_data(self):
        self.feature_data = pd.read_csv(self.input_file.get(), index_col=0)

        self.parse_features_and_conditions()
        
        
    def parse_features_and_conditions(self):
        columns = self.feature_data.columns.tolist()

        base_feature_names = set()
        gold_cols = [c for c in columns if c.endswith('_gold')]
        
        if not gold_cols:
            gold_cols = [c for c in columns if c.endswith('_original')]
        
        if not gold_cols:
            messagebox.showerror("输入文件错误", "输入文件中必须包含带有 '_gold' 或 '_original' 后缀的列作为金标准。")
            raise ValueError("未找到金标准列 ('_gold' 或 '_original')。")

        for col in gold_cols:
            if col.endswith('_gold'):
                base_feature_names.add(col[:-5])
            elif col.endswith('_original'):
                base_feature_names.add(col[:-9])

        sorted_base_features = sorted(list(base_feature_names), key=len, reverse=True)

        self.features = sorted_base_features
        self.conditions = []
        feature_condition_map = {}
        
        for col in columns:
            matched = False
            for feature in self.features:
                if col.startswith(feature + '_'):
                    condition = col[len(feature) + 1:]
                    if condition and condition not in self.conditions:
                        self.conditions.append(condition)
                    feature_condition_map[col] = (feature, condition)
                    matched = True
                    break
        
        self.feature_condition_map = feature_condition_map
        self.cases = self.feature_data.index.tolist()

        if 'gold' in self.conditions:
            self.conditions.remove('gold')
            self.conditions.insert(0, 'gold')

            
    def calculate_icc(self):
        self.feature_data = self.feature_data.replace([np.inf, -np.inf], np.nan)
        
        self.feature_data = self.feature_data.dropna(axis=1, how='all')
        
        numeric_columns = self.feature_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_data = self.feature_data[col]
            if col_data.std() > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                self.feature_data[col] = col_data.where(
                    (col_data <= mean_val + 10*std_val) & 
                    (col_data >= mean_val - 10*std_val), 
                    np.nan
                )

        num_features = len(self.features)
        columns_for_icc = [c for c in self.conditions if c not in ['gold', 'original']]
        self.icc_matrix = pd.DataFrame(index=self.features, columns=columns_for_icc)

        for feature in self.features:
            for condition in columns_for_icc:
                gold_col = f"{feature}_gold"
                if gold_col not in self.feature_data.columns:
                    gold_col = f"{feature}_original"
                
                condition_col = f"{feature}_{condition}"
                
                if gold_col in self.feature_data.columns and condition_col in self.feature_data.columns:
                    data_for_icc = []
                    for case in self.cases:
                        gold_value = self.feature_data.loc[case, gold_col]
                        condition_value = self.feature_data.loc[case, condition_col]
                        data_for_icc.append({'case': case, 'rater': 'gold', 'value': gold_value})
                        data_for_icc.append({'case': case, 'rater': condition, 'value': condition_value})

                    icc_df = pd.DataFrame(data_for_icc).dropna(subset=['value'])

                    n_cases = icc_df['case'].nunique()
                    n_raters = icc_df['rater'].nunique()
                    value_var = icc_df['value'].var()

                    log_prefix = f"跳过 {feature} vs {condition}:"
                    if n_cases < 2:
                        self.log_message(f"{log_prefix} 病例数不足 ({n_cases})")
                        self.icc_matrix.loc[feature, condition] = np.nan
                        continue
                    if n_raters < 2:
                        self.log_message(f"{log_prefix} 测量/条件数不足 ({n_raters})")
                        self.icc_matrix.loc[feature, condition] = np.nan
                        continue
                    if value_var < 1e-8:
                        self.log_message(f"{log_prefix} 特征值方差为零或过小")
                        self.icc_matrix.loc[feature, condition] = np.nan
                        continue

                    try:
                        icc_result = pg.intraclass_corr(
                            data=icc_df,
                            targets='case',
                            raters='rater',
                            ratings='value'
                        ).set_index('Type')

                        icc_key_to_extract = 'ICC3k' if self.icc_type.get() == "ICC(3,k)" else 'ICC2k'

                        if icc_key_to_extract in icc_result.index:
                            icc_value = icc_result.loc[icc_key_to_extract, 'ICC']
                            
                            if pd.notna(icc_value) and -1.0 <= icc_value <= 1.0:
                                self.icc_matrix.loc[feature, condition] = icc_value
                            else:
                                self.icc_matrix.loc[feature, condition] = np.nan
                        else:
                            self.icc_matrix.loc[feature, condition] = np.nan

                    except Exception as e:
                        self.log_message(f"计算 {feature} vs {condition} 的ICC时出错: {str(e)}")
                        self.icc_matrix.loc[feature, condition] = np.nan

        self.icc_matrix = self.icc_matrix.apply(pd.to_numeric, errors='coerce')

        self.icc_matrix.dropna(axis=1, how='all', inplace=True)

        self.icc_matrix['mean_icc'] = self.icc_matrix.mean(axis=1)
        self.icc_matrix['min_icc'] = self.icc_matrix.min(axis=1)

        self.icc_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if self.icc_matrix.empty or self.icc_matrix['mean_icc'].isnull().all():
            self.log_message("错误：所有特征的ICC值均未能成功计算，无法进行后续分析。")
            messagebox.showerror("计算错误", "所有特征的ICC值均未能成功计算，请检查输入数据和日志。")
            return

        nan_features_count = self.icc_matrix['mean_icc'].isnull().sum()
        if nan_features_count > 0:
            nan_features = self.icc_matrix[self.icc_matrix['mean_icc'].isnull()].index.tolist()
            self.log_message(f"警告: {nan_features_count} 个特征的ICC值未能成功计算: {', '.join(nan_features[:3])}...")
            self.icc_matrix.dropna(subset=['mean_icc'], inplace=True)
            self.features = self.icc_matrix.index.tolist()
        
        self.log_message(f"ICC计算完成，平均ICC范围: {self.icc_matrix['mean_icc'].min():.3f} - {self.icc_matrix['mean_icc'].max():.3f}")

    def _filter_by_icc_threshold(self):
        if not self.enable_icc_filter.get():
            return

        min_thresh = self.min_icc_threshold.get()
        mean_thresh = self.mean_icc_threshold.get()
        
        self.log_message(f"执行ICC预筛选：min_icc >= {min_thresh} AND mean_icc >= {mean_thresh}")
        
        initial_feature_count = len(self.icc_matrix)
        
        mask = (self.icc_matrix['min_icc'] >= min_thresh) & (self.icc_matrix['mean_icc'] >= mean_thresh)
        
        self.icc_matrix = self.icc_matrix.loc[mask]
        
        self.features = self.icc_matrix.index.tolist()
        
        final_feature_count = len(self.icc_matrix)
        removed_count = initial_feature_count - final_feature_count
        
        self.log_message(f"ICC预筛选完成：移除了 {removed_count} 个不稳健特征，保留 {final_feature_count} 个。")

    def perform_clustering(self):
        cluster_data = self.icc_matrix.drop(['mean_icc', 'min_icc', 'cluster'], axis=1, errors='ignore')
        
        cluster_data.dropna(how='all', inplace=True)
        
        cluster_data = cluster_data.apply(lambda row: row.fillna(row.mean()), axis=1)

        cluster_data.fillna(0, inplace=True)
        
        if cluster_data.shape[0] < 2:
            self.log_message("错误：有效特征数量不足（<2），无法进行聚类分析")
            return

        if self.distance_metric.get() == 'correlation':
            cluster_data.loc[(cluster_data==0).all(axis=1)] = np.random.rand(cluster_data.shape[1]) * 1e-6
            dist_matrix = pdist(cluster_data, metric='correlation')
        else:
            dist_matrix = pdist(cluster_data, metric='euclidean')

        self.linkage_matrix = hierarchy.linkage(dist_matrix, method=self.linkage_method.get())

        n_clusters = self.n_clusters.get()
        self.log_message(f"使用聚类数量: {n_clusters}")

        cluster_labels = hierarchy.fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')

        self.icc_matrix['cluster'] = pd.Series(cluster_labels, index=cluster_data.index)

        best_cluster = self._select_best_cluster()
        
        self.selected_features = self.icc_matrix[self.icc_matrix['cluster'] == best_cluster].index.tolist()
        
        self.log_message(f"聚类完成，选择了簇 {best_cluster}，包含 {len(self.selected_features)} 个特征")

    def suggest_optimal_clusters(self):
        if self.icc_matrix is None:
            messagebox.showwarning(self.get_text('warning'), self.get_text('calculate_icc_first'))
            return
        
        try:
            optimal_k = self._find_optimal_clusters()
            self.n_clusters.set(optimal_k)
            self.log_message(f"{self.get_text('suggested_clusters')}{optimal_k}")
            messagebox.showinfo(self.get_text('info'), 
                            f"{self.get_text('suggested_clusters')}{optimal_k}{self.get_text('adjust_value_hint')}")
        except Exception as e:
            self.log_message(f"{self.get_text('cannot_calculate_suggestion')}{str(e)}")
            messagebox.showerror(self.get_text('error'), f"{self.get_text('cannot_calculate_suggestion')}{str(e)}")
        
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
        if 'cluster' not in self.icc_matrix.columns or self.icc_matrix['cluster'].isnull().all():
            self.log_message("警告: 未能成功分配聚类标签，无法选择最佳簇。")
            return 1

        cluster_stats = self.icc_matrix.groupby('cluster').agg(
            avg_of_mean_icc=('mean_icc', 'mean'),
            min_of_min_icc=('min_icc', 'min')
        )

        selection_strategy = self.cluster_selection.get()
        best_cluster = None

        if selection_strategy == 'mean_icc':
            best_cluster = cluster_stats['avg_of_mean_icc'].idxmax()
            self.log_message(f"簇选择策略('mean_icc'): 选择平均ICC最高的簇 -> 簇 {best_cluster}")
        
        elif selection_strategy == 'min_icc':
            best_cluster = cluster_stats['min_of_min_icc'].idxmax()
            self.log_message(f"簇选择策略('min_icc'): 选择最小ICC值最高的簇 -> 簇 {best_cluster}")

        else:
            if not cluster_stats.empty:
                best_cluster = cluster_stats.index[0]
            else:
                best_cluster = 1
            self.log_message(f"簇选择策略('manual'): 默认选择第一个可用簇 -> 簇 {best_cluster}")

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

        final_features_df = pd.DataFrame({
            'feature': self.final_features,
            'mean_icc': [self.icc_matrix.loc[f, 'mean_icc'] for f in self.final_features],
            'min_icc': [self.icc_matrix.loc[f, 'min_icc'] for f in self.final_features]
        })
        final_file = os.path.join(output_dir, 'final_robust_features.csv')
        final_features_df.to_csv(final_file, index=False)
        self.log_message(f"{self.get_text('results_saved')} {final_file}")

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
