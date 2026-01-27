from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import ctypes
import threading
from pathlib import Path
from queue import Queue

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from utils.robustness_analysis import analyze_robustness, save_analysis_result


try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


class RobustnessGUI:
    def __init__(self, parent: tk.Widget):
        self.parent = parent

        self.unperturbed_csv = tk.StringVar()
        self.perturbed_csv = tk.StringVar()
        self.pfirrmann_csv = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.enable_pfirrmann = tk.BooleanVar(value=True)
        self.alpha = tk.DoubleVar(value=0.05)
        self.icc_cluster_count = tk.IntVar(value=3)
        self.dup_corr_threshold = tk.DoubleVar(value=0.99)

        self._queue: Queue = Queue()
        self._result = None

        self._setup_gui()

    def _setup_gui(self) -> None:
        main = ttk.Frame(self.parent, padding=12)
        main.pack(fill="both", expand=True)

        self._setup_file_frame(main)
        self._setup_settings_frame(main)
        self._setup_controls(main)
        self._setup_log(main)

        self._log("🎯 稳健性相关性分析模块已就绪。")

    def _setup_file_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="📁 文件选择", padding=10)
        frame.pack(fill="x", pady=6)

        csv_filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")]
        grade_filetypes = [
            ("CSV/Excel Files", "*.csv *.xlsx *.xls"),
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx *.xls"),
            ("All Files", "*.*"),
        ]

        self._file_row(
            frame,
            label="未扰动特征CSV:",
            var=self.unperturbed_csv,
            title="选择未扰动特征CSV文件",
            filetypes=csv_filetypes,
        )
        self._file_row(
            frame,
            label="扰动后特征CSV:",
            var=self.perturbed_csv,
            title="选择扰动后特征CSV文件",
            filetypes=csv_filetypes,
        )
        self._file_row(
            frame,
            label="Pfirrmann分级文件(可选):",
            var=self.pfirrmann_csv,
            title="选择 Pfirrmann 分级文件（CSV/Excel）",
            filetypes=grade_filetypes,
        )

        out_row = ttk.Frame(frame)
        out_row.pack(fill="x", pady=2)
        ttk.Label(out_row, text="输出目录:", width=18).pack(side="left")
        ttk.Entry(out_row, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(out_row, text="💾 选择", command=self._select_output_dir).pack(side="left")

    def _file_row(
        self,
        parent: ttk.Frame,
        *,
        label: str,
        var: tk.StringVar,
        title: str,
        filetypes: list[tuple[str, str]],
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="📂 选择", command=lambda: self._select_file(var, title, filetypes)).pack(side="left")

    def _setup_settings_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="🔧 分析设置", padding=10)
        frame.pack(fill="x", pady=6)

        left = ttk.Frame(frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        mid = ttk.Frame(frame)
        mid.pack(side="left", fill="both", expand=True, padx=10)
        right = ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        rel = ttk.LabelFrame(left, text="步骤1：Pfirrmann相关性预筛 (Spearman)", padding=10)
        rel.pack(fill="x", pady=4)
        ttk.Checkbutton(rel, text="启用预筛（需要分级文件）", variable=self.enable_pfirrmann).pack(anchor="w")

        arow = ttk.Frame(rel)
        arow.pack(fill="x", pady=2)
        ttk.Label(arow, text="显著性水平 α:", width=14).pack(side="left")
        ttk.Spinbox(arow, from_=0.001, to=0.2, increment=0.001, textvariable=self.alpha, width=10).pack(
            side="left", padx=6
        )

        icc = ttk.LabelFrame(mid, text="步骤2：扰动鲁棒性 (ICC(2,1) + 聚类)", padding=10)
        icc.pack(fill="x", pady=4)
        crow = ttk.Frame(icc)
        crow.pack(fill="x", pady=2)
        ttk.Label(crow, text="聚类簇数 C:", width=14).pack(side="left")
        ttk.Spinbox(crow, from_=2, to=10, increment=1, textvariable=self.icc_cluster_count, width=10).pack(
            side="left", padx=6
        )

        red = ttk.LabelFrame(right, text="步骤3：特征去冗余 (Spearman)", padding=10)
        red.pack(fill="x", pady=4)
        trow = ttk.Frame(red)
        trow.pack(fill="x", pady=2)
        ttk.Label(trow, text="冗余阈值 T_dup:", width=14).pack(side="left")
        ttk.Spinbox(trow, from_=0.80, to=0.999, increment=0.001, textvariable=self.dup_corr_threshold, width=10).pack(
            side="left", padx=6
        )

    def _setup_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="🚀 执行控制", padding=10)
        frame.pack(fill="x", pady=6)

        self._start_btn = ttk.Button(frame, text="开始分析", command=self.start_analysis)
        self._start_btn.pack(side="left")
        ttk.Button(frame, text="查看ICC分布", command=self.show_icc_distribution).pack(side="left", padx=8)
        ttk.Button(frame, text="查看ICC树状图", command=self.show_icc_dendrogram).pack(side="left")

    def _setup_log(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="📝 运行日志", padding=10)
        frame.pack(fill="both", expand=True, pady=6)
        self._log_text = scrolledtext.ScrolledText(frame, height=12, wrap=tk.WORD)
        self._log_text.pack(fill="both", expand=True)

    def _select_file(self, var: tk.StringVar, title: str, filetypes: list[tuple[str, str]]) -> None:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
        )
        if path:
            var.set(path)

    def _select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_dir.set(path)

    def _log(self, msg: str) -> None:
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)

    def _queue_log(self, msg: str) -> None:
        self._queue.put(("log", msg))

    def _poll_queue(self) -> None:
        try:
            while True:
                typ, payload = self._queue.get_nowait()
                if typ == "log":
                    self._log(payload)
                elif typ == "done":
                    self._start_btn.config(state="normal")
                    messagebox.showinfo("完成", payload)
                elif typ == "error":
                    self._start_btn.config(state="normal")
                    messagebox.showerror("错误", payload)
        except Exception:
            pass
        self.parent.after(120, self._poll_queue)

    def start_analysis(self) -> None:
        un_path = self.unperturbed_csv.get().strip()
        pert_path = self.perturbed_csv.get().strip()
        out_dir = self.output_dir.get().strip()

        if not un_path:
            messagebox.showerror("错误", "请先选择未扰动特征CSV。")
            return
        if not pert_path:
            messagebox.showerror("错误", "请先选择扰动后特征CSV。")
            return
        if not out_dir:
            messagebox.showerror("错误", "请选择输出目录。")
            return

        enable_p = bool(self.enable_pfirrmann.get())
        pf_path = self.pfirrmann_csv.get().strip() or None
        if enable_p and not pf_path:
            messagebox.showwarning("提示", "未提供 Pfirrmann 分级文件，将跳过步骤1相关性预筛。")
            enable_p = False

        self._start_btn.config(state="disabled")
        self._log_text.delete(1.0, tk.END)

        self._queue_log("加载并对齐数据...")
        self._queue_log(f"- 未扰动: {un_path}")
        self._queue_log(f"- 扰动后: {pert_path}")
        self._queue_log(f"- 输出目录: {out_dir}")
        if enable_p:
            self._queue_log(f"- Pfirrmann: {pf_path}")
        else:
            self._queue_log("- Pfirrmann: 跳过")

        worker = threading.Thread(
            target=self._run_analysis,
            kwargs={
                "un_path": un_path,
                "pert_path": pert_path,
                "pf_path": pf_path,
                "enable_p": enable_p,
                "out_dir": out_dir,
            },
            daemon=True,
        )
        worker.start()
        self.parent.after(120, self._poll_queue)

    def _run_analysis(self, *, un_path: str, pert_path: str, pf_path: str | None, enable_p: bool, out_dir: str) -> None:
        try:
            result = analyze_robustness(
                unperturbed_csv=un_path,
                perturbed_csv=pert_path,
                pfirrmann_csv=pf_path,
                enable_pfirrmann_filter=enable_p,
                alpha=float(self.alpha.get()),
                icc_cluster_count=int(self.icc_cluster_count.get()),
                dup_corr_threshold=float(self.dup_corr_threshold.get()),
            )
            save_analysis_result(result, output_dir=out_dir)
            self._result = result

            chosen_cluster = None
            chosen_mean = float("nan")
            if result.robust_features:
                try:
                    chosen_cluster = int(result.icc_cluster.loc[result.robust_features[0]])
                    chosen_mean = float(result.icc.loc[result.robust_features].mean())
                except Exception:
                    chosen_cluster = None

            try:
                icc_valid = result.icc.dropna()
                clusters = result.icc_cluster.dropna()
                if not clusters.empty:
                    summary = (
                        clusters.groupby(clusters)
                        .apply(
                            lambda s: pd.Series(
                                {
                                    "size": int(s.size),
                                    "mean_icc": float(icc_valid.loc[s.index].mean()),
                                }
                            )
                        )
                        .sort_values(by="mean_icc", ascending=False)
                    )
                    self._queue.put(("log", "ICC cluster summary (size, mean ICC):"))
                    for idx, row in summary.iterrows():
                        self._queue.put(("log", f"- cluster {int(idx)}: n={int(row['size'])}, mean={float(row['mean_icc']):.4f}"))
                    if chosen_cluster is not None:
                        self._queue.put(("log", f"Selected cluster: {chosen_cluster} (mean ICC={chosen_mean:.4f})"))
            except Exception:
                pass

            self._queue.put(
                (
                    "done",
                    "分析完成！\n"
                    f"- 最终特征数: {len(result.final_features)}\n"
                    f"- 鲁棒特征数(最佳ICC簇): {len(result.robust_features)}"
                    + (f" (cluster={chosen_cluster}, mean ICC={chosen_mean:.4f})\n" if chosen_cluster is not None else "\n")
                    + f"结果已保存到: {Path(out_dir).resolve()}",
                )
            )
        except Exception as e:
            self._queue.put(("error", str(e)))

    def show_icc_distribution(self) -> None:
        if self._result is None:
            messagebox.showwarning("提示", "请先完成一次分析。")
            return

        win = tk.Toplevel(self.parent)
        win.title("ICC(2,1) Distribution")
        win.geometry("800x500")

        fig = Figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        icc = self._result.icc.dropna().to_numpy()
        ax.hist(icc, bins=50, color="#4c78a8", alpha=0.85)
        ax.set_xlabel("ICC(2,1)")
        ax.set_ylabel("Feature Count")
        ax.set_title("ICC(2,1) Distribution Histogram")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_icc_dendrogram(self) -> None:
        if self._result is None:
            messagebox.showwarning("提示", "请先完成一次分析。")
            return

        icc_vals = self._result.icc.dropna().to_numpy().reshape(-1, 1)
        if icc_vals.shape[0] < 2:
            messagebox.showwarning("提示", "ICC 数据量不足，无法绘制树状图。")
            return

        model = AgglomerativeClustering(
            distance_threshold=0,
            n_clusters=None,
            linkage="ward",
            compute_distances=True,
        )
        model.fit(icc_vals)

        counts = np.zeros(model.children_.shape[0], dtype=np.int64)
        n_samples = icc_vals.shape[0]
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

        win = tk.Toplevel(self.parent)
        win.title("ICC(2,1) Dendrogram")
        win.geometry("900x600")

        fig = Figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        dendrogram(
            linkage_matrix,
            truncate_mode="lastp",
            p=30,
            show_leaf_counts=True,
            leaf_rotation=0,
            leaf_font_size=9,
            ax=ax,
        )
        ax.set_title("ICC(2,1) Hierarchical Clustering Dendrogram (Truncated)")
        ax.set_xlabel("Cluster (truncated)")
        ax.set_ylabel("Ward Distance")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


__all__ = ["RobustnessGUI"]
