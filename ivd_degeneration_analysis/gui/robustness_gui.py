from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import ctypes
import threading
from pathlib import Path
from queue import Queue

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
        self.statistics_csv = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.ft_classic = tk.BooleanVar(value=True)
        self.ft_pyradiomics = tk.BooleanVar(value=True)
        self.ft_deep = tk.BooleanVar(value=True)
        self.ft_tensor = tk.BooleanVar(value=True)

        self.enable_pca = tk.BooleanVar(value=True)
        self.pca_eta = tk.DoubleVar(value=0.95)
        self.pca_m_cap = tk.IntVar(value=50)

        self.icc_threshold = tk.DoubleVar(value=0.60)
        self.alpha_fdr = tk.DoubleVar(value=0.05)
        self.rho_min = tk.DoubleVar(value=0.0)
        self.dup_corr_threshold = tk.DoubleVar(value=0.95)

        self.enable_step5 = tk.BooleanVar(value=True)

        self.enet_l1_ratio = tk.DoubleVar(value=0.80)
        self.enable_auto_lambda_cv = tk.BooleanVar(value=True)
        self.lambda_value = tk.DoubleVar(value=0.01)
        self.lambda_cv_folds = tk.IntVar(value=5)
        self.lambda_cv_n_alphas = tk.IntVar(value=100)
        self.lambda_cv_epsilon = tk.DoubleVar(value=0.01)
        self.lambda_cv_use_1se = tk.BooleanVar(value=False)
        self.enable_lambda_size_tuning = tk.BooleanVar(value=False)

        self.bootstrap_B = tk.IntVar(value=100)
        self.stability_delta = tk.DoubleVar(value=1e-4)
        self.stability_tau = tk.DoubleVar(value=0.30)
        self.k_max = tk.IntVar(value=0)

        self.force_vars: dict[str, tk.BooleanVar] = {
            "classic_dhi_dhi": tk.BooleanVar(value=True),
            "classic_dhi_disc_height": tk.BooleanVar(value=True),
            "classic_t2si_si_ratio": tk.BooleanVar(value=True),
            "classic_t2si_roi_si": tk.BooleanVar(value=True),
            "classic_t2si_csf_si": tk.BooleanVar(value=True),
            "classic_asi_peak_diff": tk.BooleanVar(value=True),
            "classic_asi_asi": tk.BooleanVar(value=True),
            "classic_fd_fd": tk.BooleanVar(value=True),
        }

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
            label="Pfirrmann分级文件:",
            var=self.pfirrmann_csv,
            title="选择 Pfirrmann 分级文件（CSV/Excel）",
            filetypes=grade_filetypes,
        )
        self._file_row(
            frame,
            label="病例统计信息(可选):",
            var=self.statistics_csv,
            title="选择病例统计信息 statistics.csv（可选，用于队列/批次分布图）",
            filetypes=csv_filetypes,
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

        left_row = ttk.Frame(left)
        left_row.pack(fill="both", expand=True)

        types_frame = ttk.LabelFrame(left_row, text="特征处理对象", padding=10)
        types_frame.pack(side="left", fill="y", pady=4, padx=(0, 10))

        ttk.Label(types_frame, text="仅处理所选类型：").pack(anchor="w", pady=(0, 4))
        ttk.Checkbutton(types_frame, text="classic", variable=self.ft_classic).pack(anchor="w")
        ttk.Checkbutton(types_frame, text="pyradiomics", variable=self.ft_pyradiomics).pack(anchor="w")
        ttk.Checkbutton(types_frame, text="deep", variable=self.ft_deep, command=self._toggle_pca_state).pack(anchor="w")
        ttk.Checkbutton(types_frame, text="tensor", variable=self.ft_tensor, command=self._toggle_pca_state).pack(anchor="w")

        pca = ttk.LabelFrame(left_row, text="步骤1：PCA 预降维（深度/张量）", padding=10)
        pca.pack(side="left", fill="both", expand=True, pady=4)
        self._enable_pca_cb = ttk.Checkbutton(
            pca,
            text="启用步骤1（PCA）",
            variable=self.enable_pca,
            command=self._toggle_pca_state,
        )
        self._enable_pca_cb.pack(anchor="w", pady=(0, 6))
        prow = ttk.Frame(pca)
        prow.pack(fill="x", pady=2)
        ttk.Label(prow, text="eta_pca:", width=14).pack(side="left")
        self._pca_eta_spin = ttk.Spinbox(prow, from_=0.50, to=0.999, increment=0.01, textvariable=self.pca_eta, width=10)
        self._pca_eta_spin.pack(side="left", padx=6)
        mrow = ttk.Frame(pca)
        mrow.pack(fill="x", pady=2)
        ttk.Label(mrow, text="m_cap:", width=14).pack(side="left")
        self._pca_m_cap_spin = ttk.Spinbox(mrow, from_=1, to=500, increment=1, textvariable=self.pca_m_cap, width=10)
        self._pca_m_cap_spin.pack(side="left", padx=6)

        core = ttk.LabelFrame(mid, text="步骤2-4：ICC + Spearman(FDR) + 去冗余", padding=10)
        core.pack(fill="x", pady=4)
        crow = ttk.Frame(core)
        crow.pack(fill="x", pady=2)
        ttk.Label(crow, text="T_ICC:", width=14).pack(side="left")
        ttk.Spinbox(crow, from_=0.0, to=1.0, increment=0.01, textvariable=self.icc_threshold, width=10).pack(
            side="left", padx=6
        )

        arow = ttk.Frame(core)
        arow.pack(fill="x", pady=2)
        ttk.Label(arow, text="alpha_FDR:", width=14).pack(side="left")
        ttk.Spinbox(arow, from_=0.001, to=0.5, increment=0.001, textvariable=self.alpha_fdr, width=10).pack(
            side="left", padx=6
        )

        rrow = ttk.Frame(core)
        rrow.pack(fill="x", pady=2)
        ttk.Label(rrow, text="rho_min:", width=14).pack(side="left")
        ttk.Spinbox(rrow, from_=0.00, to=1.0, increment=0.01, textvariable=self.rho_min, width=10).pack(
            side="left", padx=6
        )

        trow = ttk.Frame(core)
        trow.pack(fill="x", pady=2)
        ttk.Label(trow, text="T_dup:", width=14).pack(side="left")
        ttk.Spinbox(trow, from_=0.80, to=0.999, increment=0.001, textvariable=self.dup_corr_threshold, width=10).pack(
            side="left", padx=6
        )

        right_row = ttk.Frame(right)
        right_row.pack(fill="both", expand=True)

        enet = ttk.LabelFrame(right_row, text="步骤5：稳定选择（ElasticNet）", padding=10)
        enet.pack(side="left", fill="both", expand=True, pady=4, padx=(0, 10))

        ttk.Checkbutton(
            enet,
            text="启用步骤5（稳定选择）",
            variable=self.enable_step5,
            command=self._toggle_step5_state,
        ).pack(anchor="w", pady=(0, 6))

        self._step5_widgets: list[tk.Widget] = []

        l1row = ttk.Frame(enet)
        l1row.pack(fill="x", pady=2)
        ttk.Label(l1row, text="l1_ratio:", width=14).pack(side="left")
        self._l1_ratio_spin = ttk.Spinbox(l1row, from_=0.0, to=1.0, increment=0.05, textvariable=self.enet_l1_ratio, width=10)
        self._l1_ratio_spin.pack(side="left", padx=6)
        self._step5_widgets.append(self._l1_ratio_spin)

        auto_row = ttk.Frame(enet)
        auto_row.pack(fill="x", pady=2)
        self._auto_lambda_cb = ttk.Checkbutton(
            auto_row,
            text="自动选择 λ（patient-grouped CV）",
            variable=self.enable_auto_lambda_cv,
            command=self._toggle_lambda_mode,
        )
        self._auto_lambda_cb.pack(anchor="w")
        self._step5_widgets.append(self._auto_lambda_cb)

        lam_row = ttk.Frame(enet)
        lam_row.pack(fill="x", pady=2)
        ttk.Label(lam_row, text="手动 λ:", width=14).pack(side="left")
        self._lambda_entry = ttk.Entry(lam_row, textvariable=self.lambda_value, width=12)
        self._lambda_entry.pack(side="left", padx=6)
        self._step5_widgets.append(self._lambda_entry)

        cv_row1 = ttk.Frame(enet)
        cv_row1.pack(fill="x", pady=2)
        ttk.Label(cv_row1, text="K_lambda:", width=14).pack(side="left")
        self._cv_folds_spin = ttk.Spinbox(cv_row1, from_=2, to=20, increment=1, textvariable=self.lambda_cv_folds, width=10)
        self._cv_folds_spin.pack(side="left", padx=6)
        self._step5_widgets.append(self._cv_folds_spin)

        cv_row2 = ttk.Frame(enet)
        cv_row2.pack(fill="x", pady=2)
        ttk.Label(cv_row2, text="L(λ路径):", width=14).pack(side="left")
        self._cv_L_spin = ttk.Spinbox(cv_row2, from_=2, to=200, increment=1, textvariable=self.lambda_cv_n_alphas, width=10)
        self._cv_L_spin.pack(side="left", padx=6)
        self._step5_widgets.append(self._cv_L_spin)

        cv_row3 = ttk.Frame(enet)
        cv_row3.pack(fill="x", pady=2)
        ttk.Label(cv_row3, text="epsilon:", width=14).pack(side="left")
        self._cv_eps_entry = ttk.Entry(cv_row3, textvariable=self.lambda_cv_epsilon, width=12)
        self._cv_eps_entry.pack(side="left", padx=6)
        self._step5_widgets.append(self._cv_eps_entry)

        self._cv_1se_cb = ttk.Checkbutton(enet, text="1-SE 规则（更稀疏）", variable=self.lambda_cv_use_1se)
        self._cv_1se_cb.pack(anchor="w", pady=(2, 0))
        self._step5_widgets.append(self._cv_1se_cb)

        self._lambda_size_tuning_cb = ttk.Checkbutton(
            enet,
            text="尺寸约束微调（沿 λ 路径调整以尽量满足 K_max，可选）",
            variable=self.enable_lambda_size_tuning,
        )
        self._lambda_size_tuning_cb.pack(anchor="w", pady=(2, 0))
        self._step5_widgets.append(self._lambda_size_tuning_cb)

        brow = ttk.Frame(enet)
        brow.pack(fill="x", pady=2)
        ttk.Label(brow, text="bootstrap_B:", width=14).pack(side="left")
        self._bootstrap_spin = ttk.Spinbox(brow, from_=1, to=2000, increment=10, textvariable=self.bootstrap_B, width=10)
        self._bootstrap_spin.pack(side="left", padx=6)
        self._step5_widgets.append(self._bootstrap_spin)

        drow = ttk.Frame(enet)
        drow.pack(fill="x", pady=2)
        ttk.Label(drow, text="delta:", width=14).pack(side="left")
        self._delta_entry = ttk.Entry(drow, textvariable=self.stability_delta, width=12)
        self._delta_entry.pack(side="left", padx=6)
        self._step5_widgets.append(self._delta_entry)

        taurow = ttk.Frame(enet)
        taurow.pack(fill="x", pady=2)
        ttk.Label(taurow, text="tau:", width=14).pack(side="left")
        self._tau_spin = ttk.Spinbox(taurow, from_=0.0, to=1.0, increment=0.05, textvariable=self.stability_tau, width=10)
        self._tau_spin.pack(side="left", padx=6)
        self._step5_widgets.append(self._tau_spin)

        krow = ttk.Frame(enet)
        krow.pack(fill="x", pady=2)
        ttk.Label(krow, text="K_max(0=不限):", width=14).pack(side="left")
        self._kmax_spin = ttk.Spinbox(krow, from_=0, to=10000, increment=10, textvariable=self.k_max, width=10)
        self._kmax_spin.pack(side="left", padx=6)
        self._step5_widgets.append(self._kmax_spin)

        force_frame = ttk.LabelFrame(right_row, text="强制纳入特征列表", padding=10)
        force_frame.pack(side="left", fill="both", expand=True, pady=4)

        ttk.Label(force_frame, text="若筛选后未纳入，则强制加入最终输出：").pack(anchor="w", pady=(0, 4))

        display_map = [
            ("classic_dhi_dhi", "classic_{节段}_dhi_dhi"),
            ("classic_dhi_disc_height", "classic_{节段}_dhi_disc_height"),
            ("classic_t2si_si_ratio", "classic_{节段}_t2si_si_ratio"),
            ("classic_t2si_roi_si", "classic_{节段}_t2si_roi_si"),
            ("classic_t2si_csf_si", "classic_{节段}_t2si_csf_si"),
            ("classic_asi_peak_diff", "classic_{节段}_asi_peak_diff"),
            ("classic_asi_asi", "classic_{节段}_asi_asi"),
            ("classic_fd_fd", "classic_{节段}_fd_fd"),
        ]
        for base, label in display_map:
            var = self.force_vars.get(base)
            if var is None:
                continue
            ttk.Checkbutton(force_frame, text=label, variable=var).pack(anchor="w")

        self._toggle_pca_state()
        self._toggle_step5_state()

    def _toggle_lambda_mode(self) -> None:
        if hasattr(self, "enable_step5") and not bool(self.enable_step5.get()):
            for wname in ("_lambda_entry", "_cv_folds_spin", "_cv_L_spin", "_cv_eps_entry", "_cv_1se_cb", "_lambda_size_tuning_cb"):
                if hasattr(self, wname):
                    getattr(self, wname).config(state="disabled")
            return

        auto = bool(self.enable_auto_lambda_cv.get())

        if hasattr(self, "_lambda_entry"):
            self._lambda_entry.config(state=("disabled" if auto else "normal"))
        for wname in ("_cv_folds_spin", "_cv_L_spin", "_cv_eps_entry", "_cv_1se_cb"):
            if hasattr(self, wname):
                getattr(self, wname).config(state=("normal" if auto else "disabled"))
        if hasattr(self, "_lambda_size_tuning_cb"):
            self._lambda_size_tuning_cb.config(state=("normal" if auto else "disabled"))

    def _toggle_pca_state(self) -> None:

        has_deep_tensor = bool(self.ft_deep.get()) or bool(self.ft_tensor.get())
        if hasattr(self, "_enable_pca_cb"):
            self._enable_pca_cb.config(state=("normal" if has_deep_tensor else "disabled"))
        enable = has_deep_tensor and bool(self.enable_pca.get())
        state = "normal" if enable else "disabled"
        if hasattr(self, "_pca_eta_spin"):
            self._pca_eta_spin.config(state=state)
        if hasattr(self, "_pca_m_cap_spin"):
            self._pca_m_cap_spin.config(state=state)

    def _toggle_step5_state(self) -> None:

        enabled = bool(self.enable_step5.get())
        state = "normal" if enabled else "disabled"

        if hasattr(self, "_step5_widgets"):
            for w in self._step5_widgets:
                try:
                    w.config(state=state)
                except Exception:
                    pass

        self._toggle_lambda_mode()

    def _setup_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="🚀 执行控制", padding=10)
        frame.pack(fill="x", pady=6)

        self._start_btn = ttk.Button(frame, text="开始分析", command=self.start_analysis)
        self._start_btn.pack(side="left")
        ttk.Button(frame, text="查看ICC分布", command=self.show_icc_distribution).pack(side="left", padx=8)

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
        pf_path = self.pfirrmann_csv.get().strip()
        stats_path = self.statistics_csv.get().strip()
        out_dir = self.output_dir.get().strip()

        if not un_path:
            messagebox.showerror("错误", "请先选择未扰动特征CSV。")
            return
        if not pert_path:
            messagebox.showerror("错误", "请先选择扰动后特征CSV。")
            return
        if not pf_path:
            messagebox.showerror("错误", "请先选择 Pfirrmann 分级文件。")
            return
        if not out_dir:
            messagebox.showerror("错误", "请选择输出目录。")
            return

        feature_types: list[str] = []
        if bool(self.ft_classic.get()):
            feature_types.append("classic")
        if bool(self.ft_pyradiomics.get()):
            feature_types.append("pyradiomics")
        if bool(self.ft_deep.get()):
            feature_types.append("deep")
        if bool(self.ft_tensor.get()):
            feature_types.append("tensor")
        if not feature_types:
            messagebox.showerror("错误", "请至少选择一种“特征处理对象”。")
            return

        enable_step5 = bool(self.enable_step5.get())
        force_include = [k for k, v in self.force_vars.items() if bool(v.get())]
        enable_pca = bool(self.enable_pca.get())

        self._start_btn.config(state="disabled")
        self._log_text.delete(1.0, tk.END)

        self._queue_log("加载并对齐数据...")
        self._queue_log(f"- 未扰动: {un_path}")
        self._queue_log(f"- 扰动后: {pert_path}")
        self._queue_log(f"- 输出目录: {out_dir}")
        self._queue_log(f"- Pfirrmann: {pf_path}")
        if stats_path:
            self._queue_log(f"- statistics.csv: {stats_path}")
        self._queue_log("参数：")

        self._queue_log(f"- feature_types={feature_types}")
        self._queue_log(f"- enable_pca={enable_pca}")
        self._queue_log(f"- enable_step5={enable_step5}")
        self._queue_log(f"- force_include_features(n={len(force_include)})")

        if enable_pca:
            self._queue_log(f"- eta_pca={float(self.pca_eta.get()):g} m_cap={int(self.pca_m_cap.get())}")
        elif ("deep" in feature_types) or ("tensor" in feature_types):
            self._queue_log("- 步骤1：PCA 已关闭（将直接使用原始 deep/tensor 特征）")
        self._queue_log(f"- T_ICC={float(self.icc_threshold.get()):g}")
        self._queue_log(f"- alpha_FDR={float(self.alpha_fdr.get()):g} rho_min={float(self.rho_min.get()):g}")
        self._queue_log(f"- T_dup={float(self.dup_corr_threshold.get()):g}")

        auto_lambda = bool(self.enable_auto_lambda_cv.get())
        if enable_step5:
            self._queue_log(f"- enet_l1_ratio={float(self.enet_l1_ratio.get()):g}")
            if auto_lambda:
                self._queue_log(
                    f"- λ: auto_cv K_lambda={int(self.lambda_cv_folds.get())} L={int(self.lambda_cv_n_alphas.get())} "
                    f"epsilon={float(self.lambda_cv_epsilon.get()):g} 1se={bool(self.lambda_cv_use_1se.get())}"
                )
                self._queue_log(f"- λ 尺寸约束微调: {bool(self.enable_lambda_size_tuning.get())}")
            else:
                self._queue_log(f"- λ: manual {float(self.lambda_value.get()):g}")

            self._queue_log(
                f"- bootstrap_B={int(self.bootstrap_B.get())} delta={float(self.stability_delta.get()):g} tau={float(self.stability_tau.get()):g}"
            )
            self._queue_log(f"- K_max={int(self.k_max.get())} (0=不限)")
        else:
            self._queue_log("- 步骤5：未启用（跳过稳定选择；最终=步骤4去冗余结果 + 强制纳入）")

        worker = threading.Thread(
            target=self._run_analysis,
            kwargs={
                "un_path": un_path,
                "pert_path": pert_path,
                "pf_path": pf_path,
                "statistics_csv": (stats_path if stats_path else None),
                "out_dir": out_dir,
                "feature_types": feature_types,
                "enable_pca": enable_pca,
                "enable_stability_selection": enable_step5,
                "force_include_features": force_include,
                "pca_eta": float(self.pca_eta.get()),
                "pca_m_cap": int(self.pca_m_cap.get()),
                "icc_threshold": float(self.icc_threshold.get()),
                "alpha_fdr": float(self.alpha_fdr.get()),
                "rho_min": float(self.rho_min.get()),
                "dup_corr_threshold": float(self.dup_corr_threshold.get()),
                "enet_l1_ratio": float(self.enet_l1_ratio.get()),
                "enable_auto_lambda_cv": auto_lambda,
                "lambda_value": float(self.lambda_value.get()),
                "lambda_cv_folds": int(self.lambda_cv_folds.get()),
                "lambda_cv_n_alphas": int(self.lambda_cv_n_alphas.get()),
                "lambda_cv_epsilon": float(self.lambda_cv_epsilon.get()),
                "lambda_cv_use_1se": bool(self.lambda_cv_use_1se.get()),
                "enable_lambda_size_tuning": bool(self.enable_lambda_size_tuning.get()),
                "bootstrap_B": int(self.bootstrap_B.get()),
                "stability_delta": float(self.stability_delta.get()),
                "stability_tau": float(self.stability_tau.get()),
                "k_max": (None if int(self.k_max.get()) <= 0 else int(self.k_max.get())),
            },
            daemon=True,
        )
        worker.start()
        self.parent.after(120, self._poll_queue)

    def _run_analysis(
        self,
        *,
        un_path: str,
        pert_path: str,
        pf_path: str,
        out_dir: str,
        feature_types: list[str],
        enable_pca: bool,
        enable_stability_selection: bool,
        force_include_features: list[str],
        statistics_csv: str | None,
        pca_eta: float,
        pca_m_cap: int,
        icc_threshold: float,
        alpha_fdr: float,
        rho_min: float,
        dup_corr_threshold: float,
        enet_l1_ratio: float,
        enable_auto_lambda_cv: bool,
        lambda_value: float,
        lambda_cv_folds: int,
        lambda_cv_n_alphas: int,
        lambda_cv_epsilon: float,
        lambda_cv_use_1se: bool,
        enable_lambda_size_tuning: bool,
        bootstrap_B: int,
        stability_delta: float,
        stability_tau: float,
        k_max: int | None,
    ) -> None:
        try:
            result = analyze_robustness(
                unperturbed_csv=un_path,
                perturbed_csv=pert_path,
                pfirrmann_csv=pf_path,
                feature_types=feature_types,
                enable_pca=enable_pca,
                enable_stability_selection=enable_stability_selection,
                force_include_features=force_include_features,
                pca_eta=pca_eta,
                pca_m_cap=pca_m_cap,
                icc_threshold=icc_threshold,
                alpha_fdr=alpha_fdr,
                rho_min=rho_min,
                dup_corr_threshold=dup_corr_threshold,
                enet_l1_ratio=enet_l1_ratio,
                enable_auto_lambda_cv=enable_auto_lambda_cv,
                lambda_value=lambda_value,
                lambda_cv_folds=lambda_cv_folds,
                lambda_cv_n_alphas=lambda_cv_n_alphas,
                lambda_cv_epsilon=lambda_cv_epsilon,
                lambda_cv_use_1se=lambda_cv_use_1se,
                enable_lambda_size_tuning=enable_lambda_size_tuning,
                bootstrap_B=bootstrap_B,
                stability_delta=stability_delta,
                stability_tau=stability_tau,
                k_max=k_max,
            )
            save_analysis_result(
                result,
                output_dir=out_dir,
                pfirrmann_csv=pf_path,
                statistics_csv=statistics_csv,
            )
            self._result = result

            self._queue.put(("log", f"清洗：初始特征数={result.n_features_initial} 清洗后={result.n_features_after_cleaning}"))
            self._queue.put(
                (
                    "log",
                    f"剔除：缺失/无穷={len(result.dropped_nan_features)} 常量={len(result.dropped_constant_features)}",
                )
            )
            if not bool(enable_pca):
                self._queue.put(("log", "步骤1：PCA 已关闭（使用原始 deep/tensor 特征）"))
            else:
                deep_k = int(result.pca_deep_info.shape[0]) if result.pca_deep_info is not None else 0
                tensor_k = int(result.pca_tensor_info.shape[0]) if result.pca_tensor_info is not None else 0
                if deep_k == 0 and tensor_k == 0:
                    self._queue.put(("log", "步骤1：PCA 跳过（未选择 deep/tensor 或未检测到对应特征）"))
                else:
                    self._queue.put(("log", f"步骤1：PCA 维度 Deep={deep_k} Tensor={tensor_k}"))

            self._queue.put(("log", f"步骤2：ICC>=T_ICC 保留 {len(result.robust_features)}"))
            self._queue.put(("log", f"步骤3：Spearman+FDR 通过 {len(result.selected_by_spearman)}"))
            self._queue.put(("log", f"步骤4：去冗余后保留 {int(result.pass_dedup.astype(bool).sum())}"))
            if bool(getattr(result, "stability_selection_enabled", True)):
                self._queue.put(
                    (
                        "log",
                        f"步骤5：稳定选择通过 {int((result.stable_pi.astype(float) >= float(result.stability_tau)).sum())} "
                        f"最终 {len(result.final_features)}（λ={float(result.lambda_value):g}, {result.lambda_mode}）",
                    )
                )
            else:
                added = len(getattr(result, "force_included_added", []) or [])
                self._queue.put(("log", f"步骤5：未启用，最终特征=步骤4去冗余结果 + 强制纳入（新增 {added}）"))

            self._queue.put(
                (
                    "done",
                    "分析完成！\n"
                    f"- 最终特征数: {len(result.final_features)}\n"
                    f"- ICC通过特征数: {len(result.robust_features)}\n"
                    f"- 最终模型输入: {Path(out_dir).resolve() / '最终模型输入.csv'}\n"
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


__all__ = ["RobustnessGUI"]
