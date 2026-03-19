from __future__ import annotations

import json
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .cli import build_hybrid_predictions, build_predictions, evaluate_predictions_file, map_at_k, tune_params
from .data import load_json, to_qrel_map, zip_single_file


class Task1Gui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JOKER Task 1 Retriever")
        self.root.geometry("1220x920")
        self.root.minsize(980, 720)

        self.events: queue.Queue[tuple[str, str, float | None]] = queue.Queue()
        self.running = False

        self.docs_var = tk.StringVar()
        self.queries_var = tk.StringVar()
        self.qrels_var = tk.StringVar()
        self.output_var = tk.StringVar(value="prediction.json")
        self.zip_var = tk.StringVar(value="submission.zip")
        self.run_id_var = tk.StringVar(value="team_task_1_hybrid_gui")
        self.params_in_var = tk.StringVar(value="")
        self.params_out_var = tk.StringVar(value="tuned_params.json")
        self.eval_pred_var = tk.StringVar(value="")
        self.manual_var = tk.IntVar(value=0)
        self.topk_var = tk.IntVar(value=1000)
        self.autotune_var = tk.BooleanVar(value=False)
        self.eval_after_run_var = tk.BooleanVar(value=False)

        self.pipeline_var = tk.StringVar(value="hybrid")
        self.device_var = tk.StringVar(value="cuda")
        self.batch_size_var = tk.IntVar(value=32)
        self.dense_model_var = tk.StringVar(value="BAAI/bge-small-en-v1.5")
        self.dense_index_dir_var = tk.StringVar(value="artifacts/dense_index")
        self.dense_topk_var = tk.IntVar(value=700)
        self.reranker_model_var = tk.StringVar(value="cross-encoder/ms-marco-MiniLM-L12-v2")
        self.rerank_topn_var = tk.IntVar(value=200)
        self.humor_model_dir_var = tk.StringVar(value="artifacts/humor_model")
        self.humor_train_model_var = tk.StringVar(value="roberta-base")
        self.humor_epochs_var = tk.IntVar(value=3)
        self.humor_batch_var = tk.IntVar(value=4)
        self.humor_negatives_var = tk.IntVar(value=3)
        self.learning_rate_var = tk.DoubleVar(value=2e-5)
        self.max_length_var = tk.IntVar(value=256)
        self.fusion_config_var = tk.StringVar(value="")

        self.status_var = tk.StringVar(value="Idle")
        self.resource_var = tk.StringVar(value="CPU: -- | RAM: -- | GPU: --")

        self._build_ui()
        self._poll_events()
        self._update_resources()

    def _build_ui(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        outer_scroll = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        outer_scroll.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=outer_scroll.set)

        self.scroll_frame = ttk.Frame(self.canvas, padding=12)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        self.scroll_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

        frm = self.scroll_frame

        def add_file_row(parent: ttk.Frame, label: str, var: tk.StringVar, row: int, *, directory: bool = False):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(parent, textvariable=var, width=100).grid(row=row, column=1, sticky="we", padx=6)
            if directory:
                ttk.Button(parent, text="Browse", command=lambda: self._browse_dir(var)).grid(row=row, column=2)
            else:
                ttk.Button(parent, text="Browse", command=lambda: self._browse_file(var)).grid(row=row, column=2)

        paths = ttk.LabelFrame(frm, text="Task files", padding=10)
        paths.grid(row=0, column=0, sticky="nsew")
        add_file_row(paths, "Corpus JSON", self.docs_var, 0)
        add_file_row(paths, "Queries JSON", self.queries_var, 1)
        add_file_row(paths, "Qrels JSON", self.qrels_var, 2)
        add_file_row(paths, "Output prediction.json", self.output_var, 3)
        add_file_row(paths, "Output ZIP (optional)", self.zip_var, 4)
        add_file_row(paths, "Load lexical params JSON", self.params_in_var, 5)
        add_file_row(paths, "Save lexical params JSON", self.params_out_var, 6)
        add_file_row(paths, "Dense index directory", self.dense_index_dir_var, 7, directory=True)
        add_file_row(paths, "Humor model directory", self.humor_model_dir_var, 8, directory=True)
        add_file_row(paths, "Fusion config JSON", self.fusion_config_var, 9)
        paths.columnconfigure(1, weight=1)

        controls = ttk.LabelFrame(frm, text="Run controls", padding=10)
        controls.grid(row=1, column=0, sticky="we", pady=(10, 0))
        ttk.Label(controls, text="Pipeline").grid(row=0, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self.pipeline_var, values=["baseline", "hybrid"], width=12, state="readonly").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(controls, text="Run ID").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(controls, textvariable=self.run_id_var, width=28).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(controls, text="Device").grid(row=0, column=4, sticky="w", padx=(12, 0))
        ttk.Combobox(controls, textvariable=self.device_var, values=["cuda", "cpu"], width=10, state="readonly").grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(controls, text="Batch size").grid(row=0, column=6, sticky="w", padx=(12, 0))
        ttk.Spinbox(controls, from_=1, to=256, textvariable=self.batch_size_var, width=7).grid(row=0, column=7, sticky="w", padx=4)

        opts = ttk.Frame(controls)
        opts.grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))
        ttk.Checkbutton(opts, text="Manual run", variable=self.manual_var, onvalue=1, offvalue=0).pack(side="left", padx=4)
        ttk.Checkbutton(opts, text="Auto-tune lexical weights", variable=self.autotune_var).pack(side="left", padx=10)
        ttk.Checkbutton(opts, text="Evaluate after prediction", variable=self.eval_after_run_var).pack(side="left", padx=10)
        ttk.Label(opts, text="Top-K").pack(side="left", padx=(12, 4))
        ttk.Spinbox(opts, from_=1, to=1000, textvariable=self.topk_var, width=7).pack(side="left")

        hybrid = ttk.LabelFrame(frm, text="Hybrid settings", padding=10)
        hybrid.grid(row=2, column=0, sticky="we", pady=(10, 0))
        ttk.Label(hybrid, text="Dense model").grid(row=0, column=0, sticky="w")
        ttk.Entry(hybrid, textvariable=self.dense_model_var, width=42).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Label(hybrid, text="Dense top-k").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(hybrid, from_=10, to=5000, textvariable=self.dense_topk_var, width=8).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(hybrid, text="Reranker model").grid(row=1, column=0, sticky="w")
        ttk.Entry(hybrid, textvariable=self.reranker_model_var, width=42).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Label(hybrid, text="Rerank top-n").grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(hybrid, from_=10, to=1000, textvariable=self.rerank_topn_var, width=8).grid(row=1, column=3, sticky="w", padx=4)
        hybrid.columnconfigure(1, weight=1)

        training = ttk.LabelFrame(frm, text="Humor model training", padding=10)
        training.grid(row=3, column=0, sticky="we", pady=(10, 0))
        ttk.Label(training, text="Train model").grid(row=0, column=0, sticky="w")
        ttk.Entry(training, textvariable=self.humor_train_model_var, width=36).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Label(training, text="Epochs").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(training, from_=1, to=20, textvariable=self.humor_epochs_var, width=6).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(training, text="Batch").grid(row=0, column=4, sticky="w", padx=(12, 0))
        ttk.Spinbox(training, from_=1, to=64, textvariable=self.humor_batch_var, width=6).grid(row=0, column=5, sticky="w", padx=4)
        ttk.Label(training, text="Negatives/positive").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(training, from_=1, to=10, textvariable=self.humor_negatives_var, width=6).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(training, text="Learning rate").grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(training, textvariable=self.learning_rate_var, width=10).grid(row=1, column=3, sticky="w", padx=4)
        ttk.Label(training, text="Max length").grid(row=1, column=4, sticky="w", padx=(12, 0))
        ttk.Spinbox(training, from_=64, to=512, textvariable=self.max_length_var, width=6).grid(row=1, column=5, sticky="w", padx=4)
        training.columnconfigure(1, weight=1)

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, sticky="w", pady=10)
        self.run_btn = ttk.Button(btns, text="Run Prediction", command=self.start_run)
        self.run_btn.pack(side="left", padx=4)
        self.index_btn = ttk.Button(btns, text="Build Dense Index", command=self.start_build_dense)
        self.index_btn.pack(side="left", padx=4)
        self.train_btn = ttk.Button(btns, text="Train Humor Model", command=self.start_train_humor)
        self.train_btn.pack(side="left", padx=4)
        self.eval_btn = ttk.Button(btns, text="Evaluate Existing Predictions", command=self.start_eval_only)
        self.eval_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="Clear Log", command=self.clear_log).pack(side="left", padx=4)

        eval_frame = ttk.Frame(frm)
        eval_frame.grid(row=5, column=0, sticky="we")
        ttk.Label(eval_frame, text="Prediction file to evaluate").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(eval_frame, textvariable=self.eval_pred_var, width=100).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(eval_frame, text="Browse", command=lambda: self._browse_file(self.eval_pred_var)).grid(row=0, column=2)
        eval_frame.columnconfigure(1, weight=1)

        system = ttk.LabelFrame(frm, text="Execution status and resource usage", padding=10)
        system.grid(row=6, column=0, sticky="we", pady=(10, 0))
        self.progress = ttk.Progressbar(system, orient="horizontal", mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="we")
        ttk.Label(system, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(system, textvariable=self.resource_var).grid(row=2, column=0, sticky="w", pady=(6, 0))
        system.columnconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(frm, text="Logger", padding=10)
        log_frame.grid(row=7, column=0, sticky="nsew", pady=(10, 0))
        self.log = tk.Text(log_frame, height=24, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(7, weight=1)

    def _on_frame_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if self.canvas.winfo_exists():
            direction = -1 if event.num == 4 else 1
            self.canvas.yview_scroll(direction, "units")

    def _browse_file(self, var: tk.StringVar):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _emit(self, kind: str, message: str, pct: float | None = None):
        self.events.put((kind, message, pct))

    def _set_running(self, running: bool):
        self.running = running
        state = "disabled" if running else "normal"
        self.run_btn.config(state=state)
        self.index_btn.config(state=state)
        self.train_btn.config(state=state)
        self.eval_btn.config(state=state)

    def _poll_events(self):
        try:
            while True:
                kind, message, pct = self.events.get_nowait()
                if kind == "progress":
                    if pct is not None:
                        self.progress["value"] = max(0, min(100, pct * 100))
                    self.status_var.set(message)
                    self.log.insert("end", f"[INFO] {message}\n")
                    self.log.see("end")
                elif kind == "done":
                    self._set_running(False)
                    self.progress["value"] = 100
                    self.status_var.set(message)
                    self.log.insert("end", f"[DONE] {message}\n")
                    self.log.see("end")
                    messagebox.showinfo("Completed", message)
                elif kind == "error":
                    self._set_running(False)
                    self.status_var.set("Failed")
                    self.log.insert("end", f"[ERROR] {message}\n")
                    self.log.see("end")
                    messagebox.showerror("Error", message)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_events)

    def _update_resources(self):
        self.resource_var.set(self._resource_snapshot())
        self.root.after(1500, self._update_resources)

    def _resource_snapshot(self) -> str:
        cpu_part = "CPU: unavailable"
        ram_part = "RAM: unavailable"
        gpu_part = "GPU: unavailable"
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            cpu_part = f"CPU: {cpu:.1f}%"
            ram_part = f"RAM: {mem.percent:.1f}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"
        except Exception:
            pass
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.5,
            ).strip()
            if output:
                util, mem_used, mem_total, temp = [part.strip() for part in output.splitlines()[0].split(",")]
                gpu_part = f"GPU: {util}% | VRAM: {mem_used}/{mem_total} MiB | Temp: {temp}°C"
        except Exception:
            pass
        return f"{cpu_part} | {ram_part} | {gpu_part}"

    def clear_log(self):
        self.log.delete("1.0", "end")

    def _validate_prediction_inputs(self):
        if not Path(self.docs_var.get()).exists():
            raise ValueError("Corpus JSON path is invalid.")
        if not Path(self.queries_var.get()).exists():
            raise ValueError("Queries JSON path is invalid.")
        if self.qrels_var.get() and not Path(self.qrels_var.get()).exists():
            raise ValueError("Qrels JSON path is invalid.")
        if self.autotune_var.get() and not self.qrels_var.get():
            raise ValueError("Auto-tune requires a qrels file.")
        if self.eval_after_run_var.get() and not self.qrels_var.get():
            raise ValueError("Evaluate-after-run requires a qrels file.")
        if self.pipeline_var.get() == "hybrid" and not self.dense_model_var.get().strip():
            raise ValueError("Dense model is required for the hybrid pipeline.")

    def _validate_eval_inputs(self):
        if not self.eval_pred_var.get() or not Path(self.eval_pred_var.get()).exists():
            raise ValueError("Prediction file for evaluation is invalid.")
        if not self.qrels_var.get() or not Path(self.qrels_var.get()).exists():
            raise ValueError("Qrels JSON path is required and must be valid for evaluation.")

    def _validate_dense_index_inputs(self):
        if not Path(self.docs_var.get()).exists():
            raise ValueError("Corpus JSON path is invalid.")
        if not self.dense_model_var.get().strip():
            raise ValueError("Dense model is required.")

    def _validate_humor_training_inputs(self):
        if not Path(self.docs_var.get()).exists():
            raise ValueError("Corpus JSON path is invalid.")
        if not Path(self.queries_var.get()).exists():
            raise ValueError("Queries JSON path is invalid.")
        if not self.qrels_var.get() or not Path(self.qrels_var.get()).exists():
            raise ValueError("Qrels JSON path is required for humor model training.")

    def start_run(self):
        if self.running:
            return
        try:
            self._validate_prediction_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._set_running(True)
        self.progress["value"] = 0
        self.status_var.set("Starting prediction...")
        threading.Thread(target=self._worker_prediction, daemon=True).start()

    def start_build_dense(self):
        if self.running:
            return
        try:
            self._validate_dense_index_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._set_running(True)
        self.progress["value"] = 0
        self.status_var.set("Building dense index...")
        threading.Thread(target=self._worker_build_dense, daemon=True).start()

    def start_train_humor(self):
        if self.running:
            return
        try:
            self._validate_humor_training_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._set_running(True)
        self.progress["value"] = 0
        self.status_var.set("Training humor model...")
        threading.Thread(target=self._worker_train_humor, daemon=True).start()

    def start_eval_only(self):
        if self.running:
            return
        try:
            self._validate_eval_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._set_running(True)
        self.progress["value"] = 0
        self.status_var.set("Evaluating...")
        threading.Thread(target=self._worker_eval_only, daemon=True).start()

    def _evaluate_map(self, pred_path: str, qrels_path: str, k: int) -> float:
        predictions = load_json(pred_path)
        qrels = load_json(qrels_path)
        rel_by_qid = to_qrel_map(qrels)
        pred_by_qid: dict[str, list[str]] = {}
        for row in predictions:
            pred_by_qid.setdefault(str(row["qid"]), []).append(str(row["docid"]))
        return map_at_k(pred_by_qid, rel_by_qid, k=k)

    def _worker_eval_only(self):
        try:
            self._emit("progress", "Evaluating predictions...", 0.2)
            score = evaluate_predictions_file(self.eval_pred_var.get(), self.qrels_var.get(), self.topk_var.get())
            self._emit("progress", f"MAP@{self.topk_var.get()}: {score:.6f}", 1.0)
            self._emit("done", f"Evaluation complete. MAP@{self.topk_var.get()} = {score:.6f}")
        except Exception as exc:
            self._emit("error", str(exc))

    def _worker_build_dense(self):
        try:
            from .dense import DenseRetriever

            docs = load_json(self.docs_var.get())
            retriever = DenseRetriever(
                model_name=self.dense_model_var.get().strip(),
                index_dir=self.dense_index_dir_var.get().strip(),
                device=self.device_var.get().strip() or None,
                batch_size=self.batch_size_var.get(),
            )
            retriever.build(docs, progress=lambda msg, p: self._emit("progress", msg, p))
            self._emit("done", f"Dense index created in {self.dense_index_dir_var.get().strip()}")
        except Exception as exc:
            self._emit("error", str(exc))

    def _worker_train_humor(self):
        try:
            from .humor_classifier import train_humor_pair_classifier

            docs = load_json(self.docs_var.get())
            queries = load_json(self.queries_var.get())
            qrels = load_json(self.qrels_var.get())
            metrics = train_humor_pair_classifier(
                docs=docs,
                queries=queries,
                qrels=qrels,
                output_dir=self.humor_model_dir_var.get().strip(),
                model_name=self.humor_train_model_var.get().strip(),
                device=self.device_var.get().strip() or None,
                epochs=self.humor_epochs_var.get(),
                batch_size=self.humor_batch_var.get(),
                learning_rate=float(self.learning_rate_var.get()),
                max_length=self.max_length_var.get(),
                negatives_per_positive=self.humor_negatives_var.get(),
                progress=lambda msg, p: self._emit("progress", msg, p),
            )
            self._emit("progress", f"Humor training metrics: {json.dumps(metrics)}", 1.0)
            self._emit("done", f"Humor model saved to {self.humor_model_dir_var.get().strip()}")
        except Exception as exc:
            self._emit("error", str(exc))

    def _worker_prediction(self):
        try:
            params = None
            docs_path = self.docs_var.get()
            queries_path = self.queries_var.get()
            qrels_path = self.qrels_var.get() or None
            output_path = self.output_var.get()
            zip_path = self.zip_var.get().strip()
            params_in = self.params_in_var.get().strip()
            params_out = self.params_out_var.get().strip()

            if params_in:
                self._emit("progress", f"Loading lexical params from {params_in}", 0.01)
                with Path(params_in).open("r", encoding="utf-8") as f:
                    params = json.load(f)
                self._emit("progress", f"Loaded params: {params}", 0.03)

            if self.autotune_var.get():
                if not qrels_path:
                    raise ValueError("Auto-tune requires qrels.")
                self._emit("progress", "Loading files for lexical auto-tuning...", 0.05)
                docs = load_json(docs_path)
                queries = load_json(queries_path)
                qrels = load_json(qrels_path)
                params, holdout = tune_params(
                    docs=docs,
                    queries=queries,
                    qrels=qrels,
                    top_k=self.topk_var.get(),
                    progress=lambda msg, p: self._emit("progress", msg, p),
                )
                self._emit("progress", f"Selected lexical params: {params}; holdout MAP={holdout:.6f}", 0.55)
                if params_out:
                    with Path(params_out).open("w", encoding="utf-8") as f:
                        json.dump(params, f, ensure_ascii=False, indent=2)
                    self._emit("progress", f"Saved lexical params to {params_out}", 0.58)

            if self.pipeline_var.get() == "baseline":
                rows = build_predictions(
                    docs_path=docs_path,
                    queries_path=queries_path,
                    output_path=output_path,
                    run_id=self.run_id_var.get().strip(),
                    manual=self.manual_var.get(),
                    qrels_path=qrels_path,
                    top_k=self.topk_var.get(),
                    params=params,
                    progress=lambda msg, p: self._emit("progress", msg, p),
                )
            else:
                rows = build_hybrid_predictions(
                    docs_path=docs_path,
                    queries_path=queries_path,
                    output_path=output_path,
                    run_id=self.run_id_var.get().strip(),
                    manual=self.manual_var.get(),
                    qrels_path=qrels_path,
                    top_k=self.topk_var.get(),
                    lexical_params=params,
                    dense_model=self.dense_model_var.get().strip(),
                    dense_index_dir=self.dense_index_dir_var.get().strip(),
                    dense_top_k=self.dense_topk_var.get(),
                    reranker_model=self.reranker_model_var.get().strip() or None,
                    rerank_top_n=self.rerank_topn_var.get(),
                    humor_model_dir=self.humor_model_dir_var.get().strip() or None,
                    device=self.device_var.get().strip() or None,
                    batch_size=self.batch_size_var.get(),
                    fusion_config_path=self.fusion_config_var.get().strip() or None,
                    progress=lambda msg, p: self._emit("progress", msg, p),
                )

            if zip_path:
                zip_single_file(output_path, zip_path, arcname="prediction.json")
                self._emit("progress", f"Created zip: {zip_path}", 0.98)

            if self.eval_after_run_var.get() and qrels_path:
                self._emit("progress", "Running post-prediction evaluation...", 0.99)
                score = self._evaluate_map(output_path, qrels_path, self.topk_var.get())
                self._emit("progress", f"MAP@{self.topk_var.get()} on selected data: {score:.6f}", 1.0)

            self.eval_pred_var.set(output_path)
            self._emit("done", f"Completed successfully. Rows written: {len(rows)}")
        except Exception as exc:
            self._emit("error", str(exc))



def main() -> None:
    root = tk.Tk()
    Task1Gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
