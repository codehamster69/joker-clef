from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .cli import build_predictions, tune_params
from .data import load_json, zip_single_file


class Task1Gui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JOKER Task 1 Retriever")
        self.root.geometry("900x650")

        self.events: queue.Queue[tuple[str, str, float | None]] = queue.Queue()
        self.running = False

        self.docs_var = tk.StringVar()
        self.queries_var = tk.StringVar()
        self.qrels_var = tk.StringVar()
        self.output_var = tk.StringVar(value="prediction.json")
        self.zip_var = tk.StringVar(value="submission.zip")
        self.run_id_var = tk.StringVar(value="team_task_1_hybrid_gui")
        self.manual_var = tk.IntVar(value=0)
        self.topk_var = tk.IntVar(value=1000)
        self.autotune_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._poll_events()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        def add_path_row(label: str, var: tk.StringVar, row: int):
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frm, textvariable=var, width=85).grid(row=row, column=1, sticky="we", padx=6)
            ttk.Button(frm, text="Browse", command=lambda: self._browse_file(var)).grid(row=row, column=2)

        add_path_row("Corpus JSON", self.docs_var, 0)
        add_path_row("Queries JSON", self.queries_var, 1)
        add_path_row("Qrels JSON (optional)", self.qrels_var, 2)
        add_path_row("Output prediction.json", self.output_var, 3)
        add_path_row("Output ZIP (optional)", self.zip_var, 4)

        ttk.Label(frm, text="Run ID").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self.run_id_var, width=40).grid(row=5, column=1, sticky="w", padx=6)

        opts = ttk.Frame(frm)
        opts.grid(row=6, column=1, sticky="w", pady=8)
        ttk.Checkbutton(opts, text="Manual run", variable=self.manual_var, onvalue=1, offvalue=0).pack(side="left", padx=4)
        ttk.Checkbutton(opts, text="Auto-tune (uses qrels)", variable=self.autotune_var).pack(side="left", padx=10)
        ttk.Label(opts, text="Top-K:").pack(side="left", padx=(12, 4))
        ttk.Spinbox(opts, from_=1, to=1000, textvariable=self.topk_var, width=6).pack(side="left")

        btns = ttk.Frame(frm)
        btns.grid(row=7, column=1, sticky="w", pady=10)
        self.run_btn = ttk.Button(btns, text="Run Prediction", command=self.start_run)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="Clear Log", command=self.clear_log).pack(side="left", padx=4)

        self.progress = ttk.Progressbar(frm, orient="horizontal", mode="determinate", maximum=100)
        self.progress.grid(row=8, column=0, columnspan=3, sticky="we", pady=8)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var).grid(row=9, column=0, columnspan=3, sticky="w")

        self.log = tk.Text(frm, height=22, wrap="word")
        self.log.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(8, 0))

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(10, weight=1)

    def _browse_file(self, var: tk.StringVar):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _emit(self, kind: str, message: str, pct: float | None = None):
        self.events.put((kind, message, pct))

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
                    self.running = False
                    self.run_btn.config(state="normal")
                    self.progress["value"] = 100
                    self.status_var.set(message)
                    self.log.insert("end", f"[DONE] {message}\n")
                    self.log.see("end")
                    messagebox.showinfo("Completed", message)
                elif kind == "error":
                    self.running = False
                    self.run_btn.config(state="normal")
                    self.status_var.set("Failed")
                    self.log.insert("end", f"[ERROR] {message}\n")
                    self.log.see("end")
                    messagebox.showerror("Error", message)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_events)

    def clear_log(self):
        self.log.delete("1.0", "end")

    def _validate(self):
        if not Path(self.docs_var.get()).exists():
            raise ValueError("Corpus JSON path is invalid.")
        if not Path(self.queries_var.get()).exists():
            raise ValueError("Queries JSON path is invalid.")
        if self.qrels_var.get() and not Path(self.qrels_var.get()).exists():
            raise ValueError("Qrels JSON path is invalid.")
        if self.autotune_var.get() and not self.qrels_var.get():
            raise ValueError("Auto-tune requires a qrels file.")

    def start_run(self):
        if self.running:
            return
        try:
            self._validate()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.running = True
        self.run_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("Starting...")

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        try:
            params = None
            docs_path = self.docs_var.get()
            queries_path = self.queries_var.get()
            qrels_path = self.qrels_var.get() or None
            output_path = self.output_var.get()
            zip_path = self.zip_var.get().strip()

            if self.autotune_var.get():
                self._emit("progress", "Loading files for auto-tuning...", 0.01)
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
                self._emit("progress", f"Selected params: {params}; holdout MAP={holdout:.6f}", 0.55)

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

            if zip_path:
                zip_single_file(output_path, zip_path, arcname="prediction.json")
                self._emit("progress", f"Created zip: {zip_path}", 0.99)

            self._emit("done", f"Completed successfully. Rows written: {len(rows)}")
        except Exception as exc:
            self._emit("error", str(exc))


def main() -> None:
    root = tk.Tk()
    Task1Gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
