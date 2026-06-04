import contextlib
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ga


def validate_float(text):
    if text == "":
        return True

    try:
        float(text)
        return True
    except ValueError:
        return False


class StreamRedirector:
    def __init__(self, on_line):
        self.on_line = on_line
        self.buffer = ""
        self.lines = []

    def write(self, text):
        if not text:
            return 0

        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.lines.append(line)
            self.on_line(line)
        return len(text)

    def flush(self):
        if self.buffer:
            self.lines.append(self.buffer)
            self.on_line(self.buffer)
            self.buffer = ""

    def getvalue(self):
        if self.buffer:
            return "\n".join(self.lines + [self.buffer])
        return "\n".join(self.lines)


def main():
    output_stream = None
    closing = False
    root = tk.Tk()
    root.title("Dome Optimizer")
    root.geometry("1000x900")

    # app_icon = tk.PhotoImage(file="my_icon.png")
    # root.iconphoto(True, app_icon)

    vcmd = (root.register(validate_float), "%P")

    instruction_label = tk.Label(root, text="Enter the radius of the dome:", font=("Arial", 12))
    instruction_label.pack(pady=10)

    radius_entry = tk.Entry(root, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    radius_entry.pack(pady=5)

    instruction_label_1 = tk.Label(root, text="Enter the height of the dome:", font=("Arial", 12))
    instruction_label_1.pack(pady=10)

    height_entry = tk.Entry(root, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    height_entry.pack(pady=5)

    status_var = tk.StringVar(value="Ready")
    status_label = tk.Label(root, textvariable=status_var, font=("Arial", 10))
    status_label.pack(pady=5)

    output_txt_box = ScrolledText(root, height=5)
    output_txt_box.pack(fill=tk.X, padx=10, pady=5)
    output_txt_box.configure(state=tk.DISABLED)

    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.set_title("Dome Optimization Progress")
    axis.set_xlabel("generation")
    axis.set_ylabel("strength-to-weight ratio (N/kg)")
    axis.grid(alpha=0.3)

    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    run_button: tk.Button
    output_lines = []

    def is_open():
        if closing:
            return False
        try:
            return root.winfo_exists()
        except tk.TclError:
            return False

    def safe_after(delay, callback):
        if not is_open():
            return
        try:
            root.after(delay, lambda: callback() if is_open() else None)
        except tk.TclError:
            pass

    def write_output(text, clear=False):
        if not is_open():
            return
        output_txt_box.configure(state=tk.NORMAL)
        if clear:
            output_txt_box.delete(1.0, tk.END)
        output_txt_box.insert(tk.END, text)
        output_txt_box.see(tk.END)
        output_txt_box.configure(state=tk.DISABLED)

    def on_close():
        nonlocal closing
        closing = True
        try:
            canvas.get_tk_widget().destroy()
        except tk.TclError:
            pass
        try:
            root.quit()
        except tk.TclError:
            pass
        try:
            root.destroy()
        except tk.TclError:
            pass

    def redraw(history):
        if not is_open():
            return
        ga.draw_history(history, axis)
        canvas.draw_idle()

    def append_output(line):
        if not is_open():
            return
        output_lines.append(line)
        write_output(line + "\n")

    def finish_run():
        if not is_open():
            return
        run_button.config(state=tk.NORMAL)
        status_var.set("Done")
        if output_stream is not None:
            write_output(output_stream.getvalue())

    def finish_run_with_error():
        if not is_open():
            return
        run_button.config(state=tk.NORMAL)
        status_var.set("Ready")
        if output_stream is not None:
            write_output(output_stream.getvalue())

    def start():
        nonlocal output_stream
        radius = float(radius_entry.get())
        height = float(height_entry.get())

        run_button.config(state=tk.DISABLED)
        status_var.set("Running GA...")
        output_lines.clear()
        write_output("", clear=True)

        output_stream = StreamRedirector(lambda line: safe_after(0, lambda l=line: append_output(l)))

        def worker():
            try:
                with contextlib.redirect_stdout(output_stream):
                    print(f"Radius: {radius}, Height: {height}")
                    ga.set_params(radius, height)
                    ga.run_ga(progress_callback=lambda snapshot: safe_after(0, lambda data=snapshot: redraw(data)))
                safe_after(0, finish_run)
            except Exception:
                safe_after(0, finish_run_with_error)

        threading.Thread(target=worker, daemon=True).start()

    run_button = tk.Button(root, text="Calculate", font=("Arial", 12), command=start)
    run_button.pack(pady=20)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()