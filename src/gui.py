import contextlib
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ga

# import ctypes

# # Using funky ctypes stuff from google to allow me to change the taskbar icon by registering the program as a unique app
# myappid = 'sajidmonowar.dome_optimizer' # Define a unique string
# ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)



def validate_float(text):
    if text == "":
        return True

    try:
        float(text)
        return True
    except ValueError:
        return False


def validate_int(text):
    if text == "":
        return True
    try:
        int(text)
        return True
    except ValueError:
        return False


class StreamRedirector:
    def __init__(self, on_line):
        self.on_line = on_line
        self.buffer = ""

    def write(self, text):
        if not text:
            return 0

        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.on_line(line)
        return len(text)

    def flush(self):
        if self.buffer:
            self.on_line(self.buffer)
            self.buffer = ""


def main():
    closing = False
    root = tk.Tk()
    root.title("Dome Optimizer")

    window_width = 1200
    window_height = 975

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2) - 25

    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    app_icon = tk.PhotoImage(file="icon.png")
    root.iconphoto(True, app_icon)

    enable_gui = tk.BooleanVar(value=True)

    vcmd = (root.register(validate_float), "%P")
    
    label_frame = tk.Frame(root)
    label_frame.pack(pady=10)

    radius_frame = tk.Frame(label_frame)
    radius_frame.grid(row=0, column=0, padx=10)
    instruction_label = tk.Label(radius_frame, text="Enter the radius of the dome:", font=("Arial", 12))
    instruction_label.pack(pady=10)
    radius_entry = tk.Entry(radius_frame, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    radius_entry.insert(0, "80.0")
    radius_unit = tk.Label(radius_frame, text="mm", font=("Arial", 10))
    radius_unit.pack()
    radius_entry.pack()

    height_frame = tk.Frame(label_frame)
    height_frame.grid(row=0, column=1, padx=10)
    instruction_label_1 = tk.Label(height_frame, text="Enter the height of the dome:", font=("Arial", 12))
    instruction_label_1.pack(pady=10)
    height_entry = tk.Entry(height_frame, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    height_entry.insert(0, "80.0")
    height_unit = tk.Label(height_frame, text="mm", font=("Arial", 10))
    height_unit.pack()
    height_entry.pack()

    min_thick_frame = tk.Frame(label_frame)
    min_thick_frame.grid(row=0, column=2, padx=10, pady=10)
    instruction_label_2 = tk.Label(min_thick_frame, text="Enter the minimum thickness:", font=("Arial", 12))
    instruction_label_2.pack(pady=10)
    min_thick_entry = tk.Entry(min_thick_frame, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    min_thick_entry.insert(0, "2.0")
    min_thick_unit = tk.Label(min_thick_frame, text="mm", font=("Arial", 10))
    min_thick_unit.pack()
    min_thick_entry.pack()

    max_thick_frame = tk.Frame(label_frame)
    max_thick_frame.grid(row=1, column=0, padx=10, pady=10)
    instruction_label_3 = tk.Label(max_thick_frame, text="Enter the maximum thickness:", font=("Arial", 12))
    instruction_label_3.pack(pady=10)
    max_thick_entry = tk.Entry(max_thick_frame, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    max_thick_entry.insert(0, "5.0")
    max_thick_unit = tk.Label(max_thick_frame, text="mm", font=("Arial", 10))
    max_thick_unit.pack()
    max_thick_entry.pack()

    min_offset_frame = tk.Frame(label_frame)
    min_offset_frame.grid(row=1, column=1, padx=10, pady=10)
    instruction_label_4 = tk.Label(min_offset_frame, text="Enter the minimum offset:", font=("Arial", 12))
    instruction_label_4.pack(pady=10)
    min_offset_entry = tk.Entry(min_offset_frame, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    min_offset_entry.insert(0, "-0.10")
    min_offset_unit = tk.Label(min_offset_frame, text="% as a decimal", font=("Arial", 10))
    min_offset_unit.pack()
    min_offset_entry.pack()

    max_offset_frame = tk.Frame(label_frame)
    max_offset_frame.grid(row=1, column=2, padx=10, pady=10)
    instruction_label_5 = tk.Label(max_offset_frame, text="Enter the maximum offset:", font=("Arial", 12))
    instruction_label_5.pack(pady=10)
    max_offset_entry = tk.Entry(max_offset_frame, font=("Arial", 12), width=25, validate="key", validatecommand=vcmd)
    max_offset_entry.insert(0, "0.10")
    max_offset_unit = tk.Label(max_offset_frame, text="% as a decimal", font=("Arial", 10))
    max_offset_unit.pack()
    max_offset_entry.pack()

    # GA controls: population size and number of generations
    pop_frame = tk.Frame(label_frame)
    pop_frame.grid(row=0, column=3, padx=10, pady=10)
    pop_label = tk.Label(pop_frame, text="Population size:", font=("Arial", 12))
    pop_label.pack(pady=6)
    ivcmd = (root.register(validate_int), "%P")
    pop_entry = tk.Entry(pop_frame, font=("Arial", 12), width=25, validate="key", validatecommand=ivcmd)
    pop_entry.insert(0, str(ga.POP_SIZE))
    pop_entry.pack()

    gen_frame = tk.Frame(label_frame)
    gen_frame.grid(row=1, column=3, padx=10, pady=10)
    gen_label = tk.Label(gen_frame, text="Generations:", font=("Arial", 12))
    gen_label.pack(pady=6)
    gen_entry = tk.Entry(gen_frame, font=("Arial", 12), width=25, validate="key", validatecommand=ivcmd)
    gen_entry.insert(0, str(ga.GENERATIONS))
    gen_entry.pack()

    # Dome variant selector
    variant_frame = tk.Frame(label_frame)
    variant_frame.grid(row=0, column=4, padx=10, pady=10)
    variant_label = tk.Label(variant_frame, text="Dome variant:", font=("Arial", 12))
    variant_label.pack(pady=6)
    variant_var = tk.StringVar(value=ga.DOME_VARIANT)
    variant_open = tk.Radiobutton(variant_frame, text="Open (no apex)", variable=variant_var, value="open")
    variant_full = tk.Radiobutton(variant_frame, text="Full (closed)", variable=variant_var, value="full")
    variant_open.pack()
    variant_full.pack()

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
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    dome_figure = plt.figure(figsize=(5, 5))
    dome_axis = dome_figure.add_subplot(111, projection="3d")
    dome_axis.set_title("Best Dome (current generation)")

    dome_canvas = FigureCanvasTkAgg(dome_figure, master=plot_frame)
    dome_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    run_button: tk.Button
    gui_button: tk.Checkbutton

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
            dome_canvas.get_tk_widget().destroy()
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

    def redraw(history, best_genome=None):
        if not is_open():
            return
        ga.draw_history(history, axis)
        canvas.draw_idle()
        if best_genome is not None:
            ga.draw_genome(best_genome, dome_axis)
            dome_canvas.draw_idle()

    def toggle_gui():
        if enable_gui.get():
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            dome_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        else:
            canvas.get_tk_widget().pack_forget()
            dome_canvas.get_tk_widget().pack_forget()

    def append_output(line):
        if not is_open():
            return
        write_output(line + "\n")

    def finish_run(status):
        if not is_open():
            return
        run_button.config(state=tk.NORMAL)
        status_var.set(status)

    def start():
        radius = float(radius_entry.get())
        height = float(height_entry.get())
        min_thick = float(min_thick_entry.get())
        max_thick = float(max_thick_entry.get())
        min_offset = float(min_offset_entry.get())
        max_offset = float(max_offset_entry.get())
        pop_size = int(pop_entry.get())
        generations = int(gen_entry.get())
        variant = variant_var.get()

        run_button.config(state=tk.DISABLED)
        status_var.set("Running GA...")
        write_output("", clear=True)

        output_stream = StreamRedirector(lambda line: safe_after(0, lambda l=line: append_output(l)))

        def worker():
            try:
                with contextlib.redirect_stdout(output_stream):
                    print(f"Radius: {radius}, Height: {height}")
                    ga.set_params(radius, height, min_thick, max_thick, min_offset, max_offset,
                                  pop_size=pop_size, generations=generations, variant=variant)
                    if enable_gui:
                        ga.run_ga(progress_callback=lambda snapshot, genome: safe_after(0, lambda data=snapshot, g=genome: redraw(data, g)))
                    else:
                        ga.run_ga()
            except Exception:
                pass
            finally:
                if output_stream is not None:
                    output_stream.flush()
                safe_after(0, finish_run)

        threading.Thread(target=worker, daemon=True).start()

    # Bottom control frame keeps buttons visible and accessible
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

    run_button = tk.Button(bottom_frame, text="Calculate", font=("Arial", 12), command=start)
    run_button.pack(side=tk.LEFT, padx=20)

    gui_button = tk.Checkbutton(bottom_frame, text="Show GUI", variable=enable_gui, command=toggle_gui)
    gui_button.pack(side=tk.LEFT, padx=10)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()