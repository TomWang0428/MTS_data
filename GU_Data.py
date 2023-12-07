import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import math
import tkinter as tk
from tkinter import ttk
from scipy import stats
import pandas as pd
from tkinter import PhotoImage, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def read_file(folder_path):
    crosshead = []
    load = []
    time = []
    folder_name = os.listdir(folder_path)
    folder_path = os.path.join(folder_path, str(folder_name[0]))
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            read_flag = 0
            with open(file_path, 'r') as f:
                for line in f:
                    if read_flag ==1:
                        line = line.strip()
                        parts = line.split('\t')
                        crosshead.append(parts[0])
                        load.append(parts[1])
                        time.append(parts[2])

                    if line == "mm	N	sec\n":
                        read_flag = 1
    return [float(i) for i in crosshead], [float(i) for i in load], [float(i) for i in time]

def sort_data(ls1, ls2):
    combined = sorted(zip(ls1, ls2))

    # Unzip the sorted pairs
    sorted_list1, sorted_list2 = zip(*combined)

    # Convert the tuples back to lists, if necessary
    sorted_list1 = list(sorted_list1)
    sorted_list2 = list(sorted_list2)
    return sorted_list1, sorted_list2


class comm:
    def __init__(self, crosshead, load, time, test_type, height, area, name, dirc):
        """
        initial the class

        **Parameters**
            grid_data: *list*
                list of the grid

        **Returns**
           None
        """
        self.crosshead = crosshead
        self.load = load
        self.time = time
        self.test_type = test_type
        self.height = height
        self.area = area
        self.name = name
        self.dirc = dirc
    
def plot_fig(ftpr):
    global current_path
    current_path = ftpr
    path = os.path.join(ftpr,all_test_name[test_id] + ".png")
    update_left_frame()
    img = Image.open(path)
    img_tk = ImageTk.PhotoImage(img)
    label = Label(left_frame, image=img_tk)
    label.image = img_tk  # Keep a reference!
    label.pack()



class Data:
    def __init__(self, crosshead, load, time, test_type, height, area, name, dirc):
        """
        initial the class

        **Parameters**
            grid_data: *list*
                list of the grid

        **Returns**
           None
        """
        self.crosshead = crosshead
        self.load = load
        self.time = time
        self.test_type = test_type
        self.height = height
        self.area = area
        self.name = name
        self.dirc = dirc

    def crosshead_load_graph(self):
        x = self.crosshead
        y = self.load
        fig, ax = plt.subplots()
        fig_name = self.name + " crosshead VS load"
        ax.plot(x, y, label=fig_name)
        plt.title(fig_name)
        plt.xlabel("Crosshead(mm)")
        plt.ylabel("Load(N)")
        folder_path = os.path.join(self.dirc, "Figure", "crosshead VS load")
        try:
            os.makedirs(folder_path, exist_ok=True)
        except:
            pass
        if self.test_type == 'dc':
            plasticity, zero_1, zero_2 = self.double_comp()
            ax.plot(x[zero_1], y[zero_1], 'ro', markersize=10)
            ax.plot(x[zero_2], y[zero_2], 'ro', markersize=10)
            fig.savefig(os.path.join(folder_path, self.name + ".png"))
            plt.close(fig)
            return plasticity
        if self.test_type == 'sc':
            fit_start, fit_end, slop, intercept, half_time = self.single_comp()
            x_fit = [i for i in x if i >= x[fit_start] and i <=x[fit_end]]
            y_fit = [a * slop + intercept for a in x_fit]
            ax.plot(x_fit, y_fit, 'r', markersize=5)
            fig.savefig(os.path.join(folder_path, self.name + ".png"))
            plt.close(fig)
            modulus = slop * self.height / self.area
            return modulus, half_time

    def time_load_graph(self):
        x = self.time
        y = self.load
        fig, ax = plt.subplots()
        fig_name = self.name + " time VS load"
        ax.plot(x, y, label=fig_name)
        plt.title(fig_name)
        plt.xlabel("Time(s)")
        plt.ylabel("Load(N)")
        folder_path = os.path.join(self.dirc, "Figure", "time VS load")
        try:
            os.makedirs(folder_path, exist_ok=True)
        except:
            pass
        fig.savefig(os.path.join(folder_path, self.name + ".png"))
        plt.close(fig)

    def time_crosshead_graph(self):
        x = self.time
        y = self.crosshead
        fig, ax = plt.subplots()
        fig_name = self.name + " time VS crosshead"
        ax.plot(x, y, label=fig_name)
        plt.title(fig_name)
        plt.xlabel("Time(s)")
        plt.ylabel("corsshead(mm)")
        folder_path = os.path.join(self.dirc, "Figure", "time VS crosshead")
        try:
            os.makedirs(folder_path, exist_ok=True)
        except:
            pass
        fig.savefig(os.path.join(folder_path, self.name + ".png"))
        plt.close(fig)

    def double_comp(self):
        zero_pos = []
        for i in range(len(self.load)):
            if self.load[i-1] < 0 and self.load[i] >= 0:
                zero_pos.append(i)
        zero_time = [self.time[i] for i in zero_pos]
        zero_1 = self.time.index(min(zero_time))
        del zero_time[0]
        zero_2 = self.time.index(min(zero_time))
        plasticity = self.crosshead[zero_2] - self.crosshead[zero_1]
        return plasticity, zero_1, zero_2

    def single_comp(self):
        half_time = []
        for i in range(len(self.load)):
            if self.load[i - 1] < 0 and self.load[i] >= 0:
                start = i
                break
        load_applied = max(self.load) - self.load[start]
        thr = 0.1*load_applied
        thr_2 = 0.3*load_applied
        thr_3 = load_applied/2
        for i in range(len(self.load)):
            if self.load[i - 1] < thr and self.load[i] >= thr:
                fit_start = i
            if self.load[i - 1] < thr_2 and self.load[i] >= thr_2:
                fit_end = i
            if self.load[i - 1] < thr_3 and self.load[i] >= thr_3 and i > self.load.index(max(self.load)):
                half_time = self.time[i]
        x = [i for i in self.crosshead if i >= self.crosshead[fit_start] and i <= self.crosshead[fit_end]]
        y = [i for i in self.load if i >= self.load[fit_start] and i <= self.load[fit_end]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return fit_start, fit_end, slope, intercept, half_time

def increase_test_id():
    global test_id, total_n
    test_id += 1
    if test_id > total_n - 1:
        test_id = 0
    update_left_frame()
    plot_fig(current_path)

def decrease_test_id():
    global test_id, total_n
    test_id -= 1
    if test_id < 0:
        test_id = total_n - 1
    update_left_frame()
    plot_fig(current_path)

def update_left_frame():
    for widget in left_frame.winfo_children():
        widget.destroy()
def update_left_top_frame(t_ftpr):
    for widget in left_top_frame.winfo_children():
        widget.destroy()
    button_width = 15
    folder_path1 = os.path.join(t_ftpr, "Figure", "crosshead VS load")
    folder_path2 = os.path.join(t_ftpr, "Figure", "time VS load")
    folder_path3 = os.path.join(t_ftpr, "Figure", "time VS crosshead")
    button1 = tk.Button(left_top_frame, text="Crosshead VS Load", command=lambda: plot_fig(folder_path1), width=button_width)

    button2 = tk.Button(left_top_frame, text="Time VS Load", command=lambda: plot_fig(folder_path2), width=button_width)

    button3 = tk.Button(left_top_frame, text="Time VS Crosshead", command=lambda: plot_fig(folder_path3), width=button_width)

    prev_button = tk.Button(left_top_frame, text="Previous", command=lambda: decrease_test_id(), width=button_width)

    next_button = tk.Button(left_top_frame, text="Next", command=lambda: increase_test_id(), width=button_width)


    button1.grid(row=0, column=2, pady=(3, 30))
    button2.grid(row=0, column=3, pady=(3, 30))
    button3.grid(row=0, column=4, pady=(3, 30))
    prev_button.grid(row=2, column=0)
    next_button.grid(row=2, column=6)

def update_right_frame1(sc_df):
    global table
    for widget in right_frame1.winfo_children():
        widget.destroy()
    columns = list(sc_df.columns)
    frame_width = right_frame1.winfo_width()
    col_width = frame_width // 3

    table = ttk.Treeview(right_frame1, columns=columns, show="headings")

    # Define headings based on DataFrame columns
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=col_width, anchor='center')
    for index, row in sc_df.iterrows():
        table.insert("", "end", values=list(row))
    table.bind("<ButtonRelease-1>", on_row_click)

    scrollbar = ttk.Scrollbar(right_frame1, orient="vertical", command=table.yview)
    table.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    scrollbar = ttk.Scrollbar(right_frame1, orient="horizontal", command=table.xview)
    table.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="bottom", fill="x")
    # Pack the table last so it fills the remaining space
    table.pack(fill="both", expand=True)

    style = ttk.Style()
    style.configure("Treeview",
                    background="silver",
                    foreground="black",
                    rowheight=25,
                    fieldbackground="light grey")

    style.map('Treeview',
              background=[('selected', 'green')])

def update_right_frame2(dc_df):
    frame_width = right_frame2.winfo_width()
    col_width = frame_width // 2
    for widget in right_frame2.winfo_children():
        widget.destroy()

    columns = list(dc_df.columns)
    # Create the treeview
    table = ttk.Treeview(right_frame2, columns=columns, show="headings")

    # Define headings based on DataFrame columns
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=col_width, anchor='center')

    for index, row in dc_df.iterrows():
        table.insert("", "end", values=list(row))

    scrollbar = ttk.Scrollbar(right_frame2, orient="vertical", command=table.yview)
    table.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    scrollbar = ttk.Scrollbar(right_frame2, orient="horizontal", command=table.xview)
    table.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="bottom", fill="x")

    table.pack(fill="both", expand=True)
    style = ttk.Style()
    style.configure("Treeview",
                    background="silver",
                    foreground="black",
                    rowheight=25,
                    fieldbackground="light grey")
    style.map('Treeview',
              background=[('selected', 'green')])

def export_to_excel():
    with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
        sc_df.to_excel(writer, sheet_name='SC Data')
        dc_df.to_excel(writer, sheet_name='DC Data')

def save_values():
    global sc_df, dc_df, test_id, total_n, all_test_name
    test_id = 0
    total_n = 0
    t_ftpr = entry1.get()
    height = entry2.get()
    r = entry3.get()
    plasticity = []
    modulus = []
    half_time = []
    area = 2 * math.pi ** int(r)
    dc_name = []
    sc_name = []
    all_test_name = []
    for filename in os.listdir(t_ftpr):
        if filename != "Figure":
            total_n += 1
            ftpr = os.path.join(t_ftpr, filename)
            if ftpr[len(ftpr) - 1] == 'c' and ftpr[len(ftpr) - 2] == 'd':
                test_type = 'dc'
            else:
                test_type = 'sc'
            c, l, t = read_file(ftpr)
            time, cross = sort_data(t, c)
            time, load = sort_data(t, l)
            data = Data(cross, load, time, test_type, float(height), area, filename, t_ftpr)
            data.time_load_graph()
            data.time_crosshead_graph()
            if test_type == 'dc':
                plasticity.append(data.crosshead_load_graph())
                dc_name.append(filename)
            else:
                m, hal = data.crosshead_load_graph()
                modulus.append(m)
                half_time.append(hal)
                sc_name.append(filename)
            all_test_name.append(filename)
    sc_data = {
        "Test_Name" : sc_name,
        "Modulus" : modulus,
        "Half_time" : half_time
    }
    sc_df = pd.DataFrame(sc_data)
    dc_data = {
        "Test_Name": dc_name,
        "Plasticity": plasticity
    }
    dc_df = pd.DataFrame(dc_data)
    update_left_top_frame(t_ftpr)
    update_right_frame1(sc_df)
    update_right_frame2(dc_df)


root = tk.Tk()
root.title("MTS Data Analyzer")
root.state('zoomed')
root.geometry("1500x700")

top_frame = tk.Frame(root)
top_frame.place(x=1300, y=0, width=100, height=200)
label = tk.Label(top_frame, text="Enter dir:")
label.pack()
entry1 = tk.Entry(top_frame)
entry1.pack()
label = tk.Label(top_frame, text="Enter height:")
label.pack()
entry2 = tk.Entry(top_frame)
entry2.pack()
label = tk.Label(top_frame, text="Enter radius:")
label.pack()
entry3 = tk.Entry(top_frame)
entry3.pack()

button = tk.Button(top_frame, text="Submit", command=save_values)
button.pack()
export_button = tk.Button(top_frame, text="Export to Excel", command=export_to_excel)
export_button.pack()


left_frame = tk.Frame(root)
left_frame.place(x=0, y=200, width=800, height=600)

left_top_frame = tk.Frame(root)
left_top_frame.place(x=100, y=100, width=700, height=100)

right_frame1 = tk.Frame(root)
right_frame1.place(x=800, y=200, width=420, height=500)

right_frame2 = tk.Frame(root)
right_frame2.place(x=1220, y=200, width=280, height=500)



root.mainloop()

