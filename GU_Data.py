import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import math
import tkinter as tk
from tkinter import ttk
from scipy import stats
import pandas as pd
from tkinter import Label


def read_file(folder_path):
    """
        read in the datas based on the folder path

        **Parameters**
            folder_path: *string*
                The folder_path to read from

        **Returns**
           crosshead: *list*
                The crosshead data converted from strings to floats
           load: *list*
                The load data converted from strings to floats
           time: *list*
                The time data converted from strings to floats
    """
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
                    if read_flag == 1:
                        line = line.strip()
                        parts = line.split('\t')
                        crosshead.append(parts[0])
                        load.append(parts[1])
                        time.append(parts[2])

                    if line == "mm	N	sec\n":
                        read_flag = 1
    return [float(i) for i in crosshead], [float(i) for i in load], [float(i) for i in time]


def sort_data(ls1, ls2):
    """
        Sort data based on ls1 and change ls2 accordingly

        **Parameters**
            ls1: *list*
                The list to sort in ascending order
            ls2: *list*
                The list to sort base on order of ls1

        **Returns**
           sorted_list1: *list*
                The sorted ls1
           sorted_list2: *list*
                The sorted ls2
    """
    combined = sorted(zip(ls1, ls2))

    # Unzip the sorted pairs
    sorted_list1, sorted_list2 = zip(*combined)

    # Convert the tuples back to lists, if necessary
    sorted_list1 = list(sorted_list1)
    sorted_list2 = list(sorted_list2)
    return sorted_list1, sorted_list2


class comm:
    """
        The class that includes most of commands for the GUI
    """

    def __init__(self, ftpr, current_path, t_ftpr, test_id, total_n, all_test_name):
        """
            initial the class

            **Parameters**
                ftpr: *string*
                    The directory of the folder
                current_path: *string*
                    The current figure plotted
                t_ftpr: *string*
                    The directory for the current figure plotted
                test_id: *int*
                    The id number for the current figure plotted
                total_n: *int*
                    The total number of figures
                all_test_name: *list*
                    List of all test names

            **Returns**
               None
        """
        self.ftpr = ftpr
        self.t_ftpr = t_ftpr
        self.current_path = current_path
        self.test_id = test_id
        self.total_n = total_n
        self.all_test_name = all_test_name

    def plot_fig(self):
        """
            Open figure based on the current test_id

            **Parameters**
                None

            **Returns**
               None
        """
        path = os.path.join(self.ftpr, self.all_test_name[self.test_id] + ".png")
        self.update_left_frame()
        img = Image.open(path)
        img_tk = ImageTk.PhotoImage(img)
        label = Label(left_frame, image=img_tk)
        label.image = img_tk  # Keep a reference!
        label.pack()

    def increase_test_id(self):
        """
            increase the test id by one when next bottom is pressed

            **Parameters**
                None

            **Returns**
               None
        """
        self.test_id += 1
        if self.test_id > self.total_n - 1:
            self.test_id = 0
        self.update_left_frame()
        self.plot_fig()

    def decrease_test_id(self):
        """
            decrease the test id by one when next bottom is pressed

            **Parameters**
                None

            **Returns**
               None
        """
        self.test_id -= 1
        if self.test_id < 0:
            self.test_id = self.total_n - 1
        self.update_left_frame()
        self.plot_fig()

    def update_left_frame(self):
        """
            Update the upper_left_frame by empty it

            **Parameters**
                None

            **Returns**
               None
        """
        for widget in left_frame.winfo_children():
            widget.destroy()

    def go_path(self, path):
        """
            change the syntext of the path when the first figure is plotted

            **Parameters**
                None

            **Returns**
               None
        """
        self.ftpr = path
        self.plot_fig()

    def update_left_top_frame(self):
        """
            Update the upper_left_top_frame by adding bottoms

            **Parameters**
                None

            **Returns**
               None
        """
        t_ftpr = self.t_ftpr
        for widget in left_top_frame.winfo_children():
            widget.destroy()
        button_width = 15
        folder_path1 = os.path.join(t_ftpr, "Figure", "crosshead VS load")
        folder_path2 = os.path.join(t_ftpr, "Figure", "time VS load")
        folder_path3 = os.path.join(t_ftpr, "Figure", "time VS crosshead")
        self.ftpr = folder_path1
        button1 = tk.Button(left_top_frame, text="Crosshead VS Load", command=lambda: self.go_path(folder_path1),
                            width=button_width)

        button2 = tk.Button(left_top_frame, text="Time VS Load", command=lambda: self.go_path(folder_path2),
                            width=button_width)

        button3 = tk.Button(left_top_frame, text="Time VS Crosshead", command=lambda: self.go_path(folder_path3),
                            width=button_width)

        prev_button = tk.Button(left_top_frame, text="Previous", command=lambda: self.decrease_test_id(),
                                width=button_width)

        next_button = tk.Button(left_top_frame, text="Next", command=lambda: self.increase_test_id(),
                                width=button_width)

        button1.grid(row=0, column=2, pady=(3, 30))
        button2.grid(row=0, column=3, pady=(3, 30))
        button3.grid(row=0, column=4, pady=(3, 30))
        prev_button.grid(row=2, column=0)
        next_button.grid(row=2, column=6)

    def update_right_frame1(self):
        """
            Update the right_frame1 by single compression table

            **Parameters**
                None

            **Returns**
               None
        """
        global sc_df
        for widget in right_frame1.winfo_children():
            widget.destroy()
        columns = list(sc_df.columns)
        frame_width = right_frame1.winfo_width()
        col_width = frame_width // len(columns)

        self.tree1 = ttk.Treeview(right_frame1, columns=columns, show="headings")
        for col in columns:
            self.tree1.heading(col, text=col)
            self.tree1.column(col, width=col_width, anchor='center')
        for index, row in sc_df.iterrows():
            self.tree1.insert("", "end", values=list(row))

        self.tree1.bind("<Double-1>", self.on_row_double_click1)

        scrollbar = ttk.Scrollbar(right_frame1, orient="vertical", command=self.tree1.yview)
        self.tree1.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        scrollbar = ttk.Scrollbar(right_frame1, orient="horizontal", command=self.tree1.xview)
        self.tree1.configure(xscrollcommand=scrollbar.set)
        scrollbar.pack(side="bottom", fill="x")
        self.tree1.pack(fill="both", expand=True)

    def update_right_frame2(self):
        """
            Update the right_frame2 by double compression table

            **Parameters**
                None

            **Returns**
               None
        """
        global dc_df
        for widget in right_frame2.winfo_children():
            widget.destroy()
        columns = list(dc_df.columns)
        frame_width = right_frame2.winfo_width()
        col_width = frame_width // len(columns)

        self.tree = ttk.Treeview(right_frame2, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=col_width, anchor='center')
        for index, row in dc_df.iterrows():
            self.tree.insert("", "end", values=list(row))

        self.tree.bind("<Double-1>", self.on_row_double_click)

        scrollbar = ttk.Scrollbar(right_frame2, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        scrollbar = ttk.Scrollbar(right_frame2, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=scrollbar.set)
        scrollbar.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

    def on_row_double_click(self, event):
        """
            Double click command to jump to the figure based on the table information clicked for dc_df

            **Parameters**
                None

            **Returns**
               None
        """
        selected_items = self.tree.selection()
        if not selected_items:
            return  # Exit the method if no item is selected

        item = selected_items[0]  # Get selected item
        test_name = self.tree.item(item, 'values')[0]  # Extract the name or identifier of the test
        self.test_id = self.all_test_name.index(test_name)

        # Update the left frame with this figure
        self.plot_fig()

    def on_row_double_click1(self, event):
        """
            Double click command to jump to the figure based on the table information clicked for dc_df

            **Parameters**
                None

            **Returns**
               None
        """
        selected_items = self.tree1.selection()
        if not selected_items:
            return  # Exit the method if no item is selected

        item = selected_items[0]  # Get selected item
        test_name = self.tree1.item(item, 'values')[0]  # Extract the name or identifier of the test

        # Construct the path for the corresponding figure
        self.test_id = self.all_test_name.index(test_name)

        # Update the left frame with this figure
        self.plot_fig()


class Data:
    """
        The class for test datas and data processing
    """

    def __init__(self, crosshead, load, time, test_type, height, area, name, dirc):
        """
        initial the class

        **Parameters**
            crosshead: *list*
                list of the crosshead data
            load: *list*
                list of the load data
            time: *list*
                list of the time data
            test_type: *string*
                The test type (single compression or double compression)
            height: *int*
                The height of the sample
            area: *int*
                The area of the sample
            name: *string*
                The test name
            dirc: *string*
                The directory of the test

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
        """
            To plot the crosshead VS load graph and find young's modulus and halftime if single compression, plasticity
            if double compression

            **Parameters**
                None

            **Returns**
            plasticity: *float*
                The plasticity of the sample
            modulus: *float*
                The young's modulus of the sample
            half_time: *float*
                The half_time for the sample
        """
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
            x_fit = [i for i in x if i >= x[fit_start] and i <= x[fit_end]]
            y_fit = [a * slop + intercept for a in x_fit]
            ax.plot(x_fit, y_fit, 'r', markersize=5)
            fig.savefig(os.path.join(folder_path, self.name + ".png"))
            plt.close(fig)
            modulus = slop * self.height / self.area
            return modulus, half_time

    def time_load_graph(self):
        """
            To plot the time VS load graph

            **Parameters**
                None

            **Returns**
                None
        """
        x = self.time
        y = self.load
        fig, ax = plt.subplots()
        fig_name = self.name + " time VS load"
        ax.plot(x, y, label=fig_name)
        plt.title(fig_name)
        plt.xlabel("Time(s)")
        plt.ylabel("Load(N)")
        folder_path = os.path.join(self.dirc, "Figure", "time VS load")
        os.makedirs(folder_path, exist_ok=True)
        fig.savefig(os.path.join(folder_path, self.name + ".png"))
        plt.close(fig)

    def time_crosshead_graph(self):
        """
            To plot the time VS crosshead graph

            **Parameters**
                None

            **Returns**
                None
        """
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
        """
            Data processing for double compression

            **Parameters**
                None

            **Returns**
                plasticity: *float*
                    The plasticity of sample
                zero_1: *float*
                    The first time load reaches 0
                zero_2: *float*
                    The second time load reaches 0
        """
        zero_pos = []
        for i in range(len(self.load)):
            if self.load[i - 1] < 0 and self.load[i] >= 0:
                zero_pos.append(i)
        zero_time = [self.time[i] for i in zero_pos]
        zero_1 = self.time.index(min(zero_time))
        del zero_time[0]
        zero_2 = self.time.index(min(zero_time))
        plasticity = self.crosshead[zero_2] - self.crosshead[zero_1]
        return plasticity, zero_1, zero_2

    def single_comp(self):
        """
            Data processing for single compression, fitting for the young's modulus

            **Parameters**
                None

            **Returns**
                fit_start: *float*
                    The start position of the fit
                fit_end: *float*
                    The end position of the fit
                slope: *float*
                    The young's modulus of the sample, or the slop for the fit
                intercept: *float*
                    The intercept for the fit
                half_time: *float*
                    The half_time for the sample
        """
        half_time = []
        for i in range(len(self.load)):
            if self.load[i - 1] < 0 and self.load[i] >= 0:
                start = i
                break
        load_applied = max(self.load) - self.load[start]
        thr = 0.1 * load_applied
        thr_2 = 0.3 * load_applied
        thr_3 = load_applied / 2
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


def export_to_excel():
    """
        Export data into an Excel file

        **Parameters**
            None

        **Returns**
           None
    """
    global sc_df, dc_df
    with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
        sc_df.to_excel(writer, sheet_name='SC Data')
        dc_df.to_excel(writer, sheet_name='DC Data')


def save_values():
    """
        Save the inputted values and call for the processing commands

        **Parameters**
            None

        **Returns**
            None
    """
    global sc_df, dc_df
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
    current_path = []
    ftpr = []
    contr = comm(ftpr, current_path, t_ftpr, test_id, total_n, all_test_name)
    for filename in os.listdir(t_ftpr):
        if filename != "Figure":
            total_n += 1
            ftpr = os.path.join(t_ftpr, filename)
            contr.ftpr = ftpr
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
        "Test_Name": sc_name,
        "Modulus": modulus,
        "Half_time": half_time
    }
    sc_df = pd.DataFrame(sc_data)
    dc_data = {
        "Test_Name": dc_name,
        "Plasticity": plasticity
    }
    dc_df = pd.DataFrame(dc_data)

    contr.update_left_top_frame()
    contr.update_right_frame1()
    contr.update_right_frame2()

    contr.total_n = total_n
    contr.all_test_name = all_test_name


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
