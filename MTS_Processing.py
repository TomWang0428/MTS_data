"""
    *MTS_data analyzer*
    Date: Dec-09-2023
    This program analyze data collected by the MTS machine and give out a GUI for easy control.
    Contributors:
        Mengke Wang
"""

import os
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import pickle
import tkinter as tk
from tkinter import ttk
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
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
    times = []
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
                        times.append(parts[2])

                    if line == "mm	N	sec\n":
                        read_flag = 1
    return [float(i) for i in crosshead], [float(i) for i in load], [float(i) for i in times]

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

def on_click(event):
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata


def create_popup():
    # Create a new Tkinter window
    popup = tk.Tk()

    # Set the title of the window
    popup.title("Notification")

    # Set the window size
    popup.geometry("300x100")

    # Create a label with a message
    message_label = tk.Label(popup, text="Export Successful", font=("Helvetica", 12))
    message_label.pack(pady=10)

    # Create a close button
    close_button = tk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack(pady=5)

    # Run the event loop
    popup.mainloop()


def bilateral_filter_1d(signal, radius, sigma_r, sigma_d):
    """
    Applies a Bilateral filter to a 1D signal.

    :param signal: The 1D input signal (numpy array).
    :param radius: The radius of the kernel (int).
    :param sigma_r: The standard deviation for intensity differences.
    :param sigma_d: The standard deviation for spatial distance.
    :return: The filtered 1D signal.
    """
    filtered_signal = np.zeros(signal.shape)

    for i in range(len(signal)):
        # Initialize weighted sum and normalization factor
        weighted_sum = 0
        Wp = 0

        # Apply the filter kernel over the window centered around the current element
        for k in range(-radius, radius + 1):
            # Handle boundary conditions by mirroring
            ik = min(max(i + k, 0), len(signal) - 1)

            # Compute the range weight (intensity difference)
            range_kernel = np.exp(-((signal[ik] - signal[i]) ** 2) / (2 * sigma_r ** 2))

            # Compute the spatial weight
            spatial_kernel = np.exp(-(k ** 2) / (2 * sigma_d ** 2))

            # Update weighted sum and normalization factor
            weighted_sum += signal[ik] * range_kernel * spatial_kernel
            Wp += range_kernel * spatial_kernel

        # Compute the filtered value
        filtered_signal[i] = weighted_sum / Wp

    return filtered_signal

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
        self.ftpr_cl = []

    def plot_fig(self):
        path = os.path.join(self.ftpr_cl, self.all_test_name[self.test_id] + ".pkl")
        self.update_left_frame()  # Clear the frame or update it as needed
        self.update_Top_frame()
        # Load the Matplotlib figure from the pickle file
        with open(path, 'rb') as file:
            fig = pickle.load(file)

        # Display the figure using FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=left_frame)  # Embedding the figure in the 'left_frame'
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add the navigation toolbar (optional)
        toolbar = NavigationToolbar2Tk(canvas, left_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Connect the on_click function to the figure's button_press_event
        fig.canvas.mpl_connect('button_press_event', on_click)


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
        self.ftpr_cl = path
        self.plot_fig()

    def redo(self):
        global ft_start, ft_end, left_entry, right_entry, redo_flag, glb_area, glb_ftpr, sc_df
        ft_start[self.test_id] = left_entry.get()
        ft_end[self.test_id] = right_entry.get()
        redo_flag = 1
        all_files = os.listdir(glb_ftpr)
        all_files = [x for x in all_files if x != 'Figure']
        modulus, _, _, _, _, _ = sig_run(glb_area, self.test_id, glb_ftpr, all_files[self.test_id], self, [], [], [], [], [],[])

        indices = sc_df[sc_df['Test_Name'] == self.all_test_name[self.test_id]].index.tolist()
        sc_df.iloc[indices, 1] = modulus
        self.plot_fig()
        self.update_right_frame1()


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
        folder_path0 = os.path.join(t_ftpr, "Figure", "Strain VS Stress")
        folder_path1 = os.path.join(t_ftpr, "Figure", "crosshead VS load")
        folder_path2 = os.path.join(t_ftpr, "Figure", "time VS load")
        folder_path3 = os.path.join(t_ftpr, "Figure", "time VS crosshead")
        self.ftpr_cl = folder_path0
        button0 = tk.Button(left_top_frame, text="Strain VS Stress", command=lambda: self.go_path(folder_path0),
                            width=button_width)

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

        button0.grid(row=0, column=1, pady=(3, 30))
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

    def update_Top_frame(self):
        """
            Update the top_frame

            **Parameters**
                None

            **Returns**
               None
        """
        global ft_start, ft_end, left_entry, right_entry
        for widget in top_frame.winfo_children():
            widget.destroy()

        label1 = tk.Label(top_frame, text="x1:")
        left_entry = tk.Entry(top_frame, validate='key', width=20)
        left_entry.insert(0, ft_start[self.test_id])  # Insert the initial value into the Entry widget
        label2 = tk.Label(top_frame, text="x2:")
        right_entry = tk.Entry(top_frame, validate='key', width=20)
        right_entry.insert(0, ft_end[self.test_id])  # Insert the initial value into the Entry widget

        export_button = tk.Button(top_frame, text="Export to Excel", command=export_to_excel)
        redo_button = tk.Button(top_frame, text="Redo With New Value", command=self.redo)

        label1.grid(row=0, column=3, pady=(3, 30))
        left_entry.grid(row=0, column=5, pady=(3, 30))
        label2.grid(row=1, column=3, pady=(3, 30))
        right_entry.grid(row=1, column=5, pady=(3, 30))
        export_button.grid(row=2, column=6, pady=(3, 30))
        redo_button.grid(row=2, column=5, pady=(3, 30))


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

    def __init__(self, crosshead, load, times, test_type, height, area, name, dirc):
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
        self.times = times
        self.test_type = test_type
        self.height = height
        self.area = area
        self.name = name
        self.dirc = dirc
        self.real_hight = 0
        self.filtered_load = []
        self.start = 0

    def power_line_filter(self):
        radius = 10
        sigma_r = 0.05
        sigma_d = 5.0
        numpy_array = np.array(self.load)
        self.filtered_load = bilateral_filter_1d(numpy_array, radius, sigma_r, sigma_d)
    """
        fs = round(1/(sum(np.diff(self.times))/(len(self.times)-1)))  # Sampling frequency, in Hz
        f0 = 1
        order = 3
        nyq = 0.5 * fs
        normal_cutoff = f0 / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        self.filtered_load = lfilter(b, a, self.load)

        fft_result = np.fft.fft(self.load)
        n = len(self.load)  # Length of the signal
        freq_bins = np.fft.fftfreq(n, d=1 / fs)

        magnitude = 20 * np.log10(np.abs(fft_result))

        # Plot the spectrum
        plt.figure(figsize=(12, 6))
        plt.plot(freq_bins[:n // 2], magnitude[:n // 2])  # Plot only the positive frequencies
        plt.xlim(0,3)
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid()
        plt.show()
    """


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
        global ft_start, ft_end
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
        with open(os.path.join(folder_path, self.name + ".pkl"), 'wb') as file:
            pickle.dump(fig, file)

    def time_load_graph(self):
        """
            To plot the time VS load graph

            **Parameters**
                None

            **Returns**
                None
        """
        hal_index = 0
        self.power_line_filter()
        x = self.times
        y = self.filtered_load
        fig, ax = plt.subplots()
        fig_name = self.name + " time VS load"
        ax.plot(x, y, label=fig_name)
        plt.title(fig_name)
        plt.xlabel("Time(s)")
        plt.ylabel("Load(N)")
        folder_path = os.path.join(self.dirc, "Figure", "time VS load")
        os.makedirs(folder_path, exist_ok=True)
        halftime = []
        if self.test_type == 'sc':
            thr = (max(self.load) - 0) / 2
            for i in range(len(self.load)):
                if self.load[i-1] < thr and self.load[i] > thr and i > self.load.index(max(self.load)):
                    hal_index = i
                    halftime = self.times[i] - self.times[self.load.index(max(self.load))]
                    ax.plot(x[hal_index], y[hal_index], 'ro', markersize=5)
                    break
            with open(os.path.join(folder_path, self.name + ".pkl"), 'wb') as file:
                pickle.dump(fig, file)

        plt.close(fig)
        return halftime

    def time_crosshead_graph(self):
        """
            To plot the time VS crosshead graph

            **Parameters**
                None

            **Returns**
                None
        """
        x = self.times
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
        with open(os.path.join(folder_path, self.name + ".pkl"), 'wb') as file:
            pickle.dump(fig, file)
        plt.close(fig)

    def Strain_Stress_graph(self):
        """
            To plot the Strain VS Stress graph

            **Parameters**
                None

            **Returns**
                None
        """
        if self.test_type == 'dc':
            zero_1, zero_2 = self.double_comp()
        if self.test_type == 'sc':
            fit_start, fit_end, slop, intercept = self.single_comp()
        x = [(float(self.real_hight) - element) / float(self.real_hight) for element in self.crosshead]
        y = [element / float(self.area) for element in self.load]
        fig, ax = plt.subplots()
        fig_name = self.name + " Strain VS Stress"
        ax.plot(x, y, label=fig_name)
        plt.title(fig_name)
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        folder_path = os.path.join(self.dirc, "Figure", "Strain VS Stress")
        try:
            os.makedirs(folder_path, exist_ok=True)
        except:
            pass
        if self.test_type == 'dc':
            ft_start.append(x[zero_1])
            ft_end.append(x[zero_2])
            plasticity = x[zero_2] - x[zero_1]
            ax.plot(x[zero_1], y[zero_1], 'ro', markersize=5)
            ax.plot(x[zero_2], y[zero_2], 'ro', markersize=5)
            with open(os.path.join(folder_path, self.name + ".pkl"), 'wb') as file:
                pickle.dump(fig, file)
            plt.close(fig)
            return plasticity
        if self.test_type == 'sc':
            ft_start.append(x[fit_start])
            ft_end.append(x[fit_end])
            x_fit = [i for i in x if i >= x[fit_start] and i <= x[fit_end]]
            y_fit = [a * slop + intercept for a in x_fit]
            ax.plot(x_fit, y_fit, 'r', markersize=5)
            with open(os.path.join(folder_path, self.name + ".pkl"), 'wb') as file:
                pickle.dump(fig, file)
            plt.close(fig)
            return slop
        with open(os.path.join(folder_path, self.name + ".pkl"), 'wb') as file:
            pickle.dump(fig, file)
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
                if len(zero_pos) == 1:
                    self.real_hight = self.crosshead[i]
        zero_cross = [self.crosshead[i] for i in zero_pos]
        zero_1 = min(zero_pos)
        del zero_cross[0]
        zero_2 = self.crosshead.index(min(zero_cross))
        return zero_1, zero_2

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
        global redo_flag, left_entry, right_entry
        start = 0
        if self.load[0] > 0:
            start_load = max(self.load[:10])
            self.load = [value - start_load for value in self.load]
        half_time = []
        for i in range(len(self.load)):
            if self.load[i - 1] < 0 and self.load[i] >= 0 and i > self.load.index(min(self.load)):
                start = i
                break
        self.real_hight = self.crosshead[start]
        x = [(float(self.real_hight) - element) / float(self.real_hight) for element in self.crosshead]
        y = [element / float(self.area) for element in self.load]
        crosshead_pos = x[y.index(max(y))] - x[start]
        self.start = start
        if redo_flag == 0:
            thr = 0.1 * crosshead_pos + x[start]
            thr_2 = 0.2 * crosshead_pos + x[start]
        else:
            thr = float(left_entry.get())
            thr_2 = float(right_entry.get())
            redo_flag = 0
        for i in range(len(y)):
            if x[i - 1] < thr and x[i] >= thr:
                fit_start = i
            if x[i - 1] < thr_2 and x[i] >= thr_2:
                fit_end = i
        fit_x = []
        fit_y = []
        for i in range(fit_start, fit_end):
            fit_x.append(x[i])
            fit_y.append(y[i])
        slope, intercept, r_value, p_value, std_err = stats.linregress(fit_x, fit_y)

        return fit_start, fit_end, slope, intercept


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
    create_popup()

def continue_execution():
    global app_state
    app_state = 'running'  # Change the application state to resume execution

# Create a 'Continue' button in the GUI

def sig_run(area, total_n, t_ftpr, filename, contr, plasticity, dc_name, modulus, half_time, sc_name, all_test_name):
    if type(area) != float:
        t_area = area[total_n]
    else:
        t_area = area
    total_n += 1
    ftpr = os.path.join(t_ftpr, filename)
    contr.ftpr = ftpr
    if ftpr[len(ftpr) - 1] == 'c' and ftpr[len(ftpr) - 2] == 'd':
        test_type = 'dc'
    else:
        test_type = 'sc'
    c, l, t = read_file(ftpr)
    times, cross = sort_data(t, c)
    times, load = sort_data(t, l)
    data = Data(cross, load, times, test_type, 0, t_area, filename, t_ftpr)

    if test_type == 'dc':
        plasticity.append(data.Strain_Stress_graph())
        hal = data.time_load_graph()
        dc_name.append(filename)
    else:
        slop = data.Strain_Stress_graph()
        modulus.append(slop)
        hal = data.time_load_graph()
        half_time.append(hal)
        sc_name.append(filename)

    all_test_name.append(filename)
    data.time_load_graph()
    data.time_crosshead_graph()
    data.crosshead_load_graph()
    return modulus, half_time, plasticity, sc_name, dc_name, total_n

def save_values():
    """
        Save the inputted values and call for the processing commands

        **Parameters**
            None

        **Returns**
            None
    """
    global sc_df, dc_df, app_state, ft_start, ft_end, redo_flag, glb_area, glb_ftpr
    redo_flag = 0
    app_state = 'paused'
    ft_start = []
    ft_end = []
    test_id = 0
    total_n = 0
    t_ftpr = entry1.get()
    area = entry3.get()
    plasticity = []
    modulus = []
    half_time = []
    dc_name = []
    sc_name = []
    all_test_name = []
    current_path = []
    ftpr = []
    contr = comm(ftpr, current_path, t_ftpr, test_id, total_n, all_test_name)
    all_files = os.listdir(t_ftpr)
    all_files = [x for x in all_files if x != 'Figure']

    if str.lower(area) == 'create':
        area_input = {
            "Test_Name": all_files,
            "Area(mm^2)": [None] * len(all_files)
        }
        area_input = pd.DataFrame(area_input)
        with pd.ExcelWriter('area_input.xlsx', engine='openpyxl') as writer:
            area_input.to_excel(writer, sheet_name='Area Data')
        info_label = tk.Label(top_frame, text="Please update 'area_input.xlsx' and click 'Continue' when done.")
        info_label.pack()
        while app_state == 'paused':
            root.update()  # This will keep the GUI responsive while waiting
            time.sleep(0.1)  # Adjust sleep time as needed to reduce CPU usage
        excel_file_path = 'area_input.xlsx'
        area_input = pd.read_excel(excel_file_path)
        area = area_input['Area(mm^2)']
    elif str.lower(area) == 'read':
        excel_file_path = 'area_input.xlsx'
        area_input = pd.read_excel(excel_file_path)
        area = area_input['Area(mm^2)']
    else:
        area = float(area)
    glb_area = area
    glb_ftpr = t_ftpr

    for filename in all_files:
        modulus, half_time, plasticity, sc_name, dc_name, total_n = sig_run(area, total_n, t_ftpr, filename, contr, plasticity, dc_name, modulus, half_time, sc_name, all_test_name)
    contr.update_Top_frame()
    contr = comm(ftpr, current_path, t_ftpr, test_id, total_n, all_test_name)
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
top_frame.place(x=800, y=0, width=700, height=200)
label = tk.Label(top_frame, text="Enter dir:")
label.pack()
entry1 = tk.Entry(top_frame)
entry1.pack()
label = tk.Label(top_frame, text="Enter Area (mm^2):")
label.pack()
entry3 = tk.Entry(top_frame)
entry3.pack()

button = tk.Button(top_frame, text="Submit", command=save_values)
button.pack()
continue_button = tk.Button(top_frame, text="Continue", command=continue_execution)
continue_button.pack()


left_frame = tk.Frame(root)
left_frame.place(x=0, y=200, width=800, height=600)

left_top_frame = tk.Frame(root)
left_top_frame.place(x=100, y=100, width=700, height=100)

right_frame1 = tk.Frame(root)
right_frame1.place(x=800, y=200, width=420, height=575)

right_frame2 = tk.Frame(root)
right_frame2.place(x=1220, y=200, width=280, height=575)

root.mainloop()
