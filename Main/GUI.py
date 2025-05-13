# import PySimpleGUI as sg
# import numpy as np
# import matplotlib.pyplot as plt
# from Main import Run
# sg.change_look_and_feel('DarkTeal9')    # look and feel theme
#
#
# # Designing layout
# layout = [[sg.Text("\t\t\tSelect_dataset"), sg.Combo(['Prostate MRI'],size=(13, 20)),sg.Text("\n")],
#           [sg.Text("\t\t\tSelect            "),sg.Combo(["TrainingData(%)","k-fold"], size=(13, 20)), sg.Text(""), sg.InputText(size=(10, 20), key='1'),sg.Button("START", size=(10, 2))],[sg.Text('\n')],
#           [sg.Text("\t\t\t   DCNN\t\t    Panoptic model\t\t    Focal-Net\t\t ResNet\t\t     Proposed HFGSO..")],
#           [sg.Text('\tAccuracy '), sg.In(key='11',size=(20,20)), sg.In(key='12',size=(20,20)), sg.In(key='13',size=(20,20)), sg.In(key='14',size=(20,20)),sg.In(key='15',size=(20,20)),sg.Text("\n")],
#           [sg.Text('\tSensitivity'), sg.In(key='21',size=(20,20)), sg.In(key='22',size=(20,20)), sg.In(key='23',size=(20,20)), sg.In(key='24',size=(20,20)),sg.In(key='25',size=(20,20)), sg.Text("\n")],
#           [sg.Text('\tSpecificity'), sg.In(key='31', size=(20, 20)), sg.In(key='32', size=(20, 20)),sg.In(key='33', size=(20, 20)), sg.In(key='34', size=(20, 20)),sg.In(key='35', size=(20, 20)), sg.Text("\n")],
#           [sg.Text('\t\t\t\t\t\t\t\t\t\t\t\t            '), sg.Button('Run Graph'), sg.Button('CLOSE')]]
#
#
# # to plot graphs
# def plot_graph(result_1, result_2, result_3):
#     plt.figure(dpi=120)
#     loc, result = [], []
#     result.append(result_1)  # appending the result
#     result.append(result_2)
#     result.append(result_3)
#     result = np.transpose(result)
#
#     # labels for bars
#     labels = ['DCNN', 'Panoptic model', 'Focal-Net','ResNet','Proposed HFGSO-based DRN ']  # x-axis labels ############################
#     tick_labels = ['Accuracy', 'Sensitivity','Specificity']  #### metrics
#     bar_width, s = 0.15, 0.025  # bar width, space between bars
#
#     for i in range(len(result)):  # allocating location for bars
#         if i == 0:  # initial location - 1st result
#             tem = []
#             for j in range(len(tick_labels)):
#                 tem.append(j + 1)
#             loc.append(tem)
#         else:  # location from 2nd result
#             tem = []
#             for j in range(len(loc[i - 1])):
#                 tem.append(loc[i - 1][j] + s + bar_width)
#             loc.append(tem)
#
#     # plotting a bar chart
#     for i in range(len(result)):
#         plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)
#
#     plt.legend(loc=(0.25, 0.25))# show a legend on the plot -- here legends are metrics
#     plt.show()  # to show the plot
#
#
# # Create the Window layout
# window = sg.Window('142705', layout)
#
# # event loop
# while True:
#     event, values = window.read()  # displays the window
#     if event == "START":
#         if values[1] == 'TrainingData(%)':
#             tp = int(values['1']) / 100
#         else:
#             tp = (int(values['1']) - 1) / int(values['1'])  # k-fold calculation
#         dataset, tr_per = values[0], tp
#         dts = dataset
#         Acc,Sen,Spe = Run.callmain(dts,tr_per)
#
#         window['11'].Update(Acc[0])
#         window['12'].Update(Acc[1])
#         window['13'].Update(Acc[2])
#         window['14'].Update(Acc[3])
#         window['15'].Update(Acc[4])
#
#         window['21'].Update(Sen[0])
#         window['22'].Update(Sen[1])
#         window['23'].Update(Sen[2])
#         window['24'].Update(Sen[3])
#         window['25'].Update(Sen[4])
#
#         window['31'].Update(Spe[0])
#         window['32'].Update(Spe[1])
#         window['33'].Update(Spe[2])
#         window['34'].Update(Spe[3])
#         window['35'].Update(Spe[4])
#
#     if event == 'Run Graph':
#         plot_graph(Acc,Sen,Spe)
#     if event == 'CLOSE':
#         break
#         window.close()
#
#
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from Main import Run

# Create the main window
root = tk.Tk()
root.title('142705')
root.configure(bg='#004d4d')  # Dark teal background similar to DarkTeal9

# Style configuration
style = ttk.Style()
style.configure('TFrame', background='#004d4d')
style.configure('TLabel', background='#004d4d', foreground='white')
style.configure('TButton', background='#004d4d', foreground='black')
style.configure('TCombobox', background='white', foreground='black')

# Main frame
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill='both', expand=True)

# Variables for storing data
dataset_var = tk.StringVar(value='Prostate MRI')
selection_var = tk.StringVar(value='TrainingData(%)')
input_var = tk.StringVar()
Acc, Sen, Spe = None, None, None

# Create result entry variables
result_entries = {}
for row_key, row_name in zip(['1', '2', '3'], ['Accuracy', 'Sensitivity', 'Specificity']):
    for col_key in range(1, 6):
        key = f'{row_key}{col_key}'
        result_entries[key] = tk.StringVar()

# Function to plot graphs
def plot_graph(result_1, result_2, result_3):
    plt.figure(dpi=120)
    loc, result = [], []
    result.append(result_1)
    result.append(result_2)
    result.append(result_3)
    result = np.transpose(result)

    labels = ['DCNN', 'Panoptic model', 'Focal-Net', 'ResNet', 'Proposed HFGSO-based DRN']
    tick_labels = ['Accuracy', 'Sensitivity', 'Specificity']
    bar_width, s = 0.15, 0.025

    for i in range(len(result)):
        if i == 0:
            tem = []
            for j in range(len(tick_labels)):
                tem.append(j + 1)
            loc.append(tem)
        else:
            tem = []
            for j in range(len(loc[i - 1])):
                tem.append(loc[i - 1][j] + s + bar_width)
            loc.append(tem)

    for i in range(len(result)):
        plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)

    plt.legend(loc=(0.25, 0.25))
    plt.show()

# Function to handle start button click
def start_process():
    global Acc, Sen, Spe

    if selection_var.get() == 'TrainingData(%)':
        tp = int(input_var.get()) / 100
    else:
        tp = (int(input_var.get()) - 1) / int(input_var.get())

    dataset, tr_per = dataset_var.get(), tp
    dts = dataset
    Acc, Sen, Spe = Run.callmain(dts, tr_per)

    # Update result entries
    for i in range(5):
        result_entries[f'1{i+1}'].set(Acc[i])
        result_entries[f'2{i+1}'].set(Sen[i])
        result_entries[f'3{i+1}'].set(Spe[i])

# Function to run graph
def run_graph():
    if Acc and Sen and Spe:
        plot_graph(Acc, Sen, Spe)

# Row 1: Dataset selection
row1 = ttk.Frame(main_frame)
row1.pack(fill='x', pady=5)
ttk.Label(row1, text="\t\t\tSelect_dataset").pack(side='left')
ttk.Combobox(row1, textvariable=dataset_var, values=['Prostate MRI'], width=13).pack(side='left', padx=5)

# Row 2: Selection type and input
row2 = ttk.Frame(main_frame)
row2.pack(fill='x', pady=5)
ttk.Label(row2, text="\t\t\tSelect            ").pack(side='left')
ttk.Combobox(row2, textvariable=selection_var, values=["TrainingData(%)", "k-fold"], width=13).pack(side='left', padx=5)
ttk.Entry(row2, textvariable=input_var, width=10).pack(side='left', padx=5)
ttk.Button(row2, text="START", command=start_process, width=10).pack(side='left', padx=5)

# Row 3: Column headers
row3 = ttk.Frame(main_frame)
row3.pack(fill='x', pady=10)
ttk.Label(row3, text="\t\t\t   DCNN\t\t    Panoptic model\t\t    Focal-Net\t\t ResNet\t\t     Proposed HFGSO..").pack()

# Create result rows
metrics = ['Accuracy', 'Sensitivity', 'Specificity']
for row_idx, metric in enumerate(metrics, 1):
    row = ttk.Frame(main_frame)
    row.pack(fill='x', pady=5)
    ttk.Label(row, text=f'\t{metric}').pack(side='left')

    for col_idx in range(1, 6):
        key = f'{row_idx}{col_idx}'
        entry = ttk.Entry(row, textvariable=result_entries[key], width=20)
        entry.pack(side='left', padx=2)

# Button row
button_row = ttk.Frame(main_frame)
button_row.pack(fill='x', pady=10)
ttk.Label(button_row, text='\t\t\t\t\t\t\t\t\t\t\t\t            ').pack(side='left')
ttk.Button(button_row, text='Run Graph', command=run_graph).pack(side='left', padx=5)
ttk.Button(button_row, text='CLOSE', command=root.destroy).pack(side='left', padx=5)

# Start the main loop
root.mainloop()