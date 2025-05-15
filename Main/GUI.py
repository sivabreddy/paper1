"""
Medical Image Analysis GUI
-------------------------
Provides interface for:
- Selecting dataset and training parameters
- Running image analysis pipeline
- Displaying performance metrics (Accuracy, Sensitivity, Specificity)
- Visualizing results in bar charts
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from Main import Run  # Main processing module

# Create and configure main application window
root = tk.Tk()
root.title('142705')  # Application title/number
root.configure(bg='#004d4d')  # Dark teal background similar to DarkTeal9 theme

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

def plot_graph(result_1, result_2, result_3):
    """
    Creates bar chart comparing performance metrics across models
    
    Args:
        result_1: Accuracy values for each model
        result_2: Sensitivity values for each model
        result_3: Specificity values for each model
    """
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

def start_process():
    """
    Handles Start button click - runs main analysis pipeline
    - Calculates training percentage based on selection
    - Calls main processing function from Run module
    - Updates UI with results
    """
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

def run_graph():
    """
    Handles Run Graph button click
    - Validates results are available
    - Calls plot_graph to visualize metrics
    """
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