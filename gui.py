import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
import pandas as pd
import os
from imblearn.pipeline import Pipeline

# Load attack detection models
try:
    binary_model = joblib.load('rf_pipeline_binary.joblib')
    print("Binary model loaded successfully")
except:
    binary_model = None
    print("Binary model not found")

try:
    multiclass_model = joblib.load('rf_pipeline_multiclass.joblib')
    print("Multiclass model loaded successfully")
except:
    multiclass_model = None
    print("Multiclass model not found")

# Create the main window
root = tk.Tk()
root.title("Network Attack Detection")
default_font = ('Helvetica', 12)

# Create file selection
file_label = tk.Label(root, text="Select test CSV file:", font=default_font)
file_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

file_path = tk.StringVar()
file_entry = tk.Entry(root, textvariable=file_path, width=50, font=default_font)
file_entry.grid(row=1, column=0, padx=10, pady=5)

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filename:
        file_path.set(filename)

browse_button = tk.Button(root, text="Browse", command=browse_file, font=default_font)
browse_button.grid(row=1, column=1, padx=5, pady=5)

# Results display
result_text = tk.Text(root, height=15, width=80, font=('Courier', 10))
result_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Function to analyze data
def analyze_data():
    try:
        if not file_path.get():
            messagebox.showerror("Error", "Please select a CSV file first!")
            return
        
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Loading data...\n")
        
        # Load CSV
        df = pd.read_csv(file_path.get())
        result_text.insert(tk.END, f"Data loaded: {len(df)} rows\n\n")
        
        # Binary prediction
        if binary_model:
            result_text.insert(tk.END, "Binary Analysis (Normal vs Attack)...\n")
            binary_pred = binary_model.predict(df)
            normal_count = sum(binary_pred == 1)
            attack_count = sum(binary_pred == 0)
            result_text.insert(tk.END, f"Normal traffic: {normal_count}\n")
            result_text.insert(tk.END, f"Attack traffic: {attack_count}\n\n")
        
        # Multiclass prediction
        if multiclass_model:
            result_text.insert(tk.END, "Multiclass Analysis (Attack Types)...\n")
            multi_pred = multiclass_model.predict(df)
            
            # Count attack types
            from collections import Counter
            attack_counts = Counter(multi_pred)
            
            result_text.insert(tk.END, f"FOUND {len(attack_counts)} DIFFERENT ATTACK CLASSES:\n")
            result_text.insert(tk.END, "="*50 + "\n")
            
            # Print all classes found
            for i, (attack_type, count) in enumerate(attack_counts.most_common(), 1):
                percentage = count / len(df) * 100
                result_text.insert(tk.END, f"{i:2d}. {attack_type:<20} {count:>6} ({percentage:5.1f}%)\n")
            
            result_text.insert(tk.END, "\n" + "="*50 + "\n")
            result_text.insert(tk.END, f"TOTAL UNIQUE ATTACK CLASSES: {len(attack_counts)}\n")
        
        result_text.insert(tk.END, "\nAnalysis completed!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during analysis: {str(e)}")

# Create analyze button
analyze_button = tk.Button(root, text="Analyze Dataset", font=default_font, command=analyze_data, bg="blue", fg="white", width=20)
analyze_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Run the application
root.mainloop()