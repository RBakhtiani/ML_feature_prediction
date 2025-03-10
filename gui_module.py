#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from eda_module import EDA
from training_module import Training


# In[6]:


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Analysis and Model Training Application")
        self.geometry("1000x700")
        
        self.eda = EDA()
        self.training = Training()
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame for Data Pre-Prosessing
        self.eda_frame = tk.LabelFrame(self, text="Data Pre-Prosessing")
        self.eda_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.load_button = tk.Button(self.eda_frame, text="Load Data", command=self.load_data)
        self.load_button.pack(side="left", padx=5, pady=5)
        
        self.describe_button = tk.Button(self.eda_frame, text="Describe Data", command=self.describe_data)
        self.describe_button.pack(side="left", padx=5, pady=5)
        
        self.distribution_button = tk.Button(self.eda_frame, text="Show Class Distribution", command=self.show_class_distribution)
        self.distribution_button.pack(side="left", padx=5, pady=5)
        
        self.clean_button = tk.Button(self.eda_frame, text="Clean Data", command=self.clean_data)
        self.clean_button.pack(side="left", padx=5, pady=5)
        
        # Frame for EDA
        self.eda_frame = tk.LabelFrame(self, text="Exploratory Data Analysis")
        self.eda_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.plot_box_button = tk.Button(self.eda_frame, text="Box Plot", command=self.plot_box)
        self.plot_box_button.pack(side="left", padx=5, pady=5)
        
        self.heatmap_button = tk.Button(self.eda_frame, text="Heatmap", command=self.plot_heatmap)
        self.heatmap_button.pack(side="left", padx=5, pady=5)
        
        self.distribution_button = tk.Button(self.eda_frame, text="Class Distribution of Cleaned Data", command=self.class_distribution_cleaned)
        self.distribution_button.pack(side="left", padx=5, pady=5)
        
        self.encode_button = tk.Button(self.eda_frame, text="Encode Categorical Variables", command=self.encode_data)
        self.encode_button.pack(side="left", padx=5, pady=5)
        
        self.corr_heatmap_button = tk.Button(self.eda_frame, text="Correlation Heatmap", command=self.plot_corr_heatmap)
        self.corr_heatmap_button.pack(side="left", padx=5, pady=5)
        
        # Frame for Model Training
        self.model_frame = tk.LabelFrame(self, text="Model Training and Evaluation")
        self.model_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.train_income_button = tk.Button(self.model_frame, text="Train for Income", command=lambda: self.train_and_evaluate("income"))
        self.train_income_button.pack(side="left", padx=5, pady=5)
        
        self.train_marital_status_button = tk.Button(self.model_frame, text="Train for Marital Status", command=lambda: self.train_and_evaluate("marital_status"))
        self.train_marital_status_button.pack(side="left", padx=5, pady=5)
        
        self.train_workclass_button = tk.Button(self.model_frame, text="Train for Workclass", command=lambda: self.train_and_evaluate("workclass"))
        self.train_workclass_button.pack(side="left", padx=5, pady=5)
        
        # Frame for displaying results
        self.results_frame = tk.LabelFrame(self, text="Results")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(self.results_frame, wrap="word", height=10)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Quit button
        self.quit_button = tk.Button(self, text="Quit", command=self.quit_application)
        self.quit_button.pack(pady=10)
        
    def load_data(self):
        self.data = self.eda.load_data()
        self.results_text.insert(tk.END, "Data Loaded Successfully\n")
        
    def describe_data(self):
        if hasattr(self, 'data'):
            desc_num = self.data.describe().round(2).to_string()
            desc_obj = self.data.describe(include=['object']).to_string()
            self.results_text.insert(tk.END, "\nDescriptive Statistics for Numerical Variables:\n")
            self.results_text.insert(tk.END, desc_num + "\n")
            self.results_text.insert(tk.END, "\nDescriptive Statistics for Categorical Variables:\n")
            self.results_text.insert(tk.END, desc_obj + "\n")
        else:
            self.results_text.insert(tk.END, "\nPlease load data first.\n")
            
    def show_class_distribution(self):
        if hasattr(self, 'data'):
#             feature = "workclass" 
            feature = simpledialog.askstring("Input", "Enter Feature:")
            if feature == '':
                messagebox.showerror("Error", "Please enter a feature.")
            elif feature not in self.data.columns:
                messagebox.showerror("Error", "Enter a valid Feature.")
            else:
                self.show_graph_in_popup(self.eda.show_feature_class_distribution, self.data, feature)
        else:
            self.results_text.insert(tk.END, "\nPlease load data first.\n")
    
    def clean_data(self):
        if hasattr(self, 'data'):
            self.cleaned_data = self.eda.clean_data(self.data)
            self.results_text.insert(tk.END, "\nData Cleaned Successfully\n")
        else:
            self.results_text.insert(tk.END, "\nPlease load data first.\n")
    
    def plot_box(self):
        if hasattr(self, 'cleaned_data'):
#             x_feature = "workclass" 
#             y_feature = "age" 
            x_feature = simpledialog.askstring("Input", "Enter X-axis Feature:")
            y_feature = simpledialog.askstring("Input", "Enter Y-axis Feature:")
            if x_feature == '' and y_feature == '':
                messagebox.showerror("Error", "Please enter a feature.")
            elif x_feature not in self.data.columns or y_feature not in self.data.columns:
                messagebox.showerror("Error", "Enter a valid Feature.")
            else:
                self.show_graph_in_popup(self.eda.box_plot, self.cleaned_data, x_feature, y_feature)
        else:
            self.results_text.insert(tk.END, "\nPlease clean data first.\n")
    
    def plot_heatmap(self):
        if hasattr(self, 'cleaned_data'):
#             x_feature = "occupation"
#             y_feature = "income"
            x_feature = simpledialog.askstring("Input", "Enter X-axis Feature:")
            y_feature = simpledialog.askstring("Input", "Enter Y-axis Feature:")
            if x_feature == '' and y_feature == '':
                messagebox.showerror("Error", "Please enter a feature.")
            elif x_feature not in self.data.columns or y_feature not in self.data.columns:
                messagebox.showerror("Error", "Enter a valid Feature.")
            else:
                self.show_graph_in_popup(self.eda.heatmap, self.cleaned_data, x_feature, y_feature)
        else:
            self.results_text.insert(tk.END, "\nPlease clean data first.\n")
            
    def class_distribution_cleaned(self):
        if hasattr(self, 'cleaned_data'):
#             feature = "workclass"
            feature = simpledialog.askstring("Input", "Enter Feature:")
            if feature == '':
                messagebox.showerror("Error", "Please enter a feature.")
            elif feature not in self.data.columns:
                messagebox.showerror("Error", "Enter a valid Feature.")
            else:
                self.show_graph_in_popup(self.eda.class_distribution_features_needed, self.cleaned_data, feature)
        else:
            self.results_text.insert(tk.END, "\nPlease clean data first.\n")
            
    def encode_data(self):
        if hasattr(self, 'cleaned_data'):
            self.encoded_data = self.training.encode_variables(self.cleaned_data)
            self.results_text.insert(tk.END, "\nCategorical variables Encoded Successfully\n")
        else:
            self.results_text.insert(tk.END, "\nPlease clean data first.\n")
    
    def plot_corr_heatmap(self):
        if hasattr(self, 'encoded_data'):
            self.show_graph_in_popup(self.eda.corr_heatmap, self.encoded_data)
        else:
            self.results_text.insert(tk.END, "\nPlease encode categorical variables first.\n")
    
    def train_and_evaluate(self, target):
        if hasattr(self, 'encoded_data'):
            results = self.training.train_and_evaluate_models(self.encoded_data, target)
            self.results_text.insert(tk.END, f"\nResults for {target}:\n")
            self.results_text.insert(tk.END, results.to_string() + "\n")
        else:
            self.results_text.insert(tk.END, "\nPlease encode categorical variables first.\n")

    def show_graph_in_popup(self, plot_function, *args):
        popup = tk.Toplevel(self)
        popup.title("Graph")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)  # Create a subplot within the figure
        plot_function(*args, ax=ax)  # Pass the subplot to the plot function

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()

        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
            
    def quit_application(self):
        messagebox.showinfo("Goodbye", "Goodbye! The application will now exit.")
        self.destroy()

