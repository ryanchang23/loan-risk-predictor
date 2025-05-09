import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List, Tuple
from .app import LoanRiskPredictor
from .config.config import ConfigManager
from sklearn.metrics import confusion_matrix
import seaborn as sns

class LoanRiskPredictorGUI:
    def __init__(self):
        self.config = ConfigManager()
        self.predictor = LoanRiskPredictor()
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Loan Risk Predictor")
        self.root.geometry("1600x900")

        # Set matplotlib style
        plt.style.use('ggplot')
 # Create frames
        self.create_control_frame()
        self.create_results_frame()
        self.create_charts_frame()
        
        # Initialize results storage
        self.results: Dict[str, Dict[str, List[float]]] = {}
        self.confusion_matrices: Dict[str, np.ndarray] = {}
        
        # Debug mode flag
        self.debug_mode = False
        
        # Feature engineering flag
        self.feature_engineering_done = False
        self.processed_features = None
        self.processed_labels = None
    
    def create_control_frame(self):
        """Create the control panel frame."""
        control_frame = ctk.CTkFrame(self.root)
        control_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=15, pady=15)
        
        # Debug mode toggle
        self.debug_var = ctk.BooleanVar(value=False)
        debug_checkbox = ctk.CTkCheckBox(
            control_frame,
            text="Debug Mode",
            variable=self.debug_var,
            command=self.toggle_debug_mode
        )
        debug_checkbox.pack(pady=5)
        
        # Feature engineering toggle
        self.feature_eng_var = ctk.BooleanVar(value=True)
        feature_eng_checkbox = ctk.CTkCheckBox(
            control_frame,
            text="Enable Feature Engineering",
            variable=self.feature_eng_var,
            command=self.toggle_feature_engineering
        )
        feature_eng_checkbox.pack(pady=5)
        
        # Train split rate
        train_frame = ctk.CTkFrame(control_frame)
        train_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(train_frame, text="Train Split Rate (%)").pack(side=ctk.LEFT, padx=5)
        self.train_split = ctk.CTkSlider(train_frame, from_=50, to=90, number_of_steps=8)
        self.train_split.set(80)
        self.train_split.pack(side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        self.train_split_label = ctk.CTkLabel(train_frame, text="80%")
        self.train_split_label.pack(side=ctk.LEFT, padx=5)
        self.train_split.configure(command=self.update_train_split_label)
        
        # Train split rate entry
        train_entry_frame = ctk.CTkFrame(control_frame)
        train_entry_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(train_entry_frame, text="Enter Train Split:").pack(side=ctk.LEFT, padx=5)
        self.train_split_entry = ctk.CTkEntry(train_entry_frame, width=60)
        self.train_split_entry.pack(side=ctk.LEFT, padx=5)
        self.train_split_entry.insert(0, "80")
        self.train_split_entry.bind('<Return>', self.update_train_split_from_entry)
        
        # K-fold number
        kfold_frame = ctk.CTkFrame(control_frame)
        kfold_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(kfold_frame, text="K-Fold Number").pack(side=ctk.LEFT, padx=5)
        self.k_fold = ctk.CTkSlider(kfold_frame, from_=2, to=10, number_of_steps=8)
        self.k_fold.set(5)
        self.k_fold.pack(side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        self.k_fold_label = ctk.CTkLabel(kfold_frame, text="5")
        self.k_fold_label.pack(side=ctk.LEFT, padx=5)
        self.k_fold.configure(command=self.update_kfold_label)
        
        # K-fold entry
        kfold_entry_frame = ctk.CTkFrame(control_frame)
        kfold_entry_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(kfold_entry_frame, text="Enter K-Fold:").pack(side=ctk.LEFT, padx=5)
        self.k_fold_entry = ctk.CTkEntry(kfold_entry_frame, width=60)
        self.k_fold_entry.pack(side=ctk.LEFT, padx=5)
        self.k_fold_entry.insert(0, "5")
        self.k_fold_entry.bind('<Return>', self.update_kfold_from_entry)
        
        # Subsample rate
        subsample_frame = ctk.CTkFrame(control_frame)
        subsample_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(subsample_frame, text="Subsample Rate (%)").pack(side=ctk.LEFT, padx=5)
        self.subsample_rate = ctk.CTkSlider(subsample_frame, from_=0.1, to=100, number_of_steps=999)
        self.subsample_rate.set(100)
        self.subsample_rate.pack(side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        self.subsample_label = ctk.CTkLabel(subsample_frame, text="100%")
        self.subsample_label.pack(side=ctk.LEFT, padx=5)
        self.subsample_rate.configure(command=self.update_subsample_label)
        
        # Subsample rate entry
        subsample_entry_frame = ctk.CTkFrame(control_frame)
        subsample_entry_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(subsample_entry_frame, text="Enter Subsample:").pack(side=ctk.LEFT, padx=5)
        self.subsample_entry = ctk.CTkEntry(subsample_entry_frame, width=60)
        self.subsample_entry.pack(side=ctk.LEFT, padx=5)
        self.subsample_entry.insert(0, "100")
        self.subsample_entry.bind('<Return>', self.update_subsample_from_entry)
        
        # Model selection
        model_frame = ctk.CTkFrame(control_frame)
        model_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(model_frame, text="Select Models").pack(anchor=ctk.W, pady=5)
        self.model_vars = {}
        for model in ["d_lstm", "mlp", "cnn_lightgbm", "dnn", "random_forest", "rnn", "xgboost"]:
            var = ctk.BooleanVar(value=True)
            self.model_vars[model] = var
            ctk.CTkCheckBox(model_frame, text=model, variable=var).pack(anchor=ctk.W, pady=2)
        
        # Run button
        self.run_button = ctk.CTkButton(
            control_frame,
            text="Run Models",
            command=self.run_models
        )
        self.run_button.pack(pady=20)
    
    def update_train_split_from_entry(self, event=None):
        """Update train split from entry field."""
        try:
            value = float(self.train_split_entry.get())
            if 50 <= value <= 90:
                self.train_split.set(value)
                self.update_train_split_label(value)
        except ValueError:
            pass
    
    def update_kfold_from_entry(self, event=None):
        """Update K-fold from entry field."""
        try:
            value = int(self.k_fold_entry.get())
            if 2 <= value <= 10:
                self.k_fold.set(value)
                self.update_kfold_label(value)
        except ValueError:
            pass
    
    def update_subsample_from_entry(self, event=None):
        """Update subsample rate from entry field."""
        try:
            value = float(self.subsample_entry.get())
            if 0.1 <= value <= 100:
                self.subsample_rate.set(value)
                self.update_subsample_label(value)
        except ValueError:
            pass
    
    def update_train_split_label(self, value):
        """Update the train split rate label."""
        self.train_split_label.configure(text=f"{int(value)}%")
        self.train_split_entry.delete(0, ctk.END)
        self.train_split_entry.insert(0, str(int(value)))
    
    def update_kfold_label(self, value):
        """Update the K-fold number label."""
        self.k_fold_label.configure(text=str(int(value)))
        self.k_fold_entry.delete(0, ctk.END)
        self.k_fold_entry.insert(0, str(int(value)))
    
    def update_subsample_label(self, value):
        """Update the subsample rate label."""
        rounded_value = round(value, 1)
        self.subsample_label.configure(text=f"{rounded_value:.1f}%")
        self.subsample_entry.delete(0, ctk.END)
        self.subsample_entry.insert(0, f"{rounded_value:.1f}")
    
    def toggle_debug_mode(self):
        """Toggle debug mode."""
        self.debug_mode = self.debug_var.get()
        if self.debug_mode:
            self.debug_text.configure(state=ctk.NORMAL)
            self.debug_text.delete("1.0", ctk.END)
            self.debug_text.insert(ctk.END, "Debug mode enabled\n")
            self.debug_text.configure(state=ctk.DISABLED)
            # Set the callback for debug logger
            self.predictor.debug_logger.set_gui_callback(self.log_debug_message)
        else:
            self.debug_text.configure(state=ctk.NORMAL)
            self.debug_text.delete("1.0", ctk.END)
            self.debug_text.configure(state=ctk.DISABLED)
            # Remove the callback
            self.predictor.debug_logger.set_gui_callback(None)
    
    def log_debug_message(self, message):
        """Log a debug message to the debug text box."""
        if self.debug_mode:
            self.debug_text.configure(state=ctk.NORMAL)
            self.debug_text.insert(ctk.END, message + "\n")
            self.debug_text.see(ctk.END)  # Scroll to the end
            self.debug_text.configure(state=ctk.DISABLED)
    
    def toggle_feature_engineering(self):
        """Toggle feature engineering."""
        if not self.feature_eng_var.get():
            self.feature_engineering_done = False
            self.processed_features = None
            self.processed_labels = None
    
    def create_results_frame(self):
        """Create the results display frame."""
        results_frame = ctk.CTkFrame(self.root)
        results_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=0, pady=20)
        
        # Results text
        ctk.CTkLabel(results_frame, text="Results").pack(pady=5)
        self.results_text = ctk.CTkTextbox(results_frame, width=550, height=350)
        self.results_text.pack(padx=0, pady=0)
        
        # Debug text
        ctk.CTkLabel(results_frame, text="Debug Messages").pack(pady=5)
        self.debug_text = ctk.CTkTextbox(results_frame, width=550, height=300)
        self.debug_text.pack(padx=0, pady=0)
        self.debug_text.configure(state=ctk.DISABLED)
    
    def create_charts_frame(self):
        """Create the charts display frame."""
        charts_frame = ctk.CTkFrame(self.root)
        charts_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=20, pady=20)
        
        # Create tabview for different chart types
        self.tabview = ctk.CTkTabview(charts_frame)
        self.tabview.pack(fill=ctk.BOTH, expand=True)
        
        # Create tabs
        self.tabview.add("Performance Metrics")
        self.tabview.add("Model Comparison")
        
        # Create figures for each tab
        self.fig1, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tabview.tab("Performance Metrics"))
        self.canvas1.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
        
        # Create two subplots for the second tab
        self.fig2, (self.ax3, self.ax4) = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tabview.tab("Model Comparison"))
        self.canvas2.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
    
    def update_charts(self):
        """Update the charts with the latest results."""
        # Clear previous charts
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Prepare data for charts
        models = list(self.results.keys())
        avg_metrics = {
            'accuracy': [np.mean(self.results[m]['accuracy']) for m in models],
            'sensitivity': [np.mean(self.results[m]['sensitivity']) for m in models],
            'specificity': [np.mean(self.results[m]['specificity']) for m in models]
        }
        
        # Set color palette for models
        model_colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # First tab: Performance Metrics
        # Bar chart for average metrics
        x = np.arange(len(models))
        width = 0.20
        bars1 = self.ax1.bar(x - width, avg_metrics['accuracy'], width, label='Accuracy', color='#2ecc71')
        bars2 = self.ax1.bar(x, avg_metrics['sensitivity'], width, label='Sensitivity', color='#3498db')
        bars3 = self.ax1.bar(x + width, avg_metrics['specificity'], width, label='Specificity', color='#e74c3c')
        
        # Add value labels on top of bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                self.ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        self.ax1.set_ylabel('Score')
        self.ax1.set_title('Average Model Performance Metrics')
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(models, rotation=45)
        self.ax1.legend()
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        # Box plot for fold-wise metrics
        fold_metrics = {
            'accuracy': [self.results[m]['accuracy'] for m in models],
            'sensitivity': [self.results[m]['sensitivity'] for m in models],
            'specificity': [self.results[m]['specificity'] for m in models]
        }

        # Create box plot with custom colors
        box_plot = self.ax2.boxplot(fold_metrics['accuracy'], labels=models, patch_artist=True)

        # Set box colors
        for box in box_plot['boxes']:
            box.set(facecolor='#2ecc71', alpha=0.5)
        
        # Add mean value labels
        for i, model in enumerate(models):
            mean_val = np.mean(fold_metrics['accuracy'][i])
            self.ax2.text(i + 1, mean_val, f'{mean_val:.3f}',
                         ha='center', va='bottom', color='black', fontweight='bold')
        
        self.ax2.set_ylabel('Score')
        self.ax2.set_title('Fold-wise Accuracy Distribution')
        self.ax2.set_xticklabels(models, rotation=45)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Second tab: Model Comparison
        # Confusion matrix comparison
        if self.confusion_matrices:
            # Calculate grid dimensions
            n_models = len(self.confusion_matrices)
            n_cols = min(3, n_models)  # Maximum 3 columns
            n_rows = (n_models + n_cols - 1) // n_cols
            
            # Create subplots for confusion matrices
            for i, (model_name, cm) in enumerate(self.confusion_matrices.items()):
                row = i // n_cols
                col = i % n_cols
                
                # Create subplot for this confusion matrix
                ax = self.ax3 if row == 0 else self.ax4
                if row > 0:
                    ax = self.ax4
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_title(f'{model_name}', fontsize=8)
                ax.set_xlabel('Predicted', fontsize=8)
                ax.set_ylabel('Actual', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Metrics comparison bar chart
        x = np.arange(3)  # 3 metrics: accuracy, sensitivity, specificity
        width = 0.8 / len(models)  # Adjust width based on number of models
        
        for i, model in enumerate(models):
            metrics = [avg_metrics['accuracy'][i], avg_metrics['sensitivity'][i], avg_metrics['specificity'][i]]
            bars = self.ax4.bar(x + i * width, metrics, width, label=model, color=model_colors[i])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                self.ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=8)
        
        self.ax4.set_ylabel('Score')
        self.ax4.set_title('Model Performance Comparison')
        self.ax4.set_xticks(x + width * (len(models) - 1) / 2)
        self.ax4.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity'])
        self.ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax4.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and draw
        self.fig1.tight_layout()
        self.fig2.tight_layout()
        self.canvas1.draw()
        self.canvas2.draw()
    
    def run_models(self):
        """Run the selected models and update the display."""
        # Update configuration
        self.config.set('models.train_percentage', int(self.train_split.get()))
        n_folds = int(self.k_fold.get())
        subsample_rate = round(self.subsample_rate.get() / 100.0, 3)
        
        # Clear previous results
        self.results_text.delete("1.0", ctk.END)
        self.results.clear()
        self.confusion_matrices.clear()
        
        # Run selected models
        for model_name, var in self.model_vars.items():
            if var.get():
                try:
                    self.results_text.insert(ctk.END, f"\nRunning {model_name}...\n")
                    self.root.update()
                    
                    metrics = self.predictor.run(
                        model_name,
                        subsample_rate,
                        n_folds,
                        debug_mode=self.debug_mode,
                        use_feature_engineering=self.feature_eng_var.get(),
                        processed_features=self.processed_features,
                        processed_labels=self.processed_labels
                    )
                    
                    # Store processed features for next model
                    if self.feature_eng_var.get() and not self.feature_engineering_done:
                        self.processed_features = metrics.pop('processed_features', None)
                        self.processed_labels = metrics.pop('processed_labels', None)
                        self.feature_engineering_done = True
                    
                    self.results[model_name] = metrics
                    
                    # Store confusion matrix if available
                    if 'confusion_matrix' in metrics:
                        self.confusion_matrices[model_name] = metrics['confusion_matrix']
                    
                    # Display average results
                    avg_metrics = {
                        'accuracy': np.mean(metrics['accuracy']),
                        'sensitivity': np.mean(metrics['sensitivity']),
                        'specificity': np.mean(metrics['specificity'])
                    }
                    
                    self.results_text.insert(ctk.END, f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n")
                    self.results_text.insert(ctk.END, f"Average Sensitivity: {avg_metrics['sensitivity']:.4f}\n")
                    self.results_text.insert(ctk.END, f"Average Specificity: {avg_metrics['specificity']:.4f}\n")
                    self.results_text.insert(ctk.END, f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
                except Exception as e:
                    self.results_text.insert(ctk.END, f"Error: {str(e)}\n")
        
        # Update charts
        self.update_charts()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop() 