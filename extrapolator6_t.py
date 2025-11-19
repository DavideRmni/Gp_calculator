"""
Python interface able to load and analyze CSV files containing spectra (fluorescence)
with support for european, american, tab delimited formats.
Includes functionalities like polinomial fitting, peaks analysis and PEAK deconvolution
useful to identify individual peaks that forms the spectra

Dipendencies needed
- pandas
- numpy  
- matplotlib
- tkinter (incluso in Python)
- chardet (per rilevamento encoding preciso)
- scikit-learn (per fitting polinomiale robusto)
- scipy (RICHIESTO per deconvoluzione spettrale e analisi picchi avanzata)

How to install dependencies:
pip install pandas numpy matplotlib chardet scikit-learn scipy

Base functions:
🔬 Loading of multi-format CSV files (EU/US/Tab)
📊 Polinomial Fitting with customizable R² 
🔍 Automatic peak detection (BETA)
🧬 SPECTRA DECONVOLUTION: tries to find individual peaks under a curve
   - MODELS: Gaussian, Lorentz, Voigt
   - Algoritms: curve_fit, differential_evolution
   - Parameter extraction: centro, ampiezza, larghezza, area, FWHM

Author: Davide Romanini

tradotto fino a riga 744
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import re
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from io import StringIO
import random

def check_dependencies():
    """CHECK THAT ALL DEPENDENCIES HAVE BEEN SUCCESSFULLY INSTALLED"""
    missing = []
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    
    if missing:
        print("❌ MISSING DEPENDENCIES:")
        for dep in missing:
            print(f"   pip install {dep}")
        return False
    
    print("✅ All needed dependencies have been found")
    return True

# Verifica scipy separatamente
try:
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import curve_fit, differential_evolution
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
    print("✅ Scipy available for advanced peak analisys")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  Scipy not available. Please install it using the string: pip install scipy")

# =============================================================================
# MATHEMATICAL METHODS FOR SPECTRA DECONVOLUTION
# =============================================================================

def gaussian_peak(x, amplitude, center, width):
    """Gaussian peak model"""
    return amplitude * np.exp(-((x - center) / width) ** 2)

def lorentzian_peak(x, amplitude, center, width):
    """Lorentzian peak model"""
    return amplitude / (1 + ((x - center) / width) ** 2)

def voigt_peak(x, amplitude, center, width_g, width_l):
    """Voigt peak model (convolution gaussian-lorentzian)"""
    # Voigt profile approximation
    fg = np.exp(-((x - center) / width_g) ** 2)
    fl = 1 / (1 + ((x - center) / width_l) ** 2)
    # weighted combination (approximation)
    eta = width_l / (width_g + width_l)  # Parametro di mixing
    return amplitude * (eta * fl + (1 - eta) * fg)

def multi_gaussian(x, *params):
    """Multiple gaussian peaks sum"""
    n_peaks = len(params) // 3
    result = np.zeros_like(x)
    for i in range(n_peaks):
        amplitude = params[i * 3]
        center = params[i * 3 + 1]
        width = params[i * 3 + 2]
        result += gaussian_peak(x, amplitude, center, width)
    return result

def multi_lorentzian(x, *params):
    """Multiple lorentzian peaks sum"""
    n_peaks = len(params) // 3
    result = np.zeros_like(x)
    for i in range(n_peaks):
        amplitude = params[i * 3]
        center = params[i * 3 + 1]
        width = params[i * 3 + 2]
        result += lorentzian_peak(x, amplitude, center, width)
    return result

def multi_voigt(x, *params):
    """Multiple Voigt peaks sum"""
    n_peaks = len(params) // 4
    result = np.zeros_like(x)
    for i in range(n_peaks):
        amplitude = params[i * 4]
        center = params[i * 4 + 1]
        width_g = params[i * 4 + 2]
        width_l = params[i * 4 + 3]
        result += voigt_peak(x, amplitude, center, width_g, width_l)
    return result

def estimate_initial_parameters(x, y, n_peaks, peak_type='gaussian', manual_centers=None, exact_centers=True):
    """
    Initial parameters estimation for multi-peak fitting
    
    Parameters:
    manual_centers: list of manual inserted peaks (None per auto-detect)
    exact_centers: if True uses exact centers, if False finds closest peaks
    """
    
    # Verify valid inputs 
    if len(x) != len(y):
        raise ValueError("x e y must have the same lenght")
    if n_peaks <= 0:
        raise ValueError("n_peaks must be positive")
    if len(x) < 3:
        raise ValueError("At least 3 points are needed")
    
    # If manual center have been given, use them
    if manual_centers is not None and len(manual_centers) > 0:
        centers_to_use = manual_centers[:n_peaks]  # Uses only the first n_peaks
        
        # If manual centers are less than needed, interpolate
        while len(centers_to_use) < n_peaks:
            if len(centers_to_use) >= 2:
                # Try to interpolate exixting peaks
                spacing = (centers_to_use[-1] - centers_to_use[0]) / (n_peaks - 1)
                new_center = centers_to_use[-1] + spacing
                if new_center <= x.max():
                    centers_to_use.append(new_center)
                else:
                    # If out of range, add before of the last
                    new_center = centers_to_use[-2] + (centers_to_use[-1] - centers_to_use[-2]) / 2
                    centers_to_use.insert(-1, new_center)
            else:
                # If there is only one center, try a uniform distribution
                x_range = x.max() - x.min()
                spacing = x_range / (n_peaks + 1)
                centers_to_use.append(x.min() + spacing * (len(centers_to_use) + 1))
        
        # Build parameters 
        params = []
        x_range = x.max() - x.min()
        
        if exact_centers:
            # USE EXACT CENTERS
            for center in centers_to_use:
                # Interpolate closest peak
                idx = np.argmin(np.abs(x - center))
                amplitude = y[idx]
                
                width = x_range / (n_peaks * 4)
                
                if peak_type == 'voigt':
                    params.extend([amplitude, center, width, width])  # center EXACT
                else:
                    params.extend([amplitude, center, width])  # center EXACT
        else:
            # USE PEAKS CLOSEST TO GIVEN DATA
            for center in centers_to_use:
                # Find the index closest to center
                idx = np.argmin(np.abs(x - center))
                amplitude = y[idx]
                actual_center = x[idx]  # Use actual data point
                
                width = x_range / (n_peaks * 4)
                
                if peak_type == 'voigt':
                    params.extend([amplitude, actual_center, width, width])
                else:
                    params.extend([amplitude, actual_center, width])
        
        return params
    
    else:
        # Use automatic identification as before
        if not SCIPY_AVAILABLE:
            # Simple estimation without scipy
            peak_indices = []
            threshold = np.mean(y) + np.std(y)
            
            for i in range(1, len(y) - 1):
                if (y[i] > y[i-1] and y[i] > y[i+1] and y[i] > threshold):
                    peak_indices.append(i)
            
            # Find highest peaks
            if len(peak_indices) > n_peaks:
                peak_heights = [y[i] for i in peak_indices]
                sorted_indices = np.argsort(peak_heights)[::-1]
                peak_indices = [peak_indices[i] for i in sorted_indices[:n_peaks]]
            
            # If not enough peaks found, distribute uniformely.
            if len(peak_indices) < n_peaks:
                additional_needed = n_peaks - len(peak_indices)
                spacing = len(x) // (additional_needed + 1)
                for i in range(1, additional_needed + 1):
                    new_idx = i * spacing
                    if new_idx < len(x) and new_idx not in peak_indices:
                        peak_indices.append(new_idx)
        else:
            # Use scipy for advanced peak identification
            try:
                height = np.mean(y) + 0.3 * np.std(y)
                distance = max(1, len(y) // (n_peaks * 3))  # Avoid distance=0
                peaks, properties = find_peaks(y, height=height, distance=distance)
                
                if len(peaks) >= n_peaks:
                    # Identify highest peaks
                    peak_heights = y[peaks]
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    peak_indices = peaks[sorted_indices[:n_peaks]].tolist()
                else:
                    # Add peaks uniformely distributed
                    peak_indices = peaks.tolist()
                    remaining = n_peaks - len(peaks)
                    if remaining > 0:
                        spacing = len(x) // (remaining + 1)
                        for i in range(1, remaining + 1):
                            new_idx = i * spacing
                            if new_idx < len(x):
                                peak_indices.append(new_idx)
            except Exception as e:
                print(f"Error scipy find_peaks: {e}, switching to simple method")
                # Fallback to simple method
                peak_indices = []
                for i in range(1, len(y) - 1):
                    if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > np.mean(y) + np.std(y):
                        peak_indices.append(i)
        
        # Check that we have exactly n_peaks
        while len(peak_indices) < n_peaks:
            # Add peaks uniformly distributed
            spacing = len(x) // (n_peaks - len(peak_indices) + 1)
            for i in range(1, n_peaks - len(peak_indices) + 1):
                new_idx = i * spacing
                if new_idx < len(x) and new_idx not in peak_indices:
                    peak_indices.append(new_idx)
                    break
            else:
                # If not able to add, use random valid indexes
                available_indices = [i for i in range(len(x)) if i not in peak_indices]
                if available_indices:
                    peak_indices.append(random.choice(available_indices))
                else:
                    break
    
    peak_indices = peak_indices[:n_peaks]  # Check for exactly n_peaks
    
    # Building initial parameters
    params = []
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    for peak_idx in peak_indices:
        if peak_idx < len(y):
            amplitude = y[peak_idx]
            center = x[peak_idx]
        else:
            amplitude = np.max(y) * 0.5
            center = x[len(x)//2]
            
        width = x_range / (n_peaks * 4)  # Adaptive width
        
        if peak_type == 'voigt':
            params.extend([amplitude, center, width, width])
        else:
            params.extend([amplitude, center, width])
    
    return params

# =============================================================================

class SpectralCSVInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spectra CSV interface - advanced analisys")
        self.root.state('zoomed')  # FULLSCREEN window
        # Cross-platform alternative:
        # self.root.attributes('-zoomed', True)  # Linux
        # self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        
        self.data = None
        self.file_path = None
        self.blank_corrected_data = None
        self.selected_blanks = []
        
        # Variables for analisys
        self.fitted_data = None
        self.polynomial_coeffs = None
        self.fitting_stats = None
        self.deconvoluted_data = None
        self.peak_components = None
        
        self.setup_ui()

    def create_export_separators_dialog(self, title="Select Separators to be used in Export"):
        """Create and manage the dialog for separators selection used in the exported file"""
        
        # Create a window for separators
        sep_dialog = tk.Toplevel(self.root)
        sep_dialog.title(title)
        sep_dialog.geometry("550x400")
        sep_dialog.resizable(False, False)
        
        # Center the window
        sep_dialog.transient(self.root)
        sep_dialog.grab_set()
        
        # Variables for separators
        export_field_sep = tk.StringVar(value=";")
        export_decimal_sep = tk.StringVar(value=",")
        export_confirmed = tk.BooleanVar(value=False)
        
        # Principal frame
        main_frame = ttk.Frame(sep_dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔧 Config export", 
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Frame for predefined profiles
        profiles_frame = ttk.LabelFrame(main_frame, text="Predefined Profiles", padding="10")
        profiles_frame.pack(fill=tk.X, pady=(0, 15))
        
        def set_export_profile(profile_type):
            if profile_type == "european":
                export_field_sep.set(";")
                export_decimal_sep.set(",")
            elif profile_type == "american":
                export_field_sep.set(",")
                export_decimal_sep.set(".")
            elif profile_type == "tab":
                export_field_sep.set("\\t")
                export_decimal_sep.set(".")
            update_export_labels()
        
        # Profiles buttons
        profiles_buttons_frame = ttk.Frame(profiles_frame)
        profiles_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(profiles_buttons_frame, text="🇪🇺 European (;,)", 
                  command=lambda: set_export_profile("european")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(profiles_buttons_frame, text="🇺🇸 American (,.)", 
                  command=lambda: set_export_profile("american")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(profiles_buttons_frame, text="📊 Tab (\\t,.)", 
                  command=lambda: set_export_profile("tab")).pack(side=tk.LEFT)
        
        # Manual configuration frame
        manual_frame = ttk.LabelFrame(main_frame, text="Manual Configuration", padding="10")
        manual_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Field separator
        field_frame = ttk.Frame(manual_frame)
        field_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(field_frame, text="Field separator:").pack(side=tk.LEFT)
        field_combo = ttk.Combobox(field_frame, textvariable=export_field_sep, 
                                  values=[",", ";", "\\t"], width=10)
        field_combo.pack(side=tk.LEFT, padx=(10, 10))
        
        field_desc_label = ttk.Label(field_frame, text="(Semicolon)", foreground="gray")
        field_desc_label.pack(side=tk.LEFT)
        
        # Decimal separator
        decimal_frame = ttk.Frame(manual_frame)
        decimal_frame.pack(fill=tk.X)
        
        ttk.Label(decimal_frame, text="Decimal separator:").pack(side=tk.LEFT)
        decimal_combo = ttk.Combobox(decimal_frame, textvariable=export_decimal_sep,
                                    values=[".", ","], width=10)
        decimal_combo.pack(side=tk.LEFT, padx=(10, 10))
        
        decimal_desc_label = ttk.Label(decimal_frame, text="(comma)", foreground="gray")
        decimal_desc_label.pack(side=tk.LEFT)
        
        def update_export_labels():
            """Update labels descriptive, used in export"""
            field_sep = export_field_sep.get()
            decimal_sep = export_decimal_sep.get()
            
            field_labels = {",": "(comma)", ";": "(semicolon)", "\\t": "(tab)"}
            decimal_labels = {".": "(period)", ",": "(comma)"}
            
            field_desc_label.config(text=field_labels.get(field_sep, ""))
            decimal_desc_label.config(text=decimal_labels.get(decimal_sep, ""))
        
        # Bind for updating the labels
        export_field_sep.trace('w', lambda *args: update_export_labels())
        export_decimal_sep.trace('w', lambda *args: update_export_labels())
        
        # Configuration preview
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        preview_frame.pack(fill=tk.X, pady=(0, 15))
        
        def update_preview():
            field_sep = export_field_sep.get()
            decimal_sep = export_decimal_sep.get()
            
            if field_sep == "\\t":
                field_display = "TAB"
            else:
                field_display = f"'{field_sep}'"
            
            preview_text = f"Esempio: Wavelength{field_sep}Sample1{field_sep}Sample2\n"
            preview_text += f"         400{decimal_sep}5{field_sep}1{decimal_sep}23{field_sep}2{decimal_sep}45"
            
            preview_label.config(text=preview_text)
        
        preview_label = ttk.Label(preview_frame, text="", font=("Courier", 9), 
                                 foreground="blue", justify=tk.LEFT)
        preview_label.pack()
        
        # Update initial preview
        update_preview()
        export_field_sep.trace('w', lambda *args: update_preview())
        export_decimal_sep.trace('w', lambda *args: update_preview())
        
        # Button Frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        def confirm_export():
            export_confirmed.set(True)
            sep_dialog.destroy()
        
        def cancel_export():
            export_confirmed.set(False)
            sep_dialog.destroy()
        
        ttk.Button(buttons_frame, text="✅ Continue Export", 
                  command=confirm_export).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(buttons_frame, text="❌ Cancel", 
                  command=cancel_export).pack(side=tk.RIGHT)
        
        # Update initial labels (export)
        update_export_labels()
        
        # Wait for window closing
        sep_dialog.wait_window()
        
        # Returns results
        if export_confirmed.get():
            selected_field_sep = export_field_sep.get()
            selected_decimal_sep = export_decimal_sep.get()
            
            # Tab management
            if selected_field_sep == "\\t":
                selected_field_sep = "\t"
                
            return selected_field_sep, selected_decimal_sep
        else:
            return None, None
        
    def setup_ui(self):
        # Principal frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Make notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # TAB 1: INPUT
        input_frame = ttk.Frame(notebook, padding="10")
        notebook.add(input_frame, text="📥 Input")
        
        # TAB 2: OUTPUT  
        output_frame = ttk.Frame(notebook, padding="10")
        notebook.add(output_frame, text="📊 Output")
        
        # =============================================================================
        # TAB INPUT: Load, Separators, Blank, Data
        # =============================================================================
        
        # File loading section
        file_frame = ttk.LabelFrame(input_frame, text="Load file", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Select CSV File", 
                  command=self.select_file).grid(row=0, column=0, padx=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No selected file")
        self.file_label.grid(row=0, column=1, sticky=tk.W)
        
        # Section separator configuration
        sep_frame = ttk.LabelFrame(input_frame, text="Separator configuration", padding="10")
        sep_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10), padx=(0, 5))
        
        # Predefined profiles
        profile_frame = ttk.Frame(sep_frame)
        profile_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(profile_frame, text="Profili:").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(profile_frame, text="🇪🇺 European (;,)", 
                  command=lambda: self.set_profile("european")).grid(row=0, column=1, padx=(10, 5))
        ttk.Button(profile_frame, text="🇺🇸 American (,.)", 
                  command=lambda: self.set_profile("american")).grid(row=0, column=2, padx=(5, 5))
        ttk.Button(profile_frame, text="📊 Tab (\\t,.)", 
                  command=lambda: self.set_profile("tab")).grid(row=0, column=3, padx=(5, 0))
        
        # Auto-detect checkbox
        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sep_frame, text="🔍 Automatic detection", 
                       variable=self.auto_detect_var,
                       command=self.toggle_manual_controls).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Field separator
        ttk.Label(sep_frame, text="Field separator:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.field_sep_var = tk.StringVar(value=";")
        self.field_sep_combo = ttk.Combobox(sep_frame, textvariable=self.field_sep_var, 
                                           values=[",", ";", "\\t"], width=10, state="disabled")
        self.field_sep_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Label showing actual separator
        self.field_sep_display = ttk.Label(sep_frame, text="(semicolon)")
        self.field_sep_display.grid(row=2, column=2, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Decimal separator
        ttk.Label(sep_frame, text="Decimal separator:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.decimal_sep_var = tk.StringVar(value=",")
        self.decimal_sep_combo = ttk.Combobox(sep_frame, textvariable=self.decimal_sep_var,
                                             values=[".", ","], width=10, state="disabled")
        self.decimal_sep_combo.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Label showing decimal separator
        self.decimal_sep_display = ttk.Label(sep_frame, text="(comma)")
        self.decimal_sep_display.grid(row=3, column=2, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Autodetection result
        self.detection_result = ttk.Label(sep_frame, text="", foreground="blue")
        self.detection_result.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Encoding
        ttk.Label(sep_frame, text="Encoding:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.encoding_var = tk.StringVar(value="utf-8")
        encoding_combo = ttk.Combobox(sep_frame, textvariable=self.encoding_var,
                                     values=["utf-8", "iso-8859-1", "cp1252"], width=15, state="disabled")
        encoding_combo.grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Load data button
        ttk.Button(sep_frame, text="📂 Load data", 
                  command=self.load_data).grid(row=6, column=0, columnspan=3, pady=(15, 0))
        
        # Bind events to update labels
        self.field_sep_var.trace('w', self.update_separator_labels)
        self.decimal_sep_var.trace('w', self.update_separator_labels)
        
        # Blank cleaning window (at the right of separaotrs)
        blank_frame = ttk.LabelFrame(input_frame, text="🔬 Blank correction", padding="10")
        blank_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 10), padx=(5, 0))
        
        # Blank selection Frame
        blank_select_frame = ttk.Frame(blank_frame)
        blank_select_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(blank_select_frame, text="Select blanks:").grid(row=0, column=0, sticky=tk.W)
        
        # Listbox for multiple blanks selection
        self.blank_listbox = tk.Listbox(blank_select_frame, selectmode=tk.MULTIPLE, height=8, width=40)
        self.blank_listbox.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=(0, 10), pady=(5, 0))
        
        blank_scroll = ttk.Scrollbar(blank_select_frame, orient=tk.VERTICAL, command=self.blank_listbox.yview)
        blank_scroll.grid(row=1, column=3, sticky=(tk.N, tk.S), pady=(5, 0))
        self.blank_listbox.configure(yscrollcommand=blank_scroll.set)
        
        # Buttons for blank control
        blank_buttons_frame = ttk.Frame(blank_frame)
        blank_buttons_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(blank_buttons_frame, text="✅ Apply correction", 
                  command=self.apply_blank_correction).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(blank_buttons_frame, text="❌ Clean list", 
                  command=self.remove_blank_correction).grid(row=0, column=1, padx=(0, 5))
        
        ttk.Button(blank_buttons_frame, text="👀 Preview blank", 
                  command=self.preview_blank).grid(row=0, column=2)
        
        # Label blank correction result
        self.blank_result = ttk.Label(blank_frame, text="", foreground="blue")
        self.blank_result.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # Sezione preview dati
        preview_frame = ttk.LabelFrame(input_frame, text="Data", padding="10")
        preview_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Treeview showing data
        self.tree = ttk.Treeview(preview_frame, height=12)
        self.tree.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Treeview scrollbar
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=0, column=3, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        ttk.Button(preview_frame, text="📊 Visualizza Spettri", 
                  command=self.plot_spectra).grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(10, 0))
        
        ttk.Button(preview_frame, text="💾 Esporta Dati", 
                  command=self.export_data).grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        
        ttk.Button(preview_frame, text="ℹ️ Info Dataset", 
                  command=self.show_info).grid(row=1, column=2, sticky="ew", pady=(10, 0))
        
        # Configurazione grid weights TAB INPUT
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)
        input_frame.rowconfigure(2, weight=1)  # Preview dati espandibile
        
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(2, weight=1)
        preview_frame.columnconfigure(3, weight=0)
        preview_frame.rowconfigure(0, weight=1)
        
        # =============================================================================
        # TAB OUTPUT: Range, Analisi, Deconvoluzione, Smoothing
        # =============================================================================
        
        # Section to define a range in wavelenght (for fitting, smoothing and deconvolution)
        range_frame = ttk.LabelFrame(output_frame, text="📏 Wavelength range", padding="10")
        range_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Checkbox for abilitating the range
        self.use_range_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(range_frame, text="📏 Limit range:", 
                       variable=self.use_range_var,
                       command=self.toggle_range_controls).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(range_frame, text="Min (nm):").grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        self.wavelength_min_var = tk.DoubleVar(value=400)
        self.wavelength_min_spin = ttk.Spinbox(range_frame, from_=0, to=2000, increment=10,
                                             textvariable=self.wavelength_min_var, width=8, state="disabled")
        self.wavelength_min_spin.grid(row=0, column=2, padx=(5, 15))
        
        ttk.Label(range_frame, text="Max (nm):").grid(row=0, column=3, sticky=tk.W)
        self.wavelength_max_var = tk.DoubleVar(value=800)
        self.wavelength_max_spin = ttk.Spinbox(range_frame, from_=0, to=2000, increment=10,
                                             textvariable=self.wavelength_max_var, width=8, state="disabled")
        self.wavelength_max_spin.grid(row=0, column=4, padx=(5, 0))
        
        # Auto-set range button
        ttk.Button(range_frame, text="🔄 Auto", 
                  command=self.auto_set_range).grid(row=0, column=5, padx=(10, 0))
        
        # Section for spectral analisys (TO THE LEFT)
        analysis_frame = ttk.LabelFrame(output_frame, text="🔬 Spectral analisys", padding="10")
        analysis_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(0, 5))
        
        # Polynomial fitting section
        poly_frame = ttk.Frame(analysis_frame)
        poly_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(poly_frame, text="Target R²:").grid(row=0, column=0, sticky=tk.W)
        self.r2_target_var = tk.DoubleVar(value=0.99)
        r2_spin = ttk.Spinbox(poly_frame, from_=0.90, to=0.999, increment=0.01, 
                             textvariable=self.r2_target_var, width=8)
        r2_spin.grid(row=0, column=1, padx=(5, 15))
        
        ttk.Label(poly_frame, text="Maximum grade:").grid(row=0, column=2, sticky=tk.W)
        self.max_degree_var = tk.IntVar(value=10)
        degree_spin = ttk.Spinbox(poly_frame, from_=2, to=20, increment=1,
                                 textvariable=self.max_degree_var, width=8)
        degree_spin.grid(row=0, column=3, padx=(5, 0))
        
        # Analysis buttons (aligned)
        buttons_frame = ttk.Frame(analysis_frame)
        buttons_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(buttons_frame, text="🧮 Calculate Fitting", 
                  command=self.polynomial_fitting).grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        ttk.Button(buttons_frame, text="📈 Visualize Fitting", 
                  command=self.plot_with_fitting).grid(row=0, column=1, padx=(0, 5), sticky="ew")
        
        ttk.Button(buttons_frame, text="🔍 Analyze Peaks", 
                  command=self.analyze_peaks).grid(row=1, column=0, padx=(0, 5), pady=(5, 0), sticky="ew")
        
        ttk.Button(buttons_frame, text="💾 Export results", 
                  command=self.export_analysis).grid(row=1, column=1, padx=(0, 5), pady=(5, 0), sticky="ew")
        
        # Configura weights per buttons_frame
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        
        # Fitting results
        self.fitting_results = ttk.Label(analysis_frame, text="", foreground="blue")
        self.fitting_results.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # Section for spectral deconvolution (ON THE RIGHT)
        deconv_frame = ttk.LabelFrame(output_frame, text="🔬 Spectra deconvolution", padding="10")
        deconv_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(5, 0))
        
        # FIRST ROW: Number of peaks, peak type, algorithm, max iteraction
        row1_frame = ttk.Frame(deconv_frame)
        row1_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(row1_frame, text="No. peaks:").grid(row=0, column=0, sticky=tk.W)
        self.n_peaks_var = tk.IntVar(value=2)
        ttk.Spinbox(row1_frame, from_=1, to=10, textvariable=self.n_peaks_var, width=5).grid(row=0, column=1, padx=(2, 8))
        
        ttk.Label(row1_frame, text="Tipo:").grid(row=0, column=2, sticky=tk.W)
        self.peak_type_var = tk.StringVar(value="voigt")  # MODIFICATO: default voigt
        ttk.Combobox(row1_frame, textvariable=self.peak_type_var,
                    values=["gaussian", "lorentzian", "voigt"], width=8).grid(row=0, column=3, padx=(2, 8))
        
        ttk.Label(row1_frame, text="Alg:").grid(row=0, column=4, sticky=tk.W)
        self.fitting_method_var = tk.StringVar(value="differential_evolution")  # MODIFICATO: default differential
        ttk.Combobox(row1_frame, textvariable=self.fitting_method_var,
                    values=["curve_fit", "differential_evolution"], width=12).grid(row=0, column=5, padx=(2, 8))
        
        ttk.Label(row1_frame, text="Max iter:").grid(row=0, column=6, sticky=tk.W)
        self.max_iter_var = tk.IntVar(value=1000)
        ttk.Spinbox(row1_frame, from_=100, to=5000, increment=100,
                   textvariable=self.max_iter_var, width=6).grid(row=0, column=7, padx=(2, 0))
        
        # Seconda riga: picchi manuali e controllo unificato esatti+fissi - MODIFICATA
        row2_frame = ttk.Frame(deconv_frame)
        row2_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.manual_peaks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2_frame, text="🎯 Manuali", 
                       variable=self.manual_peaks_var,
                       command=self.toggle_manual_peaks).grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        
        # UNIFICATO: Esatti + Fissi insieme
        self.exact_and_fixed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2_frame, text="🎯🔒 Esatti e Fissi", 
                       variable=self.exact_and_fixed_var,
                       command=self.update_exact_fixed_status).grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        
        # Manteniamo le variabili separate per compatibilità con il codice esistente
        self.exact_centers_var = tk.BooleanVar(value=True)
        self.fixed_centers_var = tk.BooleanVar(value=True)
        
        # Label informativo
        self.exact_fixed_info = ttk.Label(row2_frame, text="(Usa centri esatti e li mantiene fissi)", 
                                         foreground="gray", font=("Arial", 8))
        self.exact_fixed_info.grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # Terza riga: centri (nm)
        row3_frame = ttk.Frame(deconv_frame)
        row3_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(row3_frame, text="Centri (nm):").grid(row=0, column=0, sticky=tk.W)
        self.manual_centers_var = tk.StringVar(value="440.0, 490.0")
        self.manual_centers_entry = ttk.Entry(row3_frame, textvariable=self.manual_centers_var, 
                                            width=25, state="enabled")
        self.manual_centers_entry.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        row3_frame.columnconfigure(1, weight=1)
        
        # Quarta riga: pulsanti deconvoluzione
        buttons_deconv_frame = ttk.Frame(deconv_frame)
        buttons_deconv_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(buttons_deconv_frame, text="🧬 Deconvoluzione", 
                  command=self.automatic_deconvolution).grid(row=0, column=0, padx=(0, 2), sticky="ew")
        
        ttk.Button(buttons_deconv_frame, text="📈 Visualizza", 
                  command=self.plot_deconvolution).grid(row=0, column=1, padx=(0, 2), sticky="ew")
        
        ttk.Button(buttons_deconv_frame, text="📊 Parametri", 
                  command=self.show_peak_parameters).grid(row=1, column=0, padx=(0, 2), pady=(2, 0), sticky="ew")
        
        ttk.Button(buttons_deconv_frame, text="💾 Esporta", 
                  command=self.export_deconvolution).grid(row=1, column=1, padx=(0, 2), pady=(2, 0), sticky="ew")
        
        # Configura weights per buttons_deconv_frame
        buttons_deconv_frame.columnconfigure(0, weight=1)
        buttons_deconv_frame.columnconfigure(1, weight=1)
        
        # Risultati deconvoluzione
        self.deconv_results = ttk.Label(deconv_frame, text="", foreground="purple")
        self.deconv_results.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # NUOVO RIQUADRO: Smoothing e Deconvoluzione su Dati Originali
        smoothing_frame = ttk.LabelFrame(output_frame, text="📈 Smoothing e Deconvoluzione Dati Originali", padding="10")
        smoothing_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Prima riga: Tipo smoothing e parametri
        smooth_params_frame = ttk.Frame(smoothing_frame)
        smooth_params_frame.grid(row=0, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(smooth_params_frame, text="Smoothing:").grid(row=0, column=0, sticky=tk.W)
        self.smoothing_type_var = tk.StringVar(value="savgol")
        smoothing_combo = ttk.Combobox(smooth_params_frame, textvariable=self.smoothing_type_var,
                                      values=["none", "savgol", "moving_avg", "gaussian"], width=10)
        smoothing_combo.grid(row=0, column=1, padx=(5, 15))
        smoothing_combo.bind('<<ComboboxSelected>>', self.on_smoothing_change)
        
        ttk.Label(smooth_params_frame, text="Window:").grid(row=0, column=2, sticky=tk.W)
        self.smooth_window_var = tk.IntVar(value=11)
        self.smooth_window_spin = ttk.Spinbox(smooth_params_frame, from_=3, to=51, increment=2,
                                             textvariable=self.smooth_window_var, width=6)
        self.smooth_window_spin.grid(row=0, column=3, padx=(5, 15))
        
        ttk.Label(smooth_params_frame, text="Poly order:").grid(row=0, column=4, sticky=tk.W)
        self.smooth_poly_var = tk.IntVar(value=3)
        self.smooth_poly_spin = ttk.Spinbox(smooth_params_frame, from_=1, to=10, increment=1,
                                           textvariable=self.smooth_poly_var, width=6)
        self.smooth_poly_spin.grid(row=0, column=5, padx=(5, 0))
        
        # Seconda riga: Controlli picchi per smoothed data - MODIFICATA
        smooth_peaks_frame = ttk.Frame(smoothing_frame)
        smooth_peaks_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(smooth_peaks_frame, text="N° picchi:").grid(row=0, column=0, sticky=tk.W)
        self.smooth_n_peaks_var = tk.IntVar(value=2)
        ttk.Spinbox(smooth_peaks_frame, from_=1, to=10, textvariable=self.smooth_n_peaks_var, width=5).grid(row=0, column=1, padx=(5, 15))
        
        ttk.Label(smooth_peaks_frame, text="Tipo picco:").grid(row=0, column=2, sticky=tk.W)
        self.smooth_peak_type_var = tk.StringVar(value="voigt")  # MODIFICATO: default voigt
        ttk.Combobox(smooth_peaks_frame, textvariable=self.smooth_peak_type_var,
                    values=["gaussian", "lorentzian", "voigt"], width=10).grid(row=0, column=3, padx=(5, 15))
        
        ttk.Label(smooth_peaks_frame, text="Algoritmo:").grid(row=0, column=4, sticky=tk.W)
        self.smooth_fitting_method_var = tk.StringVar(value="differential_evolution")  # MODIFICATO: default differential
        ttk.Combobox(smooth_peaks_frame, textvariable=self.smooth_fitting_method_var,
                    values=["curve_fit", "differential_evolution"], width=12).grid(row=0, column=5, padx=(5, 15))
        
        ttk.Label(smooth_peaks_frame, text="Max iter:").grid(row=0, column=6, sticky=tk.W)
        self.smooth_max_iter_var = tk.IntVar(value=1000)
        ttk.Spinbox(smooth_peaks_frame, from_=100, to=5000, increment=100,
                   textvariable=self.smooth_max_iter_var, width=6).grid(row=0, column=7, padx=(5, 0))
        
        # NUOVA: Terza riga - Controlli manuali per smoothing
        smooth_controls_frame = ttk.Frame(smoothing_frame)
        smooth_controls_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.smooth_manual_peaks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(smooth_controls_frame, text="🎯 Manuali", 
                       variable=self.smooth_manual_peaks_var,
                       command=self.toggle_smooth_manual_peaks).grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        
        # UNIFICATO: Esatti + Fissi insieme per smoothing
        self.smooth_exact_and_fixed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(smooth_controls_frame, text="🎯🔒 Esatti e Fissi", 
                       variable=self.smooth_exact_and_fixed_var,
                       command=self.update_smooth_exact_fixed_status).grid(row=0, column=1, sticky=tk.W, padx=(0, 15))
        
        # Manteniamo le variabili separate per compatibilità con il codice esistente
        self.smooth_exact_centers_var = tk.BooleanVar(value=True)
        self.smooth_fixed_centers_var = tk.BooleanVar(value=True)
        
        # Label informativo per smoothing
        self.smooth_exact_fixed_info = ttk.Label(smooth_controls_frame, text="(Usa centri esatti e li mantiene fissi)", 
                                               foreground="gray", font=("Arial", 8))
        self.smooth_exact_fixed_info.grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # Quarta riga: Centri manuali per smoothed data (era terza riga)
        smooth_centers_frame = ttk.Frame(smoothing_frame)
        smooth_centers_frame.grid(row=3, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(smooth_centers_frame, text="Centri (nm):").grid(row=0, column=0, sticky=tk.W)
        self.smooth_centers_var = tk.StringVar(value="440.0, 490.0")
        self.smooth_centers_entry = ttk.Entry(smooth_centers_frame, textvariable=self.smooth_centers_var, 
                                             width=40, state="enabled")
        self.smooth_centers_entry.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        smooth_centers_frame.columnconfigure(1, weight=1)
        
        # Quinta riga: Pulsanti azione (era quarta riga)
        smooth_buttons_frame = ttk.Frame(smoothing_frame)
        smooth_buttons_frame.grid(row=4, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(smooth_buttons_frame, text="🧽 Applica Smoothing", 
                  command=self.apply_smoothing).grid(row=0, column=0, padx=(0, 3), sticky="ew")
        
        ttk.Button(smooth_buttons_frame, text="🧬 Deconv. su Originali", 
                  command=self.deconvolve_original_data).grid(row=0, column=1, padx=(0, 3), sticky="ew")
        
        ttk.Button(smooth_buttons_frame, text="📈 Visualizza", 
                  command=self.plot_original_deconvolution).grid(row=0, column=2, padx=(0, 3), sticky="ew")
        
        ttk.Button(smooth_buttons_frame, text="📊 Confronta Risultati", 
                  command=self.compare_deconvolution_results).grid(row=0, column=3, padx=(0, 3), sticky="ew")
        
        ttk.Button(smooth_buttons_frame, text="💾 Esporta Smoothed", 
                  command=self.export_smoothed_data).grid(row=0, column=4, padx=(0, 3), sticky="ew")
        
        ttk.Button(smooth_buttons_frame, text="🧮 Calcolo GP", 
                  command=self.calculate_gp_analysis).grid(row=0, column=5, padx=(0, 0), sticky="ew")
        
        # Configura weights per smooth_buttons_frame (6 colonne uguali)
        for i in range(6):
            smooth_buttons_frame.columnconfigure(i, weight=1)
        
        # Risultati smoothing
        self.smoothing_results = ttk.Label(smoothing_frame, text="", foreground="darkgreen")
        self.smoothing_results.grid(row=5, column=0, columnspan=6, sticky=tk.W, pady=(5, 0))
        
        # Variabili per dati smoothed
        self.smoothed_data = None
        self.original_deconvoluted_data = None
        self.original_peak_components = None
        
        # Configurazione grid weights TAB OUTPUT
        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)
        output_frame.rowconfigure(1, weight=1)  # Analisi/Deconv espandibili
        output_frame.rowconfigure(2, weight=0)  # Smoothing dimensione fissa
        
        # Configurazione grid weights MAIN
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    # =============================================================================
    # NUOVI METODI PER CONTROLLO UNIFICATO ESATTI+FISSI
    # =============================================================================
    
    def update_exact_fixed_status(self):
        """NUOVO: Aggiorna lo stato di exact_centers e fixed_centers insieme"""
        status = self.exact_and_fixed_var.get()
        self.exact_centers_var.set(status)
        self.fixed_centers_var.set(status)
        
        # Aggiorna il testo informativo
        if status:
            self.exact_fixed_info.config(text="(Usa centri esatti e li mantiene fissi)", foreground="green")
        else:
            self.exact_fixed_info.config(text="(Centri liberi di muoversi)", foreground="orange")
    
    def toggle_manual_peaks(self):
        """MODIFICATO: Versione aggiornata per gestire il controllo unificato"""
        state = "normal" if self.manual_peaks_var.get() else "disabled"
        self.manual_centers_entry.config(state=state)
        
        # Se disabilita i picchi manuali, disabilita anche esatti+fissi
        if not self.manual_peaks_var.get():
            self.exact_and_fixed_var.set(False)
            self.update_exact_fixed_status()
    
    # =============================================================================
    # NUOVI METODI PER CONTROLLI SMOOTHING
    # =============================================================================
    
    def update_smooth_exact_fixed_status(self):
        """NUOVO: Aggiorna lo stato di exact_centers e fixed_centers per smoothing"""
        status = self.smooth_exact_and_fixed_var.get()
        self.smooth_exact_centers_var.set(status)
        self.smooth_fixed_centers_var.set(status)
        
        # Aggiorna il testo informativo
        if status:
            self.smooth_exact_fixed_info.config(text="(Usa centri esatti e li mantiene fissi)", foreground="green")
        else:
            self.smooth_exact_fixed_info.config(text="(Centri liberi di muoversi)", foreground="orange")
    
    def toggle_smooth_manual_peaks(self):
        """NUOVO: Gestisce i picchi manuali nella sezione smoothing"""
        state = "normal" if self.smooth_manual_peaks_var.get() else "disabled"
        self.smooth_centers_entry.config(state=state)
        
        # Se disabilita i picchi manuali, disabilita anche esatti+fissi
        if not self.smooth_manual_peaks_var.get():
            self.smooth_exact_and_fixed_var.set(False)
            self.update_smooth_exact_fixed_status()
    
    # =============================================================================
    # RESTO DEI METODI ORIGINALI (invariati)
    # =============================================================================
    
    def detect_encoding(self, file_path):
        """Rileva l'encoding del file con chardet (più preciso) e fallback"""
        try:
            import chardet
            
            # Usa chardet per rilevamento preciso
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Legge i primi 10KB
            
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            
            # Se la confidenza è alta, usa il risultato
            if confidence > 0.7 and detected_encoding:
                return detected_encoding
            
            # Altrimenti prova i fallback comuni
            for encoding in ['utf-8', 'iso-8859-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            return detected_encoding or 'utf-8'
            
        except ImportError:
            # Fallback senza chardet
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read(2000)
                        if content and not any(ord(c) > 126 and ord(c) < 160 for c in content[:200]):
                            return encoding
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            return 'utf-8'
            
        except Exception:
            return 'utf-8'
    
    def auto_detect_separators(self):
        """Rileva automaticamente i separatori con algoritmo migliorato"""
        if not self.file_path:
            return
        
        try:
            # Rileva encoding
            detected_encoding = self.detect_encoding(self.file_path)
            self.encoding_var.set(detected_encoding)
            
            # Legge il file con l'encoding rilevato
            with open(self.file_path, 'r', encoding=detected_encoding) as f:
                lines = f.readlines()[:10]  # Prime 10 righe
            
            sample = ''.join(lines)
            
            # Rileva separatore di campo
            field_separators = [',', ';', '\t']
            field_scores = {}
            
            for sep in field_separators:
                # Conta i separatori per riga e verifica consistenza
                counts = [line.count(sep) for line in lines if line.strip()]
                if counts and len(set(counts)) <= 2:  # Massimo 2 valori diversi (header vs data)
                    field_scores[sep] = max(counts) if counts else 0
                else:
                    field_scores[sep] = 0
            
            detected_field_sep = max(field_scores, key=field_scores.get) if any(field_scores.values()) else ';'
            
            # Rileva separatore decimale
            # Analizza i pattern numerici dopo aver diviso per il separatore di campo
            decimal_patterns = {',': 0, '.': 0}
            
            for line in lines[1:]:  # Salta l'header
                if line.strip():
                    fields = line.split(detected_field_sep)
                    for field in fields[1:]:  # Salta la prima colonna (wavelength)
                        field = field.strip()
                        # Conta pattern decimali
                        if re.search(r'\d+,\d+', field):
                            decimal_patterns[','] += 1
                        elif re.search(r'\d+\.\d+', field):
                            decimal_patterns['.'] += 1
            
            detected_decimal_sep = max(decimal_patterns, key=decimal_patterns.get)
            
            # CORREZIONE: Risolve conflitto separatori identici
            if detected_field_sep == detected_decimal_sep:
                if detected_field_sep == ',':
                    # Se entrambi sono virgola, field diventa punto e virgola
                    detected_field_sep = ';'
                elif detected_field_sep == '.':
                    # Se entrambi sono punto, field diventa virgola
                    detected_field_sep = ','
            
            # Imposta i valori rilevati
            # Gestione speciale per tab
            field_sep_display = "\\t" if detected_field_sep == '\t' else detected_field_sep
            self.field_sep_var.set(field_sep_display)
            self.decimal_sep_var.set(detected_decimal_sep)
            
            # Determina il profilo rilevato
            profile_detected = ""
            if detected_field_sep == ';' and detected_decimal_sep == ',':
                profile_detected = "🇪🇺 Formato Europeo"
            elif detected_field_sep == ',' and detected_decimal_sep == '.':
                profile_detected = "🇺🇸 Formato Americano"
            elif detected_field_sep == '\t':
                profile_detected = "📊 Formato Tab-delimited"
            else:
                profile_detected = "🔍 Formato personalizzato"
            
            result_text = f"✅ Rilevato: {profile_detected} | Campo: '{detected_field_sep}' | Decimale: '{detected_decimal_sep}' | Encoding: {detected_encoding}"
            self.detection_result.config(text=result_text, foreground="green")
            
            print(f"Auto-detect completato: {result_text}")
            
        except Exception as e:
            error_msg = f"❌ Errore rilevamento: {e}"
            self.detection_result.config(text=error_msg, foreground="red")
            messagebox.showerror("Errore", f"Errore nel rilevamento automatico: {e}")
    
    def set_profile(self, profile_type):
        """Imposta profili predefiniti per diversi standard internazionali"""
        self.auto_detect_var.set(False)
        self.toggle_manual_controls()
        
        if profile_type == "european":
            self.field_sep_var.set(";")
            self.decimal_sep_var.set(",")
            self.detection_result.config(text="✅ Profilo Europeo: separatore ';' decimali ','")
        elif profile_type == "american":
            self.field_sep_var.set(",")
            self.decimal_sep_var.set(".")
            self.detection_result.config(text="✅ Profilo Americano: separatore ',' decimali '.'")
        elif profile_type == "tab":
            self.field_sep_var.set("\\t")
            self.decimal_sep_var.set(".")
            self.detection_result.config(text="✅ Profilo Tab: separatore 'TAB' decimali '.'")
    
    def update_separator_labels(self, *args):
        """Aggiorna le etichette descrittive dei separatori"""
        field_sep = self.field_sep_var.get()
        decimal_sep = self.decimal_sep_var.get()
        
        # Aggiorna etichetta separatore campi
        field_labels = {",": "(virgola)", ";": "(punto e virgola)", "\\t": "(tab)"}
        self.field_sep_display.config(text=field_labels.get(field_sep, ""))
        
        # Aggiorna etichetta separatore decimali
        decimal_labels = {".": "(punto)", ",": "(virgola)"}
        self.decimal_sep_display.config(text=decimal_labels.get(decimal_sep, ""))
    
    def toggle_manual_controls(self):
        """Abilita/disabilita i controlli manuali dei separatori"""
        state = "disabled" if self.auto_detect_var.get() else "normal"
        self.field_sep_combo.config(state=state)
        self.decimal_sep_combo.config(state=state)
        
        if self.auto_detect_var.get() and self.file_path:
            self.auto_detect_separators()
    
    def toggle_range_controls(self):
        """Abilita/disabilita i controlli del range di lunghezze d'onda"""
        state = "normal" if self.use_range_var.get() else "disabled"
        self.wavelength_min_spin.config(state=state)
        self.wavelength_max_spin.config(state=state)
    
    def auto_detect_peaks(self):
        """Rileva automaticamente i picchi dominanti e li inserisce nel campo"""
        if self.fitted_data is None:
            messagebox.showwarning("Attenzione", "Calcola prima il fitting polinomiale")
            return
        
        try:
            # Usa il primo spettro fittato disponibile per la rilevazione
            analysis_cols = self.get_analysis_columns()
            if not analysis_cols:
                messagebox.showwarning("Attenzione", "Nessuna colonna disponibile")
                return
            
            first_col = analysis_cols[0]
            fitted_col = f"{first_col}_fitted"
            
            if fitted_col not in self.fitted_data.columns:
                messagebox.showwarning("Attenzione", "Calcola prima il fitting polinomiale")
                return
            
            wavelength_col = self.fitted_data.columns[0]
            wavelengths = self.fitted_data[wavelength_col].values
            intensities = self.fitted_data[fitted_col].values
            
            # Applica range se selezionato
            wl_filtered, int_filtered = self.get_wavelength_range(wavelengths, intensities)
            
            if SCIPY_AVAILABLE:
                # Usa scipy per rilevamento avanzato
                height = np.mean(int_filtered) + 0.5 * np.std(int_filtered)
                prominence = 0.2 * (np.max(int_filtered) - np.min(int_filtered))
                distance = max(1, len(int_filtered) // 10)
                
                peaks, properties = find_peaks(int_filtered, 
                                              height=height,
                                              prominence=prominence,
                                              distance=distance)
                
                if len(peaks) > 0:
                    # Ordina per altezza e prende i più alti
                    peak_heights = int_filtered[peaks]
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    top_peaks = peaks[sorted_indices[:5]]  # Max 5 picchi
                    
                    # Converte in lunghezze d'onda
                    peak_wavelengths = wl_filtered[top_peaks]
                    peak_wavelengths_sorted = sorted(peak_wavelengths)
                    
                    # Formatta come stringa
                    centers_str = ", ".join([f"{wl:.1f}" for wl in peak_wavelengths_sorted])
                    self.manual_centers_var.set(centers_str)
                    
                    # Aggiorna il numero di picchi
                    self.n_peaks_var.set(len(peak_wavelengths_sorted))
                    
                    messagebox.showinfo("Picchi rilevati", 
                                      f"Trovati {len(peak_wavelengths_sorted)} picchi: {centers_str} nm")
                else:
                    messagebox.showwarning("Nessun picco", "Nessun picco significativo trovato")
            else:
                # Metodo semplice senza scipy
                peaks = []
                threshold = np.mean(int_filtered) + np.std(int_filtered)
                
                for i in range(1, len(int_filtered) - 1):
                    if (int_filtered[i] > int_filtered[i-1] and 
                        int_filtered[i] > int_filtered[i+1] and
                        int_filtered[i] > threshold):
                        peaks.append(i)
                
                if peaks:
                    peak_wavelengths = wl_filtered[peaks]
                    peak_wavelengths_sorted = sorted(peak_wavelengths)[:5]  # Max 5
                    
                    centers_str = ", ".join([f"{wl:.1f}" for wl in peak_wavelengths_sorted])
                    self.manual_centers_var.set(centers_str)
                    self.n_peaks_var.set(len(peak_wavelengths_sorted))
                    
                    messagebox.showinfo("Picchi rilevati", 
                                      f"Trovati {len(peak_wavelengths_sorted)} picchi: {centers_str} nm")
                else:
                    messagebox.showwarning("Nessun picco", "Nessun picco significativo trovato")
                    
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella rilevazione picchi: {e}")
    
    def parse_manual_centers(self):
        """Analizza la stringa dei centri manuali e restituisce lista di valori"""
        try:
            centers_str = self.manual_centers_var.get().strip()
            if not centers_str:
                return []
            
            # Divide per virgole e converte in float
            centers = []
            for center_str in centers_str.split(','):
                center_str = center_str.strip()
                if center_str:
                    centers.append(float(center_str))
            
            return sorted(centers)  # Ordina i centri
            
        except ValueError as e:
            raise ValueError(f"Formato centri non valido. Usa: '500, 600' o '450.5, 550.2'")
        except Exception as e:
            raise ValueError(f"Errore nel parsing centri: {e}")
    
    def auto_set_range(self):
        """Imposta automaticamente il range basato sui dati caricati"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        try:
            wavelength_col = self.data.columns[0]
            wavelengths = self.data[wavelength_col].values
            
            # Rimuove NaN per calcoli accurati
            clean_wavelengths = wavelengths[~np.isnan(wavelengths)]
            
            if len(clean_wavelengths) > 0:
                min_wl = np.min(clean_wavelengths)
                max_wl = np.max(clean_wavelengths)
                
                # Aggiunge +10 al minimo e -10 al massimo
                min_wl = min_wl + 10
                max_wl = max_wl - 10
                
                # Verifica che il range rimanga valido
                if min_wl >= max_wl:
                    min_wl = np.min(clean_wavelengths)
                    max_wl = np.max(clean_wavelengths)
                
                self.wavelength_min_var.set(min_wl)
                self.wavelength_max_var.set(max_wl)
                
                messagebox.showinfo("Range impostato", 
                                  f"Range impostato: {min_wl:.0f} - {max_wl:.0f} nm")
            else:
                messagebox.showerror("Errore", "Nessuna lunghezza d'onda valida trovata")
                
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nell'impostazione del range: {e}")
    
    def get_wavelength_range(self, wavelengths, intensities):
        """Filtra i dati per il range di lunghezze d'onda selezionato"""
        if not self.use_range_var.get():
            return wavelengths, intensities
        
        try:
            min_wl = self.wavelength_min_var.get()
            max_wl = self.wavelength_max_var.get()
            
            if min_wl >= max_wl:
                raise ValueError(f"Range non valido: min ({min_wl}) >= max ({max_wl})")
            
            # Crea maschera per il range
            mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
            
            if not np.any(mask):
                raise ValueError(f"Nessun dato nel range {min_wl}-{max_wl} nm")
            
            return wavelengths[mask], intensities[mask]
            
        except Exception as e:
            print(f"Errore nel filtraggio range: {e}")
            return wavelengths, intensities
    
    def select_file(self):
        """Seleziona il file CSV"""
        file_path = filedialog.askopenfilename(
            title="Seleziona file CSV",
            filetypes=[
                ("CSV files", "*.csv"), 
                ("TSV files", "*.tsv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=Path(file_path).name)
            
            if self.auto_detect_var.get():
                self.auto_detect_separators()
            else:
                self.detection_result.config(text="📁 File caricato. Configura i separatori manualmente.")
    
    def load_data(self):
        """Carica i dati dal file CSV con gestione migliorata"""
        if not self.file_path:
            messagebox.showwarning("Attenzione", "Seleziona prima un file CSV")
            return
        
        try:
            field_sep = self.field_sep_var.get()
            decimal_sep = self.decimal_sep_var.get()
            encoding = self.encoding_var.get()
            
            # Gestione tab
            if field_sep == "\\t":
                field_sep = "\t"
            
            # Se il separatore decimale è virgola, dobbiamo convertire
            if decimal_sep == ",":
                # Leggiamo il file come testo
                with open(self.file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                lines = content.strip().split('\n')
                
                # Gestione righe vuote
                if not lines:
                    raise ValueError("File vuoto")
                    
                header = lines[0]
                
                # Convertiamo le virgole in punti solo nei numeri (non nell'header)
                converted_lines = [header]
                for line in lines[1:]:
                    if line.strip():  # Salta righe vuote
                        # Sostituisce virgole con punti solo se sono circondate da cifre
                        converted_line = re.sub(r'(\d),(\d)', r'\1.\2', line)
                        converted_lines.append(converted_line)
                
                # Creiamo un file temporaneo in memoria
                converted_content = '\n'.join(converted_lines)
                data_io = StringIO(converted_content)
                
                self.data = pd.read_csv(data_io, sep=field_sep)
            else:
                self.data = pd.read_csv(self.file_path, sep=field_sep, encoding=encoding)
            
            # Verifica che ci siano dati
            if self.data.empty:
                raise ValueError("Il file non contiene dati validi")
                
            # Verifica almeno 2 colonne (wavelength + almeno uno spettro)
            if len(self.data.columns) < 2:
                raise ValueError("Il file deve contenere almeno 2 colonne (wavelength + spettri)")
            
            # Verifica che i dati siano numerici (eccetto la prima colonna)
            numeric_cols = self.data.columns[1:]
            for col in numeric_cols:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Verifica che ci siano dati numerici validi
            numeric_data = self.data[numeric_cols]
            if numeric_data.isna().all().all():
                raise ValueError("Nessun dato numerico valido trovato nelle colonne degli spettri")
            
            # Rimuove righe con troppi NaN
            initial_rows = len(self.data)
            self.data = self.data.dropna(thresh=len(self.data.columns) * 0.5)
            final_rows = len(self.data)
            
            # Verifica che rimangano dati dopo la pulizia
            if final_rows == 0:
                raise ValueError("Tutti i dati sono stati rimossi durante la pulizia (troppi valori mancanti)")
            
            self.update_preview()
            
            success_msg = f"✅ Dati caricati: {final_rows} righe, {len(self.data.columns)} colonne"
            if initial_rows != final_rows:
                success_msg += f" ({initial_rows - final_rows} righe con errori rimosse)"
            
            messagebox.showinfo("Successo", success_msg)
            self.detection_result.config(text=success_msg, foreground="green")
            
        except Exception as e:
            error_msg = f"❌ Errore nel caricamento: {e}"
            self.detection_result.config(text=error_msg, foreground="red")
            messagebox.showerror("Errore", f"Errore nel caricamento dei dati:\n{e}\n\nVerifica i separatori e riprova.")

    def update_preview(self):
        """Aggiorna l'anteprima nella tabella Treeview"""
        # Pulisce la treeview
        self.tree.delete(*self.tree.get_children())
    
        if self.data is not None:
            # Imposta le intestazioni della tabella
            self.tree["columns"] = list(self.data.columns)
            self.tree["show"] = "headings"
        
            for col in self.data.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, anchor="center")
        
            # Inserisce le righe nella preview
            for _, row in self.data.iterrows():
                self.tree.insert("", "end", values=list(row))
            
            # Aggiorna la lista delle colonne per la selezione bianco
            self.update_blank_list()
    
    def update_blank_list(self):
        """Aggiorna la listbox delle colonne bianco"""
        if self.data is None:
            return
        
        # Pulisce la listbox
        self.blank_listbox.delete(0, tk.END)
        
        # Aggiunge tutte le colonne spettrali (esclusa la prima che è wavelength)
        spectral_cols = self.data.columns[1:]
        for col in spectral_cols:
            self.blank_listbox.insert(tk.END, col)
    
    def apply_blank_correction(self):
        """Applica la correzione del bianco sottraendo la media dei bianchi selezionati"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        # Ottiene le colonne bianco selezionate
        selected_indices = self.blank_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Attenzione", "Seleziona almeno una colonna bianco")
            return
        
        try:
            wavelength_col = self.data.columns[0]
            spectral_cols = self.data.columns[1:]
            
            # Ottiene i nomi delle colonne bianco selezionate
            self.selected_blanks = [spectral_cols[i] for i in selected_indices]
            
            # Calcola la media dei bianchi
            blank_data = self.data[self.selected_blanks].values
            blank_mean = np.nanmean(blank_data, axis=1)
            
            # Crea dataset corretto
            self.blank_corrected_data = self.data.copy()
            
            # Sottrae la media bianco da TUTTE le colonne spettrali
            for col in spectral_cols:
                self.blank_corrected_data[col] = self.data[col] - blank_mean
            
            # Aggiorna risultato
            blank_names = ", ".join(self.selected_blanks)
            result_text = f"✅ Correzione applicata usando: {blank_names}"
            self.blank_result.config(text=result_text, foreground="green")
            
            messagebox.showinfo("Successo", f"Correzione bianco applicata.\nBianchi usati: {blank_names}")
            
        except Exception as e:
            error_msg = f"❌ Errore nella correzione: {e}"
            self.blank_result.config(text=error_msg, foreground="red")
            messagebox.showerror("Errore", f"Errore nella correzione del bianco:\n{e}")
    
    def remove_blank_correction(self):
        """Rimuove la correzione del bianco"""
        self.blank_corrected_data = None
        self.selected_blanks = []
        self.blank_result.config(text="❌ Correzione rimossa", foreground="orange")
        messagebox.showinfo("Info", "Correzione del bianco rimossa")
    
    def preview_blank(self):
        """Visualizza il bianco medio calcolato"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        selected_indices = self.blank_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Attenzione", "Seleziona almeno una colonna bianco")
            return
        
        try:
            wavelength_col = self.data.columns[0]
            wavelengths = self.data[wavelength_col].values
            spectral_cols = self.data.columns[1:]
            
            # Ottiene le colonne bianco selezionate
            selected_blanks = [spectral_cols[i] for i in selected_indices]
            
            plt.figure(figsize=(12, 8))
            
            # Plot singoli bianchi
            for col in selected_blanks:
                plt.plot(wavelengths, self.data[col].values, alpha=0.5, label=col)
            
            # Plot media bianco
            blank_data = self.data[selected_blanks].values
            blank_mean = np.nanmean(blank_data, axis=1)
            plt.plot(wavelengths, blank_mean, 'k-', linewidth=3, label='Bianco Medio')
            
            plt.xlabel('Lunghezza d\'onda (nm)')
            plt.ylabel('Intensità')
            plt.title('Preview Bianco e Media')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel preview bianco: {e}")
    
    def get_active_data(self):
        """Restituisce i dati attivi (corretti del bianco se applicato, altrimenti originali)"""
        return self.blank_corrected_data if self.blank_corrected_data is not None else self.data
    
    def get_analysis_columns(self):
        """Restituisce le colonne da usare per l'analisi (escludendo i bianchi)"""
        if self.data is None:
            return []
        
        spectral_cols = self.data.columns[1:]  # Esclude wavelength
        
        # Se c'è correzione bianco, esclude le colonne bianco dall'analisi
        if self.selected_blanks:
            analysis_cols = [col for col in spectral_cols if col not in self.selected_blanks]
        else:
            analysis_cols = list(spectral_cols)
        
        return analysis_cols
    
    def plot_spectra(self):
        """Visualizza gli spettri"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        try:
            # Usa i dati attivi (corretti del bianco se applicato)
            active_data = self.get_active_data()
            analysis_cols = self.get_analysis_columns()
            
            if not analysis_cols:
                messagebox.showwarning("Attenzione", "Nessuna colonna disponibile per l'analisi")
                return
            
            # Assume che la prima colonna sia la lunghezza d'onda
            wavelength_col = active_data.columns[0]
            wavelengths = active_data[wavelength_col].values
            
            plt.figure(figsize=(14, 10))
            
            # Plot solo le colonne di analisi (esclude bianchi)
            for col in analysis_cols:
                plt.plot(wavelengths, active_data[col].values, label=col, alpha=0.7)
            
            # Se ci sono bianchi, mostra anche la media bianco
            if self.selected_blanks and self.blank_corrected_data is not None:
                # Calcola e mostra bianco medio sui dati originali
                blank_data = self.data[self.selected_blanks].values
                blank_mean = np.nanmean(blank_data, axis=1)
                plt.plot(wavelengths, blank_mean, 'k--', linewidth=2, 
                        label=f'Bianco Medio ({len(self.selected_blanks)} col.)', alpha=0.8)
            
            plt.xlabel('Lunghezza d\'onda (nm)')
            plt.ylabel('Intensità')
            
            title = 'Spettri'
            if self.blank_corrected_data is not None:
                title += ' (Corretti del Bianco)'
            plt.title(title)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella visualizzazione: {e}")
    
    def export_data(self):
        """Esporta i dati elaborati con selezione dei separatori"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        # Ottiene i separatori dall'utente
        selected_field_sep, selected_decimal_sep = self.create_export_separators_dialog()
        
        # Se l'utente ha annullato, esce
        if selected_field_sep is None:
            return
        
        # Ora apre il dialog per salvare il file
        file_path = filedialog.asksaveasfilename(
            title="Salva dati",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("TSV files", "*.tsv"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                # Prepara i dati per l'esportazione
                export_data = self.data.copy()
                
                # Se il separatore decimale è virgola, converte i numeri
                if selected_decimal_sep == ",":
                    # Converte tutte le colonne numeriche tranne la prima (wavelength)
                    numeric_cols = export_data.columns[1:]
                    for col in numeric_cols:
                        if export_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            # Converte in stringa con virgola come separatore decimale
                            export_data[col] = export_data[col].astype(str).str.replace('.', ',', regex=False)
                
                # Esporta con i separatori selezionati
                export_data.to_csv(file_path, sep=selected_field_sep, index=False, 
                                  float_format='%.6f' if selected_decimal_sep == '.' else None)
                
                # Messaggio di successo con dettagli
                profile_name = ""
                if selected_field_sep == ";" and selected_decimal_sep == ",":
                    profile_name = " (Formato Europeo)"
                elif selected_field_sep == "," and selected_decimal_sep == ".":
                    profile_name = " (Formato Americano)"
                elif selected_field_sep == "\t":
                    profile_name = " (Formato Tab-delimited)"
                
                messagebox.showinfo("Successo", 
                                  f"Dati esportati con successo{profile_name}\n"
                                  f"Separatore campi: '{selected_field_sep if selected_field_sep != chr(9) else 'TAB'}'\n"
                                  f"Separatore decimali: '{selected_decimal_sep}'")
                
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nell'esportazione: {e}")
    
    def polynomial_fitting(self):
        """Calcola il fitting polinomiale ottimale per raggiungere l'R² target"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        try:
            # Usa dati attivi e colonne di analisi
            active_data = self.get_active_data()
            analysis_cols = self.get_analysis_columns()
            
            if not analysis_cols:
                messagebox.showwarning("Attenzione", "Nessuna colonna disponibile per l'analisi")
                return
            
            wavelength_col = active_data.columns[0]
            wavelengths = active_data[wavelength_col].values
            
            r2_target = self.r2_target_var.get()
            max_degree = self.max_degree_var.get()
            
            # Inizializza risultati con tutti i wavelengths originali
            self.fitted_data = pd.DataFrame()
            self.fitted_data[wavelength_col] = wavelengths
            self.polynomial_coeffs = {}
            self.fitting_stats = {}
            
            # Progress tracking
            total_spectra = len(analysis_cols)
            results_summary = []
            
            for i, col in enumerate(analysis_cols):
                intensities = active_data[col].values
                
                # Applica range se selezionato
                wl_filtered, int_filtered = self.get_wavelength_range(wavelengths, intensities)
                
                # Rimuove NaN
                mask = ~(np.isnan(wl_filtered) | np.isnan(int_filtered))
                x_clean = wl_filtered[mask]
                y_clean = int_filtered[mask]
                
                if len(x_clean) < 3:
                    continue
                
                # Trova il grado ottimale del polinomio
                best_degree = None
                best_r2 = 0
                best_model = None
                
                for degree in range(1, max_degree + 1):
                    try:
                        # Usa sklearn per fitting robusto
                        poly_pipeline = Pipeline([
                            ('poly', PolynomialFeatures(degree=degree)),
                            ('linear', LinearRegression())
                        ])
                        
                        poly_pipeline.fit(x_clean.reshape(-1, 1), y_clean)
                        y_pred = poly_pipeline.predict(x_clean.reshape(-1, 1))
                        r2 = r2_score(y_clean, y_pred)
                        
                        if r2 >= r2_target:
                            best_degree = degree
                            best_r2 = r2
                            best_model = poly_pipeline
                            break
                        elif r2 > best_r2:
                            best_degree = degree
                            best_r2 = r2
                            best_model = poly_pipeline
                            
                    except Exception as e:
                        print(f"Errore grado {degree} per {col}: {e}")
                        continue
                
                if best_model is not None:
                    # Genera curve fittata su TUTTI i punti wavelength (non solo il range)
                    y_fitted = best_model.predict(wavelengths.reshape(-1, 1))
                    self.fitted_data[f"{col}_fitted"] = y_fitted
                    
                    # Salva coefficienti
                    poly_coeffs = best_model.named_steps['linear'].coef_
                    self.polynomial_coeffs[col] = {
                        'degree': best_degree,
                        'coefficients': poly_coeffs,
                        'intercept': best_model.named_steps['linear'].intercept_,
                        'r2': best_r2,
                        'model': best_model,
                        'fitting_range': [x_clean.min(), x_clean.max()] if self.use_range_var.get() else None
                    }
                    
                    # Calcola statistiche sul range usato per il fitting
                    wl_fit, int_fit = self.get_wavelength_range(wavelengths, y_fitted)
                    wl_orig, int_orig = self.get_wavelength_range(wavelengths, active_data[col].values)
                    
                    residuals = int_orig - int_fit
                    rmse = np.sqrt(np.mean(residuals**2))
                    self.fitting_stats[col] = {
                        'r2': best_r2,
                        'rmse': rmse,
                        'degree': best_degree,
                        'target_achieved': best_r2 >= r2_target,
                        'fitting_range': [x_clean.min(), x_clean.max()] if self.use_range_var.get() else None
                    }
                    
                    status = "✅" if best_r2 >= r2_target else "⚠️"
                    range_info = f" (range: {x_clean.min():.0f}-{x_clean.max():.0f} nm)" if self.use_range_var.get() else ""
                    results_summary.append(f"{status} {col}: grado {best_degree}, R²={best_r2:.4f}{range_info}")
                
                # Aggiorna interfaccia con progress
                progress_text = f"Elaborazione: {i+1}/{total_spectra} spettri completati"
                self.fitting_results.config(text=progress_text)
                self.root.update()
            
            # Mostra risultati finali
            achieved_count = sum(1 for stats in self.fitting_stats.values() if stats['target_achieved'])
            total_count = len(self.fitting_stats)
            
            range_text = ""
            if self.use_range_var.get():
                range_text = f" (range: {self.wavelength_min_var.get():.0f}-{self.wavelength_max_var.get():.0f} nm)"
            
            blank_text = ""
            if self.selected_blanks:
                blank_text = f" | Bianchi esclusi: {len(self.selected_blanks)}"
            
            summary_text = f"✅ Fitting completato: {achieved_count}/{total_count} spettri raggiungono R²≥{r2_target}{range_text}{blank_text}"
            self.fitting_results.config(text=summary_text, foreground="green")
            
            # Mostra dettagli in una finestra
            self.show_fitting_details(results_summary)
            
        except Exception as e:
            error_msg = f"❌ Errore nel fitting: {e}"
            self.fitting_results.config(text=error_msg, foreground="red")
            messagebox.showerror("Errore", f"Errore nel calcolo del fitting:\n{e}")
    
    def show_fitting_details(self, results_summary):
        """Mostra i dettagli del fitting in una finestra separata"""
        details_window = tk.Toplevel(self.root)
        details_window.title("Risultati Fitting Polinomiale")
        details_window.geometry("600x400")
        
        # Frame principale con scrollbar
        main_frame = ttk.Frame(details_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text widget con scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Costruisce il report dettagliato
        report = "🔬 RISULTATI FITTING POLINOMIALE\n"
        report += "=" * 50 + "\n\n"
        
        for result in results_summary:
            report += f"{result}\n"
        
        report += "\n" + "=" * 50 + "\n"
        report += "📊 STATISTICHE DETTAGLIATE\n\n"
        
        for spectrum, stats in self.fitting_stats.items():
            report += f"Spettro: {spectrum}\n"
            report += f"  • Grado polinomio: {stats['degree']}\n"
            report += f"  • R²: {stats['r2']:.6f}\n"
            report += f"  • RMSE: {stats['rmse']:.3f}\n"
            report += f"  • Target raggiunto: {'Sì' if stats['target_achieved'] else 'No'}\n\n"
        
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def plot_with_fitting(self):
        """Visualizza gli spettri con le curve fittate (usando dati corretti del bianco se applicato)"""
        if self.fitted_data is None:
            messagebox.showwarning("Attenzione", "Calcola prima il fitting polinomiale")
            return
        
        try:
            # USA DATI ATTIVI (corretti del bianco se applicato)
            active_data = self.get_active_data()
            analysis_cols = self.get_analysis_columns()
            
            if not analysis_cols:
                messagebox.showwarning("Attenzione", "Nessuna colonna disponibile per l'analisi")
                return
            
            wavelength_col = active_data.columns[0]
            wavelengths = active_data[wavelength_col].values
            
            plt.figure(figsize=(14, 8))
            
            # Plot spettri (corretti del bianco) vs fittati
            for col in analysis_cols:
                if col in self.fitting_stats:
                    # Spettro dopo correzione bianco (se applicata)
                    data_label = f"{col}"
                    if self.blank_corrected_data is not None:
                        data_label += " (corretto bianco)"
                    else:
                        data_label += " (originale)"
                    
                    plt.plot(wavelengths, active_data[col].values, 
                            alpha=0.6, linewidth=1, label=data_label)
                    
                    # Spettro fittato
                    fitted_col = f"{col}_fitted"
                    if fitted_col in self.fitted_data.columns:
                        r2 = self.fitting_stats[col]['r2']
                        degree = self.fitting_stats[col]['degree']
                        plt.plot(wavelengths, self.fitted_data[fitted_col].values,
                                linewidth=2, linestyle='--',
                                label=f"{col} (fit: grado {degree}, R²={r2:.3f})")
            
            plt.xlabel('Lunghezza d\'onda (nm)')
            plt.ylabel('Intensità')
            
            # Titolo dinamico basato sulla correzione bianco
            title = 'Spettri vs Fitting Polinomiale'
            if self.blank_corrected_data is not None:
                title += ' (Dati Corretti del Bianco)'
            plt.title(title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella visualizzazione: {e}")
    
    def analyze_peaks(self):
        """Analizza i picchi negli spettri fittati"""
        if self.fitted_data is None:
            messagebox.showwarning("Attenzione", "Calcola prima il fitting polinomiale")
            return
        
        try:
            wavelength_col = self.data.columns[0]
            wavelengths = self.fitted_data[wavelength_col].values
            
            peaks_info = {}
            
            # Analizza ogni spettro fittato
            fitted_cols = [col for col in self.fitted_data.columns if col.endswith('_fitted')]
            
            for fitted_col in fitted_cols:
                original_col = fitted_col.replace('_fitted', '')
                intensities = self.fitted_data[fitted_col].values
                
                if SCIPY_AVAILABLE:
                    # Usa scipy per rilevamento picchi avanzato
                    # Parametri adattivi basati sui dati
                    height = np.mean(intensities) + 0.5 * np.std(intensities)
                    prominence = 0.1 * (np.max(intensities) - np.min(intensities))
                    distance = len(intensities) // 20  # Minimo 5% della lunghezza
                    
                    peaks, properties = find_peaks(intensities, 
                                                  height=height,
                                                  prominence=prominence,
                                                  distance=distance)
                else:
                    # Metodo semplice senza scipy
                    peaks = []
                    for i in range(1, len(intensities) - 1):
                        if (intensities[i] > intensities[i-1] and 
                            intensities[i] > intensities[i+1] and
                            intensities[i] > np.mean(intensities) + np.std(intensities)):
                            peaks.append(i)
                    properties = {}
                
                # Salva informazioni sui picchi
                peak_wavelengths = wavelengths[peaks]
                peak_intensities = intensities[peaks]
                
                peaks_info[original_col] = {
                    'peak_indices': peaks,
                    'peak_wavelengths': peak_wavelengths,
                    'peak_intensities': peak_intensities,
                    'num_peaks': len(peaks),
                    'properties': properties
                }
            
            # Mostra risultati
            self.show_peaks_analysis(peaks_info)
            
            # Plot dei picchi
            self.plot_peaks(peaks_info)
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nell'analisi picchi: {e}")
    
    def show_peaks_analysis(self, peaks_info):
        """Mostra i risultati dell'analisi dei picchi"""
        peaks_window = tk.Toplevel(self.root)
        peaks_window.title("Analisi Picchi Spettrali")
        peaks_window.geometry("700x500")
        
        # Text widget con scrollbar
        main_frame = ttk.Frame(peaks_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Costruisce il report
        report = "🔍 ANALISI PICCHI SPETTRALI\n"
        report += "=" * 60 + "\n\n"
        
        for spectrum, info in peaks_info.items():
            report += f"📊 Spettro: {spectrum}\n"
            report += f"Numero di picchi rilevati: {info['num_peaks']}\n\n"
            
            if info['num_peaks'] > 0:
                report += "Dettaglio picchi:\n"
                for i, (wl, intensity) in enumerate(zip(info['peak_wavelengths'], info['peak_intensities'])):
                    report += f"  Picco {i+1}: λ = {wl:.2f} nm, I = {intensity:.2f}\n"
                
                # Statistiche sui picchi
                if info['num_peaks'] >= 2:
                    separations = np.diff(info['peak_wavelengths'])
                    report += f"\nSeparazioni tra picchi:\n"
                    for i, sep in enumerate(separations):
                        report += f"  Picco {i+1} → Picco {i+2}: {sep:.2f} nm\n"
                    
                    report += f"Separazione media: {np.mean(separations):.2f} nm\n"
            else:
                report += "Nessun picco significativo rilevato.\n"
            
            report += "\n" + "-" * 40 + "\n\n"
        
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def plot_peaks(self, peaks_info):
        """Visualizza gli spettri con i picchi evidenziati"""
        try:
            wavelength_col = self.fitted_data.columns[0]
            wavelengths = self.fitted_data[wavelength_col].values
            
            plt.figure(figsize=(14, 8))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(peaks_info)))
            
            for i, (spectrum, info) in enumerate(peaks_info.items()):
                fitted_col = f"{spectrum}_fitted"
                intensities = self.fitted_data[fitted_col].values
                
                # Plot spettro
                plt.plot(wavelengths, intensities, 
                        color=colors[i], linewidth=2, label=spectrum)
                
                # Evidenzia picchi
                if info['num_peaks'] > 0:
                    plt.scatter(info['peak_wavelengths'], info['peak_intensities'],
                              color=colors[i], s=100, marker='o', 
                              edgecolors='black', linewidth=2, zorder=5)
                    
                    # Annota i picchi
                    for j, (wl, intensity) in enumerate(zip(info['peak_wavelengths'], info['peak_intensities'])):
                        plt.annotate(f'P{j+1}\n{wl:.1f}nm', 
                                   xy=(wl, intensity),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=8, ha='left',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', color='black', lw=1))
            
            plt.xlabel('Lunghezza d\'onda (nm)')
            plt.ylabel('Intensità')
            plt.title('Spettri Fittati con Picchi Rilevati')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella visualizzazione picchi: {e}")
    
    def automatic_deconvolution(self):
        """Esegue la deconvoluzione automatica degli spettri in componenti individuali"""
        if self.fitted_data is None:
            messagebox.showwarning("Attenzione", "Calcola prima il fitting polinomiale")
            return
        
        if not SCIPY_AVAILABLE:
            messagebox.showerror("Errore", "Scipy è richiesto per la deconvoluzione. Installalo con: pip install scipy")
            return
        
        try:
            # Usa dati attivi e colonne di analisi
            active_data = self.get_active_data()
            analysis_cols = self.get_analysis_columns()
            
            if not analysis_cols:
                messagebox.showwarning("Attenzione", "Nessuna colonna disponibile per l'analisi")
                return
                
            wavelength_col = self.fitted_data.columns[0]
            wavelengths = self.fitted_data[wavelength_col].values
            fitted_cols = [f"{col}_fitted" for col in analysis_cols if f"{col}_fitted" in self.fitted_data.columns]
            
            n_peaks = self.n_peaks_var.get()
            peak_type = self.peak_type_var.get()
            method = self.fitting_method_var.get()
            max_iter = self.max_iter_var.get()
            
            self.deconvoluted_data = {}
            self.peak_components = {}
            
            # Seleziona il modello appropriato
            if peak_type == 'gaussian':
                model_func = multi_gaussian
                params_per_peak = 3
            elif peak_type == 'lorentzian':
                model_func = multi_lorentzian
                params_per_peak = 3
            elif peak_type == 'voigt':
                model_func = multi_voigt
                params_per_peak = 4
            
            successful_decomp = 0
            total_spectra = len(fitted_cols)
            
            for i, fitted_col in enumerate(fitted_cols):
                original_col = fitted_col.replace('_fitted', '')
                intensities = self.fitted_data[fitted_col].values
                
                # Applica range se selezionato
                wl_filtered, int_filtered = self.get_wavelength_range(wavelengths, intensities)
                
                # Aggiorna progress
                progress_text = f"Deconvoluzione: {i+1}/{total_spectra} spettri..."
                self.deconv_results.config(text=progress_text)
                self.root.update()
                
                try:
                    # Ottiene centri manuali se specificati
                    manual_centers = None
                    if self.manual_peaks_var.get():
                        try:
                            manual_centers = self.parse_manual_centers()
                            if len(manual_centers) == 0:
                                raise ValueError("Nessun centro manuale specificato")
                        except Exception as e:
                            messagebox.showerror("Errore", f"Errore nei centri manuali: {e}")
                            continue
                    
                    # Stima parametri iniziali sul range filtrato
                    exact_centers = self.exact_centers_var.get()
                    fixed_centers = self.fixed_centers_var.get() and manual_centers is not None
                    
                    # Se i centri non sono fissi, non usare centri manuali per l'inizializzazione
                    init_centers = manual_centers if fixed_centers else None
                    
                    initial_params = estimate_initial_parameters(wl_filtered, int_filtered, 
                                                               n_peaks, peak_type, init_centers, exact_centers)
                    
                    # Definisce bounds per i parametri basati sul range filtrato
                    x_min, x_max = wl_filtered.min(), wl_filtered.max()
                    y_min, y_max = int_filtered.min(), int_filtered.max()
                    x_range = x_max - x_min
                    
                    # Ottiene informazioni sui centri fissi se applicabili
                    fixed_centers = self.fixed_centers_var.get() and manual_centers is not None
                    
                    if peak_type == 'voigt':
                        # [amplitude, center, width_g, width_l] per ogni picco
                        lower_bounds = []
                        upper_bounds = []
                        for i in range(n_peaks):
                            # Amplitude bounds
                            lower_bounds.append(0)
                            upper_bounds.append(y_max * 2)
                            
                            # Center bounds
                            if fixed_centers and i < len(manual_centers):
                                # Blocca centro con tolleranza di ±0.1 nm
                                center = manual_centers[i]
                                lower_bounds.append(center - 0.1)
                                upper_bounds.append(center + 0.1)
                            else:
                                lower_bounds.append(x_min)
                                upper_bounds.append(x_max)
                            
                            # Width bounds
                            lower_bounds.extend([x_range/100, x_range/100])
                            upper_bounds.extend([x_range/2, x_range/2])
                    else:
                        # [amplitude, center, width] per ogni picco
                        lower_bounds = []
                        upper_bounds = []
                        for i in range(n_peaks):
                            # Amplitude bounds
                            lower_bounds.append(0)
                            upper_bounds.append(y_max * 2)
                            
                            # Center bounds
                            if fixed_centers and i < len(manual_centers):
                                # Blocca centro con tolleranza di ±0.1 nm
                                center = manual_centers[i]
                                lower_bounds.append(center - 0.1)
                                upper_bounds.append(center + 0.1)
                            else:
                                lower_bounds.append(x_min)
                                upper_bounds.append(x_max)
                            
                            # Width bounds
                            lower_bounds.append(x_range/100)
                            upper_bounds.append(x_range/2)
                    
                    bounds = (lower_bounds, upper_bounds)
                    
                    # Esegue il fitting sul range filtrato
                    if method == 'differential_evolution':
                        # Usa differential evolution per ottimizzazione globale
                        def objective(params):
                            return np.sum((model_func(wl_filtered, *params) - int_filtered) ** 2)
                        
                        result = differential_evolution(objective, bounds=list(zip(lower_bounds, upper_bounds)),
                                                      maxiter=max_iter, seed=42)
                        
                        if result.success:
                            popt = result.x
                            # Calcola R² manualmente sul range filtrato
                            y_pred_filtered = model_func(wl_filtered, *popt)
                            ss_res = np.sum((int_filtered - y_pred_filtered) ** 2)
                            ss_tot = np.sum((int_filtered - np.mean(int_filtered)) ** 2)
                            r2 = 1 - (ss_res / ss_tot)
                            pcov = None
                        else:
                            raise Exception("Differential evolution non è convergente")
                    
                    else:
                        # Usa curve_fit standard sul range filtrato
                        popt, pcov = curve_fit(model_func, wl_filtered, int_filtered,
                                             p0=initial_params, bounds=bounds,
                                             maxfev=max_iter)
                        
                        # Calcola R² sul range filtrato
                        y_pred_filtered = model_func(wl_filtered, *popt)
                        ss_res = np.sum((int_filtered - y_pred_filtered) ** 2)
                        ss_tot = np.sum((int_filtered - np.mean(int_filtered)) ** 2)
                        r2 = 1 - (ss_res / ss_tot)
                    
                    # Genera spettro completo (tutti i wavelengths) per visualizzazione
                    y_pred_full = model_func(wavelengths, *popt)
                    
                    # Salva risultati
                    self.deconvoluted_data[original_col] = {
                        'fitted_spectrum': y_pred_full,
                        'parameters': popt,
                        'covariance': pcov,
                        'r2': r2,
                        'method': method,
                        'peak_type': peak_type,
                        'n_peaks': n_peaks,
                        'fitting_range': [wl_filtered.min(), wl_filtered.max()] if self.use_range_var.get() else None
                    }
                    
                    # Estrae componenti individuali sui wavelengths completi
                    components = {}
                    for peak_idx in range(n_peaks):
                        start_idx = peak_idx * params_per_peak
                        
                        if peak_type == 'voigt':
                            amplitude = popt[start_idx]
                            center = popt[start_idx + 1]
                            width_g = popt[start_idx + 2]
                            width_l = popt[start_idx + 3]
                            component = voigt_peak(wavelengths, amplitude, center, width_g, width_l)
                            
                            components[f'Peak_{peak_idx + 1}'] = {
                                'intensity': component,
                                'amplitude': amplitude,
                                'center': center,
                                'width_gaussian': width_g,
                                'width_lorentzian': width_l,
                                'area': np.trapz(component, wavelengths)
                            }
                        else:
                            amplitude = popt[start_idx]
                            center = popt[start_idx + 1]
                            width = popt[start_idx + 2]
                            
                            if peak_type == 'gaussian':
                                component = gaussian_peak(wavelengths, amplitude, center, width)
                            else:  # lorentzian
                                component = lorentzian_peak(wavelengths, amplitude, center, width)
                            
                            components[f'Peak_{peak_idx + 1}'] = {
                                'intensity': component,
                                'amplitude': amplitude,
                                'center': center,
                                'width': width,
                                'area': np.trapz(component, wavelengths),
                                'fwhm': 2 * width if peak_type == 'lorentzian' else 2.355 * width  # FWHM
                            }
                    
                    self.peak_components[original_col] = components
                    successful_decomp += 1
                    
                except Exception as e:
                    print(f"Errore nella deconvoluzione di {original_col}: {e}")
                    continue
            
            # Mostra risultati finali
            if successful_decomp > 0:
                avg_r2 = np.mean([data['r2'] for data in self.deconvoluted_data.values()])
                success_text = f"✅ Deconvoluzione completata: {successful_decomp}/{total_spectra} spettri (R² medio: {avg_r2:.4f})"
                self.deconv_results.config(text=success_text, foreground="green")
                
                # Mostra risultati dettagliati
                self.show_deconvolution_summary()
            else:
                error_text = "❌ Nessuna deconvoluzione riuscita"
                self.deconv_results.config(text=error_text, foreground="red")
            
        except Exception as e:
            error_msg = f"❌ Errore nella deconvoluzione: {e}"
            self.deconv_results.config(text=error_msg, foreground="red")
            messagebox.showerror("Errore", f"Errore nella deconvoluzione:\n{e}")
    
    def show_deconvolution_summary(self):
        """Mostra un riassunto dei risultati della deconvoluzione"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Riassunto Deconvoluzione Spettrale")
        summary_window.geometry("800x600")
        
        # Text widget con scrollbar
        main_frame = ttk.Frame(summary_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Costruisce il report
        report = "🧬 RISULTATI DECONVOLUZIONE SPETTRALE\n"
        report += "=" * 70 + "\n\n"
        
        for spectrum, data in self.deconvoluted_data.items():
            report += f"📊 Spettro: {spectrum}\n"
            report += f"Metodo: {data['method']} | Tipo picco: {data['peak_type']} | R²: {data['r2']:.6f}\n\n"
            
            if spectrum in self.peak_components:
                components = self.peak_components[spectrum]
                
                for peak_name, params in components.items():
                    report += f"  🔸 {peak_name}:\n"
                    report += f"    • Centro: {params['center']:.2f} nm\n"
                    report += f"    • Ampiezza: {params['amplitude']:.2f}\n"
                    
                    if 'width' in params:
                        report += f"    • Larghezza: {params['width']:.2f} nm\n"
                        report += f"    • FWHM: {params['fwhm']:.2f} nm\n"
                    else:
                        report += f"    • Larghezza Gaussiana: {params['width_gaussian']:.2f} nm\n"
                        report += f"    • Larghezza Lorentziana: {params['width_lorentzian']:.2f} nm\n"
                    
                    report += f"    • Area: {params['area']:.2f}\n"
                    
                    # Calcola percentuale di contributo
                    total_area = sum(comp['area'] for comp in components.values())
                    contribution = (params['area'] / total_area) * 100
                    report += f"    • Contributo: {contribution:.1f}%\n\n"
                
                # Separazioni tra picchi
                centers = [params['center'] for params in components.values()]
                if len(centers) >= 2:
                    centers.sort()
                    report += "  📏 Separazioni tra picchi:\n"
                    for i in range(len(centers) - 1):
                        separation = centers[i + 1] - centers[i]
                        report += f"    • Picco {i+1} → Picco {i+2}: {separation:.2f} nm\n"
                    report += "\n"
            
            report += "-" * 50 + "\n\n"
        
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def plot_deconvolution(self):
        """Visualizza gli spettri con le componenti individuali"""
        if self.deconvoluted_data is None or len(self.deconvoluted_data) == 0:
            messagebox.showwarning("Attenzione", "Esegui prima la deconvoluzione")
            return
        
        try:
            wavelength_col = self.fitted_data.columns[0]
            wavelengths = self.fitted_data[wavelength_col].values
            
            n_spectra = len(self.deconvoluted_data)
            
            # Crea subplot per ogni spettro
            if n_spectra == 1:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                axes = [ax]
            else:
                cols = min(3, n_spectra)
                rows = (n_spectra + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
                if n_spectra == 2:
                    axes = [axes[0], axes[1]]
                elif n_spectra > 2:
                    axes = axes.flatten()
            
            for idx, (spectrum, data) in enumerate(self.deconvoluted_data.items()):
                ax = axes[idx] if n_spectra > 1 else axes[0]
                
                # Plot spettro originale fittato
                fitted_col = f"{spectrum}_fitted"
                original_intensities = self.fitted_data[fitted_col].values
                ax.plot(wavelengths, original_intensities, 'k-', linewidth=2, 
                       label=f'{spectrum} (originale)', alpha=0.7)
                
                # Plot fitting deconvoluto
                ax.plot(wavelengths, data['fitted_spectrum'], 'r--', linewidth=2,
                       label=f'Fitting (R²={data["r2"]:.4f})')
                
                # Plot componenti individuali
                if spectrum in self.peak_components:
                    components = self.peak_components[spectrum]
                    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
                    
                    for i, (peak_name, params) in enumerate(components.items()):
                        ax.plot(wavelengths, params['intensity'], '--', 
                               color=colors[i], linewidth=1.5, alpha=0.8,
                               label=f"{peak_name} ({params['center']:.1f} nm)")
                        
                        # Annotazione centro picco
                        max_idx = np.argmax(params['intensity'])
                        ax.annotate(f'{params["center"]:.1f}', 
                                   xy=(wavelengths[max_idx], params['intensity'][max_idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, color=colors[i])
                
                ax.set_xlabel('Lunghezza d\'onda (nm)')
                ax.set_ylabel('Intensità')
                ax.set_title(f'Deconvoluzione: {spectrum}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Nasconde assi vuoti se necessario
            if n_spectra > 1:
                for idx in range(n_spectra, len(axes)):
                    axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella visualizzazione: {e}")
    
    def show_peak_parameters(self):
        """Mostra i parametri dettagliati di tutti i picchi"""
        if self.peak_components is None or len(self.peak_components) == 0:
            messagebox.showwarning("Attenzione", "Esegui prima la deconvoluzione")
            return
        
        params_window = tk.Toplevel(self.root)
        params_window.title("Parametri Picchi Individuali")
        params_window.geometry("1000x700")
        
        # Crea notebook per organizzare i dati per spettro
        notebook = ttk.Notebook(params_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for spectrum, components in self.peak_components.items():
            # Frame per ogni spettro
            spectrum_frame = ttk.Frame(notebook)
            notebook.add(spectrum_frame, text=spectrum)
            
            # Treeview per mostrare i parametri
            tree_frame = ttk.Frame(spectrum_frame)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Definisce colonne basate sul tipo di picco
            sample_component = next(iter(components.values()))
            
            if 'width_gaussian' in sample_component:
                # Picchi Voigt
                columns = ('Picco', 'Centro (nm)', 'Ampiezza', 'Width_G (nm)', 
                          'Width_L (nm)', 'Area', 'Contributo (%)')
            else:
                # Picchi Gaussiani/Lorentziani
                columns = ('Picco', 'Centro (nm)', 'Ampiezza', 'Width (nm)', 
                          'FWHM (nm)', 'Area', 'Contributo (%)')
            
            tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)
            
            # Configura headers
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=120)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Popola dati
            total_area = sum(comp['area'] for comp in components.values())
            
            for peak_name, params in components.items():
                contribution = (params['area'] / total_area) * 100
                
                if 'width_gaussian' in params:
                    # Voigt
                    values = (peak_name, 
                             f"{params['center']:.2f}",
                             f"{params['amplitude']:.2f}",
                             f"{params['width_gaussian']:.2f}",
                             f"{params['width_lorentzian']:.2f}",
                             f"{params['area']:.2f}",
                             f"{contribution:.1f}")
                else:
                    # Gaussiano/Lorentziano
                    values = (peak_name,
                             f"{params['center']:.2f}",
                             f"{params['amplitude']:.2f}",
                             f"{params['width']:.2f}",
                             f"{params['fwhm']:.2f}",
                             f"{params['area']:.2f}",
                             f"{contribution:.1f}")
                
                tree.insert('', 'end', values=values)
            
            # Aggiungi statistiche riassuntive
            stats_frame = ttk.LabelFrame(spectrum_frame, text="Statistiche", padding="10")
            stats_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            deconv_data = self.deconvoluted_data[spectrum]
            centers = [params['center'] for params in components.values()]
            
            stats_text = f"""R² deconvoluzione: {deconv_data['r2']:.6f}
Numero picchi: {len(components)}
Range centri: {min(centers):.2f} - {max(centers):.2f} nm
Separazione media: {np.mean(np.diff(sorted(centers))):.2f} nm (se > 1 picco)
Area totale: {total_area:.2f}"""
            
            stats_label = ttk.Label(stats_frame, text=stats_text, font=("Consolas", 10))
            stats_label.pack()
    
    def export_analysis(self):
        """Esporta tutti i risultati dell'analisi con selezione dei separatori"""
        if self.fitted_data is None:
            messagebox.showwarning("Attenzione", "Calcola prima il fitting polinomiale")
            return
        
        # Ottiene i separatori dall'utente
        selected_field_sep, selected_decimal_sep = self.create_export_separators_dialog("Seleziona Separatori per Esportazione Analisi")
        
        # Se l'utente ha annullato, esce
        if selected_field_sep is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salva risultati analisi",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path) as writer:
                        # Sheet 1: Dati originali e fittati
                        combined_data = self.data.copy()
                        for col in self.fitted_data.columns:
                            if col not in combined_data.columns:
                                combined_data[col] = self.fitted_data[col]
                        combined_data.to_excel(writer, sheet_name='Dati_e_Fitting', index=False)
                        
                        # Sheet 2: Statistiche fitting
                        if self.fitting_stats:
                            stats_df = pd.DataFrame(self.fitting_stats).T
                            stats_df.to_excel(writer, sheet_name='Statistiche_Fitting')
                        
                        # Sheet 3: Coefficienti polinomiali
                        if self.polynomial_coeffs:
                            coeffs_data = []
                            for spectrum, coeffs in self.polynomial_coeffs.items():
                                coeffs_data.append({
                                    'Spettro': spectrum,
                                    'Grado': coeffs['degree'],
                                    'R2': coeffs['r2'],
                                    'Intercetta': coeffs['intercept'],
                                    'Coefficienti': str(coeffs['coefficients'].tolist())
                                })
                            coeffs_df = pd.DataFrame(coeffs_data)
                            coeffs_df.to_excel(writer, sheet_name='Coefficienti_Polinomiali', index=False)
                
                else:
                    # CSV con dati combinati e separatori personalizzati
                    combined_data = self.data.copy()
                    for col in self.fitted_data.columns:
                        if col not in combined_data.columns:
                            combined_data[col] = self.fitted_data[col]
                    
                    # Applica conversione separatori decimali se necessario
                    if selected_decimal_sep == ",":
                        numeric_cols = combined_data.columns[1:]
                        for col in numeric_cols:
                            if combined_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                combined_data[col] = combined_data[col].astype(str).str.replace('.', ',', regex=False)
                    
                    combined_data.to_csv(file_path, sep=selected_field_sep, index=False,
                                       float_format='%.6f' if selected_decimal_sep == '.' else None)
                
                # Messaggio di successo
                profile_name = ""
                if selected_field_sep == ";" and selected_decimal_sep == ",":
                    profile_name = " (Formato Europeo)"
                elif selected_field_sep == "," and selected_decimal_sep == ".":
                    profile_name = " (Formato Americano)"
                elif selected_field_sep == "\t":
                    profile_name = " (Formato Tab-delimited)"
                
                messagebox.showinfo("Successo", f"Risultati analisi esportati con successo{profile_name}")
                
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nell'esportazione: {e}")
    
    def export_deconvolution(self):
        """Esporta tutti i risultati della deconvoluzione con selezione dei separatori"""
        if self.deconvoluted_data is None or len(self.deconvoluted_data) == 0:
            messagebox.showwarning("Attenzione", "Esegui prima la deconvoluzione")
            return
        
        # Ottiene i separatori dall'utente
        selected_field_sep, selected_decimal_sep = self.create_export_separators_dialog("Seleziona Separatori per Esportazione Deconvoluzione")
        
        # Se l'utente ha annullato, esce
        if selected_field_sep is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salva risultati deconvoluzione",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                wavelength_col = self.fitted_data.columns[0]
                wavelengths = self.fitted_data[wavelength_col].values
                
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path) as writer:
                        # Sheet 1: Spettri deconvoluti
                        deconv_spectra = pd.DataFrame({wavelength_col: wavelengths})
                        
                        for spectrum, data in self.deconvoluted_data.items():
                            deconv_spectra[f"{spectrum}_deconv"] = data['fitted_spectrum']
                            
                            # Aggiungi componenti individuali
                            if spectrum in self.peak_components:
                                components = self.peak_components[spectrum]
                                for peak_name, params in components.items():
                                    deconv_spectra[f"{spectrum}_{peak_name}"] = params['intensity']
                        
                        deconv_spectra.to_excel(writer, sheet_name='Spettri_Deconvoluti', index=False)
                        
                        # Sheet 2: Parametri picchi
                        all_params = []
                        for spectrum, components in self.peak_components.items():
                            total_area = sum(comp['area'] for comp in components.values())
                            
                            for peak_name, params in components.items():
                                contribution = (params['area'] / total_area) * 100
                                
                                param_row = {
                                    'Spettro': spectrum,
                                    'Picco': peak_name,
                                    'Centro_nm': params['center'],
                                    'Ampiezza': params['amplitude'],
                                    'Area': params['area'],
                                    'Contributo_perc': contribution
                                }
                                
                                if 'width' in params:
                                    param_row['Width_nm'] = params['width']
                                    param_row['FWHM_nm'] = params['fwhm']
                                else:
                                    param_row['Width_Gaussian_nm'] = params['width_gaussian']
                                    param_row['Width_Lorentzian_nm'] = params['width_lorentzian']
                                
                                all_params.append(param_row)
                        
                        params_df = pd.DataFrame(all_params)
                        params_df.to_excel(writer, sheet_name='Parametri_Picchi', index=False)
                        
                        # Sheet 3: Statistiche deconvoluzione
                        stats_data = []
                        for spectrum, data in self.deconvoluted_data.items():
                            stats_data.append({
                                'Spettro': spectrum,
                                'R2_deconvoluzione': data['r2'],
                                'Numero_picchi': data['n_peaks'],
                                'Tipo_picco': data['peak_type'],
                                'Metodo': data['method']
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        stats_df.to_excel(writer, sheet_name='Statistiche_Deconvoluzione', index=False)
                
                else:
                    # CSV semplice con spettri deconvoluti e separatori personalizzati
                    deconv_spectra = pd.DataFrame({wavelength_col: wavelengths})
                    
                    for spectrum, data in self.deconvoluted_data.items():
                        deconv_spectra[f"{spectrum}_deconv"] = data['fitted_spectrum']
                        
                        if spectrum in self.peak_components:
                            components = self.peak_components[spectrum]
                            for peak_name, params in components.items():
                                deconv_spectra[f"{spectrum}_{peak_name}"] = params['intensity']
                    
                    # Applica conversione separatori decimali se necessario
                    if selected_decimal_sep == ",":
                        numeric_cols = deconv_spectra.columns[1:]
                        for col in numeric_cols:
                            if deconv_spectra[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                deconv_spectra[col] = deconv_spectra[col].astype(str).str.replace('.', ',', regex=False)
                    
                    deconv_spectra.to_csv(file_path, sep=selected_field_sep, index=False,
                                        float_format='%.6f' if selected_decimal_sep == '.' else None)
                
                # Messaggio di successo
                profile_name = ""
                if selected_field_sep == ";" and selected_decimal_sep == ",":
                    profile_name = " (Formato Europeo)"
                elif selected_field_sep == "," and selected_decimal_sep == ".":
                    profile_name = " (Formato Americano)"
                elif selected_field_sep == "\t":
                    profile_name = " (Formato Tab-delimited)"
                
                messagebox.showinfo("Successo", f"Risultati deconvoluzione esportati con successo{profile_name}")
                
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nell'esportazione: {e}")
    
    def show_info(self):
        """Mostra informazioni sul dataset"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        info = f"""Informazioni Dataset:
        
Righe: {len(self.data)}
Colonne: {len(self.data.columns)}

Colonne:
{chr(10).join(f"- {col}" for col in self.data.columns)}

Statistiche colonne numeriche:
{self.data.describe().to_string()}
"""
        
        # Crea una finestra per mostrare le info
        info_window = tk.Toplevel(self.root)
        info_window.title("Informazioni Dataset")
        info_window.geometry("600x400")
        
        text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, info)
        text_widget.config(state=tk.DISABLED)
    
    # =============================================================================
    # METODI PER SMOOTHING E DECONVOLUZIONE SU DATI ORIGINALI
    # =============================================================================
    
    def on_smoothing_change(self, event=None):
        """Aggiorna i parametri quando cambia il tipo di smoothing"""
        smoothing_type = self.smoothing_type_var.get()
        
        if smoothing_type == "none":
            self.smooth_window_spin.config(state="disabled")
            self.smooth_poly_spin.config(state="disabled")
        elif smoothing_type == "savgol":
            self.smooth_window_spin.config(state="normal")
            self.smooth_poly_spin.config(state="normal")
            # Assicura che window sia dispari
            if self.smooth_window_var.get() % 2 == 0:
                self.smooth_window_var.set(self.smooth_window_var.get() + 1)
        elif smoothing_type == "moving_avg":
            self.smooth_window_spin.config(state="normal")
            self.smooth_poly_spin.config(state="disabled")
        elif smoothing_type == "gaussian":
            self.smooth_window_spin.config(state="normal")  # Usato come sigma
            self.smooth_poly_spin.config(state="disabled")
    
    def apply_smoothing(self):
        """Applica smoothing ai dati attivi"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        if not SCIPY_AVAILABLE and self.smoothing_type_var.get() in ["savgol", "gaussian"]:
            messagebox.showerror("Errore", "Scipy richiesto per Savitzky-Golay e Gaussian filter")
            return
        
        try:
            active_data = self.get_active_data()
            analysis_cols = self.get_analysis_columns()
            
            if not analysis_cols:
                messagebox.showwarning("Attenzione", "Nessuna colonna disponibile")
                return
            
            wavelength_col = active_data.columns[0]
            wavelengths = active_data[wavelength_col].values
            
            smoothing_type = self.smoothing_type_var.get()
            window = self.smooth_window_var.get()
            poly_order = self.smooth_poly_var.get()
            
            # Inizializza dati smoothed
            self.smoothed_data = pd.DataFrame()
            self.smoothed_data[wavelength_col] = wavelengths
            
            successful_smooth = 0
            
            for col in analysis_cols:
                intensities = active_data[col].values
                
                if smoothing_type == "none":
                    smoothed_intensities = intensities
                elif smoothing_type == "savgol":
                    if window >= len(intensities):
                        window = len(intensities) - 1 if len(intensities) % 2 == 0 else len(intensities) - 2
                        if window < 3:
                            continue
                    if window % 2 == 0:
                        window -= 1
                    if poly_order >= window:
                        poly_order = window - 1
                    smoothed_intensities = savgol_filter(intensities, window, poly_order)
                elif smoothing_type == "moving_avg":
                    # Moving average con pandas
                    smoothed_intensities = pd.Series(intensities).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
                elif smoothing_type == "gaussian":
                    sigma = window / 3.0  # Converte window in sigma
                    smoothed_intensities = gaussian_filter1d(intensities, sigma=sigma)
                
                self.smoothed_data[f"{col}_smoothed"] = smoothed_intensities
                successful_smooth += 1
            
            if successful_smooth > 0:
                result_text = f"✅ Smoothing '{smoothing_type}' applicato a {successful_smooth} spettri"
                if smoothing_type == "savgol":
                    result_text += f" (window={window}, poly={poly_order})"
                elif smoothing_type in ["moving_avg", "gaussian"]:
                    result_text += f" (window/sigma={window})"
                
                self.smoothing_results.config(text=result_text, foreground="green")
                messagebox.showinfo("Successo", result_text)
            else:
                self.smoothing_results.config(text="❌ Nessun smoothing applicato", foreground="red")
                
        except Exception as e:
            error_msg = f"❌ Errore nel smoothing: {e}"
            self.smoothing_results.config(text=error_msg, foreground="red")
            messagebox.showerror("Errore", f"Errore nell'applicazione smoothing:\n{e}")
    
    def copy_centers_from_deconv(self):
        """Copia i centri dalla deconvoluzione precedente"""
        if self.peak_components is None or len(self.peak_components) == 0:
            messagebox.showwarning("Attenzione", "Esegui prima la deconvoluzione spettrale")
            return
        
        try:
            # Raccoglie tutti i centri da tutti gli spettri
            all_centers = []
            for spectrum, components in self.peak_components.items():
                centers = [params['center'] for params in components.values()]
                all_centers.extend(centers)
            
            # Rimuove duplicati e ordina
            unique_centers = sorted(list(set(all_centers)))
            
            # Formatta come stringa
            centers_str = ", ".join([f"{center:.1f}" for center in unique_centers])
            self.smooth_centers_var.set(centers_str)
            
            # Aggiorna anche il numero di picchi
            self.smooth_n_peaks_var.set(len(unique_centers))
            
            messagebox.showinfo("Centri copiati", f"Copiati {len(unique_centers)} centri unici: {centers_str}")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella copia centri: {e}")
    
    def deconvolve_original_data(self):
        """Esegue deconvoluzione sui dati originali (eventualmente smoothed) nel range specificato"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        if not SCIPY_AVAILABLE:
            messagebox.showerror("Errore", "Scipy richiesto per deconvoluzione")
            return
        
        try:
            # Usa dati smoothed se disponibili, altrimenti dati attivi
            if self.smoothed_data is not None:
                source_data = self.smoothed_data
                analysis_cols = [col.replace('_smoothed', '') for col in source_data.columns if col.endswith('_smoothed')]
                data_suffix = '_smoothed'
                data_label = "smoothed"
            else:
                source_data = self.get_active_data()
                analysis_cols = self.get_analysis_columns()
                data_suffix = ''
                data_label = "originali"
            
            if not analysis_cols:
                messagebox.showwarning("Attenzione", "Nessuna colonna disponibile")
                return
            
            wavelength_col = source_data.columns[0]
            wavelengths = source_data[wavelength_col].values
            
            # Ottiene parametri
            n_peaks = self.smooth_n_peaks_var.get()
            peak_type = self.smooth_peak_type_var.get()
            method = self.smooth_fitting_method_var.get()
            max_iter = self.smooth_max_iter_var.get()
            
            # Ottiene centri manuali o auto-detect se vuoto
            centers_str = self.smooth_centers_var.get().strip()
            if centers_str:
                # Usa centri manuali
                manual_centers = [float(c.strip()) for c in centers_str.split(',') if c.strip()]
            else:
                # Auto-detection: usa None per far partire il rilevamento automatico
                manual_centers = None
            
            if not manual_centers:
                messagebox.showwarning("Attenzione", "Nessun centro valido specificato")
                return
            
            # Seleziona modello
            if peak_type == 'gaussian':
                model_func = multi_gaussian
                params_per_peak = 3
            elif peak_type == 'lorentzian':
                model_func = multi_lorentzian
                params_per_peak = 3
            elif peak_type == 'voigt':
                model_func = multi_voigt
                params_per_peak = 4
            
            self.original_deconvoluted_data = {}
            self.original_peak_components = {}
            successful_decomp = 0
            
            for i, col in enumerate(analysis_cols):
                intensities = source_data[f"{col}{data_suffix}"].values
                
                # Applica range se selezionato
                wl_filtered, int_filtered = self.get_wavelength_range(wavelengths, intensities)
                
                try:
                    # Stima parametri iniziali
                    # Usa exact_centers dalla sezione smoothing
                    exact_centers = self.smooth_exact_centers_var.get()
                    initial_params = estimate_initial_parameters(wl_filtered, int_filtered, 
                                                               n_peaks, peak_type, manual_centers, exact_centers)
                    
                    # Definisce bounds per i parametri basati sul range filtrato
                    x_min, x_max = wl_filtered.min(), wl_filtered.max()
                    y_min, y_max = int_filtered.min(), int_filtered.max()
                    x_range = x_max - x_min
                    
                    # Verifica se usare centri fissi (dalla sezione smoothing)
                    fixed_centers = self.smooth_fixed_centers_var.get() and manual_centers is not None
                    
                    if peak_type == 'voigt':
                        # [amplitude, center, width_g, width_l] per ogni picco
                        lower_bounds = []
                        upper_bounds = []
                        for j in range(n_peaks):
                            # Amplitude bounds
                            lower_bounds.append(0)
                            upper_bounds.append(y_max * 2)
                            
                            # Center bounds - USA CENTRI FISSI SE RICHIESTO
                            if fixed_centers and j < len(manual_centers):
                                # Blocca centro con tolleranza di ±0.1 nm
                                center = manual_centers[j]
                                lower_bounds.append(center - 0.1)
                                upper_bounds.append(center + 0.1)
                            else:
                                lower_bounds.append(x_min)
                                upper_bounds.append(x_max)
                            
                            # Width bounds
                            lower_bounds.extend([x_range/100, x_range/100])
                            upper_bounds.extend([x_range/2, x_range/2])
                    else:
                        # [amplitude, center, width] per ogni picco
                        lower_bounds = []
                        upper_bounds = []
                        for j in range(n_peaks):
                            # Amplitude bounds
                            lower_bounds.append(0)
                            upper_bounds.append(y_max * 2)
                            
                            # Center bounds - USA CENTRI FISSI SE RICHIESTO
                            if fixed_centers and j < len(manual_centers):
                                # Blocca centro con tolleranza di ±0.1 nm
                                center = manual_centers[j]
                                lower_bounds.append(center - 0.1)
                                upper_bounds.append(center + 0.1)
                            else:
                                lower_bounds.append(x_min)
                                upper_bounds.append(x_max)
                            
                            # Width bounds
                            lower_bounds.append(x_range/100)
                            upper_bounds.append(x_range/2)
                    
                    bounds = (lower_bounds, upper_bounds)
                    
                    # Esegue il fitting sul range filtrato - USA ALGORITMO SELEZIONATO
                    if method == 'differential_evolution':
                        # Usa differential evolution per ottimizzazione globale
                        def objective(params):
                            return np.sum((model_func(wl_filtered, *params) - int_filtered) ** 2)
                        
                        result = differential_evolution(objective, bounds=list(zip(lower_bounds, upper_bounds)),
                                                      maxiter=max_iter, seed=42)
                        
                        if result.success:
                            popt = result.x
                            # Calcola R² manualmente sul range filtrato
                            y_pred_filtered = model_func(wl_filtered, *popt)
                            ss_res = np.sum((int_filtered - y_pred_filtered) ** 2)
                            ss_tot = np.sum((int_filtered - np.mean(int_filtered)) ** 2)
                            r2 = 1 - (ss_res / ss_tot)
                            pcov = None
                        else:
                            raise Exception("Differential evolution non è convergente")
                    else:
                        # Usa curve_fit standard sul range filtrato
                        popt, pcov = curve_fit(model_func, wl_filtered, int_filtered,
                                             p0=initial_params, bounds=bounds,
                                             maxfev=max_iter)
                        
                        # Calcola R² sul range filtrato
                        y_pred_filtered = model_func(wl_filtered, *popt)
                        ss_res = np.sum((int_filtered - y_pred_filtered) ** 2)
                        ss_tot = np.sum((int_filtered - np.mean(int_filtered)) ** 2)
                        r2 = 1 - (ss_res / ss_tot)
                    
                    # Genera spettro completo
                    y_pred_full = model_func(wavelengths, *popt)
                    
                    # Salva risultati
                    self.original_deconvoluted_data[col] = {
                        'fitted_spectrum': y_pred_full,
                        'parameters': popt,
                        'covariance': pcov if method == 'curve_fit' else None,
                        'r2': r2,
                        'method': method,
                        'peak_type': peak_type,
                        'n_peaks': n_peaks,
                        'data_type': data_label
                    }
                    
                    # Estrae componenti
                    components = {}
                    for peak_idx in range(n_peaks):
                        start_idx = peak_idx * params_per_peak
                        
                        if peak_type == 'voigt':
                            amplitude = popt[start_idx]
                            center = popt[start_idx + 1]
                            width_g = popt[start_idx + 2]
                            width_l = popt[start_idx + 3]
                            component = voigt_peak(wavelengths, amplitude, center, width_g, width_l)
                            
                            components[f'Peak_{peak_idx + 1}'] = {
                                'intensity': component,
                                'amplitude': amplitude,
                                'center': center,
                                'width_gaussian': width_g,
                                'width_lorentzian': width_l,
                                'area': np.trapz(component, wavelengths)
                            }
                        else:
                            amplitude = popt[start_idx]
                            center = popt[start_idx + 1]
                            width = popt[start_idx + 2]
                            
                            if peak_type == 'gaussian':
                                component = gaussian_peak(wavelengths, amplitude, center, width)
                            else:
                                component = lorentzian_peak(wavelengths, amplitude, center, width)
                            
                            components[f'Peak_{peak_idx + 1}'] = {
                                'intensity': component,
                                'amplitude': amplitude,
                                'center': center,
                                'width': width,
                                'area': np.trapz(component, wavelengths),
                                'fwhm': 2 * width if peak_type == 'lorentzian' else 2.355 * width
                            }
                    
                    self.original_peak_components[col] = components
                    successful_decomp += 1
                    
                except Exception as e:
                    print(f"Errore deconvoluzione {col}: {e}")
                    continue
            
            if successful_decomp > 0:
                avg_r2 = np.mean([data['r2'] for data in self.original_deconvoluted_data.values()])
                success_text = f"✅ Deconvoluzione dati {data_label}: {successful_decomp} spettri (R² medio: {avg_r2:.4f}) | {method} | {peak_type}"
                self.smoothing_results.config(text=success_text, foreground="green")
                messagebox.showinfo("Successo", success_text)
            else:
                error_text = f"❌ Nessuna deconvoluzione su dati {data_label} riuscita"
                self.smoothing_results.config(text=error_text, foreground="red")
            
        except Exception as e:
            error_msg = f"❌ Errore deconvoluzione dati originali: {e}"
            self.smoothing_results.config(text=error_msg, foreground="red")
            messagebox.showerror("Errore", f"Errore nella deconvoluzione:\n{e}")
    
    def compare_deconvolution_results(self):
        """Confronta i risultati della deconvoluzione su dati fitted vs originali"""
        if self.original_deconvoluted_data is None:
            messagebox.showwarning("Attenzione", "Esegui prima la deconvoluzione su dati originali")
            return
        
        if self.deconvoluted_data is None:
            messagebox.showwarning("Attenzione", "Esegui prima la deconvoluzione standard")
            return
        
        # Crea finestra di confronto
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Confronto Deconvoluzione: Fitted vs Originali")
        compare_window.geometry("900x600")
        
        # Text widget con scrollbar
        main_frame = ttk.Frame(compare_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Costruisce report comparativo
        report = "🔬 CONFRONTO DECONVOLUZIONE SPETTRALE\n"
        report += "=" * 70 + "\n\n"
        
        # Confronta spettri comuni
        common_spectra = set(self.deconvoluted_data.keys()) & set(self.original_deconvoluted_data.keys())
        
        for spectrum in common_spectra:
            fitted_data = self.deconvoluted_data[spectrum]
            original_data = self.original_deconvoluted_data[spectrum]
            
            report += f"📊 Spettro: {spectrum}\n"
            report += f"Dati Fitted    - R²: {fitted_data['r2']:.6f} | Tipo: {fitted_data['peak_type']}\n"
            report += f"Dati {original_data['data_type'].title():8} - R²: {original_data['r2']:.6f} | Tipo: {original_data['peak_type']}\n"
            report += f"Differenza R²: {abs(fitted_data['r2'] - original_data['r2']):.6f}\n\n"
            
            # Confronta parametri dei picchi
            if spectrum in self.peak_components and spectrum in self.original_peak_components:
                fitted_components = self.peak_components[spectrum]
                original_components = self.original_peak_components[spectrum]
                
                report += "Confronto Parametri Picchi:\n"
                for peak_name in fitted_components.keys():
                    if peak_name in original_components:
                        fitted_params = fitted_components[peak_name]
                        original_params = original_components[peak_name]
                        
                        report += f"  {peak_name}:\n"
                        report += f"    Centro - Fitted: {fitted_params['center']:.2f} nm | Original: {original_params['center']:.2f} nm | Δ: {abs(fitted_params['center'] - original_params['center']):.2f} nm\n"
                        report += f"    Ampiezza - Fitted: {fitted_params['amplitude']:.2f} | Original: {original_params['amplitude']:.2f} | Δ: {abs(fitted_params['amplitude'] - original_params['amplitude']):.2f}\n"
                        report += f"    Area - Fitted: {fitted_params['area']:.2f} | Original: {original_params['area']:.2f} | Δ: {abs(fitted_params['area'] - original_params['area']):.2f}\n\n"
            
            report += "-" * 50 + "\n\n"
        
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def export_smoothed_data(self):
        """Esporta i dati smoothed e i risultati della deconvoluzione"""
        if self.smoothed_data is None:
            messagebox.showwarning("Attenzione", "Applica prima lo smoothing")
            return
        
        # Dialog separatori
        selected_field_sep, selected_decimal_sep = self.create_export_separators_dialog("Esportazione Dati Smoothed")
        
        if selected_field_sep is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salva dati smoothed",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    with pd.ExcelWriter(file_path) as writer:
                        # Sheet 1: Dati smoothed
                        self.smoothed_data.to_excel(writer, sheet_name='Dati_Smoothed', index=False)
                        
                        # Sheet 2: Risultati deconvoluzione originali se disponibili
                        if self.original_deconvoluted_data:
                            wavelength_col = self.smoothed_data.columns[0]
                            wavelengths = self.smoothed_data[wavelength_col].values
                            
                            deconv_original = pd.DataFrame({wavelength_col: wavelengths})
                            
                            for spectrum, data in self.original_deconvoluted_data.items():
                                deconv_original[f"{spectrum}_deconv_original"] = data['fitted_spectrum']
                                
                                if spectrum in self.original_peak_components:
                                    components = self.original_peak_components[spectrum]
                                    for peak_name, params in components.items():
                                        deconv_original[f"{spectrum}_{peak_name}_original"] = params['intensity']
                            
                            deconv_original.to_excel(writer, sheet_name='Deconv_Originali', index=False)
                
                else:
                    # CSV con separatori personalizzati
                    export_data = self.smoothed_data.copy()
                    
                    if selected_decimal_sep == ",":
                        numeric_cols = export_data.columns[1:]
                        for col in numeric_cols:
                            if export_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                export_data[col] = export_data[col].astype(str).str.replace('.', ',', regex=False)
                    
                    export_data.to_csv(file_path, sep=selected_field_sep, index=False,
                                     float_format='%.6f' if selected_decimal_sep == '.' else None)
                
                messagebox.showinfo("Successo", "Dati smoothed esportati con successo")
                
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nell'esportazione: {e}")
    
    def calculate_gp_analysis(self):
        """Calcola analisi GP completa con altezze curve, aree e altezze picchi usando centri dinamici"""
        if self.data is None:
            messagebox.showwarning("Attenzione", "Carica prima i dati")
            return
        
        # Verifica disponibilità dati
        has_deconv_fitted = self.deconvoluted_data is not None and self.peak_components is not None
        has_deconv_original = self.original_deconvoluted_data is not None and self.original_peak_components is not None
        
        if not has_deconv_fitted and not has_deconv_original:
            messagebox.showwarning("Attenzione", "Esegui almeno una deconvoluzione (standard o su dati originali)")
            return
        
        # Dialog separatori
        selected_field_sep, selected_decimal_sep = self.create_export_separators_dialog("Esportazione Analisi GP")
        
        if selected_field_sep is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salva analisi GP",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                active_data = self.get_active_data()
                analysis_cols = self.get_analysis_columns()
                wavelength_col = active_data.columns[0]
                wavelengths = active_data[wavelength_col].values
                
                # Determina i centri dei picchi principali dinamicamente
                peak_centers = self._get_main_peak_centers()
                if not peak_centers or len(peak_centers) < 2:
                    messagebox.showerror("Errore", "Non sono stati trovati almeno 2 picchi principali per il calcolo GP")
                    return
                
                # Ordina i centri (minore, maggiore)
                peak_centers_sorted = sorted(peak_centers)
                center_1 = peak_centers_sorted[0]  # Picco a lunghezza d'onda minore
                center_2 = peak_centers_sorted[1]  # Picco a lunghezza d'onda maggiore
                
                results = []
                
                for col in analysis_cols:
                    row_data = {"Nome": col}
                    
                    # 1. Altezze curve originali ai centri dinamici (interpolazione)
                    intensities = active_data[col].values
                    height_1 = np.interp(center_1, wavelengths, intensities)
                    height_2 = np.interp(center_2, wavelengths, intensities)
                    
                    row_data[f"Altezza_Curva_{center_1:.1f}nm"] = height_1
                    row_data[f"Altezza_Curva_{center_2:.1f}nm"] = height_2
                    
                    # GP altezze curve: (minore - maggiore) / (minore + maggiore)
                    if height_1 + height_2 != 0:
                        gp_curve = (height_1 - height_2) / (height_1 + height_2)
                    else:
                        gp_curve = 0
                    row_data["GP_Altezze_Curve"] = gp_curve
                    
                    # 2. Dati deconvoluzione standard (su fitted)
                    if has_deconv_fitted and col in self.peak_components:
                        components = self.peak_components[col]
                        
                        # Trova picchi più vicini ai centri dinamici
                        peak_1_fitted = self._find_closest_peak(components, center_1)
                        peak_2_fitted = self._find_closest_peak(components, center_2)
                        
                        if peak_1_fitted:
                            row_data[f"Area_Gauss_{center_1:.1f}_Deconv"] = peak_1_fitted['area']
                            row_data[f"Altezza_Picco_{center_1:.1f}_Deconv"] = peak_1_fitted['amplitude']
                        else:
                            row_data[f"Area_Gauss_{center_1:.1f}_Deconv"] = 0
                            row_data[f"Altezza_Picco_{center_1:.1f}_Deconv"] = 0
                        
                        if peak_2_fitted:
                            row_data[f"Area_Gauss_{center_2:.1f}_Deconv"] = peak_2_fitted['area']
                            row_data[f"Altezza_Picco_{center_2:.1f}_Deconv"] = peak_2_fitted['amplitude']
                        else:
                            row_data[f"Area_Gauss_{center_2:.1f}_Deconv"] = 0
                            row_data[f"Altezza_Picco_{center_2:.1f}_Deconv"] = 0
                        
                        # GP per aree deconvolute
                        area_1 = row_data[f"Area_Gauss_{center_1:.1f}_Deconv"]
                        area_2 = row_data[f"Area_Gauss_{center_2:.1f}_Deconv"]
                        if area_1 + area_2 != 0:
                            row_data["GP_Aree_Deconv"] = (area_1 - area_2) / (area_1 + area_2)
                        else:
                            row_data["GP_Aree_Deconv"] = 0
                        
                        # GP per altezze picchi deconvoluti
                        height_1_deconv = row_data[f"Altezza_Picco_{center_1:.1f}_Deconv"]
                        height_2_deconv = row_data[f"Altezza_Picco_{center_2:.1f}_Deconv"]
                        if height_1_deconv + height_2_deconv != 0:
                            row_data["GP_Altezze_Picchi_Deconv"] = (height_1_deconv - height_2_deconv) / (height_1_deconv + height_2_deconv)
                        else:
                            row_data["GP_Altezze_Picchi_Deconv"] = 0
                    else:
                        # Valori vuoti se non disponibili
                        row_data.update({
                            f"Area_Gauss_{center_1:.1f}_Deconv": 0,
                            f"Area_Gauss_{center_2:.1f}_Deconv": 0,
                            f"Altezza_Picco_{center_1:.1f}_Deconv": 0,
                            f"Altezza_Picco_{center_2:.1f}_Deconv": 0,
                            "GP_Aree_Deconv": 0,
                            "GP_Altezze_Picchi_Deconv": 0
                        })
                    
                    # 3. Dati deconvoluzione originali
                    if has_deconv_original and col in self.original_peak_components:
                        orig_components = self.original_peak_components[col]
                        
                        # Trova picchi più vicini ai centri dinamici
                        peak_1_orig = self._find_closest_peak(orig_components, center_1)
                        peak_2_orig = self._find_closest_peak(orig_components, center_2)
                        
                        if peak_1_orig:
                            row_data[f"Area_Gauss_{center_1:.1f}_Original"] = peak_1_orig['area']
                            row_data[f"Altezza_Picco_{center_1:.1f}_Original"] = peak_1_orig['amplitude']
                        else:
                            row_data[f"Area_Gauss_{center_1:.1f}_Original"] = 0
                            row_data[f"Altezza_Picco_{center_1:.1f}_Original"] = 0
                        
                        if peak_2_orig:
                            row_data[f"Area_Gauss_{center_2:.1f}_Original"] = peak_2_orig['area']
                            row_data[f"Altezza_Picco_{center_2:.1f}_Original"] = peak_2_orig['amplitude']
                        else:
                            row_data[f"Area_Gauss_{center_2:.1f}_Original"] = 0
                            row_data[f"Altezza_Picco_{center_2:.1f}_Original"] = 0
                        
                        # GP per aree originali
                        area_1_orig = row_data[f"Area_Gauss_{center_1:.1f}_Original"]
                        area_2_orig = row_data[f"Area_Gauss_{center_2:.1f}_Original"]
                        if area_1_orig + area_2_orig != 0:
                            row_data["GP_Aree_Original"] = (area_1_orig - area_2_orig) / (area_1_orig + area_2_orig)
                        else:
                            row_data["GP_Aree_Original"] = 0
                        
                        # GP per altezze picchi originali
                        height_1_orig = row_data[f"Altezza_Picco_{center_1:.1f}_Original"]
                        height_2_orig = row_data[f"Altezza_Picco_{center_2:.1f}_Original"]
                        if height_1_orig + height_2_orig != 0:
                            row_data["GP_Altezze_Picchi_Original"] = (height_1_orig - height_2_orig) / (height_1_orig + height_2_orig)
                        else:
                            row_data["GP_Altezze_Picchi_Original"] = 0
                    else:
                        # Valori vuoti se non disponibili
                        row_data.update({
                            f"Area_Gauss_{center_1:.1f}_Original": 0,
                            f"Area_Gauss_{center_2:.1f}_Original": 0,
                            f"Altezza_Picco_{center_1:.1f}_Original": 0,
                            f"Altezza_Picco_{center_2:.1f}_Original": 0,
                            "GP_Aree_Original": 0,
                            "GP_Altezze_Picchi_Original": 0
                        })
                    
                    results.append(row_data)
                
                # Crea DataFrame
                df_results = pd.DataFrame(results)
                
                # Applica conversione separatori decimali se necessario
                if selected_decimal_sep == ",":
                    numeric_cols = [col for col in df_results.columns if col != "Nome"]
                    for col in numeric_cols:
                        if df_results[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            df_results[col] = df_results[col].astype(str).str.replace('.', ',', regex=False)
                
                # Esporta
                df_results.to_csv(file_path, sep=selected_field_sep, index=False,
                                float_format='%.6f' if selected_decimal_sep == '.' else None)
                
                # Messaggio successo
                profile_name = ""
                if selected_field_sep == ";" and selected_decimal_sep == ",":
                    profile_name = " (Formato Europeo)"
                elif selected_field_sep == "," and selected_decimal_sep == ".":
                    profile_name = " (Formato Americano)"
                elif selected_field_sep == "\t":
                    profile_name = " (Formato Tab-delimited)"
                
                success_msg = f"Analisi GP esportata: {len(results)} spettri (picchi: {center_1:.1f}, {center_2:.1f} nm){profile_name}"
                self.smoothing_results.config(text=f"✅ {success_msg}", foreground="green")
                messagebox.showinfo("Successo", success_msg)
                
            except Exception as e:
                error_msg = f"❌ Errore calcolo GP: {e}"
                self.smoothing_results.config(text=error_msg, foreground="red")
                messagebox.showerror("Errore", f"Errore nel calcolo GP:\n{e}")
    
    def _get_main_peak_centers(self):
        """Determina i centri dei due picchi principali dalle deconvoluzioni disponibili"""
        all_centers = []
        
        # Raccoglie centri dalla deconvoluzione standard
        if self.peak_components:
            for spectrum, components in self.peak_components.items():
                centers = [params['center'] for params in components.values()]
                all_centers.extend(centers)
        
        # Raccoglie centri dalla deconvoluzione originali  
        if self.original_peak_components:
            for spectrum, components in self.original_peak_components.items():
                centers = [params['center'] for params in components.values()]
                all_centers.extend(centers)
        
        if not all_centers:
            return []
        
        # Trova i due centri più comuni/rappresentativi
        # Raggruppa centri simili (tolleranza ±2 nm)
        center_groups = []
        for center in all_centers:
            added_to_group = False
            for group in center_groups:
                if any(abs(center - existing) <= 2.0 for existing in group):
                    group.append(center)
                    added_to_group = True
                    break
            if not added_to_group:
                center_groups.append([center])
        
        # Calcola la media di ogni gruppo e ordina per frequenza
        group_means = []
        for group in center_groups:
            mean_center = np.mean(group)
            frequency = len(group)
            group_means.append((mean_center, frequency))
        
        # Ordina per frequenza (più comune prima)
        group_means.sort(key=lambda x: x[1], reverse=True)
        
        # Prende i primi due centri più comuni
        main_centers = [center for center, freq in group_means[:2]]
        
        return main_centers
    
    def _find_closest_peak(self, components, target_wavelength):
        """Trova il picco più vicino alla lunghezza d'onda target"""
        closest_peak = None
        min_distance = float('inf')
        
        for peak_name, params in components.items():
            distance = abs(params['center'] - target_wavelength)
            if distance < min_distance:
                min_distance = distance
                closest_peak = params
        
        # Restituisce il picco solo se è entro 20nm dalla target
        if closest_peak and min_distance <= 20:
            return closest_peak
        return None
    
    def plot_original_deconvolution(self):
        """Visualizza la deconvoluzione sui dati originali con componenti individuali"""
        if self.original_deconvoluted_data is None or len(self.original_deconvoluted_data) == 0:
            messagebox.showwarning("Attenzione", "Esegui prima la deconvoluzione su dati originali")
            return
        
        try:
            # Usa dati attivi per wavelengths
            active_data = self.get_active_data()
            wavelength_col = active_data.columns[0]
            wavelengths = active_data[wavelength_col].values
            
            n_spectra = len(self.original_deconvoluted_data)
            
            # Crea subplot per ogni spettro
            if n_spectra == 1:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                axes = [ax]
            else:
                cols = min(3, n_spectra)
                rows = (n_spectra + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
                if n_spectra == 2:
                    axes = [axes[0], axes[1]]
                elif n_spectra > 2:
                    axes = axes.flatten()
            
            for idx, (spectrum, data) in enumerate(self.original_deconvoluted_data.items()):
                ax = axes[idx] if n_spectra > 1 else axes[0]
                
                # Plot spettro originale (attivo: corretto bianco o smoothed)
                if self.smoothed_data is not None and f"{spectrum}_smoothed" in self.smoothed_data.columns:
                    # Usa dati smoothed se disponibili
                    original_intensities = self.smoothed_data[f"{spectrum}_smoothed"].values
                    data_label = f'{spectrum} (smoothed)'
                else:
                    # Usa dati attivi
                    original_intensities = active_data[spectrum].values
                    data_label = f'{spectrum} (originale)'
                
                ax.plot(wavelengths, original_intensities, 'k-', linewidth=2, 
                       label=data_label, alpha=0.7)
                
                # Plot fitting deconvoluto
                ax.plot(wavelengths, data['fitted_spectrum'], 'r--', linewidth=2,
                       label=f'Fitting (R²={data["r2"]:.4f})')
                
                # Plot componenti individuali
                if spectrum in self.original_peak_components:
                    components = self.original_peak_components[spectrum]
                    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
                    
                    for i, (peak_name, params) in enumerate(components.items()):
                        ax.plot(wavelengths, params['intensity'], '--', 
                               color=colors[i], linewidth=1.5, alpha=0.8,
                               label=f"{peak_name} ({params['center']:.1f} nm)")
                        
                        # Annotazione centro picco
                        max_idx = np.argmax(params['intensity'])
                        ax.annotate(f'{params["center"]:.1f}', 
                                   xy=(wavelengths[max_idx], params['intensity'][max_idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, color=colors[i])
                
                ax.set_xlabel('Lunghezza d\'onda (nm)')
                ax.set_ylabel('Intensità')
                ax.set_title(f'Deconvoluzione Dati Originali: {spectrum}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Nasconde assi vuoti se necessario
            if n_spectra > 1:
                for idx in range(n_spectra, len(axes)):
                    axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella visualizzazione: {e}")
    
    def run(self):
        """Avvia l'interfaccia"""
        self.root.mainloop()

# Esempio di utilizzo
if __name__ == "__main__":
    if not check_dependencies():
        print("Installa le dipendenze mancanti prima di continuare.")
        exit(1)
    
    # Avvia l'interfaccia grafica
    app = SpectralCSVInterface()
    app.run()