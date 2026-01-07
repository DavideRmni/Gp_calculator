# Spectral Extrapolator & Deconvolution Tool 🧬
<img width="1250" height="750" alt="Screenshot 2026-01-07 184057" src="https://github.com/user-attachments/assets/f8b44b0d-3077-4805-b39b-f934975a84db" />
<img width="544" height="336" alt="Immagine1" src="https://github.com/user-attachments/assets/d5ab6a43-f09c-4055-b25f-e3e117616b19" />
<img width="547" height="336" alt="Immagine2" src="https://github.com/user-attachments/assets/89e46050-295b-4f81-847d-660dbc194946" />
<img width="548" height="336" alt="Immagine3" src="https://github.com/user-attachments/assets/dca82793-e7cd-4627-b545-839b85d95bde" />

This is a Python GUI application designed for the analysis of spectral data (specifically fluorescence spectra).

It streamlines the workflow from **Data Import** (supporting various CSV standards) to **Polynomial Fitting** and **Peak Deconvolution**, allowing researchers to isolate individual peak components (Gaussian, Lorentzian, Voigt) and calculate Generalized Polarization (GP) values.

## ✨ Key Features

### 📥 Data Management
* **Smart Import:** Automatically detects file encoding (via `chardet`) and CSV formats (European `;` vs US `,` vs Tab-delimited).
* **Blank Correction:** Select specific columns to act as "Blanks" and automatically subtract their average from the active dataset.
* **Data Preview:** Integrated TreeView to inspect loaded data and column structures.

### 🔬 Spectral Analysis (Polynomial)
* **Polynomial Fitting:** Fits spectra to polynomial curves to estimate baselines or smooth trends.
* **Target R²:** Automatically determines the optimal polynomial degree to achieve a user-defined $R^2$ target (e.g., 0.99).
* **Peak Detection:** Identifies dominant peaks on the fitted data using prominence and height thresholds.

### 🧬 Advanced Deconvolution
* **Multi-Model Support:** Deconvolve spectra into constituent peaks using:
    * **Gaussian**
    * **Lorentzian**
    * **Voigt** (Convolution of Gaussian and Lorentzian)
* **Algorithms:** Choice between `curve_fit` (Levenberg-Marquardt) and `differential_evolution` (Global optimization) via `scipy`.
* **Constraints:** Supports **Fixed Centers** and **Manual Peak Picking** to constrain the solver for physical accuracy.
* **Dual Mode:** Perform deconvolution on the *Polynomial Fit* or directly on the *Original/Smoothed Data*.

### 📈 Smoothing & GP Analysis
* **Signal Smoothing:** Savitzky-Golay, Moving Average, and Gaussian filters.
* **GP Calculation:** Automated calculation of **Generalized Polarization (GP)** based on the heights and areas of the two main detected peaks.
    * Formula: $GP = \frac{I_1 - I_2}{I_1 + I_2}$

## 📋 Prerequisites

The interface is built with `tkinter` (included in standard Python), but relies on the scientific stack for calculations.

pip install pandas numpy matplotlib scikit-learn scipy chardet

Note: scipy is strictly required for the Deconvolution and Smoothing features.

🚀 Usage Guide
1. Input Tab
Load Data: Click "Select CSV File". The tool will attempt to auto-detect separators.

Blank Correction: Select columns in the "Blank correction" listbox and click "Apply correction".

Preview: Click "Visualizza Spettri" to plot the raw/corrected data.

2. Output Tab
This tab is split into Analysis (Left) and Deconvolution (Right).

A. Wavelength Range
Set the Min and Max nanometers to limit the analysis to a specific region of interest. Click Auto to detect the range from data.

B. Polynomial Fitting
Set the Target R² (e.g., 0.99) and Max Degree.

Click Calculate Fitting.

Use Analyze Peaks to find local maxima on the fit.

C. Deconvolution
Setup: Choose the number of peaks (e.g., 2), the model (e.g., voigt), and the algorithm.

Centers:

Manual: Check "Manuali" and enter centers (e.g., 440.0, 490.0).

Constraints: Check "Esatti e Fissi" to force the solver to respect these wavelengths.

Run: Click Deconvoluzione.

Visualize: Click "Visualizza" to see the original curve, the total fit, and individual peak components.

D. Smoothing & GP
Apply Savitzky-Golay smoothing to noisy data.

Click Calcolo GP to export a CSV containing the GP values calculated from the heights and areas of the two primary peaks found.

📂 Export Options
The tool provides flexible export options for every stage:

Excel (.xlsx): Saves data, fitted curves, parameters, and statistics in separate sheets.

CSV (.csv): Saves data with customizable delimiters (Field Separator and Decimal Separator) to ensure compatibility with European or US software (e.g., Origin, Excel, SigmaPlot).
