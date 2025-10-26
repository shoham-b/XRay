# X-Ray Diffraction Analysis

This repository contains a Python-based framework for analyzing X-ray diffraction (XRD) data.

---

## 1. Overview

This project provides tools to:

- **Load and manage XRD data** from Excel files.
- **Find diffraction peaks** using both naive (max intensity) and advanced (Gaussian fitting) methods.
- **Calculate d-spacing** using Bragg's Law.
- **Visualize data** with plots for raw scans and peak fits.

---

## 2. Installation Guide

### Prerequisites

- Python >=3.12
- Git (for cloning)
- [uv](https://github.com/astral-sh/uv) (for installation)

### Steps

1. **Clone the repository**
```bash
git clone <your-repo-url> xray-analysis
cd xray-analysis
```

2. **Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
uv sync --group dev
```

---

## 3. User Workflow

### Basic Usage

The primary entry point is a command-line interface (CLI).

- **Run analysis on a data file**
```bash
uv run python -m xray --input "path/to/your/data.xlsx" --output "./artifacts"
```

- **CLI Options:**
  - `--input`: Path to the input Excel file (required).
  - `--output`: Directory to save generated plots (defaults to `./artifacts`).
  - `--wavelength`: X-ray wavelength in Angstroms (defaults to 1.5406 for Cu K-alpha).

### Inspect Results

- **Console Output**: The CLI will print the found peaks and calculated d-spacings for both naive and fitting methods.
- **Artifacts Directory**: The output directory will contain:
  - A plot of the raw XRD scan (`*_scan.png`).
  - A plot of the data with the fitted Gaussian peak (`*_fit.png`).

---

## 4. Developer Guide

### Core Structure

- `src/xray/`: Main source code.
  - `__main__.py`: The Typer-based CLI application.
  - `data_handler/`: For loading and managing data.
  - `analysis/`: Peak finding algorithms.
  - `mathutils.py`: Mathematical helper functions (Bragg's Law, Gaussian).
  - `viz.py`: Plotting functions.
  - `path_manager.py`: Manages data paths.

### Development Tasks

- **Run tests**
```bash
make test
```

- **Check formatting and linting**
```bash
make lint
make format
```

---

## 5. Contact & License

- Maintainer: Shoham Baris (shoham.baris@mail.huji.ac.il)
- License: MIT
