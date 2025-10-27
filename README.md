# X-Ray Diffraction Analysis for Crystal Lattices

[![CI](https://github.com/shoham-baris/XRay/actions/workflows/ci.yaml/badge.svg)](https://github.com/shoham-baris/XRay/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/shoham-baris/XRay/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/shoham-baris/XRay)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a sophisticated Python framework for analyzing X-ray Diffraction (XRD) data to determine the lattice parameters of crystalline materials. It is specifically tailored for identifying and fitting K-alpha/K-beta doublets on a Bremsstrahlung background, a common scenario in XRD analysis.

---

## 1. Scientific Overview

The primary goal of this project is to automate the process of analyzing XRD data to determine the crystal structure of a sample. The core of the analysis involves:

- **Peak Detection:** Identifying the diffraction peaks in the raw XRD data.
- **Peak Fitting:** Fitting a physically-motivated model (a double Voigt profile on a Bremsstrahlung background) to the detected peaks to precisely locate the K-alpha and K-beta components.
- **d-spacing Calculation:** Using Bragg's Law to calculate the d-spacing for each fitted peak.
- **Lattice Parameter Refinement:** Using the calculated d-spacings to refine the crystal's lattice parameter 'a', providing a final, statistically robust result with an error estimate.

---

## 2. Features

- **Advanced Peak Fitting:** Utilizes a `double_voigt` model on a Bremsstrahlung background to accurately fit K-alpha/K-beta doublets.
- **Global Background Subtraction:** Fits a single, global background to the entire spectrum for a more robust analysis.
- **Interactive HTML Reports:** Generates a self-contained HTML report with an interactive `plotly` graph and detailed summary tables.
- **Caching:** Caches the results of expensive fitting operations to make subsequent runs significantly faster.
- **Flexible Configuration:** Supports configuration via both command-line arguments and a `.env` file.

---

## 3. Installation and Usage

### Prerequisites

- Python >=3.12
- [uv](https://github.com/astral-sh/uv) (for installation)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shoham-baris/XRay.git
   cd XRay
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Run the analysis:**
   ```bash
   python -m xray
   ```
   This will run the analysis on the default `data/dummy.csv` file. You can specify a different input file using the `--input` option or by creating a `.env` file.

4. **View the report:**
   After the analysis is complete, you will find the interactive HTML report at `artifacts/index.html`.

---

## 4. CI/CD

The project includes a CI/CD pipeline that automatically runs tests and builds the analysis report. You can view the status of the pipeline and the latest reports on the [GitHub Actions page](https://github.com/shoham-baris/XRay/actions).

---

## 5. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
