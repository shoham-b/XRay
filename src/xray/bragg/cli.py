from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from xray.bragg.main import (
    generate_summary_tables,
    load_and_prep_data,
    perform_fitting_with_predefined_peaks,
    perform_peak_analysis,
)
import subprocess
from xray.bragg.viz import create_multi_material_report, create_multi_material_tex_report
from xray.path_manager import PathManager

bragg_cli = typer.Typer(
    invoke_without_command=True,
    help="Analyzes X-ray diffraction data to find peaks and calculate d-spacing.",
)


REAL_LATTICE_CONSTANT_NACL = 5.64  # Angstroms
REAL_LATTICE_CONSTANT_LIF = 4.026  # Angstroms


def compile_tex_to_pdf(tex_path: Path, console: Console) -> None:
    """Compiles a TeX file to PDF using pdflatex."""
    console.print(f"\n[bold]--- Compiling LaTeX to PDF ---[/bold]")
    try:
        process = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                f"-output-directory={tex_path.parent}",
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        console.print(f"Successfully compiled [cyan]{tex_path}[/cyan] to PDF.")
        pdf_path = tex_path.with_suffix(".pdf")
        console.print(f"Saved PDF report to [cyan]{pdf_path.absolute()}[/cyan]")
    except FileNotFoundError:
        console.print(
            "[bold red]Error: pdflatex not found. Please ensure you have a LaTeX distribution installed and in your PATH.[/bold red]"
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Failed to compile LaTeX to PDF.[/bold red]")
        console.print(f"pdflatex stdout:\n{e.stdout}")
        console.print(f"pdflatex stderr:\n{e.stderr}")


@bragg_cli.callback()
def bragg_analysis(
        input_files: Annotated[
            list[Path] | None,
            typer.Option(
                "--input",
                help="Input data file(s) (CSV format) relative to the data directory.",
                envvar="BRAGG_INPUT_FILES",
            ),
        ] = None,
        wavelength: Annotated[
            float,
            typer.Option(
                "--wavelength", help="X-ray wavelength in Angstroms.", envvar="BRAGG_WAVELENGTH"
            ),
        ] = 0.7108,
        threshold: Annotated[
            float,
            typer.Option(
                "--threshold",
                help="Relative height threshold (0-1) for peak detection.",
                envvar="BRAGG_THRESHOLD",
            ),
        ] = 0.05,
        distance: Annotated[
            int,
            typer.Option(
                "--distance", help="Minimum number of points between peaks.", envvar="BRAGG_DISTANCE"
            ),
        ] = 5,
        window: Annotated[
            int,
            typer.Option(
                "--window",
                help="Half-window size in points used for local Voigt fitting.",
                envvar="BRAGG_WINDOW",
            ),
        ] = 20,
        prominence: Annotated[
            float | None,
            typer.Option(
                "--prominence",
                help="Relative prominence (0-1) for peak detection.",
                envvar="BRAGG_PROMINENCE",
            ),
        ] = 0.05,
        show_plots: Annotated[
            bool,
            typer.Option(
                "--show-plots", help="Show plots as they are generated.", envvar="BRAGG_SHOW_PLOTS"
            ),
        ] = False,
        clear_cache: Annotated[
            bool,
            typer.Option("--clear-cache", help="Clear the cache before running."),
        ] = False,
        # Remove real_lattice_constant from here, it will be set dynamically
        expected_nacl_angles: Annotated[
            list[float] | None,
            typer.Option(
                "--expected-nacl-angles",
                help="Expected 2θ angles for NaCl peaks (e.g., 7.7 8.5 14.1).",
                envvar="BRAGG_EXPECTED_NACL_ANGLES",
            ),
        ] = None,
        expected_lif_angles: Annotated[
            list[float] | None,
            typer.Option(
                "--expected-lif-angles",
                help="Expected 2θ angles for LiF peaks (e.g., 11.1 12.3 20.4).",
                envvar="BRAGG_EXPECTED_LIF_ANGLES",
            ),
        ] = None,
        use_predefined_angles: Annotated[
            bool,
            typer.Option(
                "--use-predefined-angles",
                help="Use predefined angles instead of performing peak analysis.",
                envvar="BRAGG_USE_PREDEFINED_ANGLES",
            ),
        ] = False,
        fit_predefined_angles: Annotated[
            bool,
            typer.Option(
                "--fit-predefined-angles",
                help="Fit predefined angles instead of performing peak analysis.",
                envvar="BRAGG_FIT_PREDEFINED_ANGLES",
            ),
        ] = True,
        tex: Annotated[
            bool,
            typer.Option(
                "--tex",
                help="Generate a LaTeX report.",
                envvar="BRAGG_TEX",
            ),
        ] = False,
        pdf: Annotated[
            bool,
            typer.Option(
                "--pdf",
                help="Generate a PDF report from the LaTeX report.",
                envvar="BRAGG_PDF",
            ),
        ] = False,
) -> int:
    """Analyzes X-ray diffraction data to find peaks and calculate d-spacing."""
    if expected_lif_angles is None:
        expected_lif_angles = [11.1, 12.3, 20.4, 22.8]
    if expected_nacl_angles is None:
        expected_nacl_angles = [7.7, 8.5, 14.1, 15.9, 20.7, 23.3, 27.7, 31.4]
    if input_files is None:
        input_files = [Path("bragg/NaCl.csv"), Path("bragg/LiF.csv")]
    console = Console()
    if clear_cache:
        from xray.cache import cache

        cache.clear()
        console.print("[bold yellow]Cache cleared.[/bold yellow]")

    path_manager = PathManager()
    output_dir = path_manager.get_artifacts_path() / "bragg"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_data_list = []
    for input_file in input_files:
        console.print(f"\n[bold]--- Analyzing {input_file.name} ---[/bold]")
        input_path = path_manager.get_data_path() / input_file
        df = load_and_prep_data(input_path, console)
        if df is None:
            continue

        material_name = input_file.stem

        # Determine real_lattice_constant based on material_name
        if "NaCl" in material_name:
            current_real_lattice_constant = REAL_LATTICE_CONSTANT_NACL
        elif "LiF" in material_name:
            current_real_lattice_constant = REAL_LATTICE_CONSTANT_LIF
        else:
            console.print(
                f"[bold red]Warning: Unknown material '{material_name}'. Using default real lattice constant for NaCl.[/bold red]"
            )
            current_real_lattice_constant = REAL_LATTICE_CONSTANT_NACL

        analysis_params = {
            "threshold": threshold,
            "distance": distance,
            "prominence": prominence,
            "window": window,
        }

        if fit_predefined_angles:
            console.print(f"Fitting predefined angles for {material_name}...")
            predefined_angles = []
            if "NaCl" in material_name:
                predefined_angles = expected_nacl_angles
            elif "LiF" in material_name:
                predefined_angles = expected_lif_angles
            else:
                console.print(
                    f"[bold red]Warning: No predefined angles for {material_name}. Skipping.[/bold red]"
                )
                continue

            analysis_results = perform_fitting_with_predefined_peaks(
                df, predefined_angles, analysis_params, console
            )
            current_predefined_angles = analysis_results["predefined_angles"]

        elif use_predefined_angles:
            console.print(f"Using predefined angles for {material_name}...")

            predefined_angles = []
            if "NaCl" in material_name:
                predefined_angles = expected_nacl_angles
            elif "LiF" in material_name:
                predefined_angles = expected_lif_angles
            else:
                console.print(
                    f"[bold red]Warning: No predefined angles for {material_name}. Skipping.[/bold red]"
                )
                continue

            # Find the closest indices to the predefined angles
            angles_array = df["Angle"].values
            intensities_array = df["Intensity"].values

            valid_fits = []
            for predefined_angle in predefined_angles:
                # Find the closest index in the data
                closest_idx = np.argmin(np.abs(angles_array - predefined_angle))
                actual_angle = angles_array[closest_idx]
                intensities_array[closest_idx]

                # Create tuple in format: (idx, None, angle)
                # None is used instead of popt since there's no Voigt fitting
                valid_fits.append((closest_idx, (0, 0, 0, 0), actual_angle))

            analysis_results = {
                "initial_peaks_idx": [fit[0] for fit in valid_fits],  # List of indices
                "initial_peaks_properties": {},
                "valid_fits": valid_fits,
                "bg_params": None,
            }
            current_predefined_angles = predefined_angles

        else:
            analysis_results = perform_peak_analysis(df, analysis_params, console)
            current_predefined_angles = []  # No predefined angles for this case

        peak_df, summary_df, fit_plot_data = generate_summary_tables(
            df, analysis_results, wavelength, current_real_lattice_constant, current_predefined_angles
        )

        analysis_data = {
            "name": input_file.stem,
            "df": df,
            "analysis_results": analysis_results,
            "peak_df": peak_df,
            "summary_df": summary_df,
            "fit_plot_data": fit_plot_data,
            "real_lattice_constant": current_real_lattice_constant,
            "expected_nacl_angles": expected_nacl_angles,
            "expected_lif_angles": expected_lif_angles,
        }
        analysis_data_list.append(analysis_data)

        # --- Console Output ---
        console.print("\n[bold]--- Peak Analysis Results ---[/bold]")
        console.print(peak_df.to_string(index=False))
        console.print("\n[bold]--- Summary ---[/bold]")
        console.print(summary_df.to_string(index=False))

    # --- HTML Report ---
    console.print("\n[bold]--- Generating HTML Report ---[/bold]")
    report_path = output_dir / "index.html"
    try:
        create_multi_material_report(analysis_data_list, out_path=report_path)
        console.print(f"Saved interactive report to [cyan]{report_path.absolute().as_uri()}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Failed to generate interactive report: {e}[/bold red]")

    # --- LaTeX Report ---
    if tex or pdf:
        console.print("\n[bold]--- Generating LaTeX Report ---[/bold]")
        tex_report_path = output_dir / "report.tex"
        try:
            create_multi_material_tex_report(analysis_data_list, out_path=tex_report_path)
            console.print(f"Saved LaTeX report to [cyan]{tex_report_path.absolute()}[/cyan]")
            # Compile to PDF whenever a LaTeX report is requested (either --tex or --pdf)
            compile_tex_to_pdf(tex_report_path, console)
        except Exception as e:
            console.print(f"[bold red]Failed to generate LaTeX report: {e}[/bold red]")

    return 0


def main() -> None:
    bragg_cli()


if __name__ == "__main__":
    main()
