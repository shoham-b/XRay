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
from xray.bragg.viz import create_multi_material_report
from xray.path_manager import PathManager

bragg_cli = typer.Typer(
    invoke_without_command=True,
    help="Analyzes X-ray diffraction data to find peaks and calculate d-spacing.",
)


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
        real_lattice_constant: Annotated[
            float,
            typer.Option(
                "--real-lattice-constant",
                help="Real lattice constant in Angstroms for comparison (e.g., 5.64 for NaCl, 4.03 for LiF).",
                envvar="BRAGG_REAL_LATTICE_CONSTANT",
            ),
        ] = 5.64,
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
            current_predefined_angles = [] # No predefined angles for this case

        peak_df, summary_df, fit_plot_data = generate_summary_tables(
            df, analysis_results, wavelength, real_lattice_constant, current_predefined_angles
        )

        analysis_data = {
            "name": input_file.stem,
            "df": df,
            "analysis_results": analysis_results,
            "peak_df": peak_df,
            "summary_df": summary_df,
            "fit_plot_data": fit_plot_data,
            "real_lattice_constant": real_lattice_constant,
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

    return 0


def main() -> None:
    bragg_cli()


if __name__ == "__main__":
    main()
