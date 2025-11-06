from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from scipy.signal import find_peaks

from xray.bragg.main import (
    generate_summary_tables,
    load_and_prep_data,
    perform_peak_analysis,
)
from xray.bragg.peak_finding import bremsstrahlung_bg, double_voigt
from xray.path_manager import PathManager
from xray.viz import create_interactive_report

bragg_cli = typer.Typer(
    invoke_without_command=True,
    help="Analyzes X-ray diffraction data to find peaks and calculate d-spacing.",
)


@bragg_cli.callback()
def bragg_analysis(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            help="Input data file (CSV format) relative to the data directory.",
            envvar="BRAGG_INPUT_FILE",
        ),
    ] = Path("bragg/NaCl1.csv"),
    wavelength: Annotated[
        float,
        typer.Option(
            "--wavelength", help="X-ray wavelength in Angstroms.", envvar="BRAGG_WAVELENGTH"
        ),
    ] = 1.5406,
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
    width: Annotated[
        int | None,
        typer.Option(
            "--width", help="Minimum peak width in points for detection.", envvar="BRAGG_WIDTH"
        ),
    ] = None,
) -> int:
    """Analyzes X-ray diffraction data to find peaks and calculate d-spacing."""
    console = Console()
    path_manager = PathManager()
    output_dir = path_manager.get_artifacts_path() / "bragg"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = path_manager.get_data_path() / input_file
    df = load_and_prep_data(input_path, console)
    if df is None:
        return 1

    analysis_params = {
        "threshold": threshold,
        "distance": distance,
        "prominence": prominence,
        "width": width,
        "window": window,
    }
    analysis_results = perform_peak_analysis(df, analysis_params, console)

    peak_df, summary_df = generate_summary_tables(df, analysis_results, wavelength)

    # --- Console Output ---
    console.print("\n[bold]--- Peak Analysis Results ---[/bold]")
    console.print(peak_df.to_string(index=False))
    console.print("\n[bold]--- Most Probable d-spacing ---[/bold]")
    console.print(summary_df.to_string(index=False))

    # --- HTML Report ---
    console.print("\n[bold]--- Generating HTML Report ---[/bold]")
    report_path = output_dir / "index.html"
    try:
        y_total_fit = bremsstrahlung_bg(df["Angle"].values, *analysis_results["bg_params"])
        for _, fit_params, _ in analysis_results["valid_fits"]:
            y_total_fit += double_voigt(df["Angle"].values, *fit_params)
        final_model_peaks_idx, _ = find_peaks(
            y_total_fit, height=y_total_fit.max() * 0.05, distance=5
        )

        create_interactive_report(
            df,
            analysis_results["initial_peaks_idx"],
            analysis_results["valid_fits"],
            analysis_results["bg_params"],
            final_model_peaks_idx,
            peak_df,
            summary_df,
            report_path,
        )
        console.print(f"Saved interactive report to [cyan]{report_path.absolute().as_uri()}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Failed to generate interactive report: {e}[/bold red]")

    return 0


def main() -> None:
    bragg_cli()
