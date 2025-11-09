from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from xray.bragg.main import (
    generate_summary_tables,
    load_and_prep_data,
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
        list[Path],
        typer.Option(
            "--input",
            help="Input data file(s) (CSV format) relative to the data directory.",
            envvar="BRAGG_INPUT_FILES",
        ),
    ] = [Path("bragg/NaCl.csv"), Path("bragg/LiF.csv")],
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
) -> int:
    """Analyzes X-ray diffraction data to find peaks and calculate d-spacing."""
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
        analysis_results = perform_peak_analysis(df, analysis_params, console)

        peak_df, summary_df, fit_plot_data = generate_summary_tables(
            df, analysis_results, wavelength, real_lattice_constant
        )

        analysis_data = {
            "name": input_file.stem,
            "df": df,
            "analysis_results": analysis_results,
            "peak_df": peak_df,
            "summary_df": summary_df,
            "fit_plot_data": fit_plot_data,
            "real_lattice_constant": real_lattice_constant,
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
