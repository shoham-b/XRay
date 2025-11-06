from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from xray.lau.main import run_lau_analysis
from xray.path_manager import PathManager

lau_cli = typer.Typer(
    invoke_without_command=True, help="Analyzes Laue diffraction patterns from an image."
)


@lau_cli.callback()
def lau_analysis(
    image_path: Annotated[
        Path,
        typer.Option(
            "--image",
            help="Input image file relative to the data directory.",
            envvar="LAU_IMAGE_FILE",
        ),
    ] = Path("lau/film.jpg"),
    big_circle_thresh: Annotated[
        int,
        typer.Option(help="Threshold for finding the big circle.", envvar="LAU_BIG_CIRCLE_THRESH"),
    ] = 10,
    small_dot_thresh: Annotated[
        int,
        typer.Option(
            help="Threshold for finding the small dots within 2 radii of the big circle center.",
            envvar="LAU_SMALL_DOT_THRESH",
        ),
    ] = 50,
    min_spot_area: Annotated[
        int,
        typer.Option(help="Minimum area for a spot to be considered.", envvar="LAU_MIN_SPOT_AREA"),
    ] = 10,
    min_circularity: Annotated[
        float,
        typer.Option(
            help="Minimum circularity for a spot to be considered.", envvar="LAU_MIN_CIRCULARITY"
        ),
    ] = 0.2,
    phys_y_mm: Annotated[
        float, typer.Option(help="Physical height of the detector in mm.", envvar="LAU_PHYS_Y_MM")
    ] = 75.0,
    phys_x_mm: Annotated[
        float, typer.Option(help="Physical width of the detector in mm.", envvar="LAU_PHYS_X_MM")
    ] = 55.0,
    l_mm: Annotated[
        float, typer.Option(help="Sample-to-film distance in mm.", envvar="LAU_L_MM")
    ] = 15.0,
    a_0_pm: Annotated[
        float, typer.Option(help="Lattice constant in picometers.", envvar="LAU_A_0_PM")
    ] = 564.02,
    small_dot_thresh_outer: Annotated[
        float,
        typer.Option(
            help="Threshold for finding the small dots beyond 2 radii of the big circle center.",
            envvar="LAU_SMALL_DOT_THRESH_OUTER",
        ),
    ] = 30.0,
    max_distance_percentage: Annotated[
        float,
        typer.Option(
            help="Max distance from center for a dot (percentage of min image dimension).",
            envvar="LAU_MAX_DISTANCE_PERCENTAGE",
        ),
    ] = 100.0,
) -> int:
    """Analyzes Laue diffraction patterns from an image."""
    console = Console()
    path_manager = PathManager()
    output_dir = path_manager.get_artifacts_path() / "lau"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = path_manager.get_data_path() / image_path

    console.print(f"Starting Laue analysis for [cyan]{input_path}[/cyan]")

    run_lau_analysis(
        image_path=str(input_path),
        output_dir=output_dir,
        big_circle_thresh=big_circle_thresh,
        small_dot_thresh=small_dot_thresh,
        min_spot_area=min_spot_area,
        min_circularity=min_circularity,
        phys_y_mm=phys_y_mm,
        phys_x_mm=phys_x_mm,
        l_mm=l_mm,
        a_0_pm=a_0_pm,
        small_dot_thresh_outer=small_dot_thresh_outer,
        max_distance_percentage=max_distance_percentage,
    )
    return 0


def main() -> None:
    lau_cli()
