from __future__ import annotations
import logging

from pathlib import Path
from typing import Annotated

import typer
import concurrent.futures
from xray.extension.analysis import process_diffraction_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("xray.extension")

extension_cli = typer.Typer(
    help="Analyzes X-ray film, generates plots, and saves results.",
)

@extension_cli.command(name="analyze")
def analyze(
    image_path: Annotated[
        str,
        typer.Argument(help="Path to the diffraction image file."),
    ],
    phys_w_mm: Annotated[
        float,
        typer.Option(help="Physical width of the film in mm."),
    ] = 50.0,
    phys_h_mm: Annotated[
        float,
        typer.Option(help="Physical height of the film in mm."),
    ] = 50.0,
    distance_l_mm: Annotated[
        float,
        typer.Option(help="Distance L in mm."),
    ] = 200.0,
    wavelength_pm: Annotated[
        float,
        typer.Option(help="Wavelength in pm."),
    ] = 71.1,
) -> None:
    """
    Analyzes X-ray film, generates plots, and saves results to a 'data' folder.
    """
    # Ensure INFO logs are shown for single image analysis
    logging.getLogger("xray.extension").setLevel(logging.INFO)
    
    process_diffraction_image(
        image_path, phys_w_mm, phys_h_mm, distance_l_mm, wavelength_pm
    )


@extension_cli.command(name="analyze-dir")
def analyze_dir(
    directory_path: Annotated[
        str,
        typer.Argument(help="Path to the directory containing diffraction image files."),
    ],
    phys_w_mm: Annotated[
        float,
        typer.Option(help="Physical width of the film in mm."),
    ] = 50.0,
    phys_h_mm: Annotated[
        float,
        typer.Option(help="Physical height of the film in mm."),
    ] = 50.0,
    distance_l_mm: Annotated[
        float,
        typer.Option(help="Distance L in mm."),
    ] = 200.0,
    wavelength_pm: Annotated[
        float,
        typer.Option(help="Wavelength in pm."),
    ] = 71.1,
) -> None:
    """
    Analyzes all X-ray film images in a directory, generates plots, and saves results.
    """

    # Suppress INFO logs for batch analysis to keep progress bar clean
    logging.getLogger("xray.extension").setLevel(logging.WARNING)
    
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        typer.echo(f"Error: Directory {directory_path} does not exist.")
        raise typer.Exit(code=1)
    
    if not dir_path.is_dir():
        typer.echo(f"Error: {directory_path} is not a directory.")
        raise typer.Exit(code=1)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    # Find all image files
    image_files = [
        f for f in dir_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        typer.echo(f"No image files found in {directory_path}")
        return
    
    typer.echo(f"Found {len(image_files)} image(s) to process...")
    
    results_list = []
    
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing images...", total=len(image_files))
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {
                executor.submit(
                    process_diffraction_image, 
                    image_file, 
                    phys_w_mm, 
                    phys_h_mm, 
                    distance_l_mm, 
                    wavelength_pm
                ): image_file for image_file in image_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                image_file = future_to_file[future]
                try:
                    result = future.result()
                    # typer.echo(f"Finished processing: {image_file.name}") # Reduced verbosity with progress bar
                    if result:
                        results_list.append(result)
                except Exception as e:
                    typer.echo(f"Error processing {image_file.name}: {e}")
                finally:
                    progress.advance(task)
    
    if results_list:
        from xray.extension import viz
        from xray.path_manager import PathManager
        
        path_manager = PathManager()
        output_dir = path_manager.get_artifacts_path() / "extension"
        
        combined_report_path = viz.generate_multi_image_report(output_dir, results_list)
        if combined_report_path:
             typer.echo(f"\nGenerated combined report: {combined_report_path.as_uri()}")

        # Create montage of center plots (Comparison: Pixels vs Clean)
        center_paths = []
        for res in results_list:
            if res.get('plot_center_pixels_path') and res.get('plot_center_clean_path'):
                center_paths.append(res['plot_center_pixels_path'])
                center_paths.append(res['plot_center_clean_path'])
            elif res.get('plot_center_path'):
                center_paths.append(res['plot_center_path'])
        if center_paths:
            montage_path = output_dir / "centers_montage.png"
            viz.create_montage(center_paths, montage_path)
            typer.echo(f"Generated center montage: {montage_path.as_uri()}")

        # Create montage of rings plots (Comparison: Emboldened vs Rings)
        rings_paths = []
        for res in results_list:
            if res.get('plot_emboldened_path') and res.get('plot_rings_path'):
                rings_paths.append(res['plot_emboldened_path'])
                rings_paths.append(res['plot_rings_path'])
            elif res.get('plot_rings_path'):
                rings_paths.append(res['plot_rings_path'])
        if rings_paths:
            rings_montage_path = output_dir / "rings_montage.png"
            viz.create_montage(rings_paths, rings_montage_path)
            typer.echo(f"Generated rings montage: {rings_montage_path.as_uri()}")

        # Create montage of Intensity vs Radius plots
        radius_paths = []
        for res in results_list:
            if res.get('plot_r_path'):
                radius_paths.append(res['plot_r_path'])
        if radius_paths:
            radius_montage_path = output_dir / "radius_plots_montage.png"
            viz.create_montage(radius_paths, radius_montage_path)
            typer.echo(f"Generated radius plots montage: {radius_montage_path.as_uri()}")

        # Create montage of Intensity vs 2-Theta plots
        theta_paths = []
        for res in results_list:
            if res.get('plot_theta_path'):
                theta_paths.append(res['plot_theta_path'])
        if theta_paths:
            theta_montage_path = output_dir / "theta_plots_montage.png"
            viz.create_montage(theta_paths, theta_montage_path)
            typer.echo(f"Generated 2-theta plots montage: {theta_montage_path.as_uri()}")

    typer.echo(f"\nCompleted processing {len(image_files)} image(s).")
