from __future__ import annotations
import logging
import logging.handlers
import multiprocessing

from pathlib import Path
from typing import Annotated

import typer
import concurrent.futures
from rich.logging import RichHandler
from xray.extension.analysis import process_diffraction_image

# Configure logging
# We will configure the root logger in the main entry points, but for now set a default
logging.basicConfig(
    level=logging.INFO, 
    format='%(message)s',
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("xray.extension")

extension_cli = typer.Typer(
    help="Analyzes X-ray film, generates plots, and saves results.",
)

def worker_init(queue):
    """
    Initializer for worker processes to setup logging to a queue.
    """
    queue_handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplication/conflict in workers
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(queue_handler)

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
    ] = 56.0, # +- 2
    phys_h_mm: Annotated[
        float,
        typer.Option(help="Physical height of the film in mm."),
    ] = 76.0, #+- 1 i cut 1 from top cause i fucking can and i think it is right
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
    
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        logger.error(f"Error: Directory {directory_path} does not exist.")
        raise typer.Exit(code=1)
    
    if not dir_path.is_dir():
        logger.error(f"Error: {directory_path} is not a directory.")
        raise typer.Exit(code=1)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    # Find all image files
    image_files = [
        f for f in dir_path.rglob('*') 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {directory_path}")
        return
    
    logger.info(f"Found {len(image_files)} image(s) to process...")
    
    results_list = []
    
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

    # Setup multiprocessing logging
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        
        # Configure the main process logger to print to the progress console
        # This ensures logs appear above the progress bar
        # We use a QueueListener to handle logs coming from workers
        
        # Create a RichHandler that writes to the progress console
        rich_handler = RichHandler(console=progress.console, rich_tracebacks=True, markup=True)
        
        # Setup QueueListener to forward worker logs to the main process's RichHandler
        listener = logging.handlers.QueueListener(queue, rich_handler)
        listener.start()
        
        # Also configure the main process to use this handler for its own logs
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        # Temporarily replace handlers
        original_handlers = root.handlers[:]
        for h in root.handlers[:]:
            root.removeHandler(h)
        root.addHandler(rich_handler)

        try:
            task = progress.add_task("[cyan]Processing images...", total=len(image_files))
            
            with concurrent.futures.ProcessPoolExecutor(initializer=worker_init, initargs=(queue,)) as executor:
                future_to_file = {}
                
                # Custom distance mapping
                distance_mapping = {
                    "7_20": 20.0,
                    "liquid": 10.0,
                    "solid": 15.0,
                    "salt": 15.0,
                    "10_1": 20.0,
                    "euctic": 15.0,
                    "eutectic": 15.0,
                }

                for image_file in image_files:
                    # Determine distance
                    current_distance = distance_l_mm
                    fname_lower = image_file.name.lower()
                    
                    for key, dist in distance_mapping.items():
                        if key in fname_lower:
                            current_distance = dist
                            break
                    
                    future = executor.submit(
                        process_diffraction_image, 
                        image_file, 
                        phys_w_mm, 
                        phys_h_mm, 
                        current_distance, 
                        wavelength_pm
                    )
                    future_to_file[future] = image_file
                
                for future in concurrent.futures.as_completed(future_to_file):
                    image_file = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            results_list.append(result)
                    except Exception as e:
                        # Log the error using the logger so it goes through the queue/rich handler
                        logger.error(f"Error processing {image_file.name}: {e}")
                    finally:
                        progress.advance(task)
        finally:
            # Restore original handlers and stop listener
            listener.stop()
            for h in root.handlers[:]:
                root.removeHandler(h)
            for h in original_handlers:
                root.addHandler(h)
    
    if results_list:
        from xray.extension import viz
        from xray.path_manager import PathManager
        
        path_manager = PathManager()
        output_dir = path_manager.get_artifacts_path() / "extension"
        
        combined_report_path = viz.generate_multi_image_report(output_dir, results_list)
        if combined_report_path:
             logger.info(f"\nGenerated combined report: {combined_report_path.as_uri()}")
        
        # Debug: Print results list
        logger.info(f"Results list contains {len(results_list)} items:")
        for res in results_list:
            logger.info(f" - {res['base_name']}")

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
            logger.info(f"Generated center montage: {montage_path.as_uri()}")

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
            logger.info(f"Generated rings montage: {rings_montage_path.as_uri()}")

        # Create montage of Intensity vs Radius plots
        radius_paths = []
        for res in results_list:
            if res.get('plot_r_path'):
                radius_paths.append(res['plot_r_path'])
        if radius_paths:
            radius_montage_path = output_dir / "radius_plots_montage.png"
            viz.create_montage(radius_paths, radius_montage_path)
            logger.info(f"Generated radius plots montage: {radius_montage_path.as_uri()}")

        # Create montage of Intensity vs 2-Theta plots
        theta_paths = []
        for res in results_list:
            if res.get('plot_theta_path'):
                theta_paths.append(res['plot_theta_path'])
        if theta_paths:
            theta_montage_path = output_dir / "theta_plots_montage.png"
            viz.create_montage(theta_paths, theta_montage_path)
            logger.info(f"Generated 2-theta plots montage: {theta_montage_path.as_uri()}")

    logger.info(f"\nCompleted processing {len(results_list)}/{len(image_files)} image(s).")
