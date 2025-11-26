"""Calibration cache management for storing/loading pyFAI calibration results."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xray.path_manager import PathManager


def get_cache_path() -> Path:
    """Get the path to the calibrations cache file."""
    path_manager = PathManager()
    cache_dir = path_manager.get_artifacts_path() / "extension" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "calibrations.json"


def load_all_calibrations() -> Dict:
    """Load all calibrations from cache file."""
    cache_path = get_cache_path()
    if not cache_path.exists():
        return {}
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load calibrations cache: {e}")
        return {}


def save_all_calibrations(calibrations: Dict) -> None:
    """Save all calibrations to cache file."""
    cache_path = get_cache_path()
    try:
        with open(cache_path, 'w') as f:
            json.dump(calibrations, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save calibrations cache: {e}")


def save_calibration(
    image_name: str,
    center: Tuple[float, float],
    ring_radii: List[float],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save calibration data for an image.
    
    Args:
        image_name: Base name of the image (without path)
        center: (center_x, center_y) tuple
        ring_radii: List of ring radii in pixels
        metadata: Optional metadata dict (e.g., physical parameters)
    """
    calibrations = load_all_calibrations()
    
    calibrations[image_name] = {
        "center": {"x": float(center[0]), "y": float(center[1])},
        "ring_radii": [float(r) for r in ring_radii],
        "metadata": metadata or {}
    }
    
    save_all_calibrations(calibrations)
    print(f"Saved calibration for {image_name}")


def load_calibration(image_name: str) -> Optional[Dict]:
    """
    Load calibration data for an image.
    
    Args:
        image_name: Base name of the image (without path)
        
    Returns:
        Dict with 'center', 'ring_radii', and 'metadata' keys, or None if not found
    """
    calibrations = load_all_calibrations()
    return calibrations.get(image_name)


def calibration_exists(image_name: str) -> bool:
    """Check if calibration exists for an image."""
    return image_name in load_all_calibrations()


def delete_calibration(image_name: str) -> bool:
    """
    Delete calibration for an image.
    
    Returns:
        True if calibration was deleted, False if it didn't exist
    """
    calibrations = load_all_calibrations()
    if image_name in calibrations:
        del calibrations[image_name]
        save_all_calibrations(calibrations)
        print(f"Deleted calibration for {image_name}")
        return True
    return False


def list_calibrations() -> List[str]:
    """Get list of all calibrated image names."""
    return list(load_all_calibrations().keys())
