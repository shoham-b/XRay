from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from .calculations import sinc_func

def plot_intensity_vs_radius(radii_mm, profile_percent, background_profile, profile_subtracted, base_name, output_dir):
    """Plots Normalized Intensity vs. Radius (mm)."""
    plt.figure(figsize=(10, 6))
    plt.plot(radii_mm, profile_percent, color='lightgray', linewidth=1.5, label='Raw Intensity')
    if background_profile is not None:
        plt.plot(radii_mm, background_profile, color='green', linestyle='--', label='Background (Poly)')
    plt.plot(radii_mm, profile_subtracted, color='blue', linewidth=1.5, label='Subtracted Intensity')
    plt.title(f"Normalized Intensity vs. Distance ($r$)\nSample: {base_name}")
    plt.xlabel("Distance from Center $r$ (mm)")
    plt.ylabel("Normalized Intensity (%)")
    plt.xlim(0, np.max(radii_mm))
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.legend()

    save_path = Path(output_dir) / f"{base_name}_intensity_vs_radius.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_center_on_image(img_arr, center_x, center_y, peak_radii_pixels, base_name, output_dir, centers_dict=None, filename_suffix="_center.png"):
    """Plots the image with the detected center marked and rings for peaks."""
    plt.figure(figsize=(10, 10))
    plt.imshow(img_arr, cmap='gray')
    
    # Plot rings - REMOVED per user request
    # "remove the plotting of the circle in the _center file since we don't use it anymore"
    # h, w = img_arr.shape
    # max_radius = np.sqrt(h**2 + w**2) * 1.5 
    # for i, r in enumerate(peak_radii_pixels):
    #     ...

    # Plot centers
    if centers_dict:
        # Define styles for different stages
        styles = {
            'CoM': {'color': 'yellow', 'marker': 'x', 'label': '1. CoM'},
            'Line': {'color': 'lime', 'marker': '*', 'label': '2. Line Proj'},
            'ExponentialWeighted': {'color': 'magenta', 'marker': 'D', 'fillstyle': 'none', 'markersize': 12, 'markeredgewidth': 2, 'label': 'Exponential Weighted CoM (96%)'}
        }
        
        for name, coords in centers_dict.items():
            if name == 'ContourMask':
                # Plot the contour pixels
                mask = coords
                y_mask, x_mask = np.where(mask)
                
                # Get intensities for "mass" visualization
                intensities = img_arr[mask]
                
                # Downsample if too many points (to avoid slow plotting/rendering)
                if len(x_mask) > 20000:
                    indices = np.random.choice(len(x_mask), 20000, replace=False)
                    x_mask = x_mask[indices]
                    y_mask = y_mask[indices]
                    intensities = intensities[indices]
                
                # Plot as scatter with color mapping based on intensity
                # Use a colormap that contrasts with the grayscale image (e.g., 'spring' or 'cool')
                scatter = plt.scatter(x_mask, y_mask, c=intensities, cmap='spring', s=15, alpha=0.8, label='Selected Pixels')
                plt.colorbar(scatter, label='Pixel Intensity', fraction=0.046, pad=0.04)
                continue

            style = styles.get(name, {'color': 'white', 'marker': '.'})
            plt.plot(coords[0], coords[1], 
                     color=style.get('color'), 
                     marker=style.get('marker'), 
                     fillstyle=style.get('fillstyle', 'full'),
                     markersize=12, 
                     markeredgewidth=2, 
                     label=style.get('label', name))
            
        # Plot Final Center
        plt.plot(center_x, center_y, 'r+', markersize=25, markeredgewidth=3, label='Final Result')
        
    else:
        plt.plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=2, label='Center')
        
    # Create custom legend to handle the patch correctly if needed, 
    # but plt.legend() should pick up the circle if it has a label.
    # However, patches added via add_patch might not show up in automatic legend unless we do this:
    handles, labels = plt.gca().get_legend_handles_labels()
    # Ensure 'Detected Peaks' is in there. If add_patch doesn't add to handles automatically:
    # We can create a proxy artist.
    from matplotlib.lines import Line2D
    # Check if 'Detected Peaks' is in labels
    # Check if 'Detected Peaks' is in labels
    # if 'Detected Peaks' not in labels and len(peak_radii_pixels) > 0:
    #     # Create proxy
    #     peak_handle = plt.Circle((0,0), 1, color='r', fill=False, linestyle='-', linewidth=2, alpha=0.6)
    #     handles.append(peak_handle)
    #     labels.append('Detected Peaks')
        
    # Sort legend? Or keep order?
    # Let's keep order but ensure Final Result is last.
    
    plt.legend(handles=handles, labels=labels, loc='upper right', framealpha=0.8)
    plt.title(f"Detected Center Refinement Steps\nSample: {base_name}")
    plt.axis('off')
    
    save_path = Path(output_dir) / f"{base_name}{filename_suffix}"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return save_path

def save_emboldened_image(img_arr, base_name, output_dir):
    """
    Creates a visualization with sharpening to 'embolden' the rings.
    User request: "expect the opposite of blurring to make it sharper"
    """
    from scipy.ndimage import gaussian_filter
    import numpy as np
    
    img_float = img_arr.astype(float)
    
    # Unsharp Masking
    # 1. Blur the image slightly to define the "unsharp" mask
    blurred = gaussian_filter(img_float, sigma=3)
    
    # 2. Calculate the mask (high frequency details)
    mask = img_float - blurred
    
    # 3. Add the mask back to the original image with a scaling factor (amount)
    # Amount > 1.0 makes it sharper
    amount = 2.0
    sharpened = img_float + amount * mask
    
    # Clip to valid range
    sharpened = np.clip(sharpened, 0, 255)
    
    plt.figure(figsize=(10, 10))
    # Use 'inferno' to emphasize intensity
    plt.imshow(sharpened, cmap='inferno')
    plt.axis('off')
    plt.title(f"Sharpened Intensity (Unsharp Mask)\nSample: {base_name}")
    
    save_path = Path(output_dir) / f"{base_name}_emboldened.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return save_path

def plot_rings_on_image(img_arr, center, radii_pixels, base_name, output_dir):
    """
    Plots the image with detected rings overlaid.
    """
    import matplotlib.patches as patches
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_arr, cmap='gray')
    
    ax = plt.gca()
    cx, cy = center
    
    # Plot center
    ax.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2)
    
    # Plot rings
    for r in radii_pixels:
        # Create a circle patch
        circle = patches.Circle((cx, cy), r, linewidth=2, edgecolor='r', facecolor='none', alpha=0.7)
        ax.add_patch(circle)
        
    plt.axis('off')
    plt.title(f"Detected Rings (Peaks)\nSample: {base_name}")
    
    save_path = Path(output_dir) / f"{base_name}_rings.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return save_path

def plot_intensity_vs_2theta(two_theta_deg, profile_percent, profile_subtracted, 
                             peak_angles, peak_intensities, d_spacings_pm, 
                             fitted_peaks, background_profile,
                             base_name, distance_L_mm, output_dir):
    """Plots Intensity vs. 2-Theta with d-spacings and fits."""
    plt.figure(figsize=(12, 7))
    
    # Plot Raw and Smoothed Data
    plt.plot(two_theta_deg, profile_percent, color='lightgray', label='Raw Data', alpha=0.5)
    plt.plot(two_theta_deg, profile_subtracted, color='darkblue', linewidth=2, label='Subtracted Intensity')
    
    # Plot Background if available
    if background_profile is not None:
        plt.plot(two_theta_deg, background_profile, color='green', linestyle='--', label='Estimated Background')

    # Plot Fits and Labels
    for i, angle in enumerate(peak_angles):
        intensity = peak_intensities[i]
        d_val = d_spacings_pm[i]
        
        # Plot Sinc Fit if available
        if fitted_peaks and i < len(fitted_peaks) and fitted_peaks[i] is not None:
            popt = fitted_peaks[i]
            # Generate points for the fit curve
            x_fit = np.linspace(angle - 2, angle + 2, 100)
            y_fit = sinc_func(x_fit, *popt)
            
            # Interpolate background at x_fit
            bg_at_fit = np.interp(x_fit, two_theta_deg, background_profile) if background_profile is not None else 0
            plt.plot(x_fit, y_fit + bg_at_fit, color='orange', linewidth=2, linestyle='-')

        plt.plot(angle, intensity, "x", color='red', markersize=8)
        plt.text(angle, intensity + 2, f"d={d_val:.0f} pm\n({angle:.1f}°)",
                 ha='center', va='bottom', fontsize=9, color='darkred', rotation=90)

    plt.title(
        f"Diffraction Pattern: Intensity vs. $2\\theta$ (with d-spacings)\nSample: {base_name} | L={distance_L_mm}mm")
    plt.xlabel("Scattering Angle $2\\theta$ (degrees)")
    plt.ylabel("Relative Intensity (%)")
    plt.xlim(0, np.max(two_theta_deg))
    plt.ylim(0, 115)
    plt.grid(True, which='both', alpha=0.3)
    
    # Create a custom legend to avoid duplicates if we loop
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()

    save_path = Path(output_dir) / f"{base_name}_intensity_vs_2theta.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

def create_interactive_plot(two_theta_deg, profile_percent, profile_subtracted, 
                            peak_angles, peak_intensities, d_spacings_pm, 
                            fitted_peaks, background_profile, base_name):
    """Creates an interactive Plotly figure."""
    fig = go.Figure()

    # 1. Raw Data
    fig.add_trace(go.Scatter(
        x=two_theta_deg, y=profile_percent,
        mode='lines', name='Raw Data',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))

    # 2. Subtracted Data
    fig.add_trace(go.Scatter(
        x=two_theta_deg, y=profile_subtracted,
        mode='lines', name='Subtracted Intensity',
        line=dict(color='darkblue', width=2)
    ))

    # 3. Background
    if background_profile is not None:
        fig.add_trace(go.Scatter(
            x=two_theta_deg, y=background_profile,
            mode='lines', name='Estimated Background',
            line=dict(color='green', width=2, dash='dash')
        ))

    # 4. Peaks and Fits
    for i, angle in enumerate(peak_angles):
        intensity = peak_intensities[i]
        d_val = d_spacings_pm[i]
        
        # Peak Marker
        fig.add_trace(go.Scatter(
            x=[angle], y=[intensity],
            mode='markers+text', name=f'Peak {i+1}',
            marker=dict(color='red', size=10, symbol='x'),
            text=[f"d={d_val:.0f} pm<br>({angle:.1f}°)"],
            textposition="top center",
            showlegend=False
        ))

        # Sinc Fit
        if fitted_peaks and i < len(fitted_peaks) and fitted_peaks[i] is not None:
            popt = fitted_peaks[i]
            x_fit = np.linspace(angle - 2, angle + 2, 100)
            # popt now includes offset
            y_fit = sinc_func(x_fit, *popt)
            # y_fit includes the offset from the fit (which models local bg residual)
            # We add the global background to it
            bg_at_fit = np.interp(x_fit, two_theta_deg, background_profile) if background_profile is not None else 0
            
            fig.add_trace(go.Scatter(
                x=x_fit, y=y_fit + bg_at_fit,
                mode='lines', name=f'Fit Peak {i+1}',
                line=dict(color='orange', width=2),
                showlegend=True
            ))

    fig.update_layout(
        title=f"Diffraction Pattern: Intensity vs. 2θ - {base_name}",
        xaxis_title="Scattering Angle 2θ (degrees)",
        yaxis_title="Relative Intensity (%)",
        template="plotly_white",
        height=600,
        hovermode="x unified"
    )
    
    return fig

def create_interactive_radius_plot(radii_mm, profile_percent, background_profile, profile_subtracted, base_name):
    """Creates an interactive Plotly figure for Intensity vs Radius."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=radii_mm, y=profile_percent,
        mode='lines', name='Raw Intensity',
        line=dict(color='lightgray', width=1.5)
    ))

    if background_profile is not None:
        fig.add_trace(go.Scatter(
            x=radii_mm, y=background_profile,
            mode='lines', name='Background (Poly)',
            line=dict(color='green', width=1.5, dash='dash')
        ))

    fig.add_trace(go.Scatter(
        x=radii_mm, y=profile_subtracted,
        mode='lines', name='Subtracted Intensity',
        line=dict(color='blue', width=1.5)
    ))

    fig.update_layout(
        title=f"Normalized Intensity vs. Distance (r) - {base_name}",
        xaxis_title="Distance from Center r (mm)",
        yaxis_title="Normalized Intensity (%)",
        template="plotly_white",
        height=600,
        hovermode="x unified"
    )
    return fig

def generate_html_report(output_dir, base_name, plot_r_path, plot_theta_path, csv_path, interactive_theta_fig=None, interactive_radius_fig=None, plot_center_path=None, preprocessed_image_path=None):
    """Generates an HTML report linking to the results."""
    
    # Calculate relative paths for HTML links
    output_dir = Path(output_dir)
    rel_plot_r = Path(plot_r_path).relative_to(output_dir)
    rel_plot_theta = Path(plot_theta_path).relative_to(output_dir)
    rel_csv = Path(csv_path).relative_to(output_dir)
    rel_plot_center = Path(plot_center_path).relative_to(output_dir) if plot_center_path else None
    rel_preprocessed = Path(preprocessed_image_path).relative_to(output_dir) if preprocessed_image_path else None
    
    theta_html = ""
    if interactive_theta_fig:
        theta_html = interactive_theta_fig.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        theta_html = f'<img src="{rel_plot_theta}" alt="Intensity vs 2-Theta">'

    radius_html = ""
    if interactive_radius_fig:
        # Don't include plotly.js again if we already included it
        include_js = False if interactive_theta_fig else 'cdn'
        radius_html = interactive_radius_fig.to_html(full_html=False, include_plotlyjs=include_js)
    else:
        radius_html = f'<img src="{rel_plot_r}" alt="Intensity vs Radius">'

    center_html = ""
    if rel_plot_center:
        center_html = f"""
            <div class="plot">
                <h2>Detected Center</h2>
                <img src="{rel_plot_center}" alt="Detected Center" style="max-width: 500px;">
            </div>
        """

    preprocessed_html = ""
    if rel_preprocessed:
        preprocessed_html = f"""
            <div class="plot">
                <h2>Preprocessed Image (No Blue/Green)</h2>
                <img src="{rel_preprocessed}" alt="Preprocessed Image" style="max-width: 500px;">
            </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>X-Ray Analysis Report: {base_name}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .container {{ display: flex; flex-direction: column; gap: 20px; }}
            .plot {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
            .data-link {{ margin-top: 20px; font-size: 1.2em; }}
        </style>
    </head>
    <body>
        <h1>X-Ray Analysis Report: {base_name}</h1>
        
        <div class="container">
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                {center_html}
                {preprocessed_html}
            </div>
            
            <div class="plot">
                <h2>Intensity vs. 2-Theta (Interactive)</h2>
                {theta_html}
            </div>
            
            <div class="plot">
                <h2>Intensity vs. Radius (Interactive)</h2>
                {radius_html}
            </div>
            
            <div class="plot">
                <h2>Static Plots (Reference)</h2>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <img src="{rel_plot_theta}" alt="Intensity vs 2-Theta" style="max-width: 48%;">
                    <img src="{rel_plot_r}" alt="Intensity vs Radius" style="max-width: 48%;">
                </div>
            </div>
        </div>
        
        <div class="data-link">
            <a href="{rel_csv}">Download Analysis Data (CSV)</a>
        </div>
    </body>
    </html>
    """
    
    html_path = Path(output_dir) / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return html_path

def generate_multi_image_report(output_dir, results_list):
    """
    Generates a single HTML report with tabs for multiple processed images.
    """
    if not results_list:
        return None

    tab_headers = []
    tab_contents = []

    for i, res in enumerate(results_list):
        base_name = res['base_name']
        
        # Calculate relative paths
        # output_dir is where the HTML is saved
        
        # Generate HTML for plots
        theta_html = ""
        if res.get('interactive_theta_fig'):
            theta_html = res['interactive_theta_fig'].to_html(full_html=False, include_plotlyjs=False)
        else:
            rel_theta = Path(res["plot_theta_path"]).relative_to(output_dir)
            theta_html = f'<img src="{rel_theta}" alt="Intensity vs 2-Theta">'

        radius_html = ""
        if res.get('interactive_radius_fig'):
            radius_html = res['interactive_radius_fig'].to_html(full_html=False, include_plotlyjs=False)
        else:
            rel_r = Path(res["plot_r_path"]).relative_to(output_dir)
            radius_html = f'<img src="{rel_r}" alt="Intensity vs Radius">'
            
        center_html = ""
        if res.get('plot_center_path'):
             rel_center = Path(res['plot_center_path']).relative_to(output_dir)
             center_html = f"""
                <div class="col-md-12 mb-4">
                    <div class="plot">
                        <h3>Detected Center</h3>
                        <img src="{rel_center}" alt="Detected Center" style="max-width: 500px;">
                    </div>
                </div>
            """
            
        csv_link = Path(res['csv_path']).name
        
        # Tab Header
        active_class = "active" if i == 0 else ""
        tab_headers.append(
            f'<li class="nav-item"><a class="nav-link {active_class}" id="tab-{i}" data-toggle="tab" href="#content-{i}" role="tab" aria-controls="content-{i}" aria-selected="{str(i==0).lower()}">{base_name}</a></li>'
        )
        
        # Tab Content
        show_class = "show active" if i == 0 else ""
        tab_contents.append(f"""
            <div class="tab-pane fade {show_class}" id="content-{i}" role="tabpanel" aria-labelledby="tab-{i}">
                <div class="container-fluid mt-3">
                    <div class="row">
                        {center_html}
                        <div class="col-md-12 mb-4">
                            <div class="plot">
                                <h3>Intensity vs. 2-Theta</h3>
                                {theta_html}
                            </div>
                        </div>
                        <div class="col-md-12 mb-4">
                            <div class="plot">
                                <h3>Intensity vs. Radius</h3>
                                {radius_html}
                            </div>
                        </div>
                    </div>
                    <div class="data-link mt-3 mb-5">
                        <a href="{csv_link}" class="btn btn-primary">Download CSV Data</a>
                    </div>
                </div>
            </div>
        """)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>X-Ray Analysis Report - Combined</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ padding: 20px; background-color: #f4f6f9; }}
            .plot {{ border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .tab-content {{ background: white; border: 1px solid #dee2e6; border-top: none; padding: 20px; border-radius: 0 0 5px 5px; }}
            .nav-tabs .nav-link {{ border-radius: 5px 5px 0 0; font-weight: 500; }}
            h1 {{ color: #2c3e50; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="mb-4">X-Ray Diffraction Analysis Report</h1>
            
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                {''.join(tab_headers)}
            </ul>
            
            <div class="tab-content" id="myTabContent">
                {''.join(tab_contents)}
            </div>
        </div>
        
        <script>
            // Fix for Plotly plots not resizing correctly in hidden tabs
            $('a[data-toggle="tab"]').on('shown.bs.tab', function (e) {{
                window.dispatchEvent(new Event('resize'));
            }})
        </script>
    </body>
    </html>
    """
    
    html_path = Path(output_dir) / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return html_path

def create_montage(image_paths, output_path, grid_width=None):
    """
    Creates a montage of images.
    """
    if not image_paths:
        return None
        
    try:
        from PIL import Image
        import math
        
        images = [Image.open(p) for p in image_paths]
        
        # Determine grid size
        n_images = len(images)
        if grid_width is None:
            grid_width = math.ceil(math.sqrt(n_images))
        
        grid_height = math.ceil(n_images / grid_width)
        
        # Assume all images are roughly same size, use the first one as reference
        w, h = images[0].size
        
        # Create blank canvas
        montage_w = w * grid_width
        montage_h = h * grid_height
        montage = Image.new('RGB', (montage_w, montage_h), (255, 255, 255))
        
        for i, img in enumerate(images):
            # Resize if different? For now assume same size or paste as is
            # If we want to be robust, resize to match first image
            if img.size != (w, h):
                img = img.resize((w, h))
                
            row = i // grid_width
            col = i % grid_width
            
            x = col * w
            y = row * h
            
            montage.paste(img, (x, y))
            
        montage.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Error creating montage: {e}")
        return None
