import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_preprocess_image(image_path):
    """Loads an image, converts it to HSV, and applies Gaussian blur."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None, None, None

    px_height, px_width, _ = image.shape
    print(f"Image dimensions: {px_width}px (width) x {px_height}px (height)")
    output_image = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    blurred_v = cv2.GaussianBlur(v, (5, 5), 0)
    return image, output_image, blurred_v, px_height, px_width


def find_big_circle(blurred_v, big_circle_thresh):
    """Finds the largest contour, assumed to be the big circle."""
    print(f"--- Pass 1: Finding Big Circle (Threshold={big_circle_thresh}) ---")
    _, thresh = cv2.threshold(blurred_v, big_circle_thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found in Pass 1. Big circle not detected.\n")
        return None, None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c_big_circle = contours[0]

    m_big = cv2.moments(c_big_circle)
    if m_big["m00"] == 0:
        print("Error: Big circle has zero area.")
        return None, None, None

    big_circle_center = (
        int(m_big["m10"] / m_big["m00"]),
        int(m_big["m01"] / m_big["m00"]),
    )
    print(f"Big Circle Center found at: {big_circle_center}")
    return c_big_circle, big_circle_center, thresh


def _apply_postprocessing(binary_image, action):
    if action is None:
        return binary_image

    if action == "dilate3":
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(binary_image, kernel, iterations=1)
    if action == "close5":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    if action == "open3":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return binary_image


def _apply_thresholding(
    masked_blurred_v,
    small_dot_thresh,
    big_circle_center,
    c_big_circle,
    px_height,
    px_width,
    small_dot_thresh_outer,
    *,
    block_size=11,
):
    """Applies dynamic or fixed adaptive thresholding based on parameters."""
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    if (
        big_circle_center is not None
        and c_big_circle is not None
        and small_dot_thresh_outer is not None
        and px_height is not None
        and px_width is not None
    ):
        (_, radius) = cv2.minEnclosingCircle(c_big_circle)
        boundary_radius = 2.2 * radius
        print(
            "Using two-step adaptive threshold: "
            f"inner_C={small_dot_thresh}, "
            f"outer_C={small_dot_thresh_outer}, "
            f"boundary_radius={boundary_radius:.2f}px, "
            f"blockSize={block_size}"
        )

        y, x = np.ogrid[:px_height, :px_width]
        dist_from_center = np.sqrt(
            (x - big_circle_center[0]) ** 2 + (y - big_circle_center[1]) ** 2
        )

        # Create a mask for the outer region
        outer_mask = dist_from_center > boundary_radius

        # Inner threshold
        thresh_inner = cv2.adaptiveThreshold(
            masked_blurred_v,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            small_dot_thresh,
        )

        # Outer threshold
        thresh_outer = cv2.adaptiveThreshold(
            masked_blurred_v,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            small_dot_thresh_outer,
        )

        # Combine
        thresh = thresh_inner
        thresh[outer_mask] = thresh_outer[outer_mask]

    else:
        print(
            f"Using fixed adaptive threshold with C={small_dot_thresh} and blockSize={block_size}"
        )
        thresh = cv2.adaptiveThreshold(
            masked_blurred_v,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            small_dot_thresh,
        )

    return thresh


def _build_threshold_attempts(small_dot_thresh, small_dot_thresh_outer):
    threshold_attempts = [
        {
            "method": "adaptive",
            "inner": small_dot_thresh,
            "outer": small_dot_thresh_outer,
            "block_size": 11,
            "label": "base adaptive parameters",
            "postprocess": None,
        }
    ]

    for delta, block_size in [(-15, 11), (-25, 13), (15, 11), (-10, 9), (5, 15)]:
        if delta == 0:
            continue
        threshold_attempts.append(
            {
                "method": "adaptive",
                "inner": small_dot_thresh + delta,
                "outer": (
                    small_dot_thresh_outer + delta if small_dot_thresh_outer is not None else None
                ),
                "block_size": block_size,
                "label": f"adaptive delta={delta} blockSize={block_size}",
                "postprocess": None,
            }
        )

    threshold_attempts.extend(
        [
            {
                "method": "adaptive",
                "inner": small_dot_thresh,
                "outer": small_dot_thresh_outer,
                "block_size": 11,
                "label": "adaptive + dilate3",
                "postprocess": "dilate3",
            },
            {
                "method": "adaptive",
                "inner": small_dot_thresh - 10,
                "outer": (
                    (small_dot_thresh_outer - 10) if small_dot_thresh_outer is not None else None
                ),
                "block_size": 9,
                "label": "adaptive delta=-10 blockSize=9 + close5",
                "postprocess": "close5",
            },
            {
                "method": "otsu",
                "label": "global Otsu fallback",
                "postprocess": None,
            },
            {
                "method": "otsu",
                "label": "global Otsu + dilate3",
                "postprocess": "dilate3",
            },
        ]
    )

    return threshold_attempts


def _select_threshold_strategy(
    threshold_attempts,
    masked_blurred_v,
    big_circle_center,
    c_big_circle,
    px_height,
    px_width,
):
    required_contours = 1 if c_big_circle is None else 2

    for attempt_number, attempt in enumerate(threshold_attempts, start=1):
        label = attempt["label"]
        print(f"Attempt {attempt_number}: {label}")

        if attempt["method"] == "adaptive":
            thresh_candidate = _apply_thresholding(
                masked_blurred_v,
                attempt["inner"],
                big_circle_center,
                c_big_circle,
                px_height,
                px_width,
                attempt["outer"],
                block_size=attempt["block_size"],
            )
        else:
            print("Using global Otsu thresholding (adaptive fallback).")
            _, thresh_candidate = cv2.threshold(
                masked_blurred_v,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )

        thresh_candidate = _apply_postprocessing(thresh_candidate, attempt.get("postprocess"))

        contours_candidate, _ = cv2.findContours(
            thresh_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours_candidate) >= required_contours:
            contours = sorted(contours_candidate, key=cv2.contourArea, reverse=True)
            print(
                f"Selected threshold strategy: {label} (found {len(contours_candidate)} contours)"
            )
            return contours, thresh_candidate

        print(
            f"Attempt {attempt_number} produced {len(contours_candidate)} contour(s); "
            "trying next strategy."
        )

    return None, None


def find_small_dots(
    blurred_v,
    small_dot_thresh,
    min_spot_area,
    min_circularity,
    big_circle_center=None,
    c_big_circle=None,
    px_height=None,
    px_width=None,
    small_dot_thresh_outer=None,
    max_distance_percentage=100.0,
):
    """Finds small, circular contours, assumed to be diffraction spots."""
    print(f"\n--- Pass 2: Finding Small Dots (Base Threshold={small_dot_thresh}) ---")

    masked_blurred_v = blurred_v.copy()
    if c_big_circle is not None:
        cv2.drawContours(masked_blurred_v, [c_big_circle], -1, 0, -1)

    threshold_attempts = _build_threshold_attempts(small_dot_thresh, small_dot_thresh_outer)
    contours, thresh = _select_threshold_strategy(
        threshold_attempts,
        masked_blurred_v,
        big_circle_center,
        c_big_circle,
        px_height,
        px_width,
    )

    if contours is None:
        print("Error: No contours found in Pass 2 after fallback strategies.")
        return [], None, None

    other_contours = contours[1:] if c_big_circle is not None and len(contours) > 1 else contours

    detected_dots = []
    rejected_dots_circularity = []
    print(f"Processing {len(other_contours)} potential small dots...")

    max_allowed_distance = (min(px_height, px_width) / 2) * (max_distance_percentage / 100.0)
    print(
        "Max allowed distance from center: "
        f"{max_allowed_distance:.2f} pixels (based on "
        f"{max_distance_percentage}% of min image dimension)"
    )

    for c in other_contours:
        area = cv2.contourArea(c)
        if area < min_spot_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter**2)

        m_small = cv2.moments(c)
        if m_small["m00"] == 0:
            continue
        c_x = int(m_small["m10"] / m_small["m00"])
        c_y = int(m_small["m01"] / m_small["m00"])

        if circularity < min_circularity:
            rejected_dots_circularity.append(
                {"center": (c_x, c_y), "contour": c, "circularity": circularity}
            )
            continue

        if big_circle_center is not None:
            dot_distance = np.sqrt(
                (c_x - big_circle_center[0]) ** 2 + (c_y - big_circle_center[1]) ** 2
            )
            if dot_distance > max_allowed_distance:
                continue

        detected_dots.append({"center": (c_x, c_y), "contour": c})

    print(f"Found {len(detected_dots)} valid small dots.")
    print(f"Rejected {len(rejected_dots_circularity)} dots due to low circularity.")
    return detected_dots, thresh, rejected_dots_circularity


def visualize_and_save_results(
    original_image,
    output_image,
    c_big_circle,
    big_circle_center,
    detected_dots,
    thresh_big_circle,
    thresh_small_dots,
    output_dir,
    rejected_dots_circularity=None,
):
    """Draws detected contours and labels on the image and saves it."""
    print("\n--- Saving Visualization ---")

    cv2.drawContours(output_image, [c_big_circle], -1, (0, 0, 255), 2)
    cv2.circle(output_image, big_circle_center, 5, (0, 0, 255), 2)

    all_small_contours = [d["contour"] for d in detected_dots]
    cv2.drawContours(output_image, all_small_contours, -1, (0, 255, 0), 1)

    for dot in detected_dots:
        center = dot["center"]
        hkl = dot.get("hkl", (0, 0, 0))

        cv2.circle(output_image, center, 5, (0, 255, 0), 2)

        hkl_text = f"({hkl[0]},{hkl[1]},{hkl[2]})"
        text_pos = (center[0] + 8, center[1] + 8)

        cv2.putText(
            output_image,
            hkl_text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if rejected_dots_circularity:
        for r_dot in rejected_dots_circularity:
            center = r_dot["center"]
            cv2.drawContours(output_image, [r_dot["contour"]], -1, (255, 0, 255), 1)
            cv2.circle(output_image, center, 5, (255, 0, 255), 2)

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detected Circles on Original Image")
    axes[1].axis("off")

    axes[2].imshow(thresh_big_circle, cmap="gray")
    axes[2].set_title("Big Circle Threshold")
    axes[2].axis("off")

    axes[3].imshow(thresh_small_dots, cmap="gray")
    axes[3].set_title("Small Dots Threshold")
    axes[3].axis("off")

    plt.tight_layout()
    output_path_combined = output_dir / "detected_spots_final_combined.png"
    plt.savefig(output_path_combined)
    plt.close()

    output_path_circles_on_original = output_dir / "detected_circles_on_original.png"
    cv2.imwrite(str(output_path_circles_on_original), output_image)

    print(
        f"Successfully saved combined visualization to '{output_path_combined.absolute().as_uri()}'"
    )
    print(
        "Successfully saved circles on original visualization to "
        f"'{output_path_circles_on_original.absolute().as_uri()}'"
    )


def save_dots_to_csv(detected_dots, output_dir):
    print("\n--- Saving Detected Dots to CSV ---")
    csv_path = output_dir / "detected_dots.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Dot_ID",
                "Center_X_px",
                "Center_Y_px",
                "x_Q_mm",
                "y_Q_mm",
                "z_Q_mm",
                "h",
                "k",
                "ell",
                "d_pm",
                "Theta_deg",
                "Lambda_pm",
            ]
        )
        for i, dot in enumerate(detected_dots):
            center_x, center_y = dot["center"]
            h, k, ell = dot["hkl"]
            writer.writerow(
                [
                    i + 1,
                    center_x,
                    center_y,
                    f"{dot['x_mm']:.3f}",
                    f"{dot['y_mm']:.3f}",
                    f"{dot['z_mm']:.1f}",
                    h,
                    k,
                    ell,
                    f"{dot['d_pm']:.2f}",
                    f"{dot['theta_deg']:.3f}",
                    f"{dot['lambda_pm']:.2f}",
                ]
            )
    print(f"Successfully saved detected dots to '{csv_path.absolute().as_uri()}'")
