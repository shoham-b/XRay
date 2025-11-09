import math

# --- 1. CORE INDEXING FUNCTIONS (Based on PDF) ---


def find_smallest_integers(x, y, z, tolerance=0.15):
    """
    Finds the smallest integer ratio (h, k, l) for (x, y, z).

    This is a simple algorithm that tries to find a common multiplier
    to convert the ratios into integers.
    """
    # Handle zero case to avoid division by zero
    if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9:
        return (0, 0, 0)

    # Find the smallest non-zero component to use as the divisor
    vec = [x, y, z]
    non_zero_vals = [abs(v) for v in vec if abs(v) > 1e-9]
    if not non_zero_vals:
        return (0, 0, 0)

    smallest = min(non_zero_vals)

    # Create ratios based on the smallest non-zero value
    ratios = [v / smallest for v in vec]

    # Try to find a multiplier (from 1 to 10) that makes them all integers
    for multiplier in range(1, 11):
        scaled_ratios = [r * multiplier for r in ratios]

        # Check if all scaled ratios are within 'tolerance' of a whole number
        if all(abs(sr - round(sr)) < tolerance for sr in scaled_ratios):
            # Success! Return the rounded integers
            return tuple(int(round(sr)) for sr in scaled_ratios)

    # If no simple multiplier is found, return the nearest integers for the
    # multiplier-of-1 ratio as a best-effort guess.
    return tuple(int(round(r)) for r in ratios)


def index_spot(x_p, y_p, L):
    """
    Calculates the (h, k, l) indices for a single Laue spot.
    Based on equations (XIII) and (XIV) from the provided PDF.

    Assumes crystal [100] axis is aligned with the beam.
    """
    # Use aliases from the PDF for clarity
    x_q = x_p
    y_q = y_p

    # Equation (XIII)
    # Handle the case L=0 to avoid division by zero
    if L == 0:
        return (0, 0, 0)

    # Calculate z_Q
    try:
        z_q = math.sqrt(x_q * 2 + y_q2 + L * 2) - L
    except ValueError:
        # This can happen if L is negative, etc.
        return (0, 0, 0)

    # Proportionality from Equation (XIV)
    # Find the smallest integer triple (h, k, l)
    hkl = find_smallest_integers(x_q, y_q, z_q)

    return hkl


# --- 2. CRYSTAL FAMILY ANALYSIS (SYMMETRY OF SPOTS) ---


def calculate_stats(distances):
    """Calculates the mean and standard deviation of a list of distances."""
    if not distances:
        return 0.0, 0.0

    n = len(distances)
    if n == 0:
        return 0.0, 0.0

    mean = sum(distances) / n

    if n < 2:  # Cannot calculate std. dev. with only one point
        return mean, 0.0

    variance = sum((x - mean) ** 2 for x in distances) / (n - 1)  # Sample std. dev.
    std_dev = math.sqrt(variance)

    return mean, std_dev


def get_partner_error(spot_list, target_x, target_y, tolerance):
    """
    Finds the closest spot to the target, if it's within tolerance.
    Returns the error distance if found, otherwise None.
    """
    min_dist = float("inf")
    found = False

    for x, y in spot_list:
        dist = math.sqrt((x - target_x) * 2 + (y - target_y) * 2)
        if dist < min_dist:
            min_dist = dist

    if min_dist <= tolerance:
        return min_dist  # Return the error of the best match
    return None  # No match found within tolerance


def analyze_pattern_symmetry(spot_list, tolerance_mm, threshold_percent):
    """
    Analyzes the (x,y) spot coordinates for geometric symmetry.
    This helps determine the Laue Class / Crystal Family.

    Assumes the pattern is centered at (0,0).
    """
    if not spot_list:
        return "Unknown"

    total_spots = len(spot_list)
    found = {"2-fold": 0, "4-fold": 0, "6-fold": 0, "mirror-x": 0, "mirror-y": 0}
    errors = {"2-fold": [], "4-fold": [], "6-fold": [], "mirror-x": [], "mirror-y": []}

    # Pre-calculate trig values for 6-fold rotation (60 degrees)
    cos60 = 0.5
    sin60 = math.sqrt(3) / 2.0  # 0.866...

    # Check each spot for its symmetric partners
    for x, y in spot_list:
        # 2-fold rotation: (x,y) -> (-x,-y)
        dist = get_partner_error(spot_list, -x, -y, tolerance_mm)
        if dist is not None:
            found["2-fold"] += 1
            errors["2-fold"].append(dist)

        # 4-fold rotation: (x,y) -> (-y,x) or (y,-x)
        # We check one direction, as the other is implied by the partner's check
        dist = get_partner_error(spot_list, -y, x, tolerance_mm)
        if dist is not None:
            found["4-fold"] += 1
            errors["4-fold"].append(dist)

        # 6-fold rotation: (x,y) -> (x*cos60 - y*sin60, x*sin60 + y*cos60)
        x_prime_60 = x * cos60 - y * sin60
        y_prime_60 = x * sin60 + y * cos60
        dist = get_partner_error(spot_list, x_prime_60, y_prime_60, tolerance_mm)
        if dist is not None:
            found["6-fold"] += 1
            errors["6-fold"].append(dist)

        # Mirror-X (y-plane): (x,y) -> (x,-y)
        dist = get_partner_error(spot_list, x, -y, tolerance_mm)
        if dist is not None:
            found["mirror-x"] += 1
            errors["mirror-x"].append(dist)

        # Mirror-Y (x-plane): (x,y) -> (-x,y)
        dist = get_partner_error(spot_list, -x, y, tolerance_mm)
        if dist is not None:
            found["mirror-y"] += 1
            errors["mirror-y"].append(dist)

    # Calculate percentages
    percentages = {
        "2-fold": (found["2-fold"] / total_spots) * 100 if total_spots > 0 else 0,
        "4-fold": (found["4-fold"] / total_spots) * 100 if total_spots > 0 else 0,
        "6-fold": (found["6-fold"] / total_spots) * 100 if total_spots > 0 else 0,
        "mirror-x": (found["mirror-x"] / total_spots) * 100 if total_spots > 0 else 0,
        "mirror-y": (found["mirror-y"] / total_spots) * 100 if total_spots > 0 else 0,
    }

    # Calculate error statistics
    stats = {}
    for key in errors:
        mean_err, std_dev_err = calculate_stats(errors[key])
        stats[key] = (mean_err, std_dev_err)

    print("--- Geometric Symmetry Analysis (Crystal Family) ---")
    print(f" (using {tolerance_mm}mm tolerance)\n")

    print(
        f"  2-fold rotation (x,y) -> (-x,-y):   {percentages['2-fold']:6.2f}% found. (Avg Error: {stats['2-fold'][0]:.2f} \u00b1 {stats['2-fold'][1]:.2f} mm)"
    )
    print(
        f"  4-fold rotation (x,y) -> (-y,x):   {percentages['4-fold']:6.2f}% found. (Avg Error: {stats['4-fold'][0]:.2f} \u00b1 {stats['4-fold'][1]:.2f} mm)"
    )
    print(
        f"  6-fold rotation (x,y) -> (60 deg):  {percentages['6-fold']:6.2f}% found. (Avg Error: {stats['6-fold'][0]:.2f} \u00b1 {stats['6-fold'][1]:.2f} mm)"
    )
    print(
        f"  Mirror-X (y-plane):               {percentages['mirror-x']:6.2f}% found. (Avg Error: {stats['mirror-x'][0]:.2f} \u00b1 {stats['mirror-x'][1]:.2f} mm)"
    )
    print(
        f"  Mirror-Y (x-plane):               {percentages['mirror-y']:6.2f}% found. (Avg Error: {stats['mirror-y'][0]:.2f} \u00b1 {stats['mirror-y'][1]:.2f} mm)"
    )

    # --- Crystal Family Similarity Scores ---
    print("\n--- Crystal Family Similarity ---")
    p_2fold = percentages["2-fold"]
    p_4fold = percentages["4-fold"]
    p_6fold = percentages["6-fold"]
    # A high score for Triclinic means a low score for all others
    p_triclinic = 100.0 - max(p_2fold, p_4fold, p_6fold, 0.0)

    print(
        f"  Cubic / Tetragonal (4-fold):        {p_4fold:6.2f}% (Avg Error: {stats['4-fold'][0]:.2f} \u00b1 {stats['4-fold'][1]:.2f} mm)"
    )
    print(
        f"  Hexagonal (6-fold):                 {p_6fold:6.2f}% (Avg Error: {stats['6-fold'][0]:.2f} \u00b1 {stats['6-fold'][1]:.2f} mm)"
    )
    print(
        f"  Orthorhombic / Monoclinic (2-fold): {p_2fold:6.2f}% (Avg Error: {stats['2-fold'][0]:.2f} \u00b1 {stats['2-fold'][1]:.2f} mm)"
    )
    print(f"  Triclinic (lack of rotation):       {p_triclinic:6.2f}% (No rotational error)")

    # --- Symmetry Conclusion (Highest Score Wins) ---
    print("\n--- Symmetry Conclusion ---")

    # 1. Create a dictionary of the rotational scores
    scores = {
        "Cubic / Tetragonal": p_4fold,
        "Hexagonal": p_6fold,
        "Orthorhombic / Monoclinic": p_2fold,
        "Triclinic": p_triclinic,
    }

    # 2. Find the family with the highest score
    best_family_key = max(scores, key=scores.get)
    best_score = scores[best_family_key]

    print(f"✅ Best fit: {best_family_key} (at {best_score:.2f}%)")

    # 3. Refine the name based on mirrors (using the user's threshold)
    has_mirror_x = percentages["mirror-x"] >= threshold_percent
    has_mirror_y = percentages["mirror-y"] >= threshold_percent
    has_mirrors = has_mirror_x or has_mirror_y

    # 4. Assign the final family name based on the winner + mirrors
    family = "Unknown"
    if best_family_key == "Cubic / Tetragonal":
        if has_mirrors:
            print("   (Mirror symmetry detected, consistent with m-3m or 4/mmm)")
            family = "Cubic / Tetragonal"
        else:
            print("   (No mirror symmetry detected, consistent with 4/m)")
            family = "Tetragonal (Low Symmetry)"  # More specific

    elif best_family_key == "Hexagonal":
        if has_mirrors:
            print("   (Mirror symmetry detected, consistent with 6/mmm)")
            family = "Hexagonal"
        else:
            print("   (No mirror symmetry detected, consistent with 6/m)")
            family = "Hexagonal (Low Symmetry)"

    elif best_family_key == "Orthorhombic / Monoclinic":
        if has_mirrors:
            print("   (Mirror symmetry detected, consistent with mmm)")
            family = "Orthorhombic"  # More specific
        else:
            print("   (No mirror symmetry detected, consistent with 2/m)")
            family = "Monoclinic"  # More specific

    elif best_family_key == "Triclinic":
        print("   (No significant rotational symmetry found)")
        family = "Triclinic"

    return family


# --- 3. LATTICE CENTERING ANALYSIS (STATISTICS OF HKL) ---


# Rule checkers for the 7 centering types
def check_primitive(h, k, l):
    """Primitive (P) - Rule: None. Always returns True."""
    return True


def check_body_centered(h, k, l):
    """Body-Centered (I) - Rule: h+k+l = even"""
    return (h + k + l) % 2 == 0


def check_face_centered(h, k, l):
    """Face-Centered (F) - Rule: h,k,l all even or all odd"""
    all_even = (h % 2 == 0) and (k % 2 == 0) and (l % 2 == 0)
    all_odd = (h % 2 != 0) and (k % 2 != 0) and (l % 2 != 0)
    return all_even or all_odd


def check_a_centered(h, k, l):
    """A-Centered (A) - Rule: k+l = even"""
    return (k + l) % 2 == 0


def check_b_centered(h, k, l):
    """B-Centered (B) - Rule: h+l = even"""
    return (h + l) % 2 == 0


def check_c_centered(h, k, l):
    """C-Centered (C) - Rule: h+k = even"""
    return (h + k) % 2 == 0


def check_rhombohedral(h, k, l):
    """Rhombohedral (R) - Rule: -h+k+l = 3n (for hexagonal setting)"""
    return (-h + k + l) % 3 == 0


def run_statistical_analysis(hkl_list, crystal_family, threshold_percent=90.0):
    """
    Checks the (hkl) list against all 7 lattice centering rules
    and reports the percentage of spots that "fit" each rule.
    """
    # Remove the (0,0,0) spot (central beam) from analysis
    cleaned_list = [hkl for hkl in hkl_list if hkl != (0, 0, 0)]

    total_spots = len(cleaned_list)
    if total_spots == 0:
        print("No valid spots to analyze.")
        return []  # Return an empty list

    print(f"\n--- Statistical Analysis of {len(cleaned_list)} Reflections ---")

    # Define all rules to check
    rules = {
        "Primitive (P)": check_primitive,
        "Body-Centered (I)": check_body_centered,
        "Face-Centered (F)": check_face_centered,
        "A-Centered (A)": check_a_centered,
        "B-Centered (B)": check_b_centered,
        "C-Centered (C)": check_c_centered,
        "Rhombohedral (R)": check_rhombohedral,
    }

    # Count violations for each rule
    violations = {rule_name: 0 for rule_name in rules}

    for h, k, l in cleaned_list:
        for rule_name, rule_func in rules.items():
            if not rule_func(h, k, l):
                # This spot is a "violation" for this rule
                violations[rule_name] += 1

    # --- Print the Results ---
    print("\n--- Results: Violations per Lattice Type ---")
    print("A 'violation' means an observed spot is FORBIDDEN by the lattice rule.")
    print(f"The 'best fit' lattices are those that meet the {threshold_percent}% fit threshold.\n")

    # Sort results by the number of violations (best fit first)
    sorted_results = sorted(violations.items(), key=lambda item: item[1])

    for rule_name, viol_count in sorted_results:
        percent_fit = 100 * (total_spots - viol_count) / total_spots
        print(f"  {rule_name:<20}: {viol_count} violations ({percent_fit:6.2f}% fit)")

    print("\n--- Centering Conclusion ---")

    # Find all rules that meet the threshold
    best_fits = []
    for rule_name, viol_count in sorted_results:
        percent_fit = 100 * (total_spots - viol_count) / total_spots
        if percent_fit >= threshold_percent:
            best_fits.append(rule_name)

    if not best_fits:
        print(f"No lattice type met the {threshold_percent}% fit threshold.")
        print("This may be due to experimental error, crystal misalignment,")
        print("or the crystal being non-cubic and mis-indexed.")
        best_fits = []
    else:
        # 'Primitive (P)' will always be in best_fits if any other rule passes,
        # so we look for the most restrictive rule that also passed.
        more_restrictive_fits = [name for name in best_fits if name != "Primitive (P)"]

        if not more_restrictive_fits:
            # This means 'Primitive (P)' was the only fit.
            print(
                f"The data is consistent with a *Primitive (P)* centering (met {threshold_percent}% threshold)."
            )
            print("No more restrictive centering rules were met.")
            best_fits = ["Primitive (P)"]
        else:
            # We found one or more restrictive fits (e.g., F, I, etc.)
            print(f"The data (met {threshold_percent}% threshold) is consistent with:")
            for name in more_restrictive_fits:
                print(f"  - *{name}*")
            # We only care about the most restrictive fits
            best_fits = more_restrictive_fits

    return best_fits  # Return the best fit centering types


# --- Main Program ---
if _name_ == "_main_":
    # --- 1. YOUR INPUTS GO HERE ---

    # Set your crystal-to-film distance (in mm)
    # This value is from the PDF article, page 4
    L_crystal = 15.0

    # Set the threshold for symmetry detection (in percent)
    # A value of 80-90% is reasonable for experimental data.
    symmetry_threshold = 85.0

    # Set the tolerance for finding a symmetric partner (in mm)
    # The article's data is quite noisy (e.g., 9.0 vs 11.0), so
    # we'll use a larger tolerance.
    symmetry_tolerance_mm = 2.0

    # Set the file to save (h,k,l) data to
    output_hkl_file = "indexed_hkl.txt"

    # --- 2. SPOT DATA FROM ARTICLE (Table 1, NaCl) ---
    # A list of (x, y) tuples from your film (in mm)
    # These are the (x_Q, y_Q) coordinates from Table 1, page 5.
    measured_spots_xy = [
        (17.2, 8.4),  # (4, 2, 2)
        (-7.8, -15.2),  # (-2, -4, 2)
        (7.8, -15.2),  # (2, -4, 2)
        (15.8, -8.0),  # (4, -2, 2)
        (12.6, -0.4),  # (6, 0, 2)
        (0.2, 14.5),  # (0, 6, 2)
        (-13.2, -0.1),  # (-6, 0, 2)
        (0.0, -11.2),  # (0, -6, 2)
        (11.5, 3.6),  # (6, 2, 2)
        (4.2, 12.4),  # (2, 6, 2)
        (-4.3, 13.0),  # (-2, 6, 2)
        (-12.2, 4.0),  # (-6, 2, 2)
        (-11.2, -3.9),  # (-6, -2, 2)
        (-3.3, -10.2),  # (-2, -6, 2)
        (3.2, -10.0),  # (2, -6, 2)
        (10.5, -3.9),  # (6, -2, 2)
        (9.0, 9.2),  # (4, 4, 2)
        (-11.0, 11.0),  # (-4, 4, 2)
        (-9.0, -9.2),  # (-4, -4, 2)
        (9.0, -9.0),  # (4, -4, 2)
        (9.0, 5.8),  # (6, 4, 2)
        (6.3, 9.2),  # (4, 6, 2)
        (-6.5, 9.8),  # (-4, 6, 2) -- Note: article has h=-6, k=4, l=2? Typo in table?
        (-9.5, 6.2),  # (-6, 4, 2)
        (-8.2, -5.5),  # (-6, -4, 2)
        (-5.2, -8.0),  # (-4, -6, 2)
        (6.4, 6.2),  # (3, 3, 1)
        (-6.8, 6.8),  # (-3, 3, 1)
        (-5.8, -5.8),  # (-3, -3, 1)
        (5.3, -5.3),  # (3, -3, 1)
        (6.8, 1.3),  # (5, 1, 1)
        (1.8, 7.8),  # (1, 5, 1)
        (-1.2, 8.0),  # (-1, 5, 1)
        (-7.2, 1.2),  # (-5, 1, 1)
        (-7.0, -1.5),  # (-5, -1, 1)
    ]

    # --- 3. RUN THE GEOMETRIC ANALYSIS ---
    print("### Step 1: Crystal Family Analysis (from spot geometry) ###")
    crystal_family = analyze_pattern_symmetry(
        measured_spots_xy, symmetry_tolerance_mm, symmetry_threshold
    )

    # --- 4. RUN THE INDEXING FOR ALL SPOTS ---
    all_hkl = []
    for spot in measured_spots_xy:
        x, y = spot
        hkl = index_spot(x, y, L_crystal)
        all_hkl.append(hkl)

    # --- 5. SAVE HKL DATA TO FILE ---
    try:
        with open(output_hkl_file, "w") as f:
            f.write(f"# (h, k, l) indices indexed from Laue pattern (L={L_crystal} mm)\n")
            f.write("# Based on PDF Equation XIV (h:k:l = x_Q:y_Q:z_Q)\n")
            f.write("# h\tk\tl\n")
            for h, k, l in all_hkl:
                f.write(f"{h}\t{k}\t{l}\n")
        print(f"\nSuccessfully saved indexed (h,k,l) data to '{output_hkl_file}'")
    except Exception as e:
        print(f"\nERROR: Could not write to file '{output_hkl_file}': {e}")

    # --- 6. RUN THE STATISTICAL ANALYSIS ---
    print("\n### Step 2: Lattice Centering Analysis (from (h,k,l) list) ###")
    centering_types = run_statistical_analysis(all_hkl, crystal_family, symmetry_threshold)

    # --- 7. FINAL COMBINED CONCLUSION ---
    print("\n" + "=" * 50)
    print("### FINAL CONCLUSION ###")
    print("=" * 50)

    if crystal_family == "Unknown" or not centering_types:
        print("Analysis inconclusive. See messages above.")
        print("This often means:")
        print(" 1. The crystal was not aligned (see README.md).")
        print(" 2. The symmetry threshold is too high for your data.")
        print(" 3. The data is not centered at (0,0).")
    else:
        # Provide a combined name, e.g., "Face-Centered Cubic"
        # This is a best-guess synthesis of the two separate analyses.

        # Simplify family name for conclusion
        if "Cubic" in crystal_family:
            family_name = "Cubic"
        elif "Tetragonal" in crystal_family:
            family_name = "Tetragonal"
        elif "Orthorhombic" in crystal_family:
            family_name = "Orthorhombic"
        elif "Monoclinic" in crystal_family:
            family_name = "Monoclinic"
        else:
            family_name = "Triclinic"

        print(f"Step 1 (Geometry) suggests Crystal Family: *{family_name}*")
        print(f"Step 2 (Statistics) suggests Centering:  *{', '.join(centering_types)}*")
        print("\n---")
        print("Best-Fit Bravais Lattice(s):")
        for centering in centering_types:
            # Handle "Primitive (P)" which doesn't need a prefix
            if centering == "Primitive (P)":
                print(f"  - *{family_name}* (Primitive)")
            else:
                print(f"  - *{centering.split(' ')[0]} {family_name}*")
