def test_get_predefined_peaks_format():
    # Create a dummy DataFrame
    angles_deg = np.linspace(0, 40, 400)
    df = pd.DataFrame({"Angle": angles_deg, "Intensity": np.zeros_like(angles_deg)})

    predefined_angles = [10.0, 20.0, 30.0]
    result = get_predefined_peaks(df, predefined_angles)

    assert isinstance(result, list)
    assert len(result) == len(predefined_angles)

    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 3
        closest_idx, popt_tuple, angle_value = item

        assert isinstance(closest_idx, (int, np.integer))
        assert isinstance(popt_tuple, tuple)
        assert len(popt_tuple) == 4
        assert popt_tuple[0] is None
        assert isinstance(popt_tuple[1], float)
        assert popt_tuple[2] is None
        assert popt_tuple[3] is None
        assert isinstance(angle_value, float)
        assert np.isclose(popt_tuple[1], angle_value)