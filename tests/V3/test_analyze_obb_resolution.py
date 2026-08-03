from myscripts.V3.analyze_obb_resolution import AR_BINS, SHORT_BINS, assign_bin


def test_assign_short_bins_are_left_closed():
    assert assign_bin(0.0, SHORT_BINS) == "<2"
    assert assign_bin(2.0, SHORT_BINS) == "2-4"
    assert assign_bin(31.999, SHORT_BINS) == "16-32"
    assert assign_bin(32.0, SHORT_BINS) == ">=32"


def test_assign_aspect_bins():
    assert assign_bin(1.0, AR_BINS) == "<3"
    assert assign_bin(5.0, AR_BINS) == "5-10"
    assert assign_bin(20.0, AR_BINS) == ">=20"
