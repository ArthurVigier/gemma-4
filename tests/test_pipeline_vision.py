import numpy as np

from arc_drone.pipeline_vision import VisionGridConverter


def test_convert_rgb_maps_colors_to_10_symbolic_bins() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :2] = np.array([255, 65, 54], dtype=np.uint8)
    image[:, 2:] = np.array([46, 204, 64], dtype=np.uint8)

    converter = VisionGridConverter()
    result = converter.convert_rgb(image)

    assert result.grid.values.shape == (30, 30)
    assert set(np.unique(result.grid.values)).issubset(set(range(10)))
    assert result.grid.values[0, 0] == 2
    assert result.grid.values[0, -1] == 3
