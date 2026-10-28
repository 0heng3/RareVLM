import pytest
from vlm_region_coordinates_v1.data import pixel_box, question


def test_rectangular_scaling_uses_each_axis_actual_frame():
    assert pixel_box([512,128,768,384],1024,512,896,448)==[448,112,672,336]


def test_outward_rounding_preserves_original_region():
    assert pixel_box([1.1,2.2,9.1,8.2],10,10,28,28)==[3,6,26,23]


def test_reject_wrong_frame_and_invalid_geometry():
    for args in (([0,0,11,9],10,10,28,28),([1,1,0,2],10,10,28,28),([1,1,2,2],10,10,0,28)):
        with pytest.raises(ValueError):pixel_box(*args)


def test_two_prompts_disclose_identical_actual_dimensions():
    for mode in ('normalized_dimensions','pixel_dimensions'):
        text=question([1,2,3,4],1036,756,mode)
        assert 'width 1036 pixels and height 756 pixels' in text
    assert 'absolute pixel' in question([1,2,3,4],1036,756,'pixel_dimensions')
