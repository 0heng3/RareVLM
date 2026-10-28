import math


def pixel_box(box, original_width, original_height, resized_width, resized_height):
    dimensions = (original_width, original_height, resized_width, resized_height)
    if any(type(v) is not int or v <= 0 for v in dimensions):
        raise ValueError('Positive integer raw and actual processed dimensions required')
    if len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in box):
        raise ValueError('Need finite xyxy coordinates')
    x0,y0,x1,y1 = box
    if not (0 <= x0 < x1 <= original_width and 0 <= y0 < y1 <= original_height):
        raise ValueError('Region must be inside its original image')
    return [math.floor(x0*resized_width/original_width), math.floor(y0*resized_height/original_height),
            math.ceil(x1*resized_width/original_width), math.ceil(y1*resized_height/original_height)]


def question(box, width, height, mode):
    if mode not in ('normalized_dimensions', 'pixel_dimensions'):
        raise ValueError('Unknown preregistered coordinate policy')
    units = ('normalized from 0 to 1000' if mode == 'normalized_dimensions' else 'absolute pixel coordinates')
    return ('Read the text inside the rectangle [' + ', '.join(map(str, box)) + ']. '
            'The rectangle coordinates [left, top, right, bottom] are ' + units
            + f' relative to the full image of width {width} pixels and height {height} pixels.')
