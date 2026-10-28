"""Training-only TextOCR geometry; no predictions or evaluation labels as inputs."""

import math


def bounds(points, width, height):
    if (len(points) < 8 or len(points) % 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in points)):
        raise ValueError('Need finite polygon coordinates')
    xs, ys = points[::2], points[1::2]
    if min(xs) < 0 or min(ys) < 0 or max(xs) > width or max(ys) > height:
        raise ValueError('Polygon lies outside the original image')
    box = [min(xs), min(ys), max(xs), max(ys)]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Zero-area polygon')
    return box


def region(ann, width, height, resized_width, resized_height, policy):
    target = str(ann.get('utf8_string', '')).strip()
    if (not target or target == '.' or len(target) > policy['max_characters']
            or not target.isprintable() or not any(c.isalnum() for c in target)):
        return None, 'unreadable_or_invalid_transcription'
    try:
        box = bounds(ann['points'], width, height)
    except (ValueError, KeyError, TypeError):
        return None, 'invalid_polygon'
    angle = math.degrees(math.atan2(ann['points'][3]-ann['points'][1], ann['points'][2]-ann['points'][0]))
    if abs(angle) > policy['max_horizontal_angle_degrees']:
        return None, 'nonhorizontal_text'
    x0, y0, x1, y1 = box
    effective = [(x1-x0)*resized_width/width, (y1-y0)*resized_height/height]
    if effective[0] < policy['min_resized_width'] or effective[1] < policy['min_resized_height']:
        return None, 'too_small_after_training_resize'
    if (x1-x0)*(y1-y0)/(width*height) > policy['max_image_area_fraction']:
        return None, 'region_too_large'
    normalized = [math.floor(x0/width*1000), math.floor(y0/height*1000),
                  math.ceil(x1/width*1000), math.ceil(y1/height*1000)]
    return {'target': target, 'original_transcription': str(ann['utf8_string']), 'box_xyxy_original_pixels': box,
            'box_xyxy_normalized_1000': normalized, 'effective_region_pixels': effective,
            'points': list(ann['points'])}, None


def excludes_other_word_centers(candidate, others):
    x0,y0,x1,y1 = candidate
    return all(not (x0 < (a+c)/2 < x1 and y0 < (b+d)/2 < y1) for a,b,c,d in others)


def region_question(box):
    return ('Read the text inside the rectangle [' + ', '.join(str(x) for x in box)
            + ']. The rectangle coordinates [left, top, right, bottom] are normalized '
              'from 0 to 1000 relative to the full image.')


def schedule(common, control, region_ids):
    if (len(common) != 1600 or len(control) != 1600 or len(region_ids) != 1600
            or len(set(common+control+region_ids)) != 4800):
        raise ValueError('Need three distinct, aligned 1600-QA catalogs')
    return {role: [{'micros': [ids[begin:begin+4] for ids, begin in
                             [(common, start), (aux, start), (common, start+4), (aux, start+4)]]}
                    for start in range(0,1600,8)]
            for role, aux in [('control',control),('region',region_ids)]}
