"""Bounded previews of replay evidence; annotations do not alter inference input."""
from collections import Counter
from html import escape


class Samples:
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.counts, self.last, self.items = Counter(), {}, []

    def add(self, frame, detail, roi):
        reason = detail['reason']
        if reason == 'ok' and detail['iris_reason'] != 'ok':
            reason = 'iris_' + detail['iris_reason']
        if self.counts[reason] >= self.limit or frame.timestamp_ms - self.last.get(reason, -1000) < 1000:
            return
        import cv2
        canvas = frame.bgr.copy()
        height, width = canvas.shape[:2]
        scale = min(1., 1280 / max(width, height))
        if scale < 1:
            canvas = cv2.resize(canvas, (round(width*scale), round(height*scale)))
        def box(bounds, color):
            x1,y1,x2,y2 = bounds
            points = [round(min(max(v, 0), maximum)*scale) for v,maximum in
                      zip((x1,y1,x2,y2), (width-1,height-1,width-1,height-1))]
            cv2.rectangle(canvas, tuple(points[:2]), tuple(points[2:]), color, 2)
        box(roi, (0,255,255))
        for candidate in detail['candidates']:
            if candidate['bbox'] is not None:
                color = (0,165,255) if candidate['invalid_reason'] else (
                    (0,255,0) if candidate['inside_roi'] else (0,0,255))
                box(candidate['bbox'], color)
        caption = f'{frame.timestamp_ms/1000:.3f}s {reason} faces={detail["detected_faces"]}'
        cv2.putText(canvas, caption, (8,24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
        filename = f'{reason}-{frame.timestamp_ms:010d}.jpg'
        if not cv2.imwrite(str(self.directory / filename), canvas):
            raise OSError('Could not write diagnostic preview')
        self.counts[reason] += 1
        self.last[reason] = frame.timestamp_ms
        self.items.append(dict(file=filename, source_pts_ms=frame.timestamp_ms, reason=reason))


def write_gallery(output, records):
    parts = ['<!doctype html><meta charset="utf-8"><title>Face replay diagnostics</title>',
             '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}img{max-width:100%}'
             'figure{margin:16px 0}code{overflow-wrap:anywhere}</style>',
             '<h1>Face replay diagnostics</h1><p>Yellow: driver ROI; green: accepted candidate; '
             'red: outside ROI; orange: invalid landmarks. Boxes are landmark bounds, not detector boxes. '
             'Samples are illustrative; use frames.csv for frequencies. No face result does not prove no person is visible.</p>']
    for record in records:
        parts.append(f'<h2>{escape(record["record_id"])} / {escape(record["view"])}</h2>')
        parts.append(f'<p>Slots: {record["slots"]}; partial: {record["partial"]}; '
                     f'head mask mismatches: {record["head_mask_mismatches"]}; '
                     f'iris mask mismatches: {record["iris_mask_mismatches"]}</p>')
        parts.append(f'<p><a href="{record["directory"]}/frames.csv">frames.csv</a> · '
                     f'<a href="{record["directory"]}/candidates.jsonl">candidates.jsonl</a></p>')
        for item in record['samples']:
            parts.append(f'<figure><figcaption>{item["source_pts_ms"]/1000:.3f}s '
                         f'{escape(item["reason"])}</figcaption><img loading="lazy" '
                         f'src="{record["directory"]}/{item["file"]}"></figure>')
    (output / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
