"""Convert CORD-v2 (Hugging Face: naver-clova-ix/cord-v2) to an Ultralytics YOLO dataset.

One box per CORD line and category, covering only the value words
(is_key == 0). Key words such as "TOTAL" or "CASH" are left out, so a
total.total_price box sits on the amount only.

Usage: python cord_to_yolo.py OUT_DIR [--splits train validation test]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# Categories with >= 90 lines in the CORD train split. Everything else is
# skipped (too rare to learn, or vague like sub_total.etc).
CLASSES = [
    'menu.nm',
    'menu.cnt',
    'menu.price',
    'menu.unitprice',
    'menu.discountprice',
    'menu.num',
    'menu.sub.nm',
    'menu.sub.cnt',
    'menu.sub.price',
    'sub_total.subtotal_price',
    'sub_total.tax_price',
    'sub_total.service_price',
    'total.total_price',
    'total.cashprice',
    'total.changeprice',
    'total.creditcardprice',
    'total.menuqty_cnt',
]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# Hugging Face split name -> YOLO folder name.
SPLIT_DIR = {'train': 'train', 'validation': 'val', 'test': 'test'}


def line_boxes(gt, width, height):
    """Yield (class_id, cx, cy, w, h) normalized YOLO boxes for one receipt."""
    for line in gt['valid_line']:
        cls = CLASS_ID.get(line['category'])
        if cls is None:
            continue
        words = [w for w in line['words'] if not w['is_key']]
        if not words:
            continue
        xs = [w['quad'][f'x{i}'] for w in words for i in range(1, 5)]
        ys = [w['quad'][f'y{i}'] for w in words for i in range(1, 5)]
        x1, x2 = max(min(xs), 0), min(max(xs), width)
        y1, y2 = max(min(ys), 0), min(max(ys), height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        yield (
            cls,
            (x1 + x2) / 2 / width,
            (y1 + y2) / 2 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        )


def convert(out_dir, splits):
    out_dir = Path(out_dir)
    for split in splits:
        name = SPLIT_DIR[split]
        img_dir = out_dir / 'images' / name
        lbl_dir = out_dir / 'labels' / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i, sample in enumerate(load_dataset('naver-clova-ix/cord-v2', split=split)):
            gt = json.loads(sample['ground_truth'])
            image = sample['image'].convert('RGB')
            # Quads are in the coordinates of meta.image_size; normalizing by
            # that size keeps boxes right even if the stored image was resized.
            size = gt.get('meta', {}).get('image_size') or {}
            width = size.get('width') or image.width
            height = size.get('height') or image.height

            stem = f'cord_{name}_{i:04d}'
            image.save(img_dir / f'{stem}.jpg', quality=95)
            rows = [
                f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'
                for c, cx, cy, w, h in line_boxes(gt, width, height)
            ]
            (lbl_dir / f'{stem}.txt').write_text('\n'.join(rows) + '\n' if rows else '')
            count += 1
        print(f'{split}: {count} images -> {img_dir}')

    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(CLASSES))
    yaml = f'path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\n'
    if 'test' in splits:
        yaml += 'test: images/test\n'
    (out_dir / 'data.yaml').write_text(f'{yaml}names:\n{names}\n')
    (out_dir / 'classes.txt').write_text('\n'.join(CLASSES) + '\n')
    print(f'wrote {out_dir / "data.yaml"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--splits', nargs='+', default=list(SPLIT_DIR), choices=list(SPLIT_DIR))
    args = parser.parse_args()
    convert(args.out_dir, args.splits)
