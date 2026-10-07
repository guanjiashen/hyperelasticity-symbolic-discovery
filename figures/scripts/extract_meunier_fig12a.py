"""Crop Meunier et al. Fig. 12(a) and remove only its embedded panel letter."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image


def main(source: Path, output: Path):
    image = Image.open(source).convert("L")
    # The publisher stores Fig. 12(a,b) as one 678 x 688 JPEG. The photograph
    # occupies the left 290 columns; the remaining columns are panel (b).
    array = np.asarray(image.crop((0, 0, 290, image.height))).copy()

    # Remove the black lower-case 'a' at x=16..35, y=18..40. Fill its mask by
    # row-wise interpolation from untouched pixels on either side so that the
    # clamp texture and every specimen pixel outside the letter remain intact.
    y0, y1, x0, x1 = 14, 45, 12, 41
    region = array[y0:y1, x0:x1]
    mask = region < 90
    expanded = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            expanded |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    for local_y in range(region.shape[0]):
        xs = np.flatnonzero(expanded[local_y])
        if not len(xs):
            continue
        left = max(0, int(xs.min()) - 2)
        right = min(region.shape[1] - 1, int(xs.max()) + 2)
        start = float(region[local_y, left])
        stop = float(region[local_y, right])
        span = max(1, right - left)
        for local_x in xs:
            alpha = (int(local_x) - left) / span
            region[local_y, local_x] = round((1.0 - alpha) * start + alpha * stop)
    array[y0:y1, x0:x1] = region

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(output, optimize=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_meunier_fig12a.py SOURCE_JPEG OUTPUT_PNG")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
