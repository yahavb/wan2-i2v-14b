"""Contact sheet: rows = dumps, columns = frames. python3 contact_sheet.py out.png frame,frame,.. dir [dir ...]"""
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw


def main():
    out, frames, dirs = sys.argv[1], [int(f) for f in sys.argv[2].split(",")], sys.argv[3:]
    rows = []
    for d in dirs:
        v = torch.load(os.path.join(d, "output.pt"))["frames"].numpy()
        tiles = []
        for f in frames:
            im = Image.fromarray(v[min(f, v.shape[0] - 1)])
            ImageDraw.Draw(im).text((4, 4), f"{os.path.basename(d.rstrip('/'))} f{f}", fill=(255, 0, 0))
            tiles.append(np.asarray(im))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print("wrote", out, Image.open(out).size)


if __name__ == "__main__":
    main()
