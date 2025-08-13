import cv2
import numpy as np
from pathlib import Path

def extract_rows_by_grid(table_dir, rows_dir):
    """
    For each table image in table_dir:
      - Detect horizontal grid lines via morphological opening
      - Compute row segments between adjacent lines
      - Crop and save each row to rows_dir
    """
    table_dir = Path(table_dir)
    rows_dir = Path(rows_dir)
    rows_dir.mkdir(parents=True, exist_ok=True)

    for tbl_path in sorted(table_dir.glob("*.png")):
        table = cv2.imread(str(tbl_path))
        if table is None:
            continue

        gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold to binary
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 51, 9
        )

        h, w = bin_img.shape
        # Detect horizontal grid lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
        horiz = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

        # Sum across rows to find line positions
        row_sum = np.sum(horiz > 0, axis=1)
        thresh = 0.5 * np.max(row_sum)
        ys = np.where(row_sum > thresh)[0]

        # Cluster contiguous ys into single line positions
        lines = []
        if ys.size:
            group = [int(ys[0])]
            for y in ys[1:]:
                if y - group[-1] <= 1:
                    group.append(int(y))
                else:
                    lines.append(int(np.mean(group)))
                    group = [int(y)]
            lines.append(int(np.mean(group)))

        # Skip if not enough lines detected
        if len(lines) < 2:
            continue

        # Rows are regions between consecutive lines
        segments = [
            (lines[i], lines[i + 1])
            for i in range(len(lines) - 1)
            if lines[i + 1] - lines[i] > 10
        ]

        # Save each row crop
        for idx, (y0, y1) in enumerate(segments):
            row_img = table[y0:y1, :]
            fname = f"{tbl_path.stem}_row_{idx:03d}.png"
            cv2.imwrite(str(rows_dir / fname), row_img)

if __name__ == "__main__":
    extract_rows_by_grid(
        table_dir="output/tables",
        rows_dir="output/rows"
    )
    print("Rows extracted to output/rows/")