import cv2
import numpy as np
from pathlib import Path

def detect_table_columns(bin_img, blank_thresh=0.02, min_width=100, fallback=3):
    """
    Split a binarized full-page image into vertical spans (tables)
    by finding columns of content. Falls back to equal splits.
    """
    h, w = bin_img.shape
    col_density = np.sum(bin_img > 0, axis=0) / float(h)
    spans, in_span = [], False
    for x in range(w):
        if col_density[x] > blank_thresh and not in_span:
            in_span, start = True, x
        elif col_density[x] <= blank_thresh and in_span:
            if x - start >= min_width:
                spans.append((start, x))
            in_span = False
    if in_span and w - start >= min_width:
        spans.append((start, w))
    if not spans:
        # fallback to equal-width splits
        width = w // fallback
        spans = [(i*width, w if i==fallback-1 else (i+1)*width) for i in range(fallback)]
    return spans

def segment_rows(table_img, min_row_height=30):
    """
    Given a table image, remove vertical lines and segment into rows.
    """
    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 51, 9
    )
    h, w = bin_img.shape
    # remove vertical grid lines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, h//30)))
    no_vert = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    row_mask = cv2.subtract(bin_img, no_vert)

    # find contours of each row band
    cnts, _ = cv2.findContours(row_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

    rows = []
    for cnt in cnts:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch >= min_row_height and cw >= w * 0.3:
            rows.append((x, y, cw, ch))
    return rows

def extract_tables_and_rows(legend_path, output_dir):
    legend_path = Path(legend_path)
    out = Path(output_dir)
    tables_dir = out / "tables"
    rows_dir   = out / "rows"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)

    # load and binarize full legend
    img = cv2.imread(str(legend_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_full = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 51, 9
    )

    # 1) extract table columns
    cols = detect_table_columns(bin_full, blank_thresh=0.02, min_width=200)
    table_paths = []
    for i, (x0, x1) in enumerate(cols):
        tbl = img[:, x0:x1]
        tbl_path = tables_dir / f"table_{i:03d}.png"
        cv2.imwrite(str(tbl_path), tbl)
        table_paths.append(tbl_path)

    # 2) from each table, extract rows
    total_rows = 0
    for tbl_path in table_paths:
        tbl = cv2.imread(str(tbl_path))
        rows = segment_rows(tbl, min_row_height=30)
        tbl_name = tbl_path.stem
        for j, (x, y, cw, ch) in enumerate(rows):
            row_img = tbl[y:y+ch, x:x+cw]
            row_path = rows_dir / f"{tbl_name}_row_{j:03d}.png"
            cv2.imwrite(str(row_path), row_img)
            total_rows += 1

    print(f"Saved {len(table_paths)} tables to '{tables_dir}'")
    print(f"Saved {total_rows} rows to '{rows_dir}'")

if __name__ == "__main__":
    # Example usage - adjust path to your actual image file
    legend_file = "data/input/your_image.png"
    extract_tables_and_rows(legend_file, "output")