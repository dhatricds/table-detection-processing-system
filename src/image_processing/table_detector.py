import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
import os
import matplotlib.pyplot as plt

class ImprovedBoundaryDetector:
    """
    Improved version with better boundary processing and classification
    """

    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode

    def crop_main_border(self, pdf_path: str, margin: float = 0.05) -> Image.Image:
        """Remove main border from PDF"""
        print(f"🔄 Converting PDF to image...")
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        img = pages[0]

        print(f"📏 Original size: {img.width} × {img.height} pixels")

        left = int(img.width * margin)
        top = int(img.height * margin)
        right = int(img.width * (1 - margin))
        bottom = int(img.height * (1 - margin))

        cropped = img.crop((left, top, right, bottom))
        print(f"✂️  Border cropped to: {cropped.width} × {cropped.height} pixels")

        return cropped

    def detect_table_boundaries_improved(self, img: Image.Image) -> list:
        """
        IMPROVED: Better boundary detection with proper filtering
        """
        print("🔍 IMPROVED table boundary detection...")

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Method 1: Contour detection
        boundaries_method1 = self.detect_by_contours(gray)

        # Method 2: Morphological detection
        boundaries_method2 = self.detect_by_morphology(gray)

        # Method 3: Smart coordinate-based detection (as backup)
        boundaries_method3 = self.smart_coordinate_detection(img.width, img.height)

        # Combine and process boundaries
        all_boundaries = boundaries_method1 + boundaries_method2

        # If automatic detection fails, use coordinate-based
        if len(all_boundaries) < 3:
            print("⚠️  Few boundaries detected, adding coordinate-based detection...")
            all_boundaries.extend(boundaries_method3)

        # Process boundaries properly
        final_boundaries = self.process_boundaries_improved(all_boundaries, img.width, img.height)

        if self.debug_mode:
            self.show_debug_improved(img, final_boundaries, boundaries_method1, boundaries_method2, boundaries_method3)

        print(f"✅ Found {len(final_boundaries)} valid table boundaries")
        return final_boundaries

    def detect_by_contours(self, gray: np.ndarray) -> list:
        """Improved contour detection"""
        # Edge detection with multiple thresholds
        edges1 = cv2.Canny(gray, 30, 100, apertureSize=3)
        edges2 = cv2.Canny(gray, 50, 150, apertureSize=3)
        edges = cv2.bitwise_or(edges1, edges2)

        # Morphological closing to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundaries = []
        min_area = gray.shape[0] * gray.shape[1] * 0.008  # At least 0.8% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Size filtering
                if w > 80 and h > 40:
                    # Aspect ratio filtering
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 15:  # More lenient
                        boundaries.append((x, y, w, h))

        return boundaries

    def detect_by_morphology(self, gray: np.ndarray) -> list:
        """Improved morphological detection"""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 5)

        # Detect horizontal and vertical lines separately
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        # Extract lines
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        # Combine
        combined = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)

        # Dilate to connect nearby elements
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.dilate(combined, kernel_dilate, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundaries = []
        min_area = gray.shape[0] * gray.shape[1] * 0.005

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 60 and h > 30:
                    boundaries.append((x, y, w, h))

        return boundaries

    def smart_coordinate_detection(self, img_width: int, img_height: int) -> list:
        """Smart coordinate detection based on typical layout"""
        return [
            # Main electrical symbols table (left)
            (int(img_width * 0.02), int(img_height * 0.05), int(img_width * 0.48), int(img_height * 0.70)),
            # Abbreviations (center-left)
            (int(img_width * 0.52), int(img_height * 0.05), int(img_width * 0.14), int(img_height * 0.35)),
            # General notes (right)
            (int(img_width * 0.68), int(img_height * 0.05), int(img_width * 0.30), int(img_height * 0.75)),
            # Drawing index (bottom-left)
            (int(img_width * 0.02), int(img_height * 0.77), int(img_width * 0.48), int(img_height * 0.21)),
            # Alternates (bottom-center)
            (int(img_width * 0.52), int(img_height * 0.42), int(img_width * 0.14), int(img_height * 0.12))
        ]

    def process_boundaries_improved(self, boundaries: list, img_width: int, img_height: int) -> list:
        """
        IMPROVED: Better boundary processing with proper overlap handling
        """
        if not boundaries:
            return []

        print(f"🔧 Processing {len(boundaries)} raw boundaries...")

        # Step 1: Remove tiny boundaries
        min_area = 5000  # Minimum area in pixels
        filtered = []
        for x, y, w, h in boundaries:
            if w * h > min_area and w > 50 and h > 30:
                filtered.append((x, y, w, h))

        print(f"   After size filtering: {len(filtered)}")

        # Step 2: Remove duplicates (very similar boundaries)
        unique = []
        for current in filtered:
            x1, y1, w1, h1 = current
            is_duplicate = False

            for existing in unique:
                x2, y2, w2, h2 = existing

                # Check if centers are close and sizes similar
                center1 = (x1 + w1/2, y1 + h1/2)
                center2 = (x2 + w2/2, y2 + h2/2)
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

                size_ratio = min(w1*h1, w2*h2) / max(w1*h1, w2*h2)

                if distance < 30 and size_ratio > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(current)

        print(f"   After duplicate removal: {len(unique)}")

        # Step 3: Handle significant overlaps (merge or keep larger)
        final = []
        for current in unique:
            x1, y1, w1, h1 = current
            should_add = True

            for i, existing in enumerate(final):
                x2, y2, w2, h2 = existing

                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                current_area = w1 * h1
                existing_area = w2 * h2
                smaller_area = min(current_area, existing_area)

                # If overlap is more than 50% of smaller rectangle
                if overlap_area > 0.5 * smaller_area:
                    # Keep the larger one
                    if current_area > existing_area:
                        final[i] = current  # Replace with larger
                    should_add = False
                    break

            if should_add:
                final.append(current)

        print(f"   After overlap processing: {len(final)}")

        # Step 4: Sort by position (top to bottom, left to right)
        final.sort(key=lambda b: (b[1], b[0]))

        # Step 5: Assign unique IDs to avoid naming conflicts
        final_with_ids = []
        for i, boundary in enumerate(final):
            x, y, w, h = boundary
            table_info = self.classify_table_improved(x, y, w, h, img_width, img_height, i)
            final_with_ids.append((x, y, w, h, table_info))

        return final_with_ids

    def classify_table_improved(self, x: int, y: int, w: int, h: int,
                               img_width: int, img_height: int, table_id: int) -> dict:
        """
        IMPROVED: Better classification with unique naming
        """
        center_x = x + w/2
        center_y = y + h/2
        rel_x = center_x / img_width
        rel_y = center_y / img_height
        aspect_ratio = w / h

        # Base classification on position
        if rel_x < 0.35:  # Left side
            if rel_y < 0.6:
                table_type = 'electrical_symbols'
                base_filename = '01_electrical_symbols'
            else:
                table_type = 'drawing_index'
                base_filename = '04_drawing_index'
        elif rel_x < 0.65:  # Center
            if rel_y < 0.4:
                table_type = 'abbreviations'
                base_filename = '02_abbreviations'
            else:
                table_type = 'alternates'
                base_filename = '05_alternates'
        else:  # Right side
            table_type = 'general_notes'
            base_filename = '03_general_notes'

        # Add unique ID to prevent conflicts
        filename = f"{base_filename}_{table_id:02d}.png"

        return {
            'type': table_type,
            'filename': filename,
            'id': table_id,
            'position': f"({rel_x:.2f}, {rel_y:.2f})",
            'aspect_ratio': f"{aspect_ratio:.2f}"
        }

    def extract_improved_tables(self, pdf_path: str, output_dir: str = "improved_extracted_tables"):
        """
        Complete IMPROVED extraction pipeline
        """
        print("🚀 Starting IMPROVED table extraction...\n")

        # Step 1: Crop main border
        cropped_img = self.crop_main_border(pdf_path)

        # Step 2: IMPROVED boundary detection
        boundaries = self.detect_table_boundaries_improved(cropped_img)

        if not boundaries:
            print("❌ No tables detected.")
            return []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save main cropped image
        main_path = os.path.join(output_dir, "00_main_cropped.png")
        cropped_img.save(main_path, "PNG", quality=95)
        print(f"💾 Main image: {main_path}")

        # Step 3: Extract each table
        print(f"\n📋 Extracting {len(boundaries)} table regions...")

        extracted_files = []

        for x, y, w, h, table_info in boundaries:
            # Add small padding
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(cropped_img.width, x + w + padding)
            y_end = min(cropped_img.height, y + h + padding)

            # Extract table
            table_img = cropped_img.crop((x_start, y_start, x_end, y_end))

            # Enhance image quality
            if table_img.mode != 'RGB':
                table_img = table_img.convert('RGB')

            # Light enhancement
            enhancer = ImageEnhance.Contrast(table_img)
            enhanced = enhancer.enhance(1.05)
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)

            # Save
            output_path = os.path.join(output_dir, table_info['filename'])
            enhanced.save(output_path, "PNG", quality=95, optimize=True)
            extracted_files.append(output_path)

            print(f"✅ {table_info['type']} (ID: {table_info['id']}): {enhanced.width} × {enhanced.height} px")
            print(f"   📄 {table_info['filename']}")
            print(f"   📍 Position: {table_info['position']}, Aspect: {table_info['aspect_ratio']}")
            print(f"   🎯 Region: ({x}, {y}) Size: {w}×{h}")

        print(f"\n🎉 IMPROVED extraction complete! Created {len(extracted_files) + 1} files.")

        # Display results
        self.display_results(extracted_files, output_dir)

        return extracted_files

    def display_results(self, extracted_files: list, output_dir: str):
        """Display extracted tables"""
        if not extracted_files:
            print("❌ No files to display")
            return

        print("\n🖼️  EXTRACTED TABLES:")

        for filepath in extracted_files:
            img = Image.open(filepath)
            filename = os.path.basename(filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   📄 {filename}: {img.width}×{img.height}px ({size_kb:.1f}KB)")

        # Show preview
        n_show = min(4, len(extracted_files))
        if n_show > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten() if n_show > 1 else [axes]

            for i in range(n_show):
                img = Image.open(extracted_files[i])
                axes[i].imshow(img)
                axes[i].set_title(os.path.basename(extracted_files[i]), fontsize=10)
                axes[i].axis('off')

            # Hide unused subplots
            for i in range(n_show, 4):
                axes[i].axis('off')

            plt.tight_layout()
            plt.show()

    def show_debug_improved(self, img, final_boundaries, method1, method2, method3):
        """Show improved debug visualization"""
        print("🔍 IMPROVED DEBUG: Showing detection process...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Original
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f"Original Image ({img.width}×{img.height})")
        axes[0, 0].axis('off')

        # Method 1 results
        img_m1 = np.array(img.copy())
        for i, (x, y, w, h) in enumerate(method1):
            cv2.rectangle(img_m1, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_m1, f"C{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        axes[0, 1].imshow(img_m1)
        axes[0, 1].set_title(f"Contour Detection ({len(method1)} found)")
        axes[0, 1].axis('off')

        # Method 2 results
        img_m2 = np.array(img.copy())
        for i, (x, y, w, h) in enumerate(method2):
            cv2.rectangle(img_m2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_m2, f"M{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        axes[1, 0].imshow(img_m2)
        axes[1, 0].set_title(f"Morphological Detection ({len(method2)} found)")
        axes[1, 0].axis('off')

        # Final result
        img_final = np.array(img.copy())
        for i, (x, y, w, h, table_info) in enumerate(final_boundaries):
            cv2.rectangle(img_final, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(img_final, f"T{table_info['id']}", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(img_final, table_info['type'][:8], (x+5, y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        axes[1, 1].imshow(img_final)
        axes[1, 1].set_title(f"Final Result ({len(final_boundaries)} tables)")
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

# SIMPLE FUNCTION TO TEST
def extract_tables_improved(pdf_path: str, debug: bool = True):
    """
    IMPROVED extraction function
    """
    detector = ImprovedBoundaryDetector(debug_mode=debug)
    return detector.extract_improved_tables(pdf_path)

if __name__ == "__main__":
    # Example usage - adjust path to your actual PDF file
    input_pdf = "data/input/your_document.pdf"
    output_dir = "data/output/extracted_tables"
    
    print("✅ IMPROVED BOUNDARY DETECTOR READY!")
    print(f"🚀 Processing: {input_pdf}")
    
    # Run the extraction
    try:
        detector = ImprovedBoundaryDetector(debug_mode=True)
        extracted_files = detector.extract_improved_tables(input_pdf, output_dir)
        print(f"\n🎉 Successfully extracted {len(extracted_files)} tables!")
        print(f"📁 Output saved to: {output_dir}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure the PDF file exists and dependencies are installed.")