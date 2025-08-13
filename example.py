#!/usr/bin/env python3
"""
Example script demonstrating how to use the Table Detection and Processing System
"""

import os
from pathlib import Path
from src.image_processing.table_detector import extract_tables_improved
from src.table_detection.detect_innertables import extract_tables_and_rows
from src.table_detection.extract_rows_cols import extract_rows_by_grid

def main():
    """
    Example workflow for table detection and processing
    """
    print("🚀 Table Detection and Processing System - Example")
    print("=" * 60)
    
    # Configuration
    input_pdf = "data/input/your_document.pdf"  # Replace with your PDF path
    output_dir = "data/output"
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract tables from PDF
    print("\n📋 Step 1: Extracting tables from PDF...")
    try:
        if os.path.exists(input_pdf):
            extracted_tables = extract_tables_improved(input_pdf, debug=False)
            print(f"✅ Extracted {len(extracted_tables)} tables")
        else:
            print(f"⚠️  PDF file not found: {input_pdf}")
            print("   Please place your PDF file in data/input/ directory")
            return
    except Exception as e:
        print(f"❌ Error extracting tables: {e}")
        return
    
    # Step 2: Process individual table images
    print("\n🔍 Step 2: Processing table images...")
    try:
        # Example: Process the first extracted table
        if extracted_tables:
            first_table = extracted_tables[0]
            print(f"   Processing: {first_table}")
            
            # Extract rows and columns from the table
            extract_tables_and_rows(first_table, f"{output_dir}/processed")
            print("✅ Table processing complete")
        else:
            print("⚠️  No tables to process")
    except Exception as e:
        print(f"❌ Error processing tables: {e}")
    
    # Step 3: Extract rows using grid detection
    print("\n📊 Step 3: Extracting rows using grid detection...")
    try:
        tables_dir = f"{output_dir}/processed/tables"
        rows_dir = f"{output_dir}/processed/rows"
        
        if os.path.exists(tables_dir):
            extract_rows_by_grid(tables_dir, rows_dir)
            print("✅ Row extraction complete")
        else:
            print(f"⚠️  Tables directory not found: {tables_dir}")
    except Exception as e:
        print(f"❌ Error extracting rows: {e}")
    
    print("\n🎉 Example workflow complete!")
    print(f"📁 Check the output directory: {output_dir}")

if __name__ == "__main__":
    main()