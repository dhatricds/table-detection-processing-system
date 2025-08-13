# Table Detection and Processing System

A computer vision system for detecting, extracting, and processing tables from PDF documents and images using advanced AI techniques.

## 🚀 Features

- **PDF to Image Conversion**: Convert PDF documents to high-quality images for processing
- **Multi-Method Table Detection**: Advanced boundary detection using contours, morphological operations, and coordinate-based methods
- **Row & Column Extraction**: Intelligent segmentation of table rows and columns
- **AI-Powered Embeddings**: Generate text and image embeddings using OpenAI and CLIP models
- **FAISS Indexing**: Fast similarity search for extracted content
- **Symbol Recognition**: Extract and categorize electrical symbols and components

## 📁 Project Structure

```
├── src/
│   ├── table_detection/        # Table detection and row/column extraction
│   │   ├── detect_innertables.py
│   │   └── extract_rows_cols.py
│   ├── image_processing/       # Advanced image processing and boundary detection
│   │   └── table_detector.py
│   └── embeddings/            # Text and image embedding generation
│       └── text_image_embeddings.py
├── data/
│   ├── input/                 # Input files (PDFs, images)
│   └── output/               # Generated outputs
├── docs/                     # Documentation
├── config/                   # Configuration files
└── requirements.txt
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/dhatricds/table-detection-processing-system.git
cd table-detection-processing-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Dependencies

- **OpenCV** (cv2) - Computer vision and image processing
- **NumPy** - Numerical computing
- **PIL/Pillow** - Image manipulation
- **pdf2image** - PDF to image conversion
- **pytesseract** - OCR capabilities
- **OpenAI API** - Text embeddings
- **PyTorch** - Deep learning framework
- **Transformers** (CLIP) - Image embeddings
- **FAISS** - Similarity search indexing
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization

## 🎯 Usage

1. **Prepare Input Files**: Place your PDF documents or images in `data/input/`

2. **Run Table Detection**:
```bash
python src/image_processing/table_detector.py
```

3. **Extract Rows and Columns**:
```bash
python src/table_detection/extract_rows_cols.py
```

4. **Generate Embeddings**:
```bash
python src/embeddings/text_image_embeddings.py
```

5. **View Results**: Check the `data/output/` directory for processed files

## 🔧 Configuration

- Modify settings in the `config/` directory
- Adjust detection parameters in individual scripts
- Set up API keys for OpenAI integration

## 📊 Output

The system generates:
- Extracted table images
- Individual row and column segments
- Symbol images with labels
- Embedding vectors and FAISS indices
- Structured data in JSON and CSV formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- OpenAI for embedding models
- FAISS team for similarity search
- CLIP model developers for image understanding capabilities