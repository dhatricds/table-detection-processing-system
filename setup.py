from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="table-detection-processing-system",
    version="1.0.0",
    author="dhatricds",
    author_email="dhatri@cdsvision.com",
    description="A computer vision system for detecting, extracting, and processing tables from PDF documents and images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhatricds/table-detection-processing-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "table-detector=src.image_processing.table_detector:main",
            "extract-tables=src.table_detection.detect_innertables:main",
            "extract-rows=src.table_detection.extract_rows_cols:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json", "*.yaml", "*.yml"],
    },
    keywords="computer-vision, table-detection, pdf-processing, image-processing, opencv, ai, embeddings",
    project_urls={
        "Bug Reports": "https://github.com/dhatricds/table-detection-processing-system/issues",
        "Source": "https://github.com/dhatricds/table-detection-processing-system",
        "Documentation": "https://github.com/dhatricds/table-detection-processing-system#readme",
    },
)