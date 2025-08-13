"""
Electrical Blueprint Multimodal RAG System
Integrates with existing table detection and symbol extraction for intelligent blueprint analysis
"""

import os
import json
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import openai
from dataclasses import dataclass
import logging

# Import existing modules
from ..image_processing.table_detector import ImprovedBoundaryDetector
from ..table_detection.detect_innertables import extract_tables_and_rows
from ..embeddings.text_image_embeddings import create_text_embeddings, create_image_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SymbolMatch:
    """Represents a matched symbol with metadata"""
    symbol_id: str
    confidence: float
    description: str
    symbol_type: str
    quantity: int
    location: Optional[str] = None
    specifications: Optional[Dict] = None

@dataclass
class InventoryItem:
    """Represents an inventory item with aggregated information"""
    symbol_type: str
    description: str
    total_quantity: int
    locations: List[str]
    specifications: Dict
    confidence_score: float

class ElectricalBlueprintRAG:
    """
    Multimodal RAG system for electrical blueprint analysis
    """
    
    def __init__(self, 
                 symbol_database_path: str = "output/final_columns.json",
                 text_index_path: str = "output/text_index.faiss",
                 image_index_path: str = "output/image_index.faiss",
                 text_map_path: str = "output/text_map.csv",
                 image_map_path: str = "output/image_map.csv",
                 openai_api_key: Optional[str] = None):
        """
        Initialize the RAG system
        
        Args:
            symbol_database_path: Path to symbol database JSON
            text_index_path: Path to FAISS text index
            image_index_path: Path to FAISS image index
            text_map_path: Path to text mapping CSV
            image_map_path: Path to image mapping CSV
            openai_api_key: OpenAI API key for text embeddings
        """
        self.symbol_database_path = symbol_database_path
        self.text_index_path = text_index_path
        self.image_index_path = image_index_path
        self.text_map_path = text_map_path
        self.image_map_path = image_map_path
        
        # Set OpenAI API key
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            logger.warning("OpenAI API key not provided. Text search will be limited.")
        
        # Initialize components
        self.symbol_database = None
        self.text_index = None
        self.image_index = None
        self.text_map = None
        self.image_map = None
        self.clip_processor = None
        self.clip_model = None
        self.table_detector = None
        
        # Load all components
        self._load_components()
    
    def _load_components(self):
        """Load all RAG components"""
        logger.info("Loading RAG components...")
        
        # Load symbol database
        try:
            with open(self.symbol_database_path, 'r') as f:
                self.symbol_database = json.load(f)
            logger.info(f"Loaded {len(self.symbol_database)} symbols from database")
        except FileNotFoundError:
            logger.error(f"Symbol database not found: {self.symbol_database_path}")
            self.symbol_database = []
        
        # Load FAISS indices
        try:
            if os.path.exists(self.text_index_path):
                self.text_index = faiss.read_index(self.text_index_path)
                logger.info("Loaded text FAISS index")
            
            if os.path.exists(self.image_index_path):
                self.image_index = faiss.read_index(self.image_index_path)
                logger.info("Loaded image FAISS index")
        except Exception as e:
            logger.error(f"Error loading FAISS indices: {e}")
        
        # Load mapping files
        try:
            if os.path.exists(self.text_map_path):
                self.text_map = pd.read_csv(self.text_map_path)
                logger.info("Loaded text mapping")
            
            if os.path.exists(self.image_map_path):
                self.image_map = pd.read_csv(self.image_map_path)
                logger.info("Loaded image mapping")
        except Exception as e:
            logger.error(f"Error loading mapping files: {e}")
        
        # Initialize CLIP model for image embeddings
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("Loaded CLIP model")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
        
        # Initialize table detector
        self.table_detector = ImprovedBoundaryDetector(debug_mode=False)
        
        logger.info("RAG system initialization complete")
    
    def extract_symbols_from_blueprint(self, blueprint_path: str) -> List[Dict]:
        """
        Extract symbols from uploaded blueprint using existing pipeline
        
        Args:
            blueprint_path: Path to blueprint PDF or image
            
        Returns:
            List of extracted symbol information
        """
        logger.info(f"Extracting symbols from blueprint: {blueprint_path}")
        
        extracted_symbols = []
        
        try:
            # Step 1: Extract tables from blueprint
            if blueprint_path.lower().endswith('.pdf'):
                tables = self.table_detector.extract_improved_tables(blueprint_path, "temp_extraction")
            else:
                # For image files, use direct processing
                tables = self._extract_from_image(blueprint_path)
            
            # Step 2: Extract symbols from each table
            for table_path in tables:
                symbols = self._extract_symbols_from_table(table_path)
                extracted_symbols.extend(symbols)
            
            logger.info(f"Extracted {len(extracted_symbols)} symbols from blueprint")
            return extracted_symbols
            
        except Exception as e:
            logger.error(f"Error extracting symbols from blueprint: {e}")
            return []
    
    def _extract_from_image(self, image_path: str) -> List[str]:
        """Extract tables from image file"""
        # For now, treat the whole image as a single table
        # This can be enhanced with more sophisticated table detection
        return [image_path]
    
    def _extract_symbols_from_table(self, table_path: str) -> List[Dict]:
        """Extract individual symbols from a table"""
        symbols = []
        
        try:
            # Use existing row extraction
            rows_dir = "temp_rows"
            extract_tables_and_rows(table_path, "temp_extraction")
            
            # Process each row as a potential symbol
            rows_path = Path("temp_extraction/rows")
            if rows_path.exists():
                for row_file in rows_path.glob("*.png"):
                    symbol_info = self._analyze_symbol_row(row_file)
                    if symbol_info:
                        symbols.append(symbol_info)
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error extracting symbols from table {table_path}: {e}")
            return []
    
    def _analyze_symbol_row(self, row_image_path: Path) -> Optional[Dict]:
        """Analyze a row image to extract symbol information"""
        try:
            # Basic symbol analysis - can be enhanced with OCR and more sophisticated detection
            img = cv2.imread(str(row_image_path))
            if img is None:
                return None
            
            # Extract basic features
            height, width = img.shape[:2]
            
            # Simple heuristic: if image is roughly square, it might be a symbol
            aspect_ratio = width / height
            if 0.5 <= aspect_ratio <= 2.0:
                return {
                    "image_path": str(row_image_path),
                    "dimensions": (width, height),
                    "aspect_ratio": aspect_ratio,
                    "confidence": 0.5  # Basic confidence
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing symbol row {row_image_path}: {e}")
            return None
    
    def multimodal_search(self, 
                         query_symbols: List[Dict], 
                         query_text: Optional[str] = None,
                         top_k: int = 10) -> List[SymbolMatch]:
        """
        Perform multimodal search to find matching symbols
        
        Args:
            query_symbols: List of symbols extracted from blueprint
            query_text: Optional text query for filtering
            top_k: Number of top matches to return
            
        Returns:
            List of symbol matches with confidence scores
        """
        logger.info(f"Performing multimodal search for {len(query_symbols)} symbols")
        
        matches = []
        
        for symbol in query_symbols:
            # Visual similarity search
            visual_matches = self._visual_similarity_search(symbol, top_k)
            
            # Text similarity search (if query provided)
            text_matches = []
            if query_text:
                text_matches = self._text_similarity_search(query_text, top_k)
            
            # Combine and rank matches
            combined_matches = self._combine_matches(visual_matches, text_matches)
            matches.extend(combined_matches)
        
        # Remove duplicates and sort by confidence
        unique_matches = self._deduplicate_matches(matches)
        unique_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(unique_matches)} unique matches")
        return unique_matches[:top_k]
    
    def _visual_similarity_search(self, symbol: Dict, top_k: int) -> List[SymbolMatch]:
        """Search for visually similar symbols"""
        if not self.image_index or not self.clip_model:
            return []
        
        try:
            # Generate embedding for query symbol
            img = Image.open(symbol["image_path"]).convert("RGB")
            inputs = self.clip_processor(images=img, return_tensors="pt")
            
            with torch.no_grad():
                query_embedding = self.clip_model.get_image_features(**inputs)[0].cpu().numpy().astype(np.float32)
            
            # Search in FAISS index
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.image_index.search(query_embedding, top_k)
            
            matches = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.image_map):
                    record_idx = self.image_map.iloc[idx]["record_idx"]
                    if record_idx < len(self.symbol_database):
                        symbol_data = self.symbol_database[record_idx]
                        confidence = 1.0 / (1.0 + distance)  # Convert distance to confidence
                        
                        match = SymbolMatch(
                            symbol_id=str(record_idx),
                            confidence=confidence,
                            description=symbol_data.get("description", ""),
                            symbol_type=symbol_data.get("symbol_type", "unknown"),
                            quantity=1
                        )
                        matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in visual similarity search: {e}")
            return []
    
    def _text_similarity_search(self, query_text: str, top_k: int) -> List[SymbolMatch]:
        """Search for textually similar symbols"""
        if not self.text_index or not openai.api_key:
            return []
        
        try:
            # Generate text embedding
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=[query_text]
            )
            query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
            
            # Search in FAISS index
            distances, indices = self.text_index.search(query_embedding, top_k)
            
            matches = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.text_map):
                    record_idx = self.text_map.iloc[idx]["record_idx"]
                    if record_idx < len(self.symbol_database):
                        symbol_data = self.symbol_database[record_idx]
                        confidence = 1.0 / (1.0 + distance)
                        
                        match = SymbolMatch(
                            symbol_id=str(record_idx),
                            confidence=confidence,
                            description=symbol_data.get("description", ""),
                            symbol_type=symbol_data.get("symbol_type", "unknown"),
                            quantity=1
                        )
                        matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in text similarity search: {e}")
            return []
    
    def _combine_matches(self, visual_matches: List[SymbolMatch], 
                        text_matches: List[SymbolMatch]) -> List[SymbolMatch]:
        """Combine visual and text matches with weighted scoring"""
        combined = {}
        
        # Add visual matches
        for match in visual_matches:
            combined[match.symbol_id] = match
        
        # Add or update with text matches
        for match in text_matches:
            if match.symbol_id in combined:
                # Average the confidence scores
                combined[match.symbol_id].confidence = (
                    combined[match.symbol_id].confidence + match.confidence
                ) / 2
            else:
                combined[match.symbol_id] = match
        
        return list(combined.values())
    
    def _deduplicate_matches(self, matches: List[SymbolMatch]) -> List[SymbolMatch]:
        """Remove duplicate matches and aggregate quantities"""
        unique_matches = {}
        
        for match in matches:
            if match.symbol_id in unique_matches:
                # Aggregate quantities and average confidence
                existing = unique_matches[match.symbol_id]
                existing.quantity += match.quantity
                existing.confidence = (existing.confidence + match.confidence) / 2
            else:
                unique_matches[match.symbol_id] = match
        
        return list(unique_matches.values())
    
    def generate_inventory_report(self, matches: List[SymbolMatch]) -> Dict:
        """
        Generate comprehensive inventory report from matches
        
        Args:
            matches: List of symbol matches
            
        Returns:
            Dictionary containing inventory report
        """
        logger.info("Generating inventory report")
        
        # Group by symbol type
        inventory_items = {}
        
        for match in matches:
            if match.symbol_type not in inventory_items:
                inventory_items[match.symbol_type] = InventoryItem(
                    symbol_type=match.symbol_type,
                    description=match.description,
                    total_quantity=match.quantity,
                    locations=[match.location] if match.location else [],
                    specifications=match.specifications or {},
                    confidence_score=match.confidence
                )
            else:
                item = inventory_items[match.symbol_type]
                item.total_quantity += match.quantity
                if match.location:
                    item.locations.append(match.location)
                item.confidence_score = max(item.confidence_score, match.confidence)
        
        # Generate summary
        total_symbols = sum(item.total_quantity for item in inventory_items.values())
        unique_types = len(inventory_items)
        
        report = {
            "summary": {
                "total_symbols": total_symbols,
                "unique_symbol_types": unique_types,
                "confidence_threshold": 0.7
            },
            "inventory_items": [
                {
                    "symbol_type": item.symbol_type,
                    "description": item.description,
                    "quantity": item.total_quantity,
                    "locations": list(set(item.locations)),  # Remove duplicates
                    "specifications": item.specifications,
                    "confidence": item.confidence_score
                }
                for item in inventory_items.values()
            ],
            "high_confidence_matches": [
                {
                    "symbol_id": match.symbol_id,
                    "description": match.description,
                    "confidence": match.confidence
                }
                for match in matches if match.confidence > 0.8
            ]
        }
        
        return report
    
    def query_blueprint(self, 
                       blueprint_path: str, 
                       query: Optional[str] = None,
                       confidence_threshold: float = 0.5) -> Dict:
        """
        Main interface for blueprint querying
        
        Args:
            blueprint_path: Path to blueprint file
            query: Optional text query for filtering
            confidence_threshold: Minimum confidence for matches
            
        Returns:
            Complete analysis report
        """
        logger.info(f"Processing blueprint query: {blueprint_path}")
        
        # Extract symbols from blueprint
        extracted_symbols = self.extract_symbols_from_blueprint(blueprint_path)
        
        if not extracted_symbols:
            return {
                "error": "No symbols found in blueprint",
                "extracted_symbols": 0
            }
        
        # Perform multimodal search
        matches = self.multimodal_search(extracted_symbols, query)
        
        # Filter by confidence threshold
        filtered_matches = [m for m in matches if m.confidence >= confidence_threshold]
        
        # Generate inventory report
        inventory_report = self.generate_inventory_report(filtered_matches)
        
        # Prepare final report
        report = {
            "blueprint_path": blueprint_path,
            "query": query,
            "extracted_symbols": len(extracted_symbols),
            "total_matches": len(matches),
            "filtered_matches": len(filtered_matches),
            "confidence_threshold": confidence_threshold,
            "inventory": inventory_report,
            "processing_time": "calculated_here"  # TODO: Add timing
        }
        
        logger.info(f"Blueprint analysis complete: {len(filtered_matches)} matches found")
        return report

# Convenience function for easy usage
def analyze_blueprint(blueprint_path: str, 
                     query: Optional[str] = None,
                     symbol_database_path: str = "output/final_columns.json",
                     openai_api_key: Optional[str] = None) -> Dict:
    """
    Convenience function to analyze a blueprint
    
    Args:
        blueprint_path: Path to blueprint file
        query: Optional query string
        symbol_database_path: Path to symbol database
        openai_api_key: OpenAI API key
        
    Returns:
        Analysis report
    """
    rag = ElectricalBlueprintRAG(
        symbol_database_path=symbol_database_path,
        openai_api_key=openai_api_key
    )
    
    return rag.query_blueprint(blueprint_path, query)

if __name__ == "__main__":
    # Example usage
    blueprint_path = "data/input/example_blueprint.pdf"
    query = "Find all circuit breakers and receptacles"
    
    if os.path.exists(blueprint_path):
        report = analyze_blueprint(blueprint_path, query)
        print(json.dumps(report, indent=2))
    else:
        print(f"Blueprint file not found: {blueprint_path}")
        print("Please provide a valid blueprint file path")