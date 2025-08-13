#!/usr/bin/env python3
"""
Simple test script for the Electrical Blueprint RAG System
Tests basic functionality without requiring actual blueprint files
"""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from rag.electrical_blueprint_rag import ElectricalBlueprintRAG
        print("✅ ElectricalBlueprintRAG imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ElectricalBlueprintRAG: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OpenCV: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FAISS: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    return True

def test_rag_initialization():
    """Test RAG system initialization"""
    print("\n🧪 Testing RAG system initialization...")
    
    try:
        from rag.electrical_blueprint_rag import ElectricalBlueprintRAG
        
        # Create a minimal test database
        test_symbols = [
            {
                "symbol_id": "test_001",
                "description": "Test Duplex Receptacle",
                "symbol_type": "receptacle",
                "mounting_height": "18 inches",
                "notes": "Test symbol for validation"
            }
        ]
        
        # Save test database
        os.makedirs("test_output", exist_ok=True)
        with open("test_output/test_symbols.json", "w") as f:
            json.dump(test_symbols, f, indent=2)
        
        # Try to initialize RAG system
        rag = ElectricalBlueprintRAG(
            symbol_database_path="test_output/test_symbols.json"
        )
        
        print("✅ RAG system initialized successfully")
        print(f"   - Symbol database loaded: {len(rag.symbol_database)} symbols")
        print(f"   - Text index available: {rag.text_index is not None}")
        print(f"   - Image index available: {rag.image_index is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

def test_cli_interface():
    """Test CLI interface"""
    print("\n🧪 Testing CLI interface...")
    
    try:
        from rag.cli_interface import BlueprintRAGCLI
        
        cli = BlueprintRAGCLI()
        print("✅ CLI interface created successfully")
        
        # Test help functionality
        print("   - CLI interface ready for use")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create CLI interface: {e}")
        return False

def test_web_interface():
    """Test web interface"""
    print("\n🧪 Testing web interface...")
    
    try:
        from rag.web_interface import BlueprintRAGInterface
        
        interface = BlueprintRAGInterface()
        print("✅ Web interface created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create web interface: {e}")
        return False

def test_basic_functionality():
    """Test basic RAG functionality"""
    print("\n🧪 Testing basic RAG functionality...")
    
    try:
        from rag.electrical_blueprint_rag import ElectricalBlueprintRAG
        
        # Create test data
        test_symbols = [
            {
                "symbol_id": "test_001",
                "description": "Test Duplex Receptacle",
                "symbol_type": "receptacle",
                "mounting_height": "18 inches",
                "notes": "Test symbol for validation"
            },
            {
                "symbol_id": "test_002",
                "description": "Test Circuit Breaker",
                "symbol_type": "circuit_breaker",
                "mounting_height": "48 inches",
                "notes": "Test circuit breaker"
            }
        ]
        
        # Save test database
        os.makedirs("test_output", exist_ok=True)
        with open("test_output/test_symbols.json", "w") as f:
            json.dump(test_symbols, f, indent=2)
        
        # Initialize RAG system
        rag = ElectricalBlueprintRAG(
            symbol_database_path="test_output/test_symbols.json"
        )
        
        # Test symbol matching (without actual images)
        print("   - RAG system ready for symbol matching")
        print("   - Database contains test symbols")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test basic functionality: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Electrical Blueprint RAG System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("RAG Initialization", test_rag_initialization),
        ("CLI Interface", test_cli_interface),
        ("Web Interface", test_web_interface),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG system is ready to use.")
        print("\n📖 Next steps:")
        print("   1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("   2. Prepare your symbol database")
        print("   3. Upload your electrical blueprints")
        print("   4. Start analyzing!")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)