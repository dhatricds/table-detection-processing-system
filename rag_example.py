#!/usr/bin/env python3
"""
Electrical Blueprint RAG System - Comprehensive Example
Demonstrates the complete multimodal RAG workflow for electrical blueprint analysis
"""

import os
import json
import sys
from pathlib import Path
import tempfile

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from rag.electrical_blueprint_rag import ElectricalBlueprintRAG, analyze_blueprint

def create_sample_symbol_database():
    """Create a sample symbol database for demonstration"""
    sample_symbols = [
        {
            "symbol_id": "001",
            "description": "Duplex Receptacle",
            "symbol_type": "receptacle",
            "mounting_height": "18 inches",
            "notes": "Standard 120V outlet",
            "symbol_image": "output/symbols/sym_duplex_receptacle.png"
        },
        {
            "symbol_id": "002", 
            "description": "GFCI Receptacle",
            "symbol_type": "receptacle",
            "mounting_height": "18 inches",
            "notes": "Ground fault circuit interrupter",
            "symbol_image": "output/symbols/sym_gfci_receptacle.png"
        },
        {
            "symbol_id": "003",
            "description": "Circuit Breaker",
            "symbol_type": "circuit_breaker",
            "mounting_height": "48 inches",
            "notes": "15A circuit breaker",
            "symbol_image": "output/symbols/sym_circuit_breaker.png"
        },
        {
            "symbol_id": "004",
            "description": "Light Switch",
            "symbol_type": "switch",
            "mounting_height": "48 inches",
            "notes": "Single pole switch",
            "symbol_image": "output/symbols/sym_light_switch.png"
        },
        {
            "symbol_id": "005",
            "description": "Ceiling Light Fixture",
            "symbol_type": "lighting",
            "mounting_height": "ceiling",
            "notes": "Standard ceiling light",
            "symbol_image": "output/symbols/sym_ceiling_light.png"
        }
    ]
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Save sample database
    with open("output/sample_symbols.json", "w") as f:
        json.dump(sample_symbols, f, indent=2)
    
    print("✅ Created sample symbol database")
    return "output/sample_symbols.json"

def demonstrate_rag_system():
    """Demonstrate the RAG system capabilities"""
    print("🚀 Electrical Blueprint RAG System - Demonstration")
    print("=" * 60)
    
    # Step 1: Create sample symbol database
    print("\n📋 Step 1: Creating sample symbol database...")
    symbol_db_path = create_sample_symbol_database()
    
    # Step 2: Initialize RAG system
    print("\n🔧 Step 2: Initializing RAG system...")
    try:
        rag = ElectricalBlueprintRAG(
            symbol_database_path=symbol_db_path,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("💡 Make sure you have the required dependencies installed")
        return
    
    # Step 3: Demonstrate different query scenarios
    print("\n🔍 Step 3: Demonstrating query scenarios...")
    
    # Scenario 1: Find all receptacles
    print("\n📋 Scenario 1: Finding all receptacles")
    try:
        # Simulate a blueprint with receptacles
        results = rag.query_blueprint(
            blueprint_path="data/input/your_blueprint.pdf",  # Replace with actual path
            query="Find all receptacles and outlet types",
            confidence_threshold=0.5
        )
        
        print("📊 Results Summary:")
        print(f"  - Extracted symbols: {results.get('extracted_symbols', 0)}")
        print(f"  - Total matches: {results.get('total_matches', 0)}")
        print(f"  - Filtered matches: {results.get('filtered_matches', 0)}")
        
        # Show inventory
        inventory = results.get("inventory", {})
        items = inventory.get("inventory_items", [])
        
        if items:
            print("\n📋 Inventory Items:")
            for item in items:
                print(f"  • {item.get('symbol_type', 'Unknown')}: {item.get('quantity', 0)} units")
                print(f"    Description: {item.get('description', 'N/A')}")
                print(f"    Confidence: {item.get('confidence', 0):.2f}")
        
    except Exception as e:
        print(f"⚠️  Demo scenario 1 failed: {e}")
        print("   (This is expected if no actual blueprint file is provided)")
    
    # Scenario 2: Circuit breaker analysis
    print("\n⚡ Scenario 2: Circuit breaker analysis")
    try:
        results = rag.query_blueprint(
            blueprint_path="data/input/your_blueprint.pdf",  # Replace with actual path
            query="Find all circuit breakers and electrical panels",
            confidence_threshold=0.6
        )
        
        print("📊 Circuit Breaker Analysis:")
        inventory = results.get("inventory", {})
        items = inventory.get("inventory_items", [])
        
        circuit_breakers = [item for item in items if "circuit" in item.get("symbol_type", "").lower()]
        
        if circuit_breakers:
            print(f"  Found {len(circuit_breakers)} circuit breaker types:")
            for cb in circuit_breakers:
                print(f"    • {cb.get('description', 'Unknown')}: {cb.get('quantity', 0)} units")
        else:
            print("  No circuit breakers found in this blueprint")
            
    except Exception as e:
        print(f"⚠️  Demo scenario 2 failed: {e}")
    
    # Scenario 3: Lighting analysis
    print("\n💡 Scenario 3: Lighting fixture analysis")
    try:
        results = rag.query_blueprint(
            blueprint_path="data/input/your_blueprint.pdf",  # Replace with actual path
            query="Find all lighting fixtures and switches",
            confidence_threshold=0.5
        )
        
        print("📊 Lighting Analysis:")
        inventory = results.get("inventory", {})
        items = inventory.get("inventory_items", [])
        
        lighting_items = [item for item in items if "light" in item.get("symbol_type", "").lower() or "switch" in item.get("symbol_type", "").lower()]
        
        if lighting_items:
            print(f"  Found {len(lighting_items)} lighting-related items:")
            for item in lighting_items:
                print(f"    • {item.get('description', 'Unknown')}: {item.get('quantity', 0)} units")
        else:
            print("  No lighting items found in this blueprint")
            
    except Exception as e:
        print(f"⚠️  Demo scenario 3 failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 RAG System Demonstration Complete!")
    print("\n💡 To use with real blueprints:")
    print("   1. Place your blueprint PDF in data/input/")
    print("   2. Update the blueprint_path in the code")
    print("   3. Run the analysis with your specific queries")
    print("\n🔧 Available interfaces:")
    print("   • Python API: Use ElectricalBlueprintRAG class directly")
    print("   • CLI: python src/rag/cli_interface.py analyze blueprint.pdf")
    print("   • Web UI: streamlit run src/rag/web_interface.py")

def demonstrate_multimodal_search():
    """Demonstrate multimodal search capabilities"""
    print("\n🔍 Multimodal Search Demonstration")
    print("-" * 40)
    
    # Create sample extracted symbols
    sample_symbols = [
        {
            "image_path": "temp_symbol1.png",
            "dimensions": (100, 100),
            "aspect_ratio": 1.0,
            "confidence": 0.8
        },
        {
            "image_path": "temp_symbol2.png", 
            "dimensions": (150, 100),
            "aspect_ratio": 1.5,
            "confidence": 0.7
        }
    ]
    
    print("📋 Sample extracted symbols:")
    for i, symbol in enumerate(sample_symbols, 1):
        print(f"  Symbol {i}: {symbol['dimensions']} px, aspect ratio: {symbol['aspect_ratio']}")
    
    print("\n🔍 Search capabilities:")
    print("  • Visual similarity search using CLIP embeddings")
    print("  • Text similarity search using OpenAI embeddings") 
    print("  • Combined scoring for better accuracy")
    print("  • Confidence-based filtering")
    print("  • Duplicate detection and aggregation")

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples")
    print("-" * 30)
    
    examples = [
        {
            "title": "Basic Analysis",
            "code": """
from rag.electrical_blueprint_rag import analyze_blueprint

# Analyze a blueprint
results = analyze_blueprint(
    blueprint_path="blueprint.pdf",
    query="Find all receptacles"
)
print(results)
            """
        },
        {
            "title": "Advanced RAG System",
            "code": """
from rag.electrical_blueprint_rag import ElectricalBlueprintRAG

# Initialize RAG system
rag = ElectricalBlueprintRAG(
    symbol_database_path="output/symbols.json",
    openai_api_key="your-api-key"
)

# Extract symbols from blueprint
symbols = rag.extract_symbols_from_blueprint("blueprint.pdf")

# Perform multimodal search
matches = rag.multimodal_search(symbols, "Find circuit breakers")

# Generate inventory report
inventory = rag.generate_inventory_report(matches)
            """
        },
        {
            "title": "CLI Usage",
            "code": """
# Analyze single blueprint
python src/rag/cli_interface.py analyze blueprint.pdf --query "Find receptacles"

# Batch analysis
python src/rag/cli_interface.py batch ./blueprints/ --output-dir ./results/

# Interactive mode
python src/rag/cli_interface.py interactive
            """
        },
        {
            "title": "Web Interface",
            "code": """
# Start web interface
streamlit run src/rag/web_interface.py

# Then open browser to http://localhost:8501
            """
        }
    ]
    
    for example in examples:
        print(f"\n📖 {example['title']}:")
        print(example['code'])

def main():
    """Main demonstration function"""
    print("⚡ Electrical Blueprint RAG System - Complete Demonstration")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected structure: project_root/rag_example.py")
        return
    
    # Run demonstrations
    demonstrate_rag_system()
    demonstrate_multimodal_search()
    show_usage_examples()
    
    print("\n" + "=" * 70)
    print("🎯 Next Steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("   3. Prepare your symbol database")
    print("   4. Upload your electrical blueprints")
    print("   5. Start analyzing!")
    print("\n📖 For more information, see the README.md file")

if __name__ == "__main__":
    main()