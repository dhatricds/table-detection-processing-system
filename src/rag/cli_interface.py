"""
Command Line Interface for Electrical Blueprint RAG System
Provides easy-to-use CLI commands for blueprint analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

# Import our RAG system
from .electrical_blueprint_rag import ElectricalBlueprintRAG, analyze_blueprint

class BlueprintRAGCLI:
    """Command Line Interface for the Electrical Blueprint RAG system"""
    
    def __init__(self):
        self.rag_system = None
        
    def initialize_rag_system(self, symbol_database_path: str, openai_api_key: Optional[str] = None):
        """Initialize the RAG system"""
        try:
            print("🔄 Initializing RAG system...")
            self.rag_system = ElectricalBlueprintRAG(
                symbol_database_path=symbol_database_path,
                openai_api_key=openai_api_key
            )
            print("✅ RAG system initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False
    
    def analyze_blueprint(self, 
                         blueprint_path: str, 
                         query: Optional[str] = None,
                         confidence_threshold: float = 0.5,
                         output_format: str = "json",
                         output_file: Optional[str] = None) -> Dict:
        """
        Analyze a blueprint and return results
        
        Args:
            blueprint_path: Path to blueprint file
            query: Optional query string
            confidence_threshold: Minimum confidence for matches
            output_format: Output format (json, csv, table)
            output_file: Optional output file path
            
        Returns:
            Analysis results
        """
        print(f"🔍 Analyzing blueprint: {blueprint_path}")
        
        if not os.path.exists(blueprint_path):
            raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
        
        # Perform analysis
        results = self.rag_system.query_blueprint(
            blueprint_path=blueprint_path,
            query=query,
            confidence_threshold=confidence_threshold
        )
        
        # Handle output
        self._handle_output(results, output_format, output_file)
        
        return results
    
    def _handle_output(self, results: Dict, output_format: str, output_file: Optional[str]):
        """Handle different output formats"""
        
        if output_format == "json":
            output_data = json.dumps(results, indent=2)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(output_data)
                print(f"📄 Results saved to: {output_file}")
            else:
                print(output_data)
        
        elif output_format == "csv":
            # Extract inventory items for CSV
            inventory_items = results.get("inventory", {}).get("inventory_items", [])
            if inventory_items:
                df = pd.DataFrame(inventory_items)
                if output_file:
                    df.to_csv(output_file, index=False)
                    print(f"📄 CSV saved to: {output_file}")
                else:
                    print(df.to_csv(index=False))
            else:
                print("No inventory items to export")
        
        elif output_format == "table":
            # Pretty print as table
            self._print_results_table(results)
        
        else:
            print(f"❌ Unknown output format: {output_format}")
    
    def _print_results_table(self, results: Dict):
        """Print results in a formatted table"""
        print("\n" + "="*80)
        print("📊 BLUEPRINT ANALYSIS RESULTS")
        print("="*80)
        
        # Summary
        summary = results.get("inventory", {}).get("summary", {})
        print(f"📋 Total Symbols: {summary.get('total_symbols', 0)}")
        print(f"🔍 Unique Types: {summary.get('unique_symbol_types', 0)}")
        print(f"✅ Extracted Symbols: {results.get('extracted_symbols', 0)}")
        print(f"🎯 Total Matches: {results.get('total_matches', 0)}")
        print(f"📊 Filtered Matches: {results.get('filtered_matches', 0)}")
        
        # Inventory items
        inventory_items = results.get("inventory", {}).get("inventory_items", [])
        if inventory_items:
            print("\n📋 INVENTORY BREAKDOWN")
            print("-" * 80)
            print(f"{'Symbol Type':<20} {'Description':<30} {'Qty':<5} {'Confidence':<10}")
            print("-" * 80)
            
            for item in inventory_items:
                symbol_type = item.get("symbol_type", "Unknown")[:19]
                description = item.get("description", "")[:29]
                quantity = item.get("quantity", 0)
                confidence = f"{item.get('confidence', 0):.2f}"
                
                print(f"{symbol_type:<20} {description:<30} {quantity:<5} {confidence:<10}")
        
        # High confidence matches
        high_confidence = results.get("inventory", {}).get("high_confidence_matches", [])
        if high_confidence:
            print(f"\n🎯 HIGH CONFIDENCE MATCHES ({len(high_confidence)})")
            print("-" * 80)
            for match in high_confidence:
                print(f"• {match.get('description', 'Unknown')} (ID: {match.get('symbol_id', 'N/A')}) - Confidence: {match.get('confidence', 0):.3f}")
        
        print("\n" + "="*80)
    
    def batch_analyze(self, 
                     blueprint_dir: str, 
                     query: Optional[str] = None,
                     confidence_threshold: float = 0.5,
                     output_dir: Optional[str] = None) -> List[Dict]:
        """
        Analyze multiple blueprints in a directory
        
        Args:
            blueprint_dir: Directory containing blueprint files
            query: Optional query string
            confidence_threshold: Minimum confidence for matches
            output_dir: Output directory for results
            
        Returns:
            List of analysis results
        """
        print(f"📁 Batch analyzing blueprints in: {blueprint_dir}")
        
        if not os.path.exists(blueprint_dir):
            raise FileNotFoundError(f"Directory not found: {blueprint_dir}")
        
        # Find blueprint files
        blueprint_files = []
        for ext in ['*.pdf', '*.png', '*.jpg', '*.jpeg']:
            blueprint_files.extend(Path(blueprint_dir).glob(ext))
        
        if not blueprint_files:
            print("❌ No blueprint files found in directory")
            return []
        
        print(f"🔍 Found {len(blueprint_files)} blueprint files")
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, blueprint_file in enumerate(blueprint_files, 1):
            print(f"\n📄 Processing {i}/{len(blueprint_files)}: {blueprint_file.name}")
            
            try:
                # Analyze blueprint
                result = self.analyze_blueprint(
                    blueprint_path=str(blueprint_file),
                    query=query,
                    confidence_threshold=confidence_threshold,
                    output_format="json",
                    output_file=os.path.join(output_dir, f"{blueprint_file.stem}_analysis.json") if output_dir else None
                )
                
                results.append({
                    "file": blueprint_file.name,
                    "result": result
                })
                
                print(f"✅ Completed: {blueprint_file.name}")
                
            except Exception as e:
                print(f"❌ Error processing {blueprint_file.name}: {e}")
                results.append({
                    "file": blueprint_file.name,
                    "error": str(e)
                })
        
        # Generate batch summary
        if output_dir:
            self._generate_batch_summary(results, output_dir)
        
        return results
    
    def _generate_batch_summary(self, results: List[Dict], output_dir: str):
        """Generate a summary report for batch analysis"""
        summary_data = []
        
        for result in results:
            if "result" in result:
                analysis = result["result"]
                summary_data.append({
                    "file": result["file"],
                    "extracted_symbols": analysis.get("extracted_symbols", 0),
                    "total_matches": analysis.get("total_matches", 0),
                    "filtered_matches": analysis.get("filtered_matches", 0),
                    "unique_types": analysis.get("inventory", {}).get("summary", {}).get("unique_symbol_types", 0),
                    "total_symbols": analysis.get("inventory", {}).get("summary", {}).get("total_symbols", 0)
                })
            else:
                summary_data.append({
                    "file": result["file"],
                    "extracted_symbols": 0,
                    "total_matches": 0,
                    "filtered_matches": 0,
                    "unique_types": 0,
                    "total_symbols": 0,
                    "error": result.get("error", "Unknown error")
                })
        
        # Save summary
        summary_file = os.path.join(output_dir, "batch_summary.csv")
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        
        print(f"📊 Batch summary saved to: {summary_file}")
    
    def interactive_mode(self):
        """Run interactive mode for blueprint analysis"""
        print("🎯 Interactive Blueprint Analysis Mode")
        print("=" * 50)
        
        # Get blueprint path
        while True:
            blueprint_path = input("📁 Enter blueprint file path (or 'quit' to exit): ").strip()
            
            if blueprint_path.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not os.path.exists(blueprint_path):
                print(f"❌ File not found: {blueprint_path}")
                continue
            
            # Get query
            query = input("🔍 Enter query (optional): ").strip()
            if not query:
                query = None
            
            # Get confidence threshold
            try:
                confidence = float(input("🎯 Enter confidence threshold (0.0-1.0, default 0.5): ").strip() or "0.5")
            except ValueError:
                confidence = 0.5
            
            # Analyze
            try:
                print(f"\n🔍 Analyzing {blueprint_path}...")
                results = self.analyze_blueprint(
                    blueprint_path=blueprint_path,
                    query=query,
                    confidence_threshold=confidence,
                    output_format="table"
                )
                
                # Ask if user wants to save results
                save_results = input("\n💾 Save results to file? (y/n): ").strip().lower()
                if save_results in ['y', 'yes']:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"analysis_{timestamp}.json"
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"📄 Results saved to: {output_file}")
                
            except Exception as e:
                print(f"❌ Error during analysis: {e}")
            
            print("\n" + "-" * 50)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Electrical Blueprint RAG System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single blueprint
  python cli_interface.py analyze blueprint.pdf --query "Find circuit breakers"
  
  # Batch analyze multiple blueprints
  python cli_interface.py batch ./blueprints/ --output-dir ./results/
  
  # Interactive mode
  python cli_interface.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single blueprint')
    analyze_parser.add_argument('blueprint_path', help='Path to blueprint file')
    analyze_parser.add_argument('--query', '-q', help='Query string for filtering')
    analyze_parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    analyze_parser.add_argument('--format', '-f', choices=['json', 'csv', 'table'], default='table', help='Output format')
    analyze_parser.add_argument('--output', '-o', help='Output file path')
    analyze_parser.add_argument('--symbol-db', default='output/final_columns.json', help='Symbol database path')
    analyze_parser.add_argument('--openai-key', help='OpenAI API key')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch analyze multiple blueprints')
    batch_parser.add_argument('blueprint_dir', help='Directory containing blueprint files')
    batch_parser.add_argument('--query', '-q', help='Query string for filtering')
    batch_parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    batch_parser.add_argument('--output-dir', '-o', help='Output directory for results')
    batch_parser.add_argument('--symbol-db', default='output/final_columns.json', help='Symbol database path')
    batch_parser.add_argument('--openai-key', help='OpenAI API key')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run in interactive mode')
    interactive_parser.add_argument('--symbol-db', default='output/final_columns.json', help='Symbol database path')
    interactive_parser.add_argument('--openai-key', help='OpenAI API key')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = BlueprintRAGCLI()
    
    # Get OpenAI API key
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    
    # Initialize RAG system
    symbol_db_path = getattr(args, 'symbol_db', 'output/final_columns.json')
    if not cli.initialize_rag_system(symbol_db_path, openai_key):
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'analyze':
            cli.analyze_blueprint(
                blueprint_path=args.blueprint_path,
                query=args.query,
                confidence_threshold=args.confidence,
                output_format=args.format,
                output_file=args.output
            )
        
        elif args.command == 'batch':
            cli.batch_analyze(
                blueprint_dir=args.blueprint_dir,
                query=args.query,
                confidence_threshold=args.confidence,
                output_dir=args.output_dir
            )
        
        elif args.command == 'interactive':
            cli.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()