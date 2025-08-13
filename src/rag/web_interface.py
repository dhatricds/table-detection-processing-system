"""
Web Interface for Electrical Blueprint RAG System
Provides a user-friendly interface for uploading blueprints and querying symbols
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go

# Import our RAG system
from .electrical_blueprint_rag import ElectricalBlueprintRAG, analyze_blueprint

# Page configuration
st.set_page_config(
    page_title="Electrical Blueprint RAG System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

class BlueprintRAGInterface:
    """Web interface for the Electrical Blueprint RAG system"""
    
    def __init__(self):
        self.rag_system = None
        self.uploaded_file = None
        self.analysis_results = None
        
    def initialize_rag_system(self):
        """Initialize the RAG system"""
        try:
            with st.spinner("Loading RAG system..."):
                self.rag_system = ElectricalBlueprintRAG()
            st.success("✅ RAG system loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading RAG system: {e}")
            return False
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">⚡ Electrical Blueprint RAG System</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Upload electrical blueprints and get instant symbol inventory analysis using AI-powered multimodal search
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.title("⚙️ Configuration")
        
        # API Key input
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for enhanced text search capabilities"
        )
        
        # Confidence threshold
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence score for symbol matches"
        )
        
        # Search options
        st.sidebar.subheader("🔍 Search Options")
        enable_visual_search = st.sidebar.checkbox("Visual Search", value=True)
        enable_text_search = st.sidebar.checkbox("Text Search", value=True)
        
        # Display options
        st.sidebar.subheader("📊 Display Options")
        show_confidence_scores = st.sidebar.checkbox("Show Confidence Scores", value=True)
        show_locations = st.sidebar.checkbox("Show Locations", value=True)
        
        return {
            "openai_api_key": openai_api_key,
            "confidence_threshold": confidence_threshold,
            "enable_visual_search": enable_visual_search,
            "enable_text_search": enable_text_search,
            "show_confidence_scores": show_confidence_scores,
            "show_locations": show_locations
        }
    
    def render_file_upload(self):
        """Render the file upload section"""
        st.header("📁 Upload Blueprint")
        
        uploaded_file = st.file_uploader(
            "Choose a blueprint file",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload an electrical blueprint in PDF or image format"
        )
        
        if uploaded_file is not None:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                self.uploaded_file = tmp_file.name
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.info(f"📄 File size: {file_size:.1f} KB")
            
            return True
        
        return False
    
    def render_query_section(self):
        """Render the query input section"""
        st.header("🔍 Query Blueprint")
        
        # Query input
        query = st.text_input(
            "Enter your query (optional)",
            placeholder="e.g., 'Find all circuit breakers and receptacles'",
            help="Describe what symbols you want to find in the blueprint"
        )
        
        # Query suggestions
        st.markdown("**💡 Query Suggestions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔌 Receptacles"):
                st.session_state.query = "Find all receptacles and outlet types"
        
        with col2:
            if st.button("⚡ Circuit Breakers"):
                st.session_state.query = "Find all circuit breakers and electrical panels"
        
        with col3:
            if st.button("💡 Lighting"):
                st.session_state.query = "Find all lighting fixtures and switches"
        
        # Use session state for query
        if 'query' in st.session_state:
            query = st.session_state.query
        
        return query
    
    def render_analysis_button(self, config):
        """Render the analysis button"""
        st.header("🚀 Analyze Blueprint")
        
        if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
            if self.uploaded_file and os.path.exists(self.uploaded_file):
                with st.spinner("Analyzing blueprint..."):
                    try:
                        # Perform analysis
                        self.analysis_results = analyze_blueprint(
                            blueprint_path=self.uploaded_file,
                            query=config.get("query", ""),
                            openai_api_key=config.get("openai_api_key")
                        )
                        
                        st.success("✅ Analysis complete!")
                        return True
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        return False
            else:
                st.error("❌ Please upload a blueprint file first")
                return False
        
        return False
    
    def render_results_summary(self):
        """Render the analysis results summary"""
        if not self.analysis_results:
            return
        
        st.header("📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Extracted Symbols",
                self.analysis_results.get("extracted_symbols", 0)
            )
        
        with col2:
            st.metric(
                "Total Matches",
                self.analysis_results.get("total_matches", 0)
            )
        
        with col3:
            st.metric(
                "Filtered Matches",
                self.analysis_results.get("filtered_matches", 0)
            )
        
        with col4:
            st.metric(
                "Unique Types",
                self.analysis_results.get("inventory", {}).get("summary", {}).get("unique_symbol_types", 0)
            )
    
    def render_inventory_table(self, config):
        """Render the inventory table"""
        if not self.analysis_results:
            return
        
        inventory = self.analysis_results.get("inventory", {})
        inventory_items = inventory.get("inventory_items", [])
        
        if not inventory_items:
            st.info("ℹ️ No inventory items found matching the criteria")
            return
        
        st.header("📋 Symbol Inventory")
        
        # Create DataFrame for display
        df_data = []
        for item in inventory_items:
            df_data.append({
                "Symbol Type": item.get("symbol_type", "Unknown"),
                "Description": item.get("description", ""),
                "Quantity": item.get("quantity", 0),
                "Confidence": f"{item.get('confidence', 0):.2f}",
                "Locations": ", ".join(item.get("locations", [])) if config.get("show_locations") else "N/A"
            })
        
        df = pd.DataFrame(df_data)
        
        # Display table
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Inventory CSV",
            data=csv,
            file_name="blueprint_inventory.csv",
            mime="text/csv"
        )
    
    def render_visualizations(self):
        """Render charts and visualizations"""
        if not self.analysis_results:
            return
        
        inventory = self.analysis_results.get("inventory", {})
        inventory_items = inventory.get("inventory_items", [])
        
        if not inventory_items:
            return
        
        st.header("📈 Visualizations")
        
        # Prepare data for charts
        symbol_types = [item.get("symbol_type", "Unknown") for item in inventory_items]
        quantities = [item.get("quantity", 0) for item in inventory_items]
        confidences = [item.get("confidence", 0) for item in inventory_items]
        
        # Bar chart for quantities
        col1, col2 = st.columns(2)
        
        with col1:
            fig_quantity = px.bar(
                x=symbol_types,
                y=quantities,
                title="Symbol Quantities",
                labels={"x": "Symbol Type", "y": "Quantity"}
            )
            fig_quantity.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_quantity, use_container_width=True)
        
        with col2:
            fig_confidence = px.bar(
                x=symbol_types,
                y=confidences,
                title="Confidence Scores",
                labels={"x": "Symbol Type", "y": "Confidence"}
            )
            fig_confidence.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Pie chart for distribution
        fig_pie = px.pie(
            values=quantities,
            names=symbol_types,
            title="Symbol Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_high_confidence_matches(self):
        """Render high confidence matches"""
        if not self.analysis_results:
            return
        
        inventory = self.analysis_results.get("inventory", {})
        high_confidence = inventory.get("high_confidence_matches", [])
        
        if high_confidence:
            st.header("🎯 High Confidence Matches")
            
            for match in high_confidence:
                with st.expander(f"🔍 {match.get('description', 'Unknown')} (ID: {match.get('symbol_id', 'N/A')})"):
                    st.write(f"**Confidence:** {match.get('confidence', 0):.3f}")
                    st.write(f"**Description:** {match.get('description', 'N/A')}")
    
    def render_raw_results(self):
        """Render raw analysis results for debugging"""
        if not self.analysis_results:
            return
        
        with st.expander("🔧 Raw Analysis Results"):
            st.json(self.analysis_results)
    
    def run(self):
        """Main interface runner"""
        # Render header
        self.render_header()
        
        # Initialize RAG system
        if not self.initialize_rag_system():
            st.stop()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["📁 Upload & Query", "📊 Results", "🔧 Advanced"])
        
        with tab1:
            # File upload
            file_uploaded = self.render_file_upload()
            
            if file_uploaded:
                # Query section
                query = self.render_query_section()
                config["query"] = query
                
                # Analysis button
                analysis_complete = self.render_analysis_button(config)
                
                if analysis_complete:
                    st.session_state.analysis_complete = True
        
        with tab2:
            if st.session_state.get("analysis_complete", False):
                # Results summary
                self.render_results_summary()
                
                # Inventory table
                self.render_inventory_table(config)
                
                # Visualizations
                self.render_visualizations()
                
                # High confidence matches
                self.render_high_confidence_matches()
            else:
                st.info("ℹ️ Please complete the analysis in the Upload & Query tab first")
        
        with tab3:
            # Advanced options and debugging
            st.header("🔧 Advanced Options")
            
            # Raw results
            self.render_raw_results()
            
            # System information
            st.subheader("System Information")
            st.write(f"RAG System Loaded: {self.rag_system is not None}")
            if self.rag_system:
                st.write(f"Symbol Database Size: {len(self.rag_system.symbol_database) if self.rag_system.symbol_database else 0}")
                st.write(f"Text Index Available: {self.rag_system.text_index is not None}")
                st.write(f"Image Index Available: {self.rag_system.image_index is not None}")

def main():
    """Main function to run the web interface"""
    interface = BlueprintRAGInterface()
    interface.run()

if __name__ == "__main__":
    main()