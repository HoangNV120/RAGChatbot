"""
t-SNE Visualization Script for Vector Store Data
This script extracts embeddings from Qdrant vector store and creates t-SNE visualizations
suitable for academic papers and research documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
import asyncio
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime
import os
from app.config import settings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class VectorStoreTSNEVisualizer:
    def __init__(self):
        """Initialize the t-SNE visualizer with Qdrant client"""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = settings.collection_name

        # Set style for academic papers
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    async def extract_embeddings_and_metadata(self, limit: int = 1000) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract embeddings and metadata from Qdrant vector store with pagination

        Args:
            limit: Maximum number of vectors to extract

        Returns:
            Tuple of (embeddings_array, metadata_list)
        """
        print(f"🔍 Extracting embeddings from collection: {self.collection_name}")

        embeddings = []
        metadata = []
        next_page_offset = None
        batch_size = 1000  # Process in smaller batches to avoid timeout
        total_extracted = 0

        while total_extracted < limit:
            current_batch_size = min(batch_size, limit - total_extracted)

            try:
                print(f"📦 Fetching batch {total_extracted // batch_size + 1}, size: {current_batch_size}")

                # Scroll through vectors in batches
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.client.scroll(
                        collection_name=self.collection_name,
                        limit=current_batch_size,
                        offset=next_page_offset,
                        with_payload=True,
                        with_vectors=True
                    )
                )

                points, next_page_offset = result

                if not points:
                    print("🔚 No more points available")
                    break

                print(f"✅ Retrieved {len(points)} points in this batch")

                for point in points:
                    # Extract dense vector (assuming you're using dense vectors)
                    if isinstance(point.vector, dict) and 'dense' in point.vector:
                        embedding = point.vector['dense']
                    else:
                        embedding = point.vector

                    embeddings.append(embedding)

                    # Extract metadata - handle nested metadata structure
                    payload = point.payload
                    metadata_nested = payload.get('metadata', {})

                    meta = {
                        'id': point.id,
                        'content': payload.get('page_content', '')[:100] + '...',  # Truncate for display
                        'source': metadata_nested.get('source', 'Unknown'),
                        'name': metadata_nested.get('name', 'Unknown'),
                        'page': metadata_nested.get('page', 0),
                        'category': metadata_nested.get('category', 'General'),
                        # Add new fields specific to your course data
                        'type': metadata_nested.get('type', ''),
                        'chunk_type': metadata_nested.get('chunk_type', ''),
                        'course_code': metadata_nested.get('course_code', ''),
                        'title': metadata_nested.get('title', ''),
                        'session_start': metadata_nested.get('session_start', 0),
                        'session_end': metadata_nested.get('session_end', 0),
                        'session_count': metadata_nested.get('session_count', 0)
                    }
                    metadata.append(meta)

                total_extracted += len(points)

                # If we got fewer points than requested, we've reached the end
                if len(points) < current_batch_size or next_page_offset is None:
                    print("🔚 Reached end of collection")
                    break

                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"⚠️ Error in batch {total_extracted // batch_size + 1}: {e}")
                if total_extracted > 0:
                    print(f"🔄 Continuing with {total_extracted} points already extracted...")
                    break
                else:
                    raise

        embeddings_array = np.array(embeddings)
        print(f"✅ Total extracted: {len(embeddings)} vectors")
        print(f"✅ Embeddings shape: {embeddings_array.shape}")

        return embeddings_array, metadata

    def apply_dimensionality_reduction(self, embeddings: np.ndarray, method: str = 'tsne',
                                     n_components: int = 2, **kwargs) -> np.ndarray:
        """
        Apply dimensionality reduction technique

        Args:
            embeddings: High-dimensional embeddings
            method: 'tsne', 'pca', or 'both'
            n_components: Number of output dimensions
            **kwargs: Additional parameters for the reduction method

        Returns:
            Reduced embeddings
        """
        print(f"📉 Applying {method.upper()} dimensionality reduction...")

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            print(f"✅ PCA completed. Explained variance ratio: {reducer.explained_variance_ratio_}")

        elif method == 'tsne':
            # Apply PCA first for large datasets (common practice)
            if embeddings.shape[1] > 50:
                print("🔄 Applying PCA preprocessing for t-SNE...")
                pca = PCA(n_components=50, random_state=42)
                embeddings = pca.fit_transform(embeddings)

            # t-SNE parameters optimized for visualization
            tsne_params = {
                'n_components': n_components,
                'perplexity': min(30, len(embeddings) - 1),
                'learning_rate': 200,
                'max_iter': 1000,  # Changed from n_iter to max_iter
                'random_state': 42,
                'metric': 'cosine'
            }
            tsne_params.update(kwargs)

            reducer = TSNE(**tsne_params)
            reduced = reducer.fit_transform(embeddings)
            print(f"✅ t-SNE completed with perplexity={tsne_params['perplexity']}")

        else:
            raise ValueError(f"Unknown method: {method}")

        return reduced

    def create_matplotlib_visualization(self, reduced_embeddings: np.ndarray, metadata: List[Dict],
                                      color_by: str = 'category', save_path: Optional[str] = None):
        """
        Create publication-ready matplotlib visualization

        Args:
            reduced_embeddings: 2D reduced embeddings
            metadata: List of metadata dictionaries
            color_by: Attribute to color points by
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Vector Store Embeddings Visualization (t-SNE)', fontsize=16, fontweight='bold')

        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'category': [m['category'] for m in metadata],
            'source': [m['source'] for m in metadata],
            'name': [m['name'] for m in metadata],
            'page': [m['page'] for m in metadata]
        })

        # Plot 1: Color by category
        unique_categories = df['category'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))

        for i, category in enumerate(unique_categories):
            mask = df['category'] == category
            axes[0, 0].scatter(df[mask]['x'], df[mask]['y'],
                             c=[colors[i]], label=category, alpha=0.7, s=50)

        axes[0, 0].set_title('Colored by Category', fontweight='bold')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Color by source
        unique_sources = df['source'].unique()
        colors_source = plt.cm.tab10(np.linspace(0, 1, len(unique_sources)))

        for i, source in enumerate(unique_sources):
            mask = df['source'] == source
            axes[0, 1].scatter(df[mask]['x'], df[mask]['y'],
                             c=[colors_source[i]], label=source[:20] + '...', alpha=0.7, s=50)

        axes[0, 1].set_title('Colored by Source', fontweight='bold')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Density plot
        axes[1, 0].hexbin(df['x'], df['y'], gridsize=20, cmap='YlOrRd', alpha=0.8)
        axes[1, 0].set_title('Density Distribution', fontweight='bold')

        # Plot 4: Color by page number (if available)
        scatter = axes[1, 1].scatter(df['x'], df['y'], c=df['page'],
                                   cmap='viridis', alpha=0.7, s=50)
        axes[1, 1].set_title('Colored by Page Number', fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 1])
        axes[1, 1].grid(True, alpha=0.3)

        # Add axis labels to all subplots
        for ax in axes.flat:
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Matplotlib visualization saved to: {save_path}")

        plt.show()

    def create_interactive_plotly_visualization(self, reduced_embeddings: np.ndarray,
                                              metadata: List[Dict], save_path: Optional[str] = None):
        """
        Create interactive Plotly visualization

        Args:
            reduced_embeddings: 2D reduced embeddings
            metadata: List of metadata dictionaries
            save_path: Path to save the HTML file
        """
        # Create DataFrame
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'category': [m['category'] for m in metadata],
            'source': [m['source'] for m in metadata],
            'name': [m['name'] for m in metadata],
            'page': [m['page'] for m in metadata],
            'content': [m['content'] for m in metadata]
        })

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Colored by Category', 'Colored by Source',
                          'Colored by Name', 'Colored by Page'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Plot 1: Category
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            fig.add_trace(
                go.Scatter(
                    x=category_df['x'], y=category_df['y'],
                    mode='markers',
                    name=category,
                    text=category_df['content'],
                    hovertemplate='<b>%{text}</b><br>Category: ' + category + '<extra></extra>',
                    marker=dict(size=8, opacity=0.7)
                ),
                row=1, col=1
            )

        # Plot 2: Source
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            fig.add_trace(
                go.Scatter(
                    x=source_df['x'], y=source_df['y'],
                    mode='markers',
                    name=source[:20] + '...',
                    text=source_df['content'],
                    hovertemplate='<b>%{text}</b><br>Source: ' + source + '<extra></extra>',
                    marker=dict(size=8, opacity=0.7),
                    showlegend=False
                ),
                row=1, col=2
            )

        # Plot 3: Name
        for name in df['name'].unique():
            name_df = df[df['name'] == name]
            fig.add_trace(
                go.Scatter(
                    x=name_df['x'], y=name_df['y'],
                    mode='markers',
                    name=name[:20] + '...',
                    text=name_df['content'],
                    hovertemplate='<b>%{text}</b><br>Name: ' + name + '<extra></extra>',
                    marker=dict(size=8, opacity=0.7),
                    showlegend=False
                ),
                row=2, col=1
            )

        # Plot 4: Page (continuous color scale)
        fig.add_trace(
            go.Scatter(
                x=df['x'], y=df['y'],
                mode='markers',
                text=df['content'],
                hovertemplate='<b>%{text}</b><br>Page: %{marker.color}<extra></extra>',
                marker=dict(
                    size=8,
                    color=df['page'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Page Number"),
                    opacity=0.7
                ),
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title='Interactive Vector Store Embeddings Visualization (t-SNE)',
            height=800,
            showlegend=True
        )

        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="t-SNE Component 1", row=i, col=j)
                fig.update_yaxes(title_text="t-SNE Component 2", row=i, col=j)

        if save_path:
            fig.write_html(save_path)
            print(f"💾 Interactive visualization saved to: {save_path}")

        fig.show()

    def generate_statistics_report(self, embeddings: np.ndarray, metadata: List[Dict]) -> str:
        """
        Generate a statistical report of the vector store data

        Args:
            embeddings: Original high-dimensional embeddings
            metadata: List of metadata dictionaries

        Returns:
            Formatted statistics report
        """
        df = pd.DataFrame(metadata)

        report = f"""
Vector Store Statistics Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

Dataset Overview:
- Total vectors: {len(embeddings)}
- Embedding dimensions: {embeddings.shape[1]}
- Collection name: {self.collection_name}

Content Distribution:
{df['category'].value_counts().to_string()}

Source Distribution:
{df['source'].value_counts().to_string()}

Document Name Distribution:
{df['name'].value_counts().to_string()}

Embedding Statistics:
- Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}
- Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}
- Min norm: {np.min(np.linalg.norm(embeddings, axis=1)):.4f}
- Max norm: {np.max(np.linalg.norm(embeddings, axis=1)):.4f}

Cosine Similarity Statistics:
- Mean pairwise similarity: {np.mean(np.dot(embeddings, embeddings.T)):.4f}
        """

        return report

    def create_single_plot_by_type(self, reduced_embeddings: np.ndarray, metadata: List[Dict],
                                   save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 8),
                                   dpi: int = 300):
        """
        Create a single, publication-ready plot showing all chunks clustered by metadata type
        Perfect for academic papers.

        Args:
            reduced_embeddings: 2D reduced embeddings
            metadata: List of metadata dictionaries
            save_path: Path to save the figure
            figsize: Figure size (width, height)
            dpi: DPI for saving (300 for publications)
        """
        # Create DataFrame for easier manipulation
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'type': [self._determine_document_type(m) for m in metadata],
            'source': [m['source'] for m in metadata],
            'name': [m['name'] for m in metadata],
            'page': [m['page'] for m in metadata],
            'content': [m['content'] for m in metadata]
        })

        # Create single figure with academic styling
        plt.figure(figsize=figsize)

        # Set academic-style font
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 12,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

        # Get unique types and assign distinct academic colors
        unique_types = sorted(df['type'].unique())

        # Use distinct, colorblind-friendly academic colors
        academic_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf'   # Cyan
        ]

        colors = academic_colors[:len(unique_types)]

        # Plot each document type with distinct colors and smaller, tighter spacing
        for i, doc_type in enumerate(unique_types):
            mask = df['type'] == doc_type
            type_data = df[mask]

            plt.scatter(type_data['x'], type_data['y'],
                       c=colors[i],
                       label=f'{doc_type} ({len(type_data)})',
                       alpha=0.8,
                       s=45,  # Smaller points for tighter spacing
                       marker='o',
                       edgecolors='white',
                       linewidth=0.3)  # Thinner edge lines

        # Academic publication styling
        plt.title('Document Embedding Visualization using t-SNE',
                 fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('t-SNE Dimension 1', fontsize=13, fontweight='normal')
        plt.ylabel('t-SNE Dimension 2', fontsize=13, fontweight='normal')

        # Tighter axis limits for closer clustering appearance
        x_margin = (df['x'].max() - df['x'].min()) * 0.05
        y_margin = (df['y'].max() - df['y'].min()) * 0.05
        plt.xlim(df['x'].min() - x_margin, df['x'].max() + x_margin)
        plt.ylim(df['y'].min() - y_margin, df['y'].max() + y_margin)

        # Clean academic legend
        legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                           frameon=True, fancybox=False, shadow=False,
                           fontsize=11, borderpad=0.5)
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_edgecolor('black')

        # Subtle grid for better readability
        plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

        # Remove top and right spines for cleaner academic look
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        # Tick styling
        ax.tick_params(axis='both', which='major', labelsize=11,
                      width=0.8, length=4, direction='out')

        # Compact statistics box
        stats_text = f"N = {len(df)} | Types = {len(unique_types)}"
        plt.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor='gray', alpha=0.9, linewidth=0.8),
                fontsize=10, verticalalignment='bottom')

        # Tight layout for academic journals
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       format='png', pil_kwargs={'optimize': True})
            print(f"💾 Academic visualization saved to: {save_path}")

        plt.show()

    def _determine_document_type(self, metadata: Dict) -> str:
        """
        Determine document type based on metadata type field only

        Args:
            metadata: Metadata dictionary

        Returns:
            Document type string
        """
        # Simply use the 'type' field from metadata
        doc_type = metadata.get('type', '')
        if doc_type:
            return doc_type.title()  # Capitalize first letter

        # Fallback if no type field
        return 'Unknown Type'

async def main():
    """Main function to run the t-SNE visualization"""
    parser = argparse.ArgumentParser(description='Generate t-SNE visualization for vector store')
    parser.add_argument('--limit', type=int, default=2000, help='Max number of vectors to visualize')
    parser.add_argument('--method', choices=['tsne', 'pca'], default='tsne', help='Dimensionality reduction method')
    parser.add_argument('--output-dir', type=str, default='./visualizations', help='Output directory for visualizations')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
    parser.add_argument('--single-plot', action='store_true', help='Generate only single plot by type (for papers)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize visualizer
    visualizer = VectorStoreTSNEVisualizer()

    try:
        # Extract embeddings and metadata
        print("🚀 Starting vector store visualization...")
        embeddings, metadata = await visualizer.extract_embeddings_and_metadata(limit=args.limit)

        if len(embeddings) == 0:
            print("❌ No embeddings found in the vector store!")
            return

        # Apply dimensionality reduction
        reduced_embeddings = visualizer.apply_dimensionality_reduction(
            embeddings,
            method=args.method,
            perplexity=args.perplexity if args.method == 'tsne' else None
        )

        # Generate statistics report
        stats_report = visualizer.generate_statistics_report(embeddings, metadata)
        stats_path = os.path.join(args.output_dir, f'vector_store_stats_{timestamp}.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(stats_report)
        print(f"📊 Statistics report saved to: {stats_path}")

        if args.single_plot:
            # Create only the single plot for papers
            single_plot_path = os.path.join(args.output_dir, f'paper_visualization_{timestamp}.png')
            visualizer.create_single_plot_by_type(
                reduced_embeddings,
                metadata,
                save_path=single_plot_path
            )
            print(f"📄 Paper-ready visualization saved to: {single_plot_path}")
        else:
            # Create all visualizations
            matplotlib_path = os.path.join(args.output_dir, f'tsne_visualization_{args.method}_{timestamp}.png')
            visualizer.create_matplotlib_visualization(
                reduced_embeddings,
                metadata,
                save_path=matplotlib_path
            )

            interactive_path = os.path.join(args.output_dir, f'interactive_tsne_{timestamp}.html')
            visualizer.create_interactive_plotly_visualization(
                reduced_embeddings,
                metadata,
                save_path=interactive_path
            )

            single_plot_path = os.path.join(args.output_dir, f'paper_visualization_{timestamp}.png')
            visualizer.create_single_plot_by_type(
                reduced_embeddings,
                metadata,
                save_path=single_plot_path
            )

        print(f"✅ Visualization complete! Check the {args.output_dir} directory for outputs.")

    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
