"""
Flight Difficulty Score Analysis - Part 4: Advanced Data Visualization

This module creates comprehensive visualizations for United Airlines flight
difficulty analysis, providing insights into operational patterns and model
performance through professional-grade charts and statistical plots.

Author: Data Science Team
Version: 1.0.0
Date: October 2025
Project: United Airlines SkyHack 3.0

Key Features:
- United Airlines operational difficulty trends
- Feature importance visualization
- Daily ranking distribution analysis
- Statistical performance metrics
- Production-ready visualization outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_ua_visualizations():
    """
    Generate comprehensive visualizations for United Airlines analysis.
    
    Returns:
        bool: Success status of visualization generation
    """
    print("🎨 UNITED AIRLINES DATA VISUALIZATION SUITE")
    print("=" * 60)
    
    # Create organized output directory
    output_path = 'Output_Files/part4_visualization/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Also ensure legacy outputs exist
    Path('outputs').mkdir(exist_ok=True)
    
    try:
        # Load United Airlines submission data
        try:
            import config
            ua_data = pd.read_csv(config.get_legacy_output_path())
        except FileNotFoundError:
            print("⚠️ Submission file not found. Please run Part 3 first.")
            return False
        
        print(f"📊 Loaded {len(ua_data):,} United Airlines flights for visualization")
        
        # Generate comprehensive visualization suite
        _create_difficulty_distribution(ua_data, output_path)
        _create_daily_trends(ua_data, output_path)
        _create_route_analysis(ua_data, output_path)
        _create_feature_analytics(ua_data, output_path)
        _create_correlation_analysis(ua_data, output_path)
        
        print(f"\n✅ VISUALIZATION SUITE COMPLETE!")
        print(f"📁 Outputs saved to: {output_path}")
        print(f"📁 Legacy copies in: outputs/")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        return False

def _create_difficulty_distribution(data, output_path):
    """Create comprehensive difficulty score distribution visualizations."""
    print("\n📈 Creating enhanced difficulty distribution charts...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('United Airlines Flight Difficulty Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Distribution histogram with statistics
    axes[0,0].hist(data['difficulty_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[0,0].axvline(data['difficulty_score'].mean(), color='red', linestyle='--', label=f'Mean: {data["difficulty_score"].mean():.3f}')
    axes[0,0].axvline(data['difficulty_score'].median(), color='orange', linestyle='--', label=f'Median: {data["difficulty_score"].median():.3f}')
    axes[0,0].set_title('Difficulty Score Distribution with Statistics')
    axes[0,0].set_xlabel('Difficulty Score')
    axes[0,0].set_ylabel('Flight Count')
    axes[0,0].legend()
    
    # 2. Enhanced category breakdown
    if 'difficulty_classification' in data.columns:
        classification_counts = data['difficulty_classification'].value_counts()
        colors = ['#ff9999', '#ffcc99', '#99ff99']  # Red, Orange, Green
        axes[0,1].pie(classification_counts.values, labels=classification_counts.index, 
                     autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0,1].set_title('Required 3-Tier Classification\n(Difficult-Medium-Easy)')
    else:
        category_counts = data['difficulty_category'].value_counts()
        axes[0,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Flight Difficulty Categories')
    
    # 3. Daily ranking distribution
    axes[0,2].hist(data['daily_rank'], bins=30, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    axes[0,2].set_title('Daily Ranking Distribution')
    axes[0,2].set_xlabel('Daily Rank (1=Most Difficult)')
    axes[0,2].set_ylabel('Flight Count')
    
    # 4. Score vs Rank correlation
    sample_data = data.sample(min(1000, len(data)))  # Sample for readability
    axes[1,0].scatter(sample_data['daily_rank'], sample_data['difficulty_score'], 
                     alpha=0.6, color='purple', s=20)
    axes[1,0].set_title('Difficulty Score vs Daily Rank')
    axes[1,0].set_xlabel('Daily Rank')
    axes[1,0].set_ylabel('Difficulty Score')
    
    # 5. Feature correlation heatmap (if features available)
    feature_cols = ['difficulty_score', 'daily_percentile']
    if 'load_factor' in data.columns:
        feature_cols.extend(['load_factor', 'time_pressure_ratio', 'total_ssrs'])
    
    correlation_matrix = data[feature_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Matrix')
    
    # 6. Top difficult routes
    if len(data) > 0:
        data['route'] = data['scheduled_departure_station_code'] + '→' + data['scheduled_arrival_station_code']
        top_difficult_routes = data.nlargest(20, 'difficulty_score')['route'].value_counts().head(10)
        
        axes[1,2].barh(range(len(top_difficult_routes)), top_difficult_routes.values, color='coral')
        axes[1,2].set_yticks(range(len(top_difficult_routes)))
        axes[1,2].set_yticklabels(top_difficult_routes.index)
        axes[1,2].set_title('Top 10 Routes by Difficulty Frequency')
        axes[1,2].set_xlabel('Count in Top Difficult Flights')
    
    # Top routes by difficulty
    route_difficulty = data.groupby(['scheduled_departure_station_code', 
                                   'scheduled_arrival_station_code'])['difficulty_score'].mean().nlargest(10)
    route_labels = [f"{idx[0]}-{idx[1]}" for idx in route_difficulty.index]
    axes[1,1].barh(range(len(route_labels)), route_difficulty.values)
    axes[1,1].set_yticks(range(len(route_labels)))
    axes[1,1].set_yticklabels(route_labels)
    axes[1,1].set_title('Top 10 Most Difficult Routes')
    axes[1,1].set_xlabel('Average Difficulty Score')
    
    plt.tight_layout()
    
    # Save to both organized and legacy locations
    organized_path = os.path.join(output_path, 'ua_difficulty_analysis.png')
    legacy_path = 'outputs/ua_difficulty_analysis.png'
    plt.savefig(organized_path, dpi=300, bbox_inches='tight')
    plt.savefig(legacy_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def _create_daily_trends(data, output_path):
    """Create daily trend visualizations."""
    print("📅 Creating daily trend analysis...")
    
    # Convert date column
    data['date'] = pd.to_datetime(data['scheduled_departure_date_local'])
    daily_stats = data.groupby('date').agg({
        'difficulty_score': ['mean', 'std', 'count'],
        'daily_rank': 'mean'
    }).round(3)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('United Airlines Daily Operational Trends', fontsize=16, fontweight='bold')
    
    # Daily average difficulty
    axes[0].plot(daily_stats.index, daily_stats[('difficulty_score', 'mean')], 
                marker='o', linewidth=2, markersize=4)
    axes[0].fill_between(daily_stats.index, 
                        daily_stats[('difficulty_score', 'mean')] - daily_stats[('difficulty_score', 'std')],
                        daily_stats[('difficulty_score', 'mean')] + daily_stats[('difficulty_score', 'std')],
                        alpha=0.3)
    axes[0].set_title('Daily Average Difficulty Score ± Standard Deviation')
    axes[0].set_ylabel('Difficulty Score')
    axes[0].grid(True, alpha=0.3)
    
    # Flight volume
    axes[1].bar(daily_stats.index, daily_stats[('difficulty_score', 'count')], alpha=0.7)
    axes[1].set_title('Daily United Airlines Flight Volume')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Number of Flights')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to both locations
    organized_path = os.path.join(output_path, 'ua_daily_trends.png')
    legacy_path = 'outputs/ua_daily_trends.png'
    plt.savefig(organized_path, dpi=300, bbox_inches='tight')
    plt.savefig(legacy_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def _create_route_analysis(data, output_path):
    """Create route-based analysis visualizations."""
    print("🛫 Creating route analysis...")
    
    # Route performance analysis
    route_stats = data.groupby(['scheduled_departure_station_code']).agg({
        'difficulty_score': ['mean', 'count'],
        'daily_percentile': 'mean'
    }).round(3)
    
    # Filter for airports with significant volume
    significant_routes = route_stats[route_stats[('difficulty_score', 'count')] >= 10]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('United Airlines Route Performance Analysis', fontsize=16, fontweight='bold')
    
    # Top departure airports by difficulty
    top_difficult = significant_routes.nlargest(15, ('difficulty_score', 'mean'))
    axes[0].barh(range(len(top_difficult)), top_difficult[('difficulty_score', 'mean')])
    axes[0].set_yticks(range(len(top_difficult)))
    axes[0].set_yticklabels(top_difficult.index)
    axes[0].set_title('Most Challenging Departure Airports')
    axes[0].set_xlabel('Average Difficulty Score')
    
    # Flight volume vs difficulty scatter
    axes[1].scatter(significant_routes[('difficulty_score', 'count')], 
                   significant_routes[('difficulty_score', 'mean')],
                   alpha=0.6, s=100)
    axes[1].set_xlabel('Flight Volume')
    axes[1].set_ylabel('Average Difficulty Score')
    axes[1].set_title('Volume vs Difficulty by Departure Airport')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to both locations
    organized_path = os.path.join(output_path, 'ua_route_analysis.png')
    legacy_path = 'outputs/ua_route_analysis.png'
    plt.savefig(organized_path, dpi=300, bbox_inches='tight')
    plt.savefig(legacy_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    """Main execution function for visualization suite."""
    success = create_ua_visualizations()
    return success

def _create_feature_analytics(data, output_path):
    """Create feature-based analytics visualizations."""
    print("🔍 Creating feature analytics...")
    
    if 'load_factor' not in data.columns:
        print("   ⚠️ Feature columns not available - skipping feature analytics")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('United Airlines Feature Analytics', fontsize=16, fontweight='bold')
    
    # Load factor vs difficulty
    axes[0,0].scatter(data['load_factor'], data['difficulty_score'], alpha=0.6, color='blue', s=20)
    axes[0,0].set_xlabel('Load Factor')
    axes[0,0].set_ylabel('Difficulty Score')
    axes[0,0].set_title('Load Factor vs Difficulty Score')
    
    # Time pressure analysis
    if 'time_pressure_ratio' in data.columns:
        axes[0,1].scatter(data['time_pressure_ratio'], data['difficulty_score'], alpha=0.6, color='red', s=20)
        axes[0,1].set_xlabel('Time Pressure Ratio')
        axes[0,1].set_ylabel('Difficulty Score')
        axes[0,1].set_title('Time Pressure vs Difficulty Score')
    
    # SSR impact
    if 'total_ssrs' in data.columns:
        axes[1,0].scatter(data['total_ssrs'], data['difficulty_score'], alpha=0.6, color='green', s=20)
        axes[1,0].set_xlabel('Total Special Service Requests')
        axes[1,0].set_ylabel('Difficulty Score')
        axes[1,0].set_title('Special Services vs Difficulty Score')
    
    # Feature distribution
    feature_cols = [col for col in ['load_factor', 'time_pressure_ratio', 'total_ssrs'] if col in data.columns]
    if feature_cols:
        data[feature_cols].hist(ax=axes[1,1], bins=20, alpha=0.7)
        axes[1,1].set_title('Feature Distributions')
    
    plt.tight_layout()
    
    # Save to both locations
    organized_path = os.path.join(output_path, 'ua_feature_analytics.png')
    legacy_path = 'outputs/ua_feature_analytics.png'
    plt.savefig(organized_path, dpi=300, bbox_inches='tight')
    plt.savefig(legacy_path, dpi=300, bbox_inches='tight')
    plt.close()

def _create_correlation_analysis(data, output_path):
    """Create correlation analysis visualizations."""
    print("📊 Creating correlation analysis...")
    
    # Identify numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        print("   ⚠️ Insufficient numeric columns for correlation analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('United Airlines Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Full correlation matrix
    correlation_matrix = data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=axes[0], fmt='.2f')
    axes[0].set_title('Complete Feature Correlation Matrix')
    
    # Difficulty score correlations
    if 'difficulty_score' in numeric_cols:
        difficulty_corr = correlation_matrix['difficulty_score'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
        axes[1].barh(range(len(difficulty_corr)), difficulty_corr.values, color='skyblue')
        axes[1].set_yticks(range(len(difficulty_corr)))
        axes[1].set_yticklabels(difficulty_corr.index)
        axes[1].set_xlabel('Absolute Correlation with Difficulty Score')
        axes[1].set_title('Feature Importance by Correlation')
    
    plt.tight_layout()
    
    # Save to both locations
    organized_path = os.path.join(output_path, 'ua_correlation_analysis.png')
    legacy_path = 'outputs/ua_correlation_analysis.png'
    plt.savefig(organized_path, dpi=300, bbox_inches='tight')
    plt.savefig(legacy_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
