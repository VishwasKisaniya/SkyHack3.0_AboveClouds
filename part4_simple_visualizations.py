"""
Simplified Enhanced Visualizations for Presentations

A robust visualization suite that works with the available United Airlines data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional presentation style
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def create_presentation_visualizations():
    """Create presentation-ready visualizations."""
    
    # Create output directory
    output_dir = Path('Output_Files/part4_enhanced_visualizations/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        import config
        df = pd.read_csv(config.get_legacy_output_path())
        print(f"📊 Loaded {len(df):,} United Airlines flights for visualization")
        
        # Add derived columns safely
        df['month'] = pd.to_datetime(df['scheduled_departure_date_local']).dt.month
        df['route'] = df['scheduled_departure_station_code'] + '→' + df['scheduled_arrival_station_code']
        
        # Create visualizations
        create_executive_summary(df, output_dir)
        create_difficulty_analysis(df, output_dir)
        create_operational_insights(df, output_dir)
        create_route_analysis(df, output_dir)
        create_temporal_analysis(df, output_dir)
        
        print("\\n✅ All presentation visualizations created successfully!")
        print(f"📁 Saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return False

def create_executive_summary(df, output_dir):
    """Executive summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('United Airlines Executive Summary Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Key metrics
    axes[0,0].axis('off')
    kpis = [
        f"Total Flights: {len(df):,}",
        f"Average Risk Score: {df['difficulty_score'].mean():.3f}",
        f"High Risk Flights: {(df['difficulty_classification'] == 'Difficult').sum():,}",
        f"Risk Rate: {(df['difficulty_classification'] == 'Difficult').mean()*100:.1f}%"
    ]
    
    y_pos = 0.8
    for kpi in kpis:
        axes[0,0].text(0.1, y_pos, f"• {kpi}", fontsize=14, fontweight='bold', 
                      transform=axes[0,0].transAxes)
        y_pos -= 0.15
    axes[0,0].set_title('Key Performance Indicators', fontweight='bold')
    
    # 2. Risk distribution
    class_counts = df['difficulty_classification'].value_counts()
    colors = ['#EF476F', '#FFD166', '#06D6A0']
    axes[0,1].pie(class_counts.values, labels=class_counts.index, 
                 autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0,1].set_title('Risk Classification Distribution', fontweight='bold')
    
    # 3. Score distribution
    axes[1,0].hist(df['difficulty_score'], bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
    mean_score = df['difficulty_score'].mean()
    axes[1,0].axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                     label=f'Mean: {mean_score:.3f}')
    axes[1,0].set_title('Risk Score Distribution', fontweight='bold')
    axes[1,0].set_xlabel('Risk Score')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    
    # 4. Monthly trend
    monthly_stats = df.groupby('month').agg({
        'difficulty_score': 'mean',
        'flight_number': 'count'
    })
    
    ax1 = axes[1,1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(monthly_stats.index, monthly_stats['difficulty_score'], 'o-', 
                    color='#E63946', linewidth=2, markersize=6)
    bars = ax2.bar(monthly_stats.index, monthly_stats['flight_number'], alpha=0.3, 
                  color='#2E86AB')
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Avg Risk Score', color='#E63946')
    ax2.set_ylabel('Flight Count', color='#2E86AB')
    ax1.set_title('Monthly Trends', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'executive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Executive summary created")

def create_difficulty_analysis(df, output_dir):
    """Detailed difficulty analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('United Airlines Difficulty Analysis', fontsize=16, fontweight='bold')
    
    # 1. Score vs Rank correlation
    sample_df = df.sample(min(1000, len(df)))
    axes[0,0].scatter(sample_df['daily_rank'], sample_df['difficulty_score'], 
                     alpha=0.6, color='purple', s=20)
    axes[0,0].set_title('Risk Score vs Daily Rank', fontweight='bold')
    axes[0,0].set_xlabel('Daily Rank')
    axes[0,0].set_ylabel('Risk Score')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Percentile analysis
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = [df['difficulty_score'].quantile(p/100) for p in percentiles]
    
    axes[0,1].plot(percentiles, percentile_values, 'o-', linewidth=2, markersize=8, color='#F72585')
    axes[0,1].set_title('Risk Score Percentiles', fontweight='bold')
    axes[0,1].set_xlabel('Percentile')
    axes[0,1].set_ylabel('Risk Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Classification breakdown
    class_counts = df['difficulty_classification'].value_counts()
    bars = axes[1,0].bar(class_counts.index, class_counts.values, 
                        color=['#EF476F', '#FFD166', '#06D6A0'], alpha=0.8)
    axes[1,0].set_title('Classification Breakdown', fontweight='bold')
    axes[1,0].set_ylabel('Number of Flights')
    
    # Add count labels
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        pct = (count / len(df)) * 100
        axes[1,0].text(bar.get_x() + bar.get_width()/2, height + 20,
                      f'{count:,}\\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # 4. Box plot by classification
    class_data = []
    labels = []
    for cat in ['Easy', 'Medium', 'Difficult']:
        if cat in df['difficulty_classification'].values:
            class_data.append(df[df['difficulty_classification'] == cat]['difficulty_score'].values)
            labels.append(cat)
    
    if class_data:
        box_plot = axes[1,1].boxplot(class_data, labels=labels, patch_artist=True, notch=True)
        colors = ['#06D6A0', '#FFD166', '#EF476F'][:len(class_data)]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    axes[1,1].set_title('Score Distribution by Classification', fontweight='bold')
    axes[1,1].set_ylabel('Risk Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Difficulty analysis created")

def create_operational_insights(df, output_dir):
    """Operational insights for management."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('United Airlines Operational Insights', fontsize=16, fontweight='bold')
    
    # 1. Load factor analysis
    axes[0,0].scatter(df['load_factor'], df['difficulty_score'], alpha=0.6, s=30, color='#4ECDC4')
    axes[0,0].set_title('Load Factor vs Risk Score', fontweight='bold')
    axes[0,0].set_xlabel('Load Factor')
    axes[0,0].set_ylabel('Risk Score')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Time pressure analysis
    axes[0,1].scatter(df['time_pressure_ratio'], df['difficulty_score'], alpha=0.6, s=30, color='#FF6B6B')
    axes[0,1].set_title('Time Pressure vs Risk Score', fontweight='bold')
    axes[0,1].set_xlabel('Time Pressure Ratio')
    axes[0,1].set_ylabel('Risk Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Passenger count analysis
    axes[1,0].scatter(df['total_passengers'], df['difficulty_score'], alpha=0.6, s=30, color='#FFE66D')
    axes[1,0].set_title('Passenger Count vs Risk Score', fontweight='bold')
    axes[1,0].set_xlabel('Total Passengers')
    axes[1,0].set_ylabel('Risk Score')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Ground time efficiency
    axes[1,1].scatter(df['ground_time_efficiency'], df['difficulty_score'], alpha=0.6, s=30, color='#95E1D3')
    axes[1,1].set_title('Ground Time Efficiency vs Risk Score', fontweight='bold')
    axes[1,1].set_xlabel('Ground Time Efficiency')
    axes[1,1].set_ylabel('Risk Score')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'operational_insights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Operational insights created")

def create_route_analysis(df, output_dir):
    """Route performance analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('United Airlines Route Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top departure airports by volume
    dept_volume = df['scheduled_departure_station_code'].value_counts().head(10)
    bars = axes[0,0].bar(range(len(dept_volume)), dept_volume.values, color='#4ECDC4')
    axes[0,0].set_xticks(range(len(dept_volume)))
    axes[0,0].set_xticklabels(dept_volume.index, rotation=45)
    axes[0,0].set_title('Top Departure Airports by Volume', fontweight='bold')
    axes[0,0].set_ylabel('Flight Count')
    
    # 2. Top arrival airports by volume
    arr_volume = df['scheduled_arrival_station_code'].value_counts().head(10)
    bars = axes[0,1].bar(range(len(arr_volume)), arr_volume.values, color='#FF6B6B')
    axes[0,1].set_xticks(range(len(arr_volume)))
    axes[0,1].set_xticklabels(arr_volume.index, rotation=45)
    axes[0,1].set_title('Top Arrival Airports by Volume', fontweight='bold')
    axes[0,1].set_ylabel('Flight Count')
    
    # 3. Most challenging routes (with sufficient data)
    route_stats = df.groupby('route').agg({
        'difficulty_score': 'mean',
        'flight_number': 'count'
    }).reset_index()
    route_stats = route_stats[route_stats['flight_number'] >= 3]  # Min 3 flights
    
    if len(route_stats) > 0:
        top_difficult_routes = route_stats.nlargest(10, 'difficulty_score')
        bars = axes[1,0].barh(range(len(top_difficult_routes)), top_difficult_routes['difficulty_score'],
                             color='#E63946', alpha=0.8)
        axes[1,0].set_yticks(range(len(top_difficult_routes)))
        axes[1,0].set_yticklabels([r[:12] for r in top_difficult_routes['route']], fontsize=10)
        axes[1,0].set_title('Most Challenging Routes', fontweight='bold')
        axes[1,0].set_xlabel('Average Risk Score')
    
    # 4. Route volume vs difficulty
    if len(route_stats) > 0:
        axes[1,1].scatter(route_stats['flight_number'], route_stats['difficulty_score'], 
                         alpha=0.6, s=60, color='purple')
        axes[1,1].set_title('Route Volume vs Average Risk', fontweight='bold')
        axes[1,1].set_xlabel('Flight Count')
        axes[1,1].set_ylabel('Average Risk Score')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'route_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Route analysis created")

def create_temporal_analysis(df, output_dir):
    """Time-based analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('United Airlines Temporal Analysis', fontsize=16, fontweight='bold')
    
    # 1. Daily ranking distribution
    axes[0,0].hist(df['daily_rank'], bins=30, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    axes[0,0].set_title('Daily Ranking Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Daily Rank')
    axes[0,0].set_ylabel('Frequency')
    
    # 2. Percentile distribution
    axes[0,1].hist(df['daily_percentile'], bins=30, alpha=0.7, color='lightblue', edgecolor='navy')
    axes[0,1].set_title('Daily Percentile Distribution', fontweight='bold')
    axes[0,1].set_xlabel('Daily Percentile')
    axes[0,1].set_ylabel('Frequency')
    
    # 3. Monthly flight volume
    monthly_counts = df.groupby('month')['flight_number'].count()
    bars = axes[1,0].bar(monthly_counts.index, monthly_counts.values, 
                        color=plt.cm.viridis(np.linspace(0, 1, len(monthly_counts))))
    axes[1,0].set_title('Monthly Flight Volume', fontweight='bold')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Flight Count')
    
    # 4. Risk score statistics
    stats_data = {
        'Mean': df['difficulty_score'].mean(),
        'Median': df['difficulty_score'].median(),
        'Std Dev': df['difficulty_score'].std(),
        '75th %ile': df['difficulty_score'].quantile(0.75),
        '90th %ile': df['difficulty_score'].quantile(0.90),
        '95th %ile': df['difficulty_score'].quantile(0.95)
    }
    
    bars = axes[1,1].bar(range(len(stats_data)), list(stats_data.values()), 
                        color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
    axes[1,1].set_xticks(range(len(stats_data)))
    axes[1,1].set_xticklabels(list(stats_data.keys()), rotation=45)
    axes[1,1].set_title('Risk Score Statistics', fontweight='bold')
    axes[1,1].set_ylabel('Score Value')
    
    # Add value labels
    for bar, value in zip(bars, stats_data.values()):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2, height + 0.005,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Temporal analysis created")

def main():
    """Main execution function."""
    print("🎨 SIMPLIFIED ENHANCED VISUALIZATION SUITE")
    print("=" * 50)
    
    success = create_presentation_visualizations()
    
    if success:
        print("\\n🎉 SUCCESS: All presentation visualizations created!")
        print("\\n📊 Generated 5 comprehensive charts:")
        print("   1. executive_summary.png - Executive overview")
        print("   2. difficulty_analysis.png - Risk analysis")
        print("   3. operational_insights.png - Operations view")
        print("   4. route_analysis.png - Route performance")
        print("   5. temporal_analysis.png - Time-based trends")
        print("\\n💡 Charts are optimized for PowerPoint presentations")
    else:
        print("\\n❌ FAILED: Could not create visualizations")
    
    return success

if __name__ == "__main__":
    main()


