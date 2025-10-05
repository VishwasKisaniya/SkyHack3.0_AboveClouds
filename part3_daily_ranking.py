"""
Flight Difficulty Score Analysis - Part 3: Daily Ranking & United Airlines Submission

This module generates daily flight difficulty rankings specifically for United Airlines
operations and produces the final submission file. The system implements advanced
ranking algorithms with statistical validation and stability analysis.

Author: Data Science Team
Version: 1.0.0
Date: October 2025
Project: United Airlines SkyHack 3.0

Key Features:
- United Airlines flight filtering
- Daily difficulty ranking with percentile analysis
- Statistical stability validation (PSI analysis)
- Multi-tier difficulty classification
- Production-ready submission file generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime
from scipy import stats
import config
import warnings
warnings.filterwarnings('ignore')

class DataDrivenRanker:
    """
    Advanced Daily Ranking System for United Airlines Operations
    
    This class implements sophisticated daily ranking algorithms specifically
    designed for United Airlines flight operations. It provides comprehensive
    difficulty assessment with statistical validation and produces production-ready
    submission files.
    
    Attributes:
        df (pd.DataFrame): Enhanced flight features dataset
        ua_df (pd.DataFrame): United Airlines flights only
        output_path (str): Organized output directory structure
        daily_stats (dict): Daily statistical summaries
    """
    
    def __init__(self, features_file='outputs/enhanced_flight_features.csv',
                 output_path='Output_Files/part3_daily_ranking/'):
        """
        Initialize the daily ranking system.
        
        Args:
            features_file (str): Path to enhanced features CSV file
            output_path (str): Directory for organized output files
        """
        # Try primary output first, fallback to legacy location
        try:
            self.df = pd.read_csv(features_file)
        except FileNotFoundError:
            legacy_path = 'enhanced_flight_features.csv'
            self.df = pd.read_csv(legacy_path)
            print(f"📋 Loaded from legacy location: {legacy_path}")
        
        self.output_path = output_path
        self.daily_stats = {}
        
        # Ensure output directory exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        # Filter for United Airlines flights only
        self.ua_df = self.df[self.df['company_id'] == 'UA'].copy()
        print(f"🎯 United Airlines Filter Applied:")
        print(f"   • Total flights in dataset: {len(self.df):,}")
        print(f"   • United Airlines flights: {len(self.ua_df):,}")
        print(f"   • UA percentage: {(len(self.ua_df)/len(self.df)*100):.1f}%")
        
        # Use UA dataset for all further processing
        self.df = self.ua_df
        
    def prepare_data(self):
        """Prepare data for ranking"""
        print("🔄 Preparing data for daily ranking...")
        
        # Parse dates
        self.df['scheduled_departure_date_local'] = pd.to_datetime(self.df['scheduled_departure_date_local']).dt.date
        
        # Ensure we have difficulty scores
        if 'difficulty_score' not in self.df.columns:
            print("❌ Error: difficulty_score column not found!")
            return self
        
        print(f"   ✓ {len(self.df):,} flights loaded")
        print(f"   ✓ Date range: {self.df['scheduled_departure_date_local'].min()} to {self.df['scheduled_departure_date_local'].max()}")
        print(f"   ✓ Difficulty score range: {self.df['difficulty_score'].min():.3f} - {self.df['difficulty_score'].max():.3f}")
        
        return self
    
    def create_daily_rankings(self):
        """Create daily rankings for each flight"""
        print("\n📅 Creating Daily Rankings:")
        
        # Rank flights within each day (1 = most difficult)
        self.df['daily_rank'] = self.df.groupby('scheduled_departure_date_local')['difficulty_score'].rank(
            method='dense',
            ascending=False
        ).astype(int)
        
        # Calculate percentile within each day
        self.df['daily_percentile'] = self.df.groupby('scheduled_departure_date_local')['difficulty_score'].rank(
            pct=True,
            ascending=False
        ) * 100
        
        # Calculate some daily statistics
        daily_summary = self.df.groupby('scheduled_departure_date_local').agg({
            'difficulty_score': ['count', 'mean', 'std', 'min', 'max'],
            'daily_rank': 'max'
        }).round(3)
        
        daily_summary.columns = ['flights_count', 'avg_difficulty', 'std_difficulty', 
                               'min_difficulty', 'max_difficulty', 'max_rank']
        
        self.daily_stats = daily_summary
        
        print(f"   ✓ Daily rankings created")
        print(f"   ✓ Daily statistics calculated for {len(daily_summary)} days")
        
        # Show sample of top difficult flights per day
        sample_difficult = self.df[self.df['daily_rank'] <= 3].groupby('scheduled_departure_date_local').first()
        print(f"   ✓ Sample of most difficult flights per day:")
        for date, row in sample_difficult.head(3).iterrows():
            print(f"     {date}: Flight {row['company_id']}{row['flight_number']} ({row['scheduled_departure_station_code']}→{row['scheduled_arrival_station_code']}) - Score: {row['difficulty_score']:.3f}")
        
        return self
    
    def classify_difficulty_levels(self):
        """Classify flights into difficulty categories"""
        print("\n🏷️ Creating Difficulty Classifications:")
        
        # Create proper 3-tier classification (as required by submission)
        # Note: Lower percentile = higher difficulty (rank 1 = top difficulty)
        self.df['difficulty_classification'] = pd.cut(
            self.df['daily_percentile'],
            bins=[0, 33.33, 66.67, 100],
            labels=['Difficult', 'Medium', 'Easy'],  # Fixed: lower percentile = more difficult
            include_lowest=True
        )
        
        # Create detailed 5-tier system for analysis
        self.df['difficulty_category'] = pd.cut(
            self.df['daily_percentile'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Difficult', 'Difficult', 'Moderate', 'Easy', 'Very Easy'],  # Fixed order
            include_lowest=True
        )
        
        # Show distribution with corrected classifications
        classification_dist = self.df['difficulty_classification'].value_counts()
        category_dist = self.df['difficulty_category'].value_counts()
        
        print(f"   ✓ 5-level difficulty distribution:")
        for category, count in category_dist.items():
            print(f"     {category}: {count:,} flights ({count/len(self.df)*100:.1f}%)")
        
        print(f"   ✓ 3-tier difficulty distribution:")
        for tier, count in classification_dist.items():
            print(f"     {tier}: {count:,} flights ({count/len(self.df)*100:.1f}%)")
        
        return self
    
    def calculate_psi_stability(self):
        """Calculate Population Stability Index to check score stability across days"""
        print("\n📊 Calculating Population Stability Index (PSI):")
        
        # Get expected distribution (overall)
        expected_dist = pd.cut(self.df['difficulty_score'], bins=10, include_lowest=True).value_counts(normalize=True).sort_index()
        
        psi_by_date = {}
        
        for date in self.df['scheduled_departure_date_local'].unique():
            daily_data = self.df[self.df['scheduled_departure_date_local'] == date]
            
            if len(daily_data) > 10:  # Only calculate if enough data
                actual_dist = pd.cut(daily_data['difficulty_score'], bins=10, include_lowest=True).value_counts(normalize=True).sort_index()
                
                # Align bins and fill missing with small value
                actual_dist = actual_dist.reindex(expected_dist.index, fill_value=0.001)
                expected_dist_aligned = expected_dist.reindex(expected_dist.index, fill_value=0.001)
                
                # Calculate PSI
                psi = np.sum((actual_dist - expected_dist_aligned) * np.log(actual_dist / expected_dist_aligned))
                psi_by_date[date] = psi
        
        avg_psi = np.mean(list(psi_by_date.values()))
        
        print(f"   ✓ Average PSI across dates: {avg_psi:.3f}")
        if avg_psi < 0.1:
            print(f"   ✅ PSI < 0.1: Model is stable across days")
        elif avg_psi < 0.25:
            print(f"   ⚠️ PSI 0.1-0.25: Some stability concerns")
        else:
            print(f"   ❌ PSI > 0.25: Significant stability issues")
        
        return self
    
    def generate_submission_file(self, username='yourusername'):
        """
        Generate the final United Airlines submission file.
        
        Args:
            username (str): Username for submission file naming
            
        Returns:
            pd.DataFrame: Final submission dataset
        """
        print(f"\n📁 Generating United Airlines Submission File: test_{username}.csv")
        
        # Validate that we have only UA flights
        unique_carriers = self.df['company_id'].unique()
        if len(unique_carriers) != 1 or unique_carriers[0] != 'UA':
            print(f"⚠️ Warning: Non-UA flights detected: {unique_carriers}")
        
        # Create enhanced submission format with features used for calculation
        feature_columns = [
            # Flight identification
            'company_id', 'flight_number', 'scheduled_departure_date_local',
            'scheduled_departure_station_code', 'scheduled_arrival_station_code',
            
            # Key features used in difficulty calculation
            'load_factor', 'time_pressure_ratio', 'ground_time_efficiency',
            'total_ssrs', 'children_ratio', 'total_passengers',
            
            # Results
            'difficulty_score', 'daily_rank', 'daily_percentile',
            'difficulty_classification', 'difficulty_category'
        ]
        
        # Select available columns (some features might not exist)
        available_columns = [col for col in feature_columns if col in self.df.columns]
        submission_df = self.df[available_columns].copy()
        
        # Add metadata for traceability
        submission_df['submission_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        submission_df['model_version'] = 'ua_focused_v1.0'
        submission_df['carrier_filter'] = 'United Airlines Only'
        
        # Sort by date and difficulty score (most difficult first)
        submission_df = submission_df.sort_values([
            'scheduled_departure_date_local', 'difficulty_score'
        ], ascending=[True, False])
        
        # Save to organized output directory
        organized_filename = os.path.join(self.output_path, config.get_submission_filename())
        submission_df.to_csv(organized_filename, index=False)
        
        # Also save to legacy outputs folder for backward compatibility
        Path('outputs').mkdir(exist_ok=True)
        legacy_filename = config.get_legacy_output_path()
        submission_df.to_csv(legacy_filename, index=False)
        
        print(f"   ✅ Primary submission file: {organized_filename}")
        print(f"   ✅ Legacy submission file: {legacy_filename}")
        print(f"   📊 Records: {len(submission_df):,} United Airlines flights")
        print(f"   📅 Date range: {submission_df['scheduled_departure_date_local'].min()} to {submission_df['scheduled_departure_date_local'].max()}")
        print(f"   🎯 Difficulty range: {submission_df['difficulty_score'].min():.3f} - {submission_df['difficulty_score'].max():.3f}")
        
        # Show sample of most difficult UA flights
        print(f"\n🔥 TOP 10 MOST DIFFICULT FLIGHTS IN SUBMISSION:")
        display_cols = ['company_id', 'flight_number', 'scheduled_departure_date_local',
                       'scheduled_departure_station_code', 'scheduled_arrival_station_code',
                       'difficulty_score', 'daily_rank', 'difficulty_classification']
        available_display_cols = [col for col in display_cols if col in submission_df.columns]
        top_10 = submission_df.head(10)[available_display_cols]
        print(top_10.to_string(index=False))
        
        # Show classification distribution
        if 'difficulty_classification' in submission_df.columns:
            print(f"\n📊 FINAL CLASSIFICATION DISTRIBUTION:")
            class_dist = submission_df['difficulty_classification'].value_counts()
            for classification, count in class_dist.items():
                pct = (count / len(submission_df)) * 100
                print(f"   • {classification}: {count:,} flights ({pct:.1f}%)")
        
        return submission_df
    
    def save_daily_statistics(self):
        """
        Save comprehensive daily statistics and analysis reports.
        
        Returns:
            DataDrivenRanker: Self reference for method chaining
        """
        # Save daily statistics to organized directory
        daily_stats_path = os.path.join(self.output_path, 'daily_difficulty_stats.csv')
        self.daily_stats.to_csv(daily_stats_path)
        
        # Create comprehensive summary report
        summary_stats = {
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'carrier_focus': 'United Airlines',
            'total_ua_flights': len(self.df),
            'total_days': self.df['scheduled_departure_date_local'].nunique(),
            'avg_flights_per_day': len(self.df) / self.df['scheduled_departure_date_local'].nunique(),
            'avg_difficulty_score': self.df['difficulty_score'].mean(),
            'std_difficulty_score': self.df['difficulty_score'].std(),
            'min_difficulty_score': self.df['difficulty_score'].min(),
            'max_difficulty_score': self.df['difficulty_score'].max(),
            'model_version': 'ua_focused_v1.0'
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_path = os.path.join(self.output_path, 'ua_scoring_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Also save to legacy outputs folder for backward compatibility
        Path('outputs').mkdir(exist_ok=True)
        legacy_daily = 'outputs/daily_difficulty_stats.csv'
        legacy_summary = 'outputs/scoring_summary.csv'
        self.daily_stats.to_csv(legacy_daily)
        summary_df.to_csv(legacy_summary, index=False)
        
        print(f"\n📈 Statistical analysis files saved:")
        print(f"   📁 Primary outputs:")
        print(f"      • Daily statistics: {daily_stats_path}")
        print(f"      • UA summary: {summary_path}")
        print(f"   📁 Legacy outputs:")
        print(f"      • Daily statistics: {legacy_daily}")
        print(f"      • Summary: {legacy_summary}")
        
        return self

def main():
    """Main execution function"""
    print("🚀 DATA-DRIVEN DAILY RANKING & SUBMISSION GENERATION")
    print("="*60)
    
    # Initialize ranker
    ranker = DataDrivenRanker()
    
    # Execute pipeline
    ranker = (ranker
             .prepare_data()
             .create_daily_rankings()
             .classify_difficulty_levels()
             .calculate_psi_stability())
    
    submission_df = ranker.generate_submission_file(config.get_username())
    ranker.save_daily_statistics()
    
    print(f"\n✅ DAILY RANKING & SUBMISSION COMPLETE!")
    print(f"🎯 Ready for hackathon submission: {config.get_submission_filename()}")
    
    return submission_df

if __name__ == "__main__":
    main()