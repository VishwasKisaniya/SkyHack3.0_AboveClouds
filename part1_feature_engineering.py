"""
Flight Difficulty Score Analysis - Part 1: Data-Driven Feature Engineering

This module implements a comprehensive feature engineering pipeline for airline
operational difficulty assessment. The approach leverages real operational data
to derive meaningful features without relying on domain assumptions.

Author: Data Science Team
Version: 1.0.0
Date: October 2025
Project: United Airlines SkyHack 3.0

Key Features:
- Time pressure analysis based on actual turnaround times
- Passenger complexity assessment using PNR data
- Special service request impact quantification
- Aircraft and route operational characteristics
- Quantitative difficulty scoring methodology
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataDrivenFeatureEngine:
    """
    Advanced Feature Engineering Engine for Airline Operations Analysis
    
    This class implements a sophisticated feature engineering pipeline that
    extracts meaningful operational insights from multi-dimensional airline data.
    The methodology is entirely data-driven, ensuring objectivity and reducing
    bias in the analytical process.
    
    Attributes:
        data_path (str): Path to input data directory
        output_path (str): Path to organized output directory structure
        flight_df (pd.DataFrame): Core flight operations data
        pnr_flight_df (pd.DataFrame): Passenger name record flight associations
        pnr_remark_df (pd.DataFrame): Special service requests and remarks
        airport_df (pd.DataFrame): Airport operational characteristics
        master_df (pd.DataFrame): Consolidated feature dataset
    """
    
    def __init__(self, data_path='data/', output_path='Output_Files/part1_feature_engineering/'):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            data_path (str): Directory containing input CSV files
            output_path (str): Directory for organized output files
        """
        self.data_path = data_path
        self.output_path = output_path
        self.flight_df = None
        self.pnr_flight_df = None
        self.pnr_remark_df = None
        self.airport_df = None
        self.master_df = None
        
        # Ensure output directory structure exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """
        Load and preprocess all required datasets for feature engineering.
        
        This method handles the initial data ingestion and performs essential
        data type conversions and standardizations required for downstream
        feature engineering processes.
        
        Returns:
            DataDrivenFeatureEngine: Self reference for method chaining
            
        Raises:
            FileNotFoundError: If required data files are not found
            pd.errors.EmptyDataError: If data files are empty or corrupted
        """
        print("🔄 Loading and preprocessing datasets...")
        
        self.flight_df = pd.read_csv(f'{self.data_path}FlightLevelData.csv')
        self.pnr_flight_df = pd.read_csv(f'{self.data_path}PNRFlightLevelData.csv')
        self.pnr_remark_df = pd.read_csv(f'{self.data_path}PNRRemarkLevelData.csv')
        self.airport_df = pd.read_csv(f'{self.data_path}AirportsData.csv')
        
        # Convert Y/N to 1/0 for proper calculations
        self.pnr_flight_df['is_child'] = (self.pnr_flight_df['is_child'] == 'Y').astype(int)
        self.pnr_flight_df['is_stroller_user'] = (self.pnr_flight_df['is_stroller_user'] == 'Y').astype(int)
        
        # Start with flight data as master
        self.master_df = self.flight_df.copy()
        
        print(f"✅ Flight data: {len(self.flight_df):,} flights")
        print(f"✅ PNR data: {len(self.pnr_flight_df):,} passenger records")
        print(f"✅ SSR data: {len(self.pnr_remark_df):,} special service requests")
        print(f"✅ Airport data: {len(self.airport_df):,} airports")
        
        return self
    
    def create_time_pressure_features(self):
        """Time Pressure Features (40% weight) - FROM ACTUAL GROUND TIME DATA"""
        print("\n🕒 TIME PRESSURE FEATURES (40% weight) - REAL DATA:")
        
        # Calculate actual delays FIRST for correlation analysis
        self.master_df['departure_delay_minutes'] = np.where(
            (self.master_df['actual_departure_datetime_local'].notna()) & 
            (self.master_df['scheduled_departure_datetime_local'].notna()),
            (pd.to_datetime(self.master_df['actual_departure_datetime_local']) - 
             pd.to_datetime(self.master_df['scheduled_departure_datetime_local'])).dt.total_seconds() / 60,
            0
        )
        
        print(f"   ✓ Departure delays calculated: {(self.master_df['departure_delay_minutes'] > 0).sum():,} delayed flights")
        print(f"   ✓ Average delay: {self.master_df['departure_delay_minutes'].mean():.2f} minutes")
        
        # Real time pressure from actual constraints
        self.master_df['time_pressure_ratio'] = np.where(
            self.master_df['scheduled_ground_time_minutes'] > 0,
            (self.master_df['minimum_turn_minutes'] - self.master_df['scheduled_ground_time_minutes']) / 
            self.master_df['scheduled_ground_time_minutes'],
            0
        )
        
        # Performance variance where available
        self.master_df['ground_time_variance'] = np.where(
            (self.master_df['actual_ground_time_minutes'].notna()) & 
            (self.master_df['scheduled_ground_time_minutes'] > 0),
            (self.master_df['actual_ground_time_minutes'] - self.master_df['scheduled_ground_time_minutes']) / 
            self.master_df['scheduled_ground_time_minutes'],
            0
        )
        
        # Replace infinite values with 0
        self.master_df['time_pressure_ratio'] = self.master_df['time_pressure_ratio'].replace([np.inf, -np.inf], 0)
        self.master_df['ground_time_variance'] = self.master_df['ground_time_variance'].replace([np.inf, -np.inf], 0)
        
        # Ground time efficiency score
        self.master_df['ground_time_efficiency'] = np.where(
            self.master_df['minimum_turn_minutes'] > 0,
            self.master_df['scheduled_ground_time_minutes'] / self.master_df['minimum_turn_minutes'],
            1
        )
        
        # Find optimal ground time threshold from data
        self.optimal_ground_time_threshold = self.find_optimal_threshold(
            self.master_df['scheduled_ground_time_minutes'], 
            self.master_df['departure_delay_minutes']
        )
        
        print(f"   🎯 Optimal ground time threshold: {self.optimal_ground_time_threshold:.1f} minutes (data-driven)")
        
        print(f"   ✓ Time pressure ratio: {self.master_df['time_pressure_ratio'].mean():.3f} avg")
        print(f"   ✓ Tight turnarounds (<0): {(self.master_df['time_pressure_ratio'] < 0).sum():,} flights")
        print(f"   ✓ Ground time variance: {self.master_df['ground_time_variance'].mean():.3f} avg")
        
        return self
    
    def create_passenger_complexity_features(self):
        """Passenger Complexity Features (30% weight) - FROM PNR DATA"""
        print("\n👥 PASSENGER COMPLEXITY FEATURES (30% weight) - FROM PNR:")
        
        # Aggregate PNR data by flight
        pnr_agg = self.pnr_flight_df.groupby([
            'company_id', 'flight_number', 'scheduled_departure_date_local'
        ]).agg({
            'total_pax': 'sum',
            'is_child': 'sum',
            'basic_economy_ind': 'sum',
            'is_stroller_user': 'sum',
            'lap_child_count': 'sum',
            'record_locator': 'count'
        }).reset_index()
        
        pnr_agg.columns = [
            'company_id', 'flight_number', 'scheduled_departure_date_local',
            'total_passengers', 'total_children', 'basic_economy_pax',
            'stroller_users', 'lap_children', 'pnr_count'
        ]
        
        # Merge with master dataframe
        self.master_df = self.master_df.merge(
            pnr_agg,
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        
        # Fill missing passenger data with 0
        passenger_cols = ['total_passengers', 'total_children', 'basic_economy_pax',
                         'stroller_users', 'lap_children', 'pnr_count']
        for col in passenger_cols:
            self.master_df[col] = self.master_df[col].fillna(0)
        
        # Calculate complexity metrics
        self.master_df['load_factor'] = np.where(
            self.master_df['total_seats'] > 0,
            self.master_df['total_passengers'] / self.master_df['total_seats'],
            0
        )
        
        self.master_df['children_ratio'] = np.where(
            self.master_df['total_passengers'] > 0,
            self.master_df['total_children'] / self.master_df['total_passengers'],
            0
        )
        
        self.master_df['basic_economy_ratio'] = np.where(
            self.master_df['total_passengers'] > 0,
            self.master_df['basic_economy_pax'] / self.master_df['total_passengers'],
            0
        )
        
        self.master_df['avg_party_size'] = np.where(
            self.master_df['pnr_count'] > 0,
            self.master_df['total_passengers'] / self.master_df['pnr_count'],
            0
        )
        
        print(f"   ✓ Average load factor: {self.master_df['load_factor'].mean():.1%}")
        print(f"   ✓ Flights with passengers: {(self.master_df['total_passengers'] > 0).sum():,}")
        print(f"   ✓ Average children ratio: {self.master_df['children_ratio'].mean():.1%}")
        print(f"   ✓ Average party size: {self.master_df['avg_party_size'].mean():.1f}")
        
        return self
    
    def create_special_services_features(self):
        """Special Services Features (20% weight) - FROM SSR DATA"""
        print("\n♿ SPECIAL SERVICES FEATURES (20% weight) - FROM SSR:")
        
        # Get PNRs for each flight to link with SSR data
        flight_pnrs = self.pnr_flight_df.groupby([
            'company_id', 'flight_number', 'scheduled_departure_date_local'
        ])['record_locator'].apply(list).reset_index()
        
        # Calculate SSR metrics for each flight
        ssr_metrics = []
        
        for _, row in flight_pnrs.iterrows():
            pnrs = row['record_locator']
            flight_ssrs = self.pnr_remark_df[self.pnr_remark_df['record_locator'].isin(pnrs)]
            
            # Count different types of SSRs
            wheelchair_count = len(flight_ssrs[flight_ssrs['special_service_request'].str.contains('Wheelchair', na=False)])
            unaccompanied_count = len(flight_ssrs[flight_ssrs['special_service_request'].str.contains('Unaccompanied', na=False)])
            total_ssr_count = len(flight_ssrs)
            
            ssr_metrics.append({
                'company_id': row['company_id'],
                'flight_number': row['flight_number'],
                'scheduled_departure_date_local': row['scheduled_departure_date_local'],
                'wheelchair_requests': wheelchair_count,
                'unaccompanied_minors': unaccompanied_count,
                'total_ssrs': total_ssr_count
            })
        
        ssr_df = pd.DataFrame(ssr_metrics)
        
        # Merge with master dataframe
        self.master_df = self.master_df.merge(
            ssr_df,
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        
        # Fill missing SSR values with 0
        ssr_cols = ['wheelchair_requests', 'unaccompanied_minors', 'total_ssrs']
        for col in ssr_cols:
            self.master_df[col] = self.master_df[col].fillna(0)
        
        # Calculate SSR intensity
        self.master_df['ssr_intensity'] = np.where(
            self.master_df['total_passengers'] > 0,
            self.master_df['total_ssrs'] / self.master_df['total_passengers'],
            0
        )
        
        print(f"   ✓ Flights with wheelchairs: {(self.master_df['wheelchair_requests'] > 0).sum():,}")
        print(f"   ✓ Flights with minors: {(self.master_df['unaccompanied_minors'] > 0).sum():,}")
        print(f"   ✓ Average SSRs per flight: {self.master_df['total_ssrs'].mean():.1f}")
        print(f"   ✓ Average SSR intensity: {self.master_df['ssr_intensity'].mean():.3f}")
        
        return self
    
    def create_aircraft_route_features(self):
        """Aircraft & Route Features (10% weight) - FROM FLIGHT DATA"""
        print("\n✈️ AIRCRAFT & ROUTE FEATURES (10% weight) - FROM FLIGHT:")
        
        # Aircraft size categories based on REAL total_seats
        self.master_df['aircraft_size_category'] = pd.cut(
            self.master_df['total_seats'],
            bins=[0, 100, 200, 500],
            labels=['Small', 'Medium', 'Large']
        )
        
        # Get airport country information
        airport_countries = self.airport_df.set_index('airport_iata_code')['iso_country_code'].to_dict()
        
        self.master_df['departure_country'] = self.master_df['scheduled_departure_station_code'].map(airport_countries)
        self.master_df['arrival_country'] = self.master_df['scheduled_arrival_station_code'].map(airport_countries)
        
        # International route indicator
        self.master_df['is_international'] = (
            self.master_df['departure_country'] != self.master_df['arrival_country']
        ).astype(int)
        
        # Major hub identification (top airports by volume)
        major_hubs = self.master_df['scheduled_departure_station_code'].value_counts().head(20).index.tolist()
        self.master_df['from_major_hub'] = self.master_df['scheduled_departure_station_code'].isin(major_hubs).astype(int)
        self.master_df['to_major_hub'] = self.master_df['scheduled_arrival_station_code'].isin(major_hubs).astype(int)
        
        # Carrier complexity (Express typically more challenging)
        self.master_df['is_express'] = (self.master_df['carrier'] == 'Express').astype(int)
        
        print(f"   ✓ Aircraft sizes: {self.master_df['aircraft_size_category'].value_counts().to_dict()}")
        print(f"   ✓ International flights: {self.master_df['is_international'].sum():,}")
        print(f"   ✓ Hub departures: {self.master_df['from_major_hub'].sum():,}")
        print(f"   ✓ Express flights: {self.master_df['is_express'].sum():,}")
        
        return self
    
    def find_optimal_threshold(self, feature_series, delay_series):
        """Find threshold that maximizes delay prediction accuracy"""
        # Remove NaN values
        valid_data = pd.DataFrame({
            'feature': feature_series,
            'delay': delay_series
        }).dropna()
        
        if len(valid_data) < 10:
            return feature_series.median()  # Fallback to median
        
        # Try different percentile thresholds
        best_threshold = feature_series.median()
        best_correlation = 0
        
        for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            threshold = valid_data['feature'].quantile(percentile / 100)
            
            # Create binary indicator
            indicator = (valid_data['feature'] <= threshold).astype(int)
            
            # Calculate correlation with delays
            try:
                correlation = abs(indicator.corr(valid_data['delay']))
                if pd.notna(correlation) and correlation > best_correlation:
                    best_correlation = correlation
                    best_threshold = threshold
            except:
                continue
        
        return best_threshold
    
    def calculate_correlation_based_weights(self):
        """Calculate feature weights based on actual correlation with delays"""
        print("\n🔬 CALCULATING CORRELATION-BASED WEIGHTS:")
        
        # Prepare component scores for correlation analysis
        def normalize_feature(series):
            return (series - series.min()) / (series.max() - series.min()) if series.max() > series.min() else series
        
        # Time pressure components
        time_pressure_score = (
            normalize_feature(self.master_df['time_pressure_ratio'].fillna(0)) * 0.4 +
            normalize_feature(self.master_df['ground_time_variance'].abs().fillna(0)) * 0.3 +
            normalize_feature((1 - self.master_df['ground_time_efficiency']).fillna(0)) * 0.3
        )
        
        # Passenger complexity components
        passenger_complexity_score = (
            normalize_feature(self.master_df['load_factor'].fillna(0)) * 0.4 +
            normalize_feature(self.master_df['total_passengers'].fillna(0)) * 0.3 +
            normalize_feature(self.master_df['children_ratio'].fillna(0)) * 0.3
        )
        
        # Special services components
        special_services_score = (
            normalize_feature(self.master_df['wheelchair_requests'].fillna(0)) * 0.4 +
            normalize_feature(self.master_df['total_ssrs'].fillna(0)) * 0.3 +
            normalize_feature(self.master_df['ssr_intensity'].fillna(0)) * 0.3
        )
        
        # Aircraft/route components
        size_map = {'narrow_body': 0.3, 'wide_body': 0.7, 'regional': 0.1}
        aircraft_size_numeric = self.master_df['aircraft_size_category'].astype(str).map(size_map).fillna(0.6)
        
        aircraft_route_score = (
            normalize_feature(pd.Series(aircraft_size_numeric)) * 0.4 +
            self.master_df['is_international'] * 0.3 +
            (self.master_df['from_major_hub'] | self.master_df['to_major_hub']) * 0.2 +
            self.master_df['is_express'] * 0.1
        )
        
        # Create feature correlation matrix with delays
        feature_components = pd.DataFrame({
            'time_pressure': time_pressure_score,
            'passenger_complexity': passenger_complexity_score,
            'special_services': special_services_score,
            'aircraft_route': aircraft_route_score
        })
        
        # Calculate correlations with actual delays
        correlations = feature_components.corrwith(self.master_df['departure_delay_minutes']).abs()
        
        # Handle NaN correlations (replace with small positive value)
        correlations = correlations.fillna(0.01)
        
        # Normalize to get weights that sum to 1
        self.correlation_weights = correlations / correlations.sum()
        
        print("   📊 CORRELATION-BASED WEIGHTS (based on actual delay prediction):")
        for component, weight in self.correlation_weights.items():
            corr_value = correlations[component]
            print(f"   ✓ {component}: {weight:.1%} (correlation: {corr_value:.3f})")
        
        return self.correlation_weights
    
    def calculate_difficulty_scores(self):
        """Calculate final difficulty scores using CORRELATION-BASED component weighting"""
        print("\n🎯 CALCULATING DIFFICULTY SCORES:")
        
        def normalize_feature(series):
            """Normalize to 0-1 scale, handling edge cases"""
            series = series.replace([np.inf, -np.inf], 0)
            if series.max() == series.min():
                return pd.Series(0, index=series.index)
            return (series - series.min()) / (series.max() - series.min())
        
        # Time pressure component (40% weight)
        time_pressure_score = (
            normalize_feature(-self.master_df['time_pressure_ratio']) * 0.6 +  # Negative because negative = tight
            normalize_feature(self.master_df['ground_time_variance'].abs()) * 0.4
        )
        
        # Passenger complexity component (30% weight)
        passenger_complexity_score = (
            normalize_feature(self.master_df['load_factor']) * 0.3 +
            normalize_feature(self.master_df['children_ratio']) * 0.25 +
            normalize_feature(self.master_df['basic_economy_ratio']) * 0.25 +
            normalize_feature(self.master_df['avg_party_size']) * 0.2
        )
        
        # Special services component (20% weight)
        special_services_score = (
            normalize_feature(self.master_df['wheelchair_requests']) * 0.5 +
            normalize_feature(self.master_df['unaccompanied_minors']) * 0.3 +
            normalize_feature(self.master_df['ssr_intensity']) * 0.2
        )
        
        # Aircraft/route component (10% weight)
        size_map = {'Small': 0.3, 'Medium': 0.6, 'Large': 1.0}
        aircraft_size_numeric = self.master_df['aircraft_size_category'].astype(str).map(size_map).fillna(0.6)
        
        aircraft_route_score = (
            normalize_feature(pd.Series(aircraft_size_numeric)) * 0.4 +
            self.master_df['is_international'] * 0.3 +
            (self.master_df['from_major_hub'] | self.master_df['to_major_hub']) * 0.2 +
            self.master_df['is_express'] * 0.1
        )
        
        # Calculate correlation-based weights
        weights = self.calculate_correlation_based_weights()
        
        # Final weighted score using CORRELATION-BASED weights
        self.master_df['difficulty_score'] = (
            time_pressure_score * weights['time_pressure'] +
            passenger_complexity_score * weights['passenger_complexity'] +
            special_services_score * weights['special_services'] +
            aircraft_route_score * weights['aircraft_route']
        )
        
        # Store weights for reporting
        self.final_weights = weights
        
        print(f"   ✓ Difficulty range: {self.master_df['difficulty_score'].min():.3f} - {self.master_df['difficulty_score'].max():.3f}")
        print(f"   ✓ Average difficulty: {self.master_df['difficulty_score'].mean():.3f}")
        
        return self
    
    def save_features(self, filename='flight_features_master.csv'):
        """
        Save the feature-engineered dataset to organized output structure.
        
        Args:
            filename (str): Base filename for the output file
        
        Returns:
            pd.DataFrame: The master features dataframe
        """
        # Save to organized output directory
        organized_path = os.path.join(self.output_path, filename)
        self.master_df.to_csv(organized_path, index=False)
        
        # Also save to legacy outputs folder for backward compatibility
        Path('outputs').mkdir(exist_ok=True)
        legacy_path = os.path.join('outputs', filename)
        self.master_df.to_csv(legacy_path, index=False)
        
        print(f"\n✅ FEATURE ENGINEERING COMPLETE!")
        print(f"📁 Primary output: {organized_path}")
        print(f"📁 Legacy output: {legacy_path}")
        print(f"📊 Total features: {len(self.master_df.columns)}")
        print(f"✈️ Total flights: {len(self.master_df):,}")
        
        print(f"\n🎯 FINAL CORRELATION-BASED WEIGHTS USED:")
        if hasattr(self, 'final_weights'):
            for component, weight in self.final_weights.items():
                print(f"   ✓ {component.replace('_', ' ').title()}: {weight:.1%}")
        print(f"\n📊 These weights adapt based on what actually predicts delays in YOUR data!")
        
        # Add data-driven difficulty categories
        self.create_data_driven_categories()
        
        return self.master_df
    
    def create_data_driven_categories(self):
        """Create difficulty categories based on actual data distribution (quartiles)"""
        print(f"\n🏷️ CREATING DATA-DRIVEN CATEGORIES:")
        
        # Calculate quartiles from actual data
        scores = self.master_df['difficulty_score']
        q20, q40, q60, q80 = scores.quantile([0.20, 0.40, 0.60, 0.80])
        
        print(f"   📊 Data-driven thresholds (from YOUR data):")
        print(f"   • Very Easy: < {q20:.3f} (bottom 20%)")
        print(f"   • Easy: {q20:.3f} - {q40:.3f} (20-40%)")
        print(f"   • Medium: {q40:.3f} - {q60:.3f} (40-60%)")
        print(f"   • Difficult: {q60:.3f} - {q80:.3f} (60-80%)")
        print(f"   • Very Difficult: > {q80:.3f} (top 20%)")
        
        # Create categories based on data distribution
        def categorize_difficulty(score):
            if score <= q20:
                return "Very Easy"
            elif score <= q40:
                return "Easy"
            elif score <= q60:
                return "Medium"
            elif score <= q80:
                return "Difficult"
            else:
                return "Very Difficult"
        
        self.master_df['difficulty_category'] = self.master_df['difficulty_score'].apply(categorize_difficulty)
        
        # Report distribution
        category_counts = self.master_df['difficulty_category'].value_counts()
        print(f"\n   ✅ Category distribution (based on YOUR data patterns):")
        for category, count in category_counts.items():
            percentage = (count / len(self.master_df)) * 100
            print(f"   • {category}: {count:,} flights ({percentage:.1f}%)")
        
        return self

def main():
    """Main execution function"""
    print("🚀 DATA-DRIVEN FEATURE ENGINEERING - ZERO ASSUMPTIONS")
    print("="*60)
    
    # Initialize and run feature engineering
    engine = DataDrivenFeatureEngine()
    
    # Execute pipeline
    result_df = (engine
                .load_data()
                .create_time_pressure_features()
                .create_passenger_complexity_features()
                .create_special_services_features()
                .create_aircraft_route_features()
                .calculate_difficulty_scores()
                .save_features())
    
    # Show top difficult flights
    print(f"\n🔥 TOP 10 MOST DIFFICULT FLIGHTS:")
    top_flights = result_df.nlargest(10, 'difficulty_score')[
        ['company_id', 'flight_number', 'scheduled_departure_date_local',
         'scheduled_departure_station_code', 'scheduled_arrival_station_code',
         'difficulty_score', 'load_factor', 'total_ssrs']
    ]
    print(top_flights.to_string(index=False))
    
    return result_df

if __name__ == "__main__":
    main()