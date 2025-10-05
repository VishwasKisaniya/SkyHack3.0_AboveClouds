"""
Flight Difficulty Score Analysis - Part 2: Advanced Feature Selection & Validation

This module implements sophisticated feature selection techniques to identify
the most predictive features for flight operational difficulty. It leverages
multiple statistical methods to ensure robust feature ranking and selection.

Author: Data Science Team
Version: 1.0.0
Date: October 2025
Project: United Airlines SkyHack 3.0

Key Features:
- Multi-method feature importance analysis
- Statistical significance testing
- Feature stability validation
- Correlation-based feature selection
- Performance-driven feature ranking
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataDrivenFeatureSelector:
    """
    Advanced Feature Selection Engine for Predictive Analytics
    
    This class implements multiple feature selection methodologies to identify
    the most predictive and stable features for operational difficulty assessment.
    The approach combines statistical methods with machine learning techniques
    to ensure robust feature ranking.
    
    Attributes:
        df (pd.DataFrame): Input features dataset
        output_path (str): Organized output directory structure
        feature_importance (dict): Feature importance scores from different methods
        selected_features (list): Final selected feature set
    """
    
    def __init__(self, features_file='outputs/flight_features_master.csv', 
                 output_path='Output_Files/part2_feature_selection/'):
        """
        Initialize the feature selection pipeline.
        
        Args:
            features_file (str): Path to input features CSV file
            output_path (str): Directory for organized output files
        """
        # Try primary output first, fallback to legacy location
        try:
            self.df = pd.read_csv(features_file)
        except FileNotFoundError:
            legacy_path = 'flight_features_master.csv'
            self.df = pd.read_csv(legacy_path)
            print(f"📋 Loaded from legacy location: {legacy_path}")
        
        self.output_path = output_path
        self.feature_importance = {}
        self.selected_features = []
        
        # Ensure output directory exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self):
        """Prepare features for selection"""
        print("🔄 Preparing features for selection...")
        
        # Identify numeric features (exclude ID columns and target)
        exclude_cols = ['company_id', 'flight_number', 'scheduled_departure_date_local',
                       'scheduled_departure_station_code', 'scheduled_arrival_station_code',
                       'scheduled_departure_datetime_local', 'scheduled_arrival_datetime_local',
                       'actual_departure_datetime_local', 'actual_arrival_datetime_local',
                       'fleet_type', 'carrier', 'aircraft_size_category', 
                       'departure_country', 'arrival_country']
        
        # Get numeric columns
        numeric_cols = []
        for col in self.df.columns:
            if col not in exclude_cols:
                if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_cols.append(col)
        
        # Remove target from features
        if 'difficulty_score' in numeric_cols:
            numeric_cols.remove('difficulty_score')
        
        self.feature_cols = numeric_cols
        print(f"   ✓ {len(self.feature_cols)} numeric features identified")
        
        # Handle missing values
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        
        # Remove infinite values
        for col in self.feature_cols:
            self.df[col] = self.df[col].replace([np.inf, -np.inf], 0)
        
        return self
    
    def random_forest_importance(self):
        """Use Random Forest to identify important features"""
        print("\n🌲 Random Forest Feature Importance:")
        
        X = self.df[self.feature_cols]
        y = self.df['difficulty_score']
        
        # Use regression since we have continuous target
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_scores = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest'] = importance_scores
        
        print(f"   ✓ Top 10 important features:")
        for _, row in importance_scores.head(10).iterrows():
            print(f"     {row['feature']}: {row['importance']:.3f}")
        
        return self
    
    def correlation_analysis(self):
        """Analyze correlation with difficulty score"""
        print("\n📊 Correlation Analysis:")
        
        correlations = []
        for col in self.feature_cols:
            corr = self.df[col].corr(self.df['difficulty_score'])
            if not np.isnan(corr):
                correlations.append({'feature': col, 'correlation': abs(corr)})
        
        correlation_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        self.feature_importance['correlation'] = correlation_df
        
        print(f"   ✓ Top 10 correlated features:")
        for _, row in correlation_df.head(10).iterrows():
            print(f"     {row['feature']}: {row['correlation']:.3f}")
        
        return self
    
    def statistical_significance(self):
        """Test statistical significance of features"""
        print("\n📈 Statistical Significance Testing:")
        
        X = self.df[self.feature_cols]
        y = self.df['difficulty_score']
        
        # Use F-regression for continuous target
        f_scores, p_values = f_regression(X, y)
        
        significance_scores = pd.DataFrame({
            'feature': self.feature_cols,
            'f_score': f_scores,
            'p_value': p_values
        }).sort_values('f_score', ascending=False)
        
        # Only keep statistically significant features (p < 0.05)
        significant_features = significance_scores[significance_scores['p_value'] < 0.05]
        
        self.feature_importance['statistical'] = significant_features
        
        print(f"   ✓ {len(significant_features)} statistically significant features (p < 0.05)")
        print(f"   ✓ Top 10 by F-score:")
        for _, row in significant_features.head(10).iterrows():
            print(f"     {row['feature']}: F={row['f_score']:.1f}, p={row['p_value']:.3e}")
        
        return self
    
    def recursive_feature_elimination(self):
        """Use RFE to select optimal feature subset"""
        print("\n🔄 Recursive Feature Elimination:")
        
        X = self.df[self.feature_cols]
        y = self.df['difficulty_score']
        
        # Use linear regression as base estimator
        estimator = LinearRegression()
        
        # Try different numbers of features
        best_score = -np.inf
        best_n_features = 10
        
        for n_features in [5, 10, 15, 20]:
            rfe = RFE(estimator, n_features_to_select=n_features)
            rfe.fit(X, y)
            
            # Get selected features
            selected = X.columns[rfe.support_].tolist()
            
            # Evaluate performance
            X_selected = X[selected]
            estimator.fit(X_selected, y)
            y_pred = estimator.predict(X_selected)
            score = r2_score(y, y_pred)
            
            print(f"   {n_features} features: R² = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_n_features = n_features
                self.selected_features = selected
        
        print(f"   ✓ Best: {best_n_features} features (R² = {best_score:.3f})")
        print(f"   ✓ Selected features: {self.selected_features}")
        
        return self
    
    def create_feature_ranking(self):
        """Combine all methods to create final feature ranking"""
        print("\n🏆 Creating Final Feature Ranking:")
        
        # Combine rankings from different methods
        all_features = set()
        for method_name, importance_df in self.feature_importance.items():
            all_features.update(importance_df['feature'].tolist())
        
        # Score each feature based on rankings
        feature_scores = {}
        
        for feature in all_features:
            score = 0
            
            # Random Forest ranking (normalized)
            if 'random_forest' in self.feature_importance:
                rf_df = self.feature_importance['random_forest']
                if feature in rf_df['feature'].values:
                    rank = rf_df[rf_df['feature'] == feature].index[0] + 1
                    score += (len(rf_df) - rank + 1) / len(rf_df) * 3  # Weight: 3
            
            # Correlation ranking (normalized)
            if 'correlation' in self.feature_importance:
                corr_df = self.feature_importance['correlation']
                if feature in corr_df['feature'].values:
                    rank = corr_df[corr_df['feature'] == feature].index[0] + 1
                    score += (len(corr_df) - rank + 1) / len(corr_df) * 2  # Weight: 2
            
            # Statistical significance (binary + F-score weight)
            if 'statistical' in self.feature_importance:
                stat_df = self.feature_importance['statistical']
                if feature in stat_df['feature'].values:
                    f_score = stat_df[stat_df['feature'] == feature]['f_score'].iloc[0]
                    score += min(f_score / 100, 1.0) * 2  # Weight: 2, capped at 1
            
            # RFE selection bonus
            if feature in self.selected_features:
                score += 2  # Bonus: 2
            
            feature_scores[feature] = score
        
        # Create final ranking
        final_ranking = pd.DataFrame([
            {'feature': feature, 'combined_score': score}
            for feature, score in feature_scores.items()
        ]).sort_values('combined_score', ascending=False)
        
        self.final_ranking = final_ranking
        
        print(f"   ✓ Final feature ranking created")
        print(f"   ✓ Top 15 features:")
        for _, row in final_ranking.head(15).iterrows():
            print(f"     {row['feature']}: {row['combined_score']:.2f}")
        
        return self
    
    def validate_selected_features(self):
        """Validate the selected features by predicting difficulty scores"""
        print("\n✅ Validating Selected Features:")
        
        # Use top 12 features for final model
        top_features = self.final_ranking.head(12)['feature'].tolist()
        
        X = self.df[top_features]
        y = self.df['difficulty_score']
        
        # Train a simple linear model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = np.mean(np.abs(y - y_pred))
        
        print(f"   ✓ R-squared: {r2:.3f}")
        print(f"   ✓ RMSE: {rmse:.3f}")
        print(f"   ✓ MAE: {mae:.3f}")
        
        # Show feature coefficients
        coefficients = pd.DataFrame({
            'feature': top_features,
            'coefficient': model.coef_
        }).sort_values('coefficient', ascending=False, key=abs)
        
        print(f"   ✓ Feature coefficients:")
        for _, row in coefficients.iterrows():
            print(f"     {row['feature']}: {row['coefficient']:.3f}")
        
        return self
    
    def save_results(self):
        """
        Save feature selection results to organized output structure.
        
        Returns:
            pd.DataFrame: Enhanced dataset with selected features
        """
        # Save feature ranking to organized directory
        ranking_path = os.path.join(self.output_path, 'feature_ranking.csv')
        self.final_ranking.to_csv(ranking_path, index=False)
        
        # Save enhanced dataset with validated features
        top_features = self.final_ranking.head(15)['feature'].tolist()
        essential_cols = ['company_id', 'flight_number', 'scheduled_departure_date_local',
                         'scheduled_departure_station_code', 'scheduled_arrival_station_code',
                         'difficulty_score'] + top_features
        
        enhanced_df = self.df[essential_cols]
        enhanced_path = os.path.join(self.output_path, 'enhanced_flight_features.csv')
        enhanced_df.to_csv(enhanced_path, index=False)
        
        # Also save to legacy outputs folder for backward compatibility
        Path('outputs').mkdir(exist_ok=True)
        legacy_ranking = 'outputs/feature_ranking.csv'
        legacy_enhanced = 'outputs/enhanced_flight_features.csv'
        self.final_ranking.to_csv(legacy_ranking, index=False)
        enhanced_df.to_csv(legacy_enhanced, index=False)
        
        print(f"\n✅ FEATURE SELECTION COMPLETE!")
        print(f"📁 Primary outputs:")
        print(f"   • Feature ranking: {ranking_path}")
        print(f"   • Enhanced dataset: {enhanced_path}")
        print(f"📁 Legacy outputs:")
        print(f"   • Feature ranking: {legacy_ranking}")
        print(f"   • Enhanced dataset: {legacy_enhanced}")
        print(f"📊 Final dataset: {len(enhanced_df)} flights, {len(enhanced_df.columns)} columns")
        
        return enhanced_df

def main():
    """Main execution function"""
    print("🚀 DATA-DRIVEN FEATURE SELECTION")
    print("="*50)
    
    # Initialize and run feature selection
    selector = DataDrivenFeatureSelector()
    
    # Execute pipeline
    result_df = (selector
                .prepare_features()
                .random_forest_importance()
                .correlation_analysis()
                .statistical_significance()
                .recursive_feature_elimination()
                .create_feature_ranking()
                .validate_selected_features()
                .save_results())
    
    return result_df

if __name__ == "__main__":
    main()