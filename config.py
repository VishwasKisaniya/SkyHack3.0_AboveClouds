"""
🔧 CENTRALIZED CONFIGURATION FOR UNITED AIRLINES FLIGHT DIFFICULTY ANALYSIS
===========================================================================

This file contains all configurable parameters for the flight difficulty analysis system.
Change the USERNAME here and it will be reflected across all modules automatically.

Author: Flight Analysis Team
Version: 1.0
"""

# =============================================================================
# 🎯 MAIN CONFIGURATION - CHANGE USERNAME HERE
# =============================================================================

# Username for output files (CHANGE THIS TO YOUR DESIRED NAME)
USERNAME = "AboveClouds"

# Derived filenames (DO NOT CHANGE - these are automatically generated)
SUBMISSION_FILENAME = f"test_{USERNAME}.csv"
ORGANIZED_OUTPUT_PATH = f"Output_Files/part3_daily_ranking/{SUBMISSION_FILENAME}"
LEGACY_OUTPUT_PATH = f"outputs/{SUBMISSION_FILENAME}"

# =============================================================================
# 📁 FILE PATHS CONFIGURATION
# =============================================================================

# Input data paths
INPUT_DATA_PATHS = {
    'flight_data': 'data/FlightLevelData.csv',
    'pnr_flight_data': 'data/PNRFlightLevelData.csv',
    'pnr_remark_data': 'data/PNRRemarkLevelData.csv',
    'bag_data': 'data/BagLevelData.csv',
    'airports_data': 'data/AirportsData.csv'
}

# Output directory paths
OUTPUT_DIRECTORIES = {
    'part1': 'Output_Files/part1_feature_engineering/',
    'part2': 'Output_Files/part2_feature_selection/',
    'part3': 'Output_Files/part3_daily_ranking/',
    'part4': 'Output_Files/part4_visualization/',
    'part4_enhanced': 'Output_Files/part4_enhanced_visualizations/',
    'eda_analysis': 'Output_Files/eda_analysis/',
    'legacy': 'outputs/'
}

# =============================================================================
# 🎯 SYSTEM CONFIGURATION
# =============================================================================

# Carrier focus
TARGET_CARRIER = "UA"  # United Airlines

# Feature engineering parameters
FEATURE_CONFIG = {
    'min_ground_time_domestic': 35,  # minutes
    'min_ground_time_international': 55,  # minutes
    'peak_hours': [7, 8, 9, 17, 18, 19],
    'hub_airports': ['ORD', 'DEN', 'IAH', 'EWR', 'SFO', 'LAX', 'IAD']
}

# Classification thresholds (percentiles)
DIFFICULTY_THRESHOLDS = {
    'difficult': 0.33,  # 0-33rd percentile
    'medium': 0.67,     # 33rd-67th percentile
    'easy': 1.0         # 67th-100th percentile
}

# =============================================================================
# 📊 VISUALIZATION CONFIGURATION
# =============================================================================

VISUALIZATION_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'font_size': 12,
    'title_size': 16
}

# =============================================================================
# 🔍 UTILITY FUNCTIONS
# =============================================================================

def get_submission_filename():
    """Get the current submission filename"""
    return SUBMISSION_FILENAME

def get_organized_output_path():
    """Get the organized output path"""
    return ORGANIZED_OUTPUT_PATH

def get_legacy_output_path():
    """Get the legacy output path"""
    return LEGACY_OUTPUT_PATH

def get_username():
    """Get the current username"""
    return USERNAME

def print_config_info():
    """Print current configuration information"""
    print("🔧 CURRENT CONFIGURATION")
    print("=" * 50)
    print(f"📝 Username: {USERNAME}")
    print(f"📄 Submission File: {SUBMISSION_FILENAME}")
    print(f"📁 Organized Path: {ORGANIZED_OUTPUT_PATH}")
    print(f"📁 Legacy Path: {LEGACY_OUTPUT_PATH}")
    print(f"✈️ Target Carrier: {TARGET_CARRIER}")
    print("=" * 50)

if __name__ == "__main__":
    print_config_info()