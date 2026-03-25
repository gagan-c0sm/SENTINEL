import pandas as pd
from sqlalchemy import create_engine
import sys, os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config.settings import get_settings

def run_exploratory_data_analysis():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    print("\n=============================================")
    print("   SENTINEL EXTENSIVE DATA EXPLORATION       ")
    print("=============================================\n")
    
    # 1. Overall volume
    print("--- 1. OVERALL DATA VOLUME BY TYPE ---")
    volume_query = """
        SELECT type, type_name, COUNT(*) as total_rows 
        FROM raw.eia_region_data 
        GROUP BY type, type_name
        ORDER BY total_rows DESC;
    """
    try:
        print(pd.read_sql(volume_query, engine).to_string(index=False))
    except Exception as e:
        print(f"Error querying Volume: {e}")

    # 2. Look for impossible zero/negative demand
    # Sometimes sensors die and report 0 MW, or reverse polarity as negative.
    print("\n--- 2. CRITICAL ANOMALIES: 0 MW OR NEGATIVE DEMAND ---")
    zero_query = """
        SELECT respondent, COUNT(*) as anomaly_count
        FROM raw.eia_region_data 
        WHERE type='D' AND value <= 0
        GROUP BY respondent
        ORDER BY anomaly_count DESC
        LIMIT 10;
    """
    try:
        df_zero = pd.read_sql(zero_query, engine)
        if df_zero.empty:
             print("Excellent! No zero or negative demand values found.")
        else:
             print("Found critical anomalies (0 or negative MW):")
             print(df_zero.to_string(index=False))
    except Exception as e:
        print(f"Error querying Zeros: {e}")

    # 3. Time Series Gaps (Missing Hours)
    print("\n--- 3. MISSING HOURS ANALYSIS (TEXAS GRID) ---")
    try:
        erco_query = "SELECT period, value FROM raw.eia_region_data WHERE respondent='ERCO' AND type='D' ORDER BY period ASC"
        df_erco = pd.read_sql(erco_query, engine)
        df_erco['period'] = pd.to_datetime(df_erco['period'], utc=True)
        df_erco = df_erco.set_index('period')
        
        # Resample to strict 1H intervals and find where values became NaN
        full_grid = df_erco.resample('1h').asfreq()
        missing_hours = full_grid['value'].isna().sum()
        total_hours = len(full_grid)
        missing_pct = (missing_hours / total_hours) * 100
        
        print(f"Texas (ERCO) Total Expected Hours (2021-2026): {total_hours:,}")
        print(f"Missing Hours Found: {missing_hours:,} ({missing_pct:.2f}% of timeline)")
        print(f"This mathematically proves we must use interpolation in the clean.py script.")
    except Exception as e:
        print(f"Error querying Texas Gaps: {e}")

    # 4. Spike Extremes (Demand > 130% of Forecast)
    print("\n--- 4. EXTREME FORECAST MISSES (SURPRISE METRIC) ---")
    spike_query = """
        SELECT d.respondent, d.period, d.value as actual_demand, f.value as forecasted_demand, 
               (d.value - f.value) as surprise_mw,
               ROUND(CAST(((d.value / NULLIF(f.value, 0)) - 1) * 100 AS numeric), 2) as error_pct
        FROM raw.eia_region_data d
        JOIN raw.eia_region_data f 
          ON d.period = f.period AND d.respondent = f.respondent
        WHERE d.type='D' AND f.type='DF' 
          AND d.value > (f.value * 1.30)  -- Actual demand was 30%+ higher than expected
        ORDER BY error_pct DESC
        LIMIT 5;
    """
    try:
        df_spikes = pd.read_sql(spike_query, engine)
        if df_spikes.empty:
             print("No extreme mathematical misses >30% found in the sample.")
        else:
             print("Found the top 5 largest prediction blunders by grid operators:")
             print(df_spikes.to_string(index=False))
    except Exception as e:
        print(f"Error querying Spikes: {e}")

if __name__ == "__main__":
    run_exploratory_data_analysis()
