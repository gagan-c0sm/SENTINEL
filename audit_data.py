"""
SENTINEL Data Integrity Audit — Fixed Connection Pooling
Cross-references orphaned staging tables against the master interchange hypertable.
"""
from sqlalchemy import create_engine, text
from src.config.settings import get_settings

def check_integrity():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    bas = ['AECI', 'AVA', 'CISO', 'ERCO', 'ISNE', 'TVA', 'LGEE', 'BPAT', 'FPL', 'NEVP', 'SWPP',
           'PJM', 'MISO', 'NYIS', 'SOCO', 'DUK', 'SWPP', 'PACW', 'SC', 'SCEG', 'SEC', 'PACE', 'PSCO']

    print("\n" + "=" * 85)
    print(f"{'BA CODE':<12} | {'MASTER ROWS':>12} | {'STAGING TABLE':<18} | {'STATUS'}")
    print("-" * 85)

    # Single connection — no pool exhaustion
    with engine.connect() as conn:
        for ba in bas:
            try:
                master_cnt = conn.execute(
                    text("SELECT count(*) FROM raw.eia_interchange_data WHERE respondent = :ba"),
                    {"ba": ba}
                ).scalar()

                staging_exists = conn.execute(
                    text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'raw' AND table_name = :t"),
                    {"t": f"temp_csv_{ba}"}
                ).scalar()
                # Also check lowercase variant
                staging_exists_lower = conn.execute(
                    text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'raw' AND table_name = :t"),
                    {"t": f"temp_csv_{ba.lower()}"}
                ).scalar()

                has_staging = bool(staging_exists or staging_exists_lower)
                status = "[OK] IN DB" if master_cnt > 0 else "[!!] MISSING"
                staging_str = "EXISTS" if has_staging else "CLEANED"

                print(f"{ba:<12} | {master_cnt:>12,} | {staging_str:<18} | {status}")

            except Exception as e:
                print(f"{ba:<12} | ERROR: {str(e)[:50]}")

    print("=" * 85)

    # Timezone Analysis
    print("\n--- TIMEZONE ANALYSIS ---")
    with engine.connect() as conn:
        sample = conn.execute(
            text("SELECT period, respondent, fromba, toba FROM raw.eia_interchange_data LIMIT 5")
        ).fetchall()
        for row in sample:
            print(f"  period={row[0]}  respondent={row[1]}  fromba={row[2]}  toba={row[3]}")

        # Check the raw timezone info stored in Postgres
        tz_info = conn.execute(
            text("SELECT period, pg_typeof(period), period AT TIME ZONE 'UTC' as utc_time, period AT TIME ZONE 'US/Eastern' as eastern FROM raw.eia_interchange_data WHERE respondent = 'PJM' LIMIT 3")
        ).fetchall()
        print("\nPJM (East Coast) Timezone Probe:")
        for row in tz_info:
            print(f"  Stored: {row[0]} | Type: {row[1]} | As UTC: {row[2]} | As Eastern: {row[3]}")

        tz_info2 = conn.execute(
            text("SELECT period, period AT TIME ZONE 'UTC' as utc_time, period AT TIME ZONE 'US/Central' as central FROM raw.eia_interchange_data WHERE respondent = 'ERCO' LIMIT 3")
        ).fetchall()
        print("\nERCO/Texas (Central) Timezone Probe:")
        for row in tz_info2:
            print(f"  Stored: {row[0]} | As UTC: {row[1]} | As Central: {row[2]}")

        tz_info3 = conn.execute(
            text("SELECT period, period AT TIME ZONE 'UTC' as utc_time, period AT TIME ZONE 'US/Pacific' as pacific FROM raw.eia_interchange_data WHERE respondent = 'CISO' LIMIT 3")
        ).fetchall()
        print("\nCISO/California (Pacific) Timezone Probe:")
        for row in tz_info3:
            print(f"  Stored: {row[0]} | As UTC: {row[1]} | As Pacific: {row[2]}")

if __name__ == "__main__":
    check_integrity()
