import pandas as pd

# ─── DEFINE PATHS RELATIVE TO sentiment-store ROOT ─────────────────────────────
INPUT_PATH  = 'data/06_Features/merged.csv'
OUTPUT_PATH = 'data/06_Features/merged_features.csv'
# ────────────────────────────────────────────────────────────────────────────────

def engineer_features(df):
    # 1) day of week
    df['bin_end_time'] = pd.to_datetime(df['bin_end_time'])
    df['date']         = pd.to_datetime(df['date']).dt.date
    df['day_of_week']  = df['bin_end_time'].dt.dayofweek

    # 2) bin slot number & opening/closing flags
    df = df.sort_values(['ticker', 'date', 'bin_end_time'])
    df['bin_slot_number'] = df.groupby(['ticker','date']).cumcount() + 1
    last_slot = df.groupby(['ticker','date'])['bin_slot_number'].transform('max')
    df['is_opening_bin'] = (df['bin_slot_number'] == 1).astype(int)
    df['is_closing_bin'] = (df['bin_slot_number'] == last_slot).astype(int)

    # 3) lagged features (previous bin)
    df['lagged_return']               = df.groupby('ticker')['bin_return']             .shift(1).fillna(0)
    df['lagged_after_close_compound_mean'] = df.groupby('ticker')['after_close_compound_mean'] .shift(1).fillna(0)
    df['lagged_pre_open_compound_mean']   = df.groupby('ticker')['pre_open_compound_mean']    .shift(1).fillna(0)
    df['lagged_trading_compound_mean']    = df.groupby('ticker')['trading_compound_mean']     .shift(1).fillna(0)

    return df

if __name__ == '__main__':
    print(f"▶ Reading input : {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    df = engineer_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✔ Wrote output : {OUTPUT_PATH}")
