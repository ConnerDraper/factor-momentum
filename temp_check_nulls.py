import polars as pl
import numpy as np

def check_file(path_name):
    df = pl.read_parquet(f'results/full/alphas/{path_name}')
    print(f'\\n{path_name}')
    print('Shape:', df.shape)
    
    # Check for actual nulls
    nulls = df.null_count().sum_horizontal()[0]
    print('Nulls:', nulls)
    
    # Check for NaN / Inf (which are different from polars nulls)
    # The error specifically said: "ValueError: Problem data contains NaN or Inf"
    df_pd = df.to_pandas()
    print('Has NaN:', df_pd.isna().sum().sum())
    print('Has Inf:', np.isinf(df_pd.select_dtypes(include=np.number)).sum().sum())
    
check_file('factor_momentum_style.parquet')
check_file('factor_momentum_industry.parquet')
check_file('factor_momentum_all.parquet')
