import polars as pl
import sys

print(f"Polars version: {pl.__version__}")

try:
    # Simulate the sparse data scenario
    data = [
        {'t': 0, 'a': 1},
        {'t': 1, 'a': 2, 'b': 3}
    ]
    
    # Method 1: from_dicts with schema inference
    try:
        df = pl.from_dicts(data, infer_schema_length=None)
        print("Success: from_dicts")
        print(df)
    except Exception as e:
        print(f"Failed: from_dicts: {e}")

    # Method 2: Diagonal concat (Current Fix)
    try:
        row_dfs = [pl.DataFrame([row], infer_schema_length=None) for row in data]
        df = pl.concat(row_dfs, how='diagonal')
        print("Success: diagonal concat")
        print(df)
    except Exception as e:
        print(f"Failed: diagonal concat: {e}")
        import traceback
        traceback.print_exc()

except ImportError:
    pass
