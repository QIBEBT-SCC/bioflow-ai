import polars as pl


def read_origin_data():
    origin_df = pl.read_parquet('test_run.parquet')
    print(origin_df)
    print(origin_df.filter(pl.col('test_run_id') == 'tool_52_bedtools').select('question').item())


if __name__ == '__main__':
    read_origin_data()
