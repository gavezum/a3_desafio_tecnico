import pandas as pd

def read_data_frame(file_name, version):
    df = pd.read_csv(f'data/processado/{file_name}_{version}.csv', index_col = 'customerID')
    return df
