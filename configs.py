import os
import pandas as pd
import numpy as np

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__)) #os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
OUTPUTS_FOLDER = os.path.join(ROOT_FOLDER, "outputs") 
MODEL_FOLDER = os.path.join(OUTPUTS_FOLDER, "models")
DATA_FOLDER = os.path.join(OUTPUTS_FOLDER, "data")

def get_datamap():
    # Load data
    data = pd.read_csv(os.path.join(DATA_FOLDER, 
            "oxfordmanrealizedvolatilityindices.csv"), index_col=0)
    data.index = pd.to_datetime(data.index)

    data.sort_index(inplace=True)

    cols_of_interest = ['open_to_close', 'rv5_ss', 'open_price']
    data['rv5_ss'] = np.log(data['rv5_ss'].replace(0.0, np.nan))
    
    T = len(data)
    split_point = int(T*0.8)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]

    def get_df_map(data):
        df_map = {}

        for name, grp in data.groupby(data['Symbol']):
            for col in cols_of_interest:
                srs = grp[col]
                srs.name = name
                if col not in df_map:
                    df_map[col] = srs.to_frame()
                else:
                    df_map[col][name] = srs
        
        for name in df_map:
            df = df_map[name]
            df = df.fillna(method='ffill').dropna()
            df_map[name] = df
        return df_map

    train_map, test_map = get_df_map(train_data), get_df_map(test_data)
    return train_map, test_map
