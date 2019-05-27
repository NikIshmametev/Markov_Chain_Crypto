import pandas as pd
from multiprocessing import Pool
from datetime import timedelta
import time
from EM import f, EM
from names import Names

dir = r'/Users/nik/PycharmProjects/Diploma/Data_daily/All_Coins'
codes = Names.coins


def do_work(asset, returns, start_dates, end_dates):
    arr = []
    for start, end in zip(start_dates, end_dates):
        asset_probs, parameters = {}, {}
        output = EM(returns.loc[(returns.index >= start) & (returns.index <= end)].values)
        parameters[asset] = output[1:-1]
        dist_params = output[1:5]
        asset_probs[asset] = output[0]
        if parameters[asset][2] > parameters[asset][3]:
            new_list = [output[-1][-1]]
            new_list.extend(dist_params)
            arr.append(new_list)
        else:
            new_list = [1 - output[-1][-1]]
            new_list.extend(dist_params)
            arr.append(new_list)

    return {asset: arr}


if __name__ == '__main__':

    # --- Day data ---
    # --- DataFrame df_all will consist all data ---

    df_all = pd.DataFrame()
    for code in codes:
        df = pd.read_csv(dir + '/{code}.csv'.format(code=code), index_col=0)
        cols = df.columns
        cols_new = []
        for col in cols:
            cols_new.append(col + '_{}'.format(code))
        df.columns = cols_new
        df_all = pd.concat([df_all, df], axis=1)

    df_all.index = pd.to_datetime(df_all.index)

    # --- Choose columns for calculating daily returns ---
    new_cols = [col for col in df_all.columns.values if 'open' in col]
    returns = pd.DataFrame()
    for col, code in zip(new_cols, codes):
        returns = pd.concat([returns, df_all[col].pct_change()], axis=1)

    returns.columns = codes
    returns.index = pd.to_datetime(returns.index)
    returns.dropna(inplace=True)

    # --- DataFrame with prices in USD ---
    prices_open = pd.DataFrame()
    for col, code in zip(new_cols, codes):
        prices_open = pd.concat([prices_open, df_all[col]], axis=1)
    prices_open.columns = codes
    prices_open.index = pd.to_datetime(prices_open.index)
    prices_open.dropna(inplace=True)
    prices_open = prices_open.iloc[1:]

    start_time = time.time()

    IS_first_date = '2017-09-03'

    window = 45
    first_date_back = pd.to_datetime(IS_first_date) - timedelta(days=window - 1)
    start_dates = returns.loc[returns.index >= first_date_back].index[:-window + 1]
    end_dates = returns.loc[returns.index >= IS_first_date].index

    pool = Pool(5)
    args, results = [], []
    for asset in codes:
        args.append((asset, returns.loc[:,asset], start_dates, end_dates))

    results = pool.starmap(do_work, args)
    results_dict = {}
    for res in results:
        results_dict.update(res)

    res_df = pd.DataFrame()
    for code in codes:
        arrays = [[code]*len(end_dates), end_dates]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['coin', 'date'])
        temp = pd.DataFrame(results_dict[code], columns=['Prob', 'mu0', 'mu1', 'sig0', 'sig1'], index=index)
        res_df = pd.concat([res_df, temp])

    res_df.to_csv(dir + r'/Probs_{w}_{date}.csv'.format(w   =int(window),
                                                        date=IS_first_date))

    print(time.time() - start_time)
