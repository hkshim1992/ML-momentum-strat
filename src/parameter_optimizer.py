import numpy as np
import pandas as pd
from config.config import shared_windows

short_window = shared_windows['short_window']
long_window = shared_windows['long_window']

def parameter_optimizer1(input_df, strat):
    ret_list = []  

    for x1, x2, in [(a,b) for a in short_window for b in long_window]:
        df = input_df.copy()
        _, ret = strat(df, x1, x2, verbose=False)
        ret_list.append((x1, x2, ret))

    optimal_params = max(ret_list, key=lambda x:x[2])
    param_results_df = pd.DataFrame(ret_list, \
                              columns=['short_window','long_window','ror'])
    print(f'Optimal Parameters:{optimal_params[0], optimal_params[1]}, '
    f'Optimized Return:{100*optimal_params[2]:.2f}%')

    optimal_df = strat(df, *optimal_params[:2])[0]

    return optimal_params, optimal_df, param_results_df


def parameter_optimizer1b(input_df, strat, **kwargs): 
    ret_list = []  

    for x1, x2, in [(a,b) for a in short_window for b in long_window]:
        df = input_df.copy()
        _, ret = strat(df, x1, x2, **kwargs, verbose=False)
        ret_list.append((x1, x2, ret))

    max_ror = max(ret_list, key=lambda x:x[2])[2]
    max_tups = [tup for tup in ret_list if tup[2] == max_ror]
    params1 = [tup[0] for tup in max_tups]
    params2 = [tup[1] for tup in max_tups]
    opt_param1 = int(np.median(params1))
    opt_param2 = int(np.median(params2))
    
    param_results_df = pd.DataFrame(ret_list, \
                              columns=['short_window','long_window','ror'])
    print(f'Max Tuples:{max_tups}')
    print(f'Optimal Parameters:{opt_param1, opt_param2}, '
    f'Optimized Return:{100*max_ror:.2f}%')

    optimal_df = strat(df, opt_param1, opt_param2, **kwargs)[0]

    return (opt_param1, opt_param2), optimal_df, param_results_df