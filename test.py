import pandas as pd

x = pd.read_pickle('/home/deivison/my_projects/graphic_analysis/input_opencpop/complete_analysis.pkl')

sex = ['F' for y in range(len(x))]

x['Gender'] = sex

x.to_pickle('/home/deivison/my_projects/graphic_analysis/output_opencpop/complete_analysis.pkl')

