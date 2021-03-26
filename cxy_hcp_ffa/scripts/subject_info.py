import numpy as np
import pandas as pd

# inputs
subj_file = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/' \
            'analysis/s2/subject_id'

subj_ids = [int(i) for i in open(subj_file).read().splitlines()]
info1_file = '/nfs/m1/hcp/S1200_behavior_restricted.csv'
info1_df = pd.read_csv(info1_file)

valid_indices = []
for idx in info1_df.index:
    if info1_df.loc[idx, 'Subject'] in subj_ids:
        valid_indices.append(idx)
info1_df = info1_df.loc[valid_indices]
assert info1_df['Subject'].to_list() == subj_ids

print('#Male:', np.sum(info1_df['Gender'] == 'M'))
print('#Female:', np.sum(info1_df['Gender'] == 'F'))
print('Mean age:', np.mean(info1_df['Age_in_Yrs']))
print('Age std:', np.std(info1_df['Age_in_Yrs']))
print(f"Age range: {np.min(info1_df['Age_in_Yrs'])} to "
      f"{np.max(info1_df['Age_in_Yrs'])}")
