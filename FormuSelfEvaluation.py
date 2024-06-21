import numpy as np

weight_proj1 = 25
weight_proj2 = 30
weight_proj3 = 30
weight_self = 15
weight = [weight_proj1, weight_proj2, weight_proj3, weight_self]

scor_proj1 = 140  # SP: 140; HG: 120; SD: 100
scor_proj2 = 120
scor_proj3 = 120
scor_self = 120
scor = [scor_proj1, scor_proj2, scor_proj3, scor_self]

def score(proj_id):
    if np.sum(weight) - 100 == 0:
        final_scor = weight[int(proj_id)] * scor[int(proj_id)] / 100
    else:
        final_scor = 0
    return final_scor


sum = score(0) + score(1) + score(2) + score(3)
print("Final score is: ", sum)