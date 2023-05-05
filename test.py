import numpy as np
from npufuncs import evalcondition

print(evalcondition.evalcondition(np.array([1,0,1]).astype(np.int8), np.array([40,42,17]).astype(np.int8), np.int8(0)))


print(evalcondition.evalcondition(np.array([[1,0,1],[0,0,1]]).astype(np.int8), np.array([[40,42,17], [40,42,17]]).astype(np.int8), np.int8(0)))
