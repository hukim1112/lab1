import numpy as np

continuous_sample_points = np.linspace(-1.0, 1.0, 10)
continuous_code = []
for _ in range(5):
    cur_sample = np.random.normal(size=[1, 3])
    continuous_code.extend([cur_sample] * 10)
s = np.tile(continuous_sample_points, 5)
continuous_code = np.concatenate(continuous_code)

print(s)

continuous_code[:, 0] = s
print(continuous_code)

