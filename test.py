import visualizations
import numpy as np

visualizations.varying_noise_continuous_ndim('hi', 3, 10, 5, 32, 1000, 'ss')

# categorical_sample_points = np.array(range(10))
# categorical_noise = []
# for _ in range(10):
#     cur_sample = np.random.choice(categorical_sample_points)
#     categorical_noise.extend([cur_sample] * 10)
# categorical_noise = np.array(categorical_noise)
# print(categorical_noise)