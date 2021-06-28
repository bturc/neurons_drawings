import numpy as np
import temporal

from matplotlib import pyplot as plt

image = np.zeros((64,64))
main_nodes = (9, 55)
min_nodes, max_nodes = main_nodes[0], main_nodes[1]
min_array, max_array = np.asarray((min_nodes, min_nodes)), np.asarray((max_nodes, max_nodes))

fibre_rand = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                               "pos": [np.zeros((1, 2)) + min_array,
                                                       image.shape - max_array],
                                               "scale": (1, 5)}, seed=27)

rows, cols = fibre_rand.return_shape(image.shape)

# for row, col in zip(rows, cols):
#     image[row, col] = 1
image[rows, cols] = 1

plt.imshow(image)
plt.title(f"before fattening")
plt.show()

base_rows, base_cols = fibre_rand.return_shape(image.shape)
base_pos = np.stack((base_rows, base_cols), axis=-1)

fibre_rand.fatten(base_pos, image.shape, n_fatten=5)

roi = ((0, 0), image.shape)
ensemble = temporal.Ensemble(roi=roi)
ensemble.append(fibre_rand)

xd = ensemble.return_frame()

plt.imshow(xd)
plt.title(f"Ensemble")
plt.show()
