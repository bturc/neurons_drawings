import numpy as np
import temporal

from matplotlib import pyplot as plt


image = np.zeros((64, 64))
main_nodes = (9, 55)
min_nodes, max_nodes = main_nodes[0], main_nodes[1]
min_array, max_array = np.asarray((min_nodes, min_nodes)), np.asarray((max_nodes, max_nodes))

fibre_rand = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                               "pos": [np.zeros((1, 2)) + min_array,
                                                       image.shape - max_array],
                                               "scale": (1, 5)}, seed=None)

roi = ((0, 0), image.shape)
ensemble_b4_fatten = temporal.Ensemble(roi=roi)
ensemble_b4_fatten.append(fibre_rand)
frame_before_fatten = ensemble_b4_fatten.return_frame()

base_rows, base_cols = fibre_rand.return_shape(image.shape)
base_pos = np.stack((base_rows, base_cols), axis=-1)

fibre_rand.fatten(base_pos, image.shape, n_fatten=5)

ensemble_after_fatten = temporal.Ensemble(roi=roi)
ensemble_after_fatten.append(fibre_rand)

frame_after_fatten = ensemble_after_fatten.return_frame()

fig, axes = plt.subplots(1, 2)

axes[0].imshow(frame_before_fatten)
axes[0].set_title(f"before fattening")

axes[1].imshow(frame_after_fatten)
axes[1].set_title(f"after fattening 5x")

plt.show()
