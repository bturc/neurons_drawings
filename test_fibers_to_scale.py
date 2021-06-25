import numpy as np
import temporal

from matplotlib import pyplot as plt


seed = 27
frame_shape = 64
min_nodes, max_nodes = 9, 55
min_array, max_array = np.asarray((min_nodes, min_nodes)), np.asarray((max_nodes, max_nodes))
image = np.zeros((frame_shape, frame_shape))
print("test!")
fibre_rand = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                               "pos": [np.zeros((1, 2)) + min_array,
                                                       image.shape - max_array],
                                               "scale": (1, 5)}, seed=None)
roi = ((0, 0), image.shape)
ensemble = temporal.Ensemble(roi=roi)
ensemble.append(fibre_rand)

unfattened = ensemble.return_frame()

# fibre_rand.fatten(10)
fibre_rand.fatten_simple(3)
# exit()

fattened = ensemble.return_frame()

nodes_pos = fibre_rand.nodes_position
# print(nodes_pos)
nodes_array = np.zeros((frame_shape, frame_shape))
for node in nodes_pos:
    if 0 <= np.round(node[0]) < frame_shape and 0 <= np.round(node[1]) < frame_shape:
        nodes_array[int(np.round(node[0])), int(np.round(node[1]))] += 1
        # fattened[int(np.round(node[0])), int(np.round(node[1]))] += 1

fig, axes = plt.subplots(1, 2)
axes[0].imshow(unfattened)
axes[0].set_title(f"unfattened")
axes[1].imshow(fattened)
axes[1].set_title(f"fattened")
plt.show()
