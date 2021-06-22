import numpy as np
import utils
from matplotlib import pyplot as plt

frame_shape = (64, 64)
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5),
                                                              seed=155)
# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
# flat_synapses_list = [item for sublist in synapses_list for item in sublist]

poils_frame = ensemble_func.return_frame().astype(int)

plt.imshow(poils_frame)
plt.show()
