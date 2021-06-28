import numpy
import random
import temporal


from scipy.spatial.distance import cdist


def generate_secondary_fibers(datamap_shape, main_fiber, n_sec, min_dist=10, sec_len=(2, 6), seed=None):
    """
    This function allows to spawn secondary fibers branching from a main fiber
    :param datamap_shape: The shape of the datamap in which the main fiber resides
    :param main_fiber: The main fiber object to which we will add secondary fiber branches
    :param n_sec: The interval for the number of secondary branches we wish to spawn (tuple)
    :param min_dist: The min distance between spawned secondary fiber, to ensure they are not all clumped
    :param sec_len: The interval for the length of the secondary fibers (tuple)
    :param seed: Random number generator seed
    :return: a list containing the secondary fiber objects
    """
    n_added = 0
    sec_fiber_positions = numpy.empty((0, 2))
    angle_at_position = []
    n_secondary = int(random.uniform(*n_sec))
    n_loops = 0
    while n_added != n_secondary:
        n_loops += 1
        if n_loops >= 100 * n_secondary:
            break
        # sampled_node = numpy.asarray(random.sample(list(main_fiber.nodes_position), 1)[0].astype(int))
        sample_idx = numpy.random.randint(len(main_fiber.nodes_position))
        sampled_node = main_fiber.nodes_position[sample_idx, :].astype(int)
        if numpy.less_equal(sampled_node, 0).any() or \
                numpy.greater_equal(sampled_node, datamap_shape - numpy.ones((1, 1))).any():
            continue
        if n_added == 0:
            sec_fiber_positions = numpy.append(sec_fiber_positions, sampled_node)
            sec_fiber_positions = numpy.expand_dims(sec_fiber_positions, 0).astype(int)
            angle_at_position.append(main_fiber.angles[sample_idx])
            n_added += 1
            continue
        else:
            sample_to_verify = numpy.expand_dims(numpy.copy(sampled_node), axis=0).astype(int)
            sec_fiber_positions = numpy.append(sec_fiber_positions, sample_to_verify, axis=0).astype(int)
            distances = cdist(sec_fiber_positions, sec_fiber_positions)
            distances[n_added, n_added] = min_dist + 1
            if numpy.less_equal(distances[n_added, :], min_dist).any():
                # at least 1 elt is closer than 10 pixels to an already present elt so remove it :)
                sec_fiber_positions = numpy.delete(sec_fiber_positions, n_added, axis=0)
            else:
                # good to add to the list
                angle_at_position.append(main_fiber.angles[sample_idx])
                n_added += 1

    sec_fibers_list = []
    for node in sec_fiber_positions:
        sec_fiber = temporal.Fiber(random_params={"num_points": sec_len,
                                                  "pos": [node, node],
                                                  "scale": (1, 3),
                                                  "angle": (- 0.25, 0.25)}, seed=seed)
        sec_fibers_list.append(sec_fiber)

    return sec_fibers_list


def generate_synapses_on_fiber(datamap_shape, main_fiber, n_syn, min_dist, synapse_scale=(5, 10)):
    """
    Generates polygon objects (representing synapses) on the main fiber
    *** add a seed? ***
    :param datamap_shape: The shape of the datamap on which the main_fiber lies
    :param main_fiber: Fiber object representing the main branch
    :param n_syn: The interval from which we will sample the number of synapses to spawn (tuple)
    :param min_dist: The minimal distance between 2 synapses, to prevent clumping
    :param synapse_scale: The interval form which we will sample each synapses' size
    :return: A list containing all the synapses on the main fiber
    """
    n_added = 0
    synapse_positions = numpy.empty((0, 2))
    n_synapses = int(random.uniform(*n_syn))
    n_loops = 0
    while n_added != n_synapses:
        n_loops += 1
        if n_loops >= 100 * n_synapses:
            break
        sampled_node = numpy.asarray(random.sample(list(main_fiber.nodes_position), 1)[0].astype(int))
        if numpy.less_equal(sampled_node, 0).any() or \
                numpy.greater_equal(sampled_node, datamap_shape - numpy.ones((1, 1))).any():
            continue
        if n_added == 0:
            synapse_positions = numpy.append(synapse_positions, sampled_node)
            synapse_positions = numpy.expand_dims(synapse_positions, 0).astype(int)
            n_added += 1
            continue
        # comparer la distance du point samplé à tous les points dans la liste
        # vérifier qu'elle est plus grande que min_distance pour tous les points déjà présents,
        # si c'est le cas, l'ajouter à la liste, sinon continuer le while :)
        else:
            sample_to_verify = numpy.expand_dims(numpy.copy(sampled_node), axis=0).astype(int)
            synapse_positions = numpy.append(synapse_positions, sample_to_verify, axis=0).astype(int)
            distances = cdist(synapse_positions, synapse_positions)
            distances[n_added, n_added] = min_dist + 1
            if numpy.less_equal(distances[n_added, :], min_dist).any():
                # at least 1 elt is closer than 10 pixels to an already present elt so remove it :)
                synapse_positions = numpy.delete(synapse_positions, n_added, axis=0)
            else:
                # good to add to the list
                n_added += 1

    synapse_list = []
    for node in synapse_positions:
        polygon = temporal.Polygon(random_params={"pos": [node, node],
                                                  "scale": synapse_scale})
        synapse_list.append(polygon)

    return synapse_list


def generate_synaptic_fibers(image_shape, main_nodes, n_sec_fibers, n_synapses, min_fiber_dist=3, min_synapse_dist=1,
                             sec_fiber_len=(10, 20), synapse_scale=(5, 5), seed=None):
    """
    This function wraps up the generation of fibers with secondary branches and synapses in a neat little package :)
    - Add variable number of synapses, distances, ?
    - Add "position identifiers" to the synapses so I can easily make them flash after
    - ??
    :param image_shape: The shape of the ROI in which we want to spawn stuff
    :param main_nodes: ???
    :param n_sec_fibers: The interval for the number of secondary fibers branching from the main fiber (tuple)
    :param n_synapses: The interval for the number of synapses (tuple)
    :param min_fiber_dist: The minimum distance separating the secondary fibers
    :param min_synapse_dist: The minimum distance separating the synapses
    :param sec_fiber_len: The interval for the lengths of the secondary fibers
    :param synapse_scale: The interval for the size of the synapses
    :param seed: Random number generator seed
    :return: An array containing the disposition of molecules corresponding to the generated shape and a list
             containing all the synapses (Polygon objects)
    """
    # generate an empty image
    image = numpy.zeros(image_shape)

    # generate the main fiber
    min_nodes, max_nodes = main_nodes[0], main_nodes[1]
    min_array, max_array = numpy.asarray((min_nodes, min_nodes)), numpy.asarray((max_nodes, max_nodes))
    fibre_rand = temporal.Fiber(random_params={"num_points": (min_nodes, max_nodes),
                                               "pos": [numpy.zeros((1, 2)) + min_array,
                                                       image.shape - max_array],
                                               "scale": (1, 5)}, seed=seed)

    # generate secondary fibers
    # ces params là devraient être ajustables
    sec_fibers = generate_secondary_fibers(image_shape, fibre_rand, n_sec_fibers, min_fiber_dist,
                                           sec_len=sec_fiber_len, seed=seed)

    # generate synapses attached to the secondary fibers
    synapses_lists = []
    for secondary_fiber in sec_fibers:
        ith_fiber_synapses = generate_synapses_on_fiber(image_shape, secondary_fiber, n_synapses, min_synapse_dist,
                                                        synapse_scale=synapse_scale)
        synapses_lists.append(ith_fiber_synapses)

    roi = ((0, 0), image_shape)  # jtrouve que la façon de gérer la shape de l'ensemble est weird
    ensemble_test = temporal.Ensemble(roi=roi)
    ensemble_test.append(fibre_rand)
    for idx, sec_fiber in enumerate(sec_fibers):
        ensemble_test.append(sec_fiber)
        for synapse in synapses_lists[idx]:
            ensemble_test.append(synapse)

    # frame = ensemble_test.return_frame()

    return ensemble_test, synapses_lists


def n_fatten_dpxsz_converter():
    """
    this func uses the datamap_pixelsize and the desired fiber width to figure out the number of times the fiber needs
    to be fattened. The fiber is initially created with a width of 1 pixel which needs to be adjusted based on
    :return:
    """
    pass
