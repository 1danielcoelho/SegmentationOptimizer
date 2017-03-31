import numpy as np
from segmentation_tools import check_ndimage, check_seeds, get_neighbors, timeit_context


class RegionGrowing(object):
    def __init__(self, image, seeds, sigma=1):
        self.image = image
        self.seeds = seeds
        self.sigma = sigma
        self.num_loops_to_yield = 100

    def __call__(self, *args, **kwargs):
        # Sanitize inputs
        try:
            check_seeds(self.seeds)
            check_ndimage(self.image)
        except:
            raise  # raises last exception

        # Pad image and offset seeds
        padding_value = np.iinfo(np.int32).min + 1
        padded = np.lib.pad(self.image, 1, 'constant', constant_values=padding_value)
        self.seeds += np.array([1, 1, 1])

        # By [200, 10, 0] we mean "x=200, y=10, z=0", but if we use this to index, we'll be saying
        # "line 200, column 10, slice 0", which is the opposite of what we want, so we invert here
        for s in self.seeds:
            s[0], s[1] = s[1], s[0]

        explored = np.zeros(padded.shape, dtype=np.bool_)
        seg = np.zeros(padded.shape, dtype=np.bool_)
        queue = []

        # Get neighboring spel positions of all seeds
        base_ground = []
        for s in self.seeds:
            for n in get_neighbors(s):
                base_ground.append(n)
            base_ground.append(s)

        # Remove repeats from neighboring positions
        # Curlies turn it into a set (no repeats)
        # Need tuples since lists are unhashable
        no_repeats = {tuple(row) for row in base_ground}

        # Convert back to ndarray for fancy indexing
        no_repeats = np.array([list(row) for row in no_repeats])

        # When doing multidimensional indexing, we need to pass all Xs, then Ys, then Zs
        no_repeats_xs = no_repeats[:, 0]
        no_repeats_ys = no_repeats[:, 1]
        no_repeats_zs = no_repeats[:, 2]

        # Grab samples at the neighboring positions
        base_samples = padded[no_repeats_xs, no_repeats_ys, no_repeats_zs]

        # Discard all instances of the padding value
        base_samples = [samp for samp in base_samples if samp != padding_value]

        seg_average = np.mean(base_samples)
        tolerance = self.sigma * max(np.std(base_samples), 1)

        print("Segment values is " + str(seg_average) + " +/- " + str(tolerance))

        for s in self.seeds:
            queue.append(s)
            explored[tuple(s)] = True

        count = 0

        with timeit_context('Region Growing'):
            while len(queue) > 0:
                curr_pos = queue[0]
                queue = queue[1:]

                for n in get_neighbors(curr_pos):
                    tn = tuple(n)

                    n_val = padded[tn]

                    if n_val == padding_value:
                        explored[tn] = True
                        continue

                    if explored[tn]:
                        continue

                    explored[tn] = True

                    # Node belongs: Mark it and queue it to be visited later
                    if abs(n_val - seg_average) < tolerance:
                        seg[tn] = True
                        queue.append(n)

                count += 1
                if count % self.num_loops_to_yield == 0:
                    yield seg[1:-1, 1:-1, 1:-1]

