import numpy as np
from itertools import combinations
from segmentation_tools import check_ndimage, check_seeds, get_neighbors
from dial_cache import DialCache
from timeit_context import timeit_context


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def relative_difference(c, d):
    return abs(c - d) / (c + d)


def average(c, d):
    return 0.5 * (c + d)


def affinity_one(c, d, mean_ave, sigma_ave, mean_reldiff, sigma_reldiff):
    ave = gaussian(average(c, d), mean_ave, sigma_ave)
    rel = gaussian(relative_difference(c, d), mean_reldiff, sigma_reldiff)
    return min(ave, rel)


class FuzzyConnectedness(object):
    def __init__(self, image, seeds, object_threshold=0.1, num_loops_to_yield=100):
        self.image = image
        self.seeds = seeds
        self.object_threshold = object_threshold
        self.num_loops_to_yield = num_loops_to_yield

    def run(self):
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

        seg = np.zeros(padded.shape, dtype=np.float32)

        # Get neighboring spel positions of all spels
        reference_spels = []
        for s in self.seeds:
            for n in get_neighbors(s):
                reference_spels.append(n)
            reference_spels.append(s)

        # Remove repeats from neighboring positions
        # Curlies turn it into a set (no repeats)
        # Need tuples since lists are unhashable
        no_repeats = {tuple(row) for row in reference_spels}

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

        # Get all ave and reldiff combinations between all base samples
        combs = [c for c in combinations(base_samples, 2)]
        ave_vals = [average(pair[0], pair[1]) for pair in combs]
        reldiff_vals = [relative_difference(pair[0], pair[1]) for pair in combs if pair[0] != -pair[1]]

        # Calculate means and sigmas
        mean_ave = np.mean(ave_vals)
        mean_reldiff = np.mean(reldiff_vals)
        sigma_ave = np.std(ave_vals)
        sigma_reldiff = np.std(reldiff_vals)

        # Initialize DialCache
        dial = DialCache()
        for s in self.seeds:
            seg[tuple(s)] = 1.0
            dial.push(tuple(s), 1.0)

        # Main loop
        itercount = 0
        with timeit_context('Fuzzy Connectedness inner loop'):
            while len(dial) > 0:
                c = dial.pop()
                c_val = padded[c]

                neighs = get_neighbors(c)
                for e in neighs:
                    e_val = padded[e]

                    # No point in marking it as visited: For 6-neighbor, pad spels only ever have 1 valid neighbor
                    if e_val == padding_value:
                        continue

                    aff_c_e = affinity_one(c_val, e_val, mean_ave, sigma_ave, mean_reldiff, sigma_reldiff)

                    if aff_c_e < self.object_threshold:
                        continue

                    f_min = min(seg[c], aff_c_e)
                    if f_min > seg[e]:
                        seg[e] = f_min

                        if dial.contains(e):
                            dial.update_spel(e, f_min)
                        else:
                            dial.push(e, f_min)

                itercount += 1
                if itercount % self.num_loops_to_yield == 0:
                    yield seg[1:-1, 1:-1, 1:-1]

        yield seg[1:-1, 1:-1, 1:-1]
