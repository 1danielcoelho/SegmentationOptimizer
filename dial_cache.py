import numpy as np


class DialCache(object):
    max_index = 65535

    def __init__(self):
        self.largest_affinity = 0  # [0, max_index] affinity value
        self.size = 0
        self.main_list = [[] for _ in range(self.max_index+1)]  # List of lists of 1x3 ndarrays (spel positions)
        self.pointer_array = {}  # Maps tuple(spel position) to [0, max_index] affinity values

    def __len__(self):
        """        
        :return: How many spels are in the DialCache 
        """
        return self.size

    def update_largest_index(self):
        """
        Starting at self.largest_affinity and going down in affinity value, looks for
        the first non-empty bucket it can find and sets largest_affinity to its index
        :return: None
        """
        while len(self.main_list[self.largest_affinity]) < 1:
            self.largest_affinity -= 1

    def push(self, spel_pos, affinity):
        """
        Adds a spel to DialCache assigned to the given affinity value
        :param spel_pos: 3-tuple (needs to be hashable) of the spel position pixel coordinates
        :param affinity: affinity value value in the [0.0, 1.0] float range
        :return: None
        """
        affinity = int(round(affinity * DialCache.max_index))

        # We don't need to check if pointer_array contains spel_pos already
        # since that is done in the main loop itself
        self.main_list[affinity].append(spel_pos)
        self.size += 1

        self.pointer_array[spel_pos] = affinity

        self.largest_affinity = max(self.largest_affinity, affinity)

    def pop(self):
        """
        Returns the first spel (FIFO) in the highest affinity bucket 
        :return: 3-tuple of the spel position pixel coordinates  
        """
        spel_to_return = self.main_list[self.largest_affinity].pop(0)
        self.size -= 1

        if len(self.main_list[self.largest_affinity]) == 0 and self.size > 0:
            self.update_largest_index()

        return spel_to_return

    def update_spel(self, spel_pos, new_affinity):
        """
        Updates the affinity value assigned to spel_pos in our internal data structures
        :param spel_pos: 3-tuple (needs to be hashable) of the spel position pixel coordinates
        :param new_affinity: new affinity value value in the [0.0, 1.0] float range
        :return: None
        """
        new_affinity = int(round(new_affinity * DialCache.max_index))
        old_affinity = self.pointer_array[spel_pos]

        if old_affinity == new_affinity:
            return

        self.main_list[old_affinity].remove(spel_pos)
        self.main_list[new_affinity].append(spel_pos)

        self.pointer_array[spel_pos] = new_affinity

        # We only ever update upwards: No need to check if largest_affinity decreased
        self.largest_affinity = max(self.largest_affinity, new_affinity)

    def contains(self, spel_pos):
        """
        Checks if spel_pos is in the DialCache by using an internal
        hash table to keep track of spels
        :param spel_pos: 3-tuple of the spel position pixel coordinates  
        :return: True if spel_pos is already in DialCache
        """
        return spel_pos in self.pointer_array
