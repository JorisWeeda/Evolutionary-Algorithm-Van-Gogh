import numpy as np
from vangogh.util import REFERENCE_IMAGE

class Population:
    def __init__(self, population_size, genotype_length, initialization):
        self.genes = np.empty(shape=(population_size, genotype_length), dtype=int)
        self.fitnesses = np.zeros(shape=(population_size,))
        self.initialization = initialization

    def initialize(self, feature_intervals):
        n = self.genes.shape[0]
        l = self.genes.shape[1]
        print("l:", l)
        print("n:", n)
        if self.initialization == "RANDOM":
            for i in range(l):
                init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                                        high=feature_intervals[i][1], size=n)
                self.genes[:, i] = init_feat_i
            print("len: ", self.genes.shape)
            
        elif self.initialization == "PSEUDORANDOM":
            max_colors = 100000
            imgcolors = REFERENCE_IMAGE.getcolors(max_colors)
            colors = [color for count, color in imgcolors]
            #print(len(colors[0]))
            for i in range(n):
                #init_feat_i = np.random.choice(colors, size=n)
                #self.genes[:, i] = init_feat_i
                col = np.array([])
                
                for j in range(int(l/5)):
                    sampled_indices = np.random.choice(len(colors))
                    # Use the sampled indices to get the actual colors
                    r, g, b = colors[sampled_indices]
                    x,y = np.random.randint(low=feature_intervals[i][0],
                                                            high=feature_intervals[i][1], size=2)
                    col = np.append(col, [x,y,r,g,b])

                self.genes[i, :] = col
        else:
            raise Exception("Unknown initialization method")

    def stack(self, other):
        self.genes = np.vstack((self.genes, other.genes))
        self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

    def shuffle(self):
        random_order = np.random.permutation(self.genes.shape[0])
        self.genes = self.genes[random_order, :]
        self.fitnesses = self.fitnesses[random_order]

    def is_converged(self):
        return len(np.unique(self.genes, axis=0)) < 2

    def delete(self, indices):
        self.genes = np.delete(self.genes, indices, axis=0)
        self.fitnesses = np.delete(self.fitnesses, indices)
