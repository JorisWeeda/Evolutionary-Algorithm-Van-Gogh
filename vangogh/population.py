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
        points = []

        if self.initialization == "RANDOM":
            for i in range(l):
                init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                                        high=feature_intervals[i][1], size=n)
                self.genes[:, i] = init_feat_i
            print("len: ", self.genes.shape)
            print("len tot: ", len(self.genes[0]))

        elif self.initialization == "RANDOM_GRADIENT":
            
            gray_image = REFERENCE_IMAGE.convert("L")
            image_array = np.array(gray_image)

            # Define the size of the cells
            cell_size_x = 20
            cell_size_y = 20

            # Calculate the number of cells
            num_cells_x = image_array.shape[1] // cell_size_x
            num_cells_y = image_array.shape[0] // cell_size_y

            # Function to calculate the gradient of an image cell
            def calculate_gradients(cell):
                gy, gx = np.gradient(cell)
                magnitude = np.sqrt(gx**2 + gy**2)
                return np.mean(magnitude)

            # Calculate the average gradients for each cell
            gradients = []
            for i in range(num_cells_x):
                for j in range(num_cells_y):
                    cell = image_array[j*cell_size_y:(j+1)*cell_size_y, i*cell_size_x:(i+1)*cell_size_x]
                    avg_gradient = calculate_gradients(cell)
                    gradients.append((avg_gradient, (i, j)))

            # Sort cells by gradient
            gradients.sort(reverse=True, key=lambda x: x[0])
            top_25_percent_index = int(0.25 * len(gradients))
            bottom_75_percent_index = len(gradients) - top_25_percent_index

            top_cells = gradients[:top_25_percent_index]
            bottom_cells = gradients[top_25_percent_index:bottom_75_percent_index]

            # Allocate points based on gradient
            num_top_points = int(0.75 * n)
            num_bottom_points = n - num_top_points

            top_cells_points = np.random.choice(len(top_cells), num_top_points, replace=True)
            bottom_cells_points = np.random.choice(len(bottom_cells), num_bottom_points, replace=True)

            for idx in top_cells_points:
                i, j = top_cells[idx][1]
                x = np.random.randint(i*cell_size_x, (i+1)*cell_size_x)
                y = np.random.randint(j*cell_size_y, (j+1)*cell_size_y)
                points.append((x, y))
            
            for idx in bottom_cells_points:
                i, j = bottom_cells[idx][1]
                x = np.random.randint(i*cell_size_x, (i+1)*cell_size_x)
                y = np.random.randint(j*cell_size_y, (j+1)*cell_size_y)
                points.append((x, y))

            points = np.array(points)
            self.genes[:, 0] = points[:, 0]
            self.genes[:, 1] = points[:, 1]
        
        elif self.initialization == "FULL_SAMPLE":
            max_colors = 100000
            imgcolors = REFERENCE_IMAGE.getcolors(max_colors)
            colors = [color for count, color in imgcolors]
            gray_image = REFERENCE_IMAGE.convert("L")
            image_array = np.array(gray_image)

            # Define the size of the cells
            cell_size_x = 20
            cell_size_y = 20

            # Calculate the number of cells
            num_cells_x = image_array.shape[1] // cell_size_x
            num_cells_y = image_array.shape[0] // cell_size_y

            # Function to calculate the gradient of an image cell
            def calculate_gradients(cell):
                gy, gx = np.gradient(cell)
                magnitude = np.sqrt(gx**2 + gy**2)
                return np.mean(magnitude)

            # Calculate the average gradients for each cell
            gradients = []
            for i in range(num_cells_x):
                for j in range(num_cells_y):
                    cell = image_array[j*cell_size_y:(j+1)*cell_size_y, i*cell_size_x:(i+1)*cell_size_x]
                    avg_gradient = calculate_gradients(cell)
                    gradients.append((avg_gradient, (i, j)))

            # Sort cells by gradient
            gradients.sort(reverse=True, key=lambda x: x[0])
            top_25_percent_index = int(0.25 * len(gradients))
            bottom_75_percent_index = len(gradients) - top_25_percent_index

            top_cells = gradients[:top_25_percent_index]
            bottom_cells = gradients[top_25_percent_index:bottom_75_percent_index]

            # Allocate points based on gradient
            num_top_points = int(0.75 * n)
            num_bottom_points = n - num_top_points

            top_cells_points = np.random.choice(len(top_cells), num_top_points, replace=True)
            bottom_cells_points = np.random.choice(len(bottom_cells), num_bottom_points, replace=True)
            col = np.array([])
            for idx in top_cells_points:
                i, j = top_cells[idx][1]
                x = np.random.randint(i*cell_size_x, (i+1)*cell_size_x)
                y = np.random.randint(j*cell_size_y, (j+1)*cell_size_y)
                sampled_indices = np.random.choice(len(colors))
                r, g, b = colors[sampled_indices]
                col = np.append(col, [x,y,r,g,b])
            
            for idx in bottom_cells_points:
                i, j = bottom_cells[idx][1]
                x = np.random.randint(i*cell_size_x, (i+1)*cell_size_x)
                y = np.random.randint(j*cell_size_y, (j+1)*cell_size_y)
                sampled_indices = np.random.choice(len(colors))
                r, g, b = colors[sampled_indices]
                col = np.append(col, [x,y,r,g,b])

            points = np.array(points)
            self.genes[i, :] = col
            
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
                    x,y = np.random.randint(low=feature_intervals[j][0],
                                                            high=feature_intervals[j][1], size=2)
                    col = np.append(col, [x,y,r,g,b])
                
                self.genes[i, :] = col
            print("len: ", self.genes.shape)
            print("len tot: ", len(self.genes[0]))
        else:
            raise Exception("Unknown initialization method")
        return points

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
