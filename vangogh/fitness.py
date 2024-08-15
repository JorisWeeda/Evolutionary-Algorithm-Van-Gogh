import numpy as np
from PIL import Image
from imgcompare import image_diff
from multiprocess import Pool, cpu_count
from scipy.spatial import KDTree
from vangogh.util import NUM_VARIABLES_PER_POINT

# Global variable to store query points for Voronoi calculations
QUERY_POINTS = []

def draw_voronoi_matrix(genotype, img_width, img_height, scale=1):
    """
    Generates a Voronoi diagram as a matrix based on the provided genotype.
    
    Args:
        genotype (list): A list representing the genotype of the individual. The genotype is 
                         expected to contain coordinates and color values for each point.
        img_width (int): The width of the output image.
        img_height (int): The height of the output image.
        scale (int, optional): The scale factor for the image. Defaults to 1.

    Returns:
        np.ndarray: A 3D NumPy array representing the RGB values of the generated Voronoi diagram.
    """
    # Scale the image dimensions based on the scale factor
    scaled_img_width = int(img_width * scale)
    scaled_img_height = int(img_height * scale)
    
    # Number of points is determined by the length of the genotype and the number of variables per point
    num_points = int(len(genotype) / NUM_VARIABLES_PER_POINT)
    coords = []
    colors = []

    # Extract coordinates and colors from the genotype
    for r in range(num_points):
        p = r * NUM_VARIABLES_PER_POINT
        x, y, r, g, b = genotype[p:p + NUM_VARIABLES_PER_POINT]
        coords.append((x * scale, y * scale))
        colors.append((r, g, b))

    # Create a KDTree for the Voronoi diagram
    voronoi_kdtree = KDTree(coords)
    
    # If scale is 1, use precomputed query points; otherwise, generate new ones
    if scale == 1:
        query_points = QUERY_POINTS
    else:
        query_points = [(x, y) for x in range(scaled_img_width) for y in range(scaled_img_height)]

    # Query the KDTree to find the nearest Voronoi region for each query point
    _, query_point_regions = voronoi_kdtree.query(query_points)

    # Initialize an empty data array for the image
    data = np.zeros((scaled_img_height, scaled_img_width, 3), dtype='uint8')
    
    # Assign colors to each pixel based on the closest Voronoi region
    i = 0
    for x in range(scaled_img_width):
        for y in range(scaled_img_height):
            for j in range(3):
                data[y, x, j] = colors[query_point_regions[i]][j]
            i += 1

    return data

def draw_voronoi_image(genotype, img_width, img_height, scale=1) -> Image:
    """
    Draws a Voronoi diagram and returns it as an image.
    
    Args:
        genotype (list): A list representing the genotype of the individual.
        img_width (int): The width of the output image.
        img_height (int): The height of the output image.
        scale (int, optional): The scale factor for the image. Defaults to 1.
    
    Returns:
        Image: A PIL Image object representing the Voronoi diagram.
    """
    data = draw_voronoi_matrix(genotype, img_width, img_height, scale)
    img = Image.fromarray(data, 'RGB')
    return img

def compute_difference(genotype, reference_image: Image) -> float:
    """
    Computes the difference between a generated Voronoi image and a reference image.
    
    Args:
        genotype (list): A list representing the genotype of the individual.
        reference_image (Image): A PIL Image object representing the reference image.
    
    Returns:
        float: A value representing the difference between the generated and reference images.
    """
    actual = draw_voronoi_matrix(genotype, reference_image.width, reference_image.height)
    diff = image_diff(Image.fromarray(actual, 'RGB'), reference_image)
    return diff

def worker(args):
    """
    Worker function for parallel processing that sets the global QUERY_POINTS 
    and computes the difference for a single genotype.
    
    Args:
        args (tuple): A tuple containing the genotype, reference image, and query points.
    
    Returns:
        float: The computed difference for the given genotype.
    """
    global QUERY_POINTS
    QUERY_POINTS = args[2]
    return compute_difference(args[0], args[1])

def drawing_fitness_function(genes, reference_image: Image) -> np.ndarray:
    """
    Evaluates the fitness of a population of genotypes by comparing their Voronoi diagrams 
    to a reference image.
    
    Args:
        genes (np.ndarray): A 2D NumPy array where each row is a genotype to be evaluated.
        reference_image (Image): A PIL Image object representing the reference image.
    
    Returns:
        np.ndarray: An array of fitness values corresponding to each genotype.
    """
    # Initialize the QUERY_POINTS if not already done
    if len(QUERY_POINTS) == 0:
        QUERY_POINTS.extend([(x, y) for x in range(reference_image.width) for y in range(reference_image.height)])

    # Use multiprocessing to evaluate the fitness in parallel
    with Pool(min(max(cpu_count() - 1, 1), 4)) as p:
        fitness_values = list(p.map(worker, zip(genes, [reference_image] * genes.shape[0], [QUERY_POINTS] * genes.shape[0])))
    
    return np.array(fitness_values)
