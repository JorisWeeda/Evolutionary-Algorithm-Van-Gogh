from PIL import Image

NUM_VARIABLES_PER_POINT = 5
IMAGE_SHRINK_SCALE = 6

REFERENCE_IMAGE = Image.open(f"./res/starry_night.jpg").convert('RGB')
