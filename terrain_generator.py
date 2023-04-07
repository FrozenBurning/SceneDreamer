import numpy as np
from scipy.spatial import Voronoi
from skimage.draw import polygon
from PIL import Image
from noise import snoise3
from skimage import exposure
from scipy.interpolate import interp1d
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation

from argparse import ArgumentParser

def save_height_map(height_map, file_name):
    #input height map should be float, raw output of noise map
    normalized_height_map = (((height_map - height_map.min()) / (height_map.max() - height_map.min()))*255).astype(np.uint8)
    cv2.imwrite(file_name, normalized_height_map)
    np.save(file_name[:-4] + '.npy', height_map)

def get_boundary(vor_map, size, kernel=1):
    boundary_map = np.zeros_like(vor_map, dtype=bool)
    n, m = vor_map.shape
    
    clip = lambda x: max(0, min(size-1, x))
    def check_for_mult(a):
        b = a[0]
        for i in range(len(a)-1):
            if a[i] != b: return 1
        return 0
    
    for i in range(n):
        for j in range(m):
            boundary_map[i, j] = check_for_mult(vor_map[
                clip(i-kernel):clip(i+kernel+1),
                clip(j-kernel):clip(j+kernel+1),
            ].flatten())
            
    return boundary_map

def histeq(img,  alpha=1):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    img_eq = np.interp(img, bin_centers, img_cdf)
    img_eq = np.interp(img_eq, (0, 1), (-1, 1))
    return alpha * img_eq + (1 - alpha) * img

def voronoi(points, size):
    # Add points at edges to eliminate infinite ridges
    edge_points = size*np.array([[-1, -1], [-1, 2], [2, -1], [2, 2]])
    new_points = np.vstack([points, edge_points])
    
    # Calculate Voronoi tessellation
    vor = Voronoi(new_points)
    
    return vor

def voronoi_map(vor, size):
    # Calculate Voronoi map
    vor_map = np.zeros((size, size), dtype=np.uint32)

    for i, region in enumerate(vor.regions):
        # Skip empty regions and infinte ridge regions
        if len(region) == 0 or -1 in region: continue
        # Get polygon vertices    
        x, y = np.array([vor.vertices[i][::-1] for i in region]).T
        # Get pixels inside polygon
        rr, cc = polygon(x, y)
        # Remove pixels out of image bounds
        in_box = np.where((0 <= rr) & (rr < size) & (0 <= cc) & (cc < size))
        rr, cc = rr[in_box], cc[in_box]
        # Paint image
        vor_map[rr, cc] = i

    return vor_map

# Lloyd's relaxation
def relax(points, size, k=10):
    new_points = points.copy()
    for _ in range(k):
        vor = voronoi(new_points, size)
        new_points = []
        for i, region in enumerate(vor.regions):
            if len(region) == 0 or -1 in region: continue
            poly = np.array([vor.vertices[i] for i in region])
            center = poly.mean(axis=0)
            new_points.append(center)
        new_points = np.array(new_points).clip(0, size)
    return new_points

def noise_map(size, res, seed, octaves=1, persistence=0.5, lacunarity=2.0):
    scale = size/res
    return np.array([[
        snoise3(
            (x+0.1)/scale,
            y/scale,
            seed,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
        for x in range(size)]
        for y in range(size)
    ])

def average_cells(vor, data):
    """Returns the average value of data inside every voronoi cell"""
    size = vor.shape[0]
    count = np.max(vor)+1

    sum_ = np.zeros(count)
    count = np.zeros(count)

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            count[p] += 1
            sum_[p] += data[i, j]

    average = sum_/ (count + 1e-3)
    average[count==0] = 0

    return average

def fill_cells(vor, data):
    size = vor.shape[0]
    image = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image

def color_cells(vor, data, dtype=int):
    size = vor.shape[0]
    image = np.zeros((size, size, 3))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image.astype(dtype)

def quantize(data, n):
    bins = np.linspace(-1, 1, n+1)
    return (np.digitize(data, bins) - 1).clip(0, n-1)


def bezier(x1, y1, x2, y2, a):
    p1 = np.array([0, 0])
    p2 = np.array([x1, y1])
    p3 = np.array([x2, y2])
    p4 = np.array([1, a])

    return lambda t: ((1-t)**3 * p1 + 3*(1-t)**2*t * p2 + 3*(1-t)*t**2 * p3 + t**3 * p4)

def bezier_lut(x1, y1, x2, y2, a):
    t = np.linspace(0, 1, 256)
    f = bezier(x1, y1, x2, y2, a)
    curve = np.array([f(t_) for t_ in t])

    return interp1d(*curve.T)

def filter_map(h_map, smooth_h_map, x1, y1, x2, y2, a, b):
    f = bezier_lut(x1, y1, x2, y2, a)
    output_map = b*h_map + (1-b)*smooth_h_map
    output_map = f(output_map.clip(0, 1))
    return output_map

def filter_inbox(pts, size):
    inidx = np.all(pts < size, axis=1)
    return pts[inidx]

def generate_trees(n, size):
    trees = np.random.randint(0, size-1, (n, 2))
    trees = relax(trees, size, k=10).astype(np.uint32)
    trees = filter_inbox(trees, size)
    return trees

def place_trees(river_land_mask, adjusted_height_river_map, n, mask, size, a=0.5):
    trees= generate_trees(n, size)
    rr, cc = trees.T

    output_trees = np.zeros((size, size), dtype=bool)
    output_trees[rr, cc] = True
    output_trees = output_trees*(mask>a)*river_land_mask*(adjusted_height_river_map<0.5)

    output_trees = np.array(np.where(output_trees == 1))[::-1].T    
    return output_trees


def PCGGen(map_size, nbins = 256, seed = 3407):
    biome_names = [
        # sand and rock
        "desert",
        # grass gravel rock stone 
        "savanna", # mixed woodland and grassland
        # trees flower
        "tropical_woodland", # rainforest
        # dirt grass gravel rock stone
        "tundra", # no trees
        # trees flower
        "seasonal_forest", 
        # trees
        "rainforest",
        # trees
        "temperate_forest",
        # trees
        "temperate_rainforest",
        # snow rock tree
        "boreal_forest" # taiga, snow forest 
    ]
    biome_colors = [
        [255, 255, 178],
        [184, 200, 98],
        [188, 161, 53],
        [190, 255, 242],
        [106, 144, 38],
        [33, 77, 41],
        [86, 179, 106],
        [34, 61, 53],
        [35, 114, 94]
    ]

    size = map_size
    n = nbins
    map_seed = seed

    # start generation
    points = np.random.randint(0, size, (514, 2))
    points = relax(points, size, k=100)
    vor = voronoi(points, size)
    vor_map = voronoi_map(vor, size)

    boundary_displacement = 8
    boundary_noise = np.dstack([noise_map(size, 32, 200 + map_seed, octaves=8), noise_map(size, 32, 250 + map_seed, octaves=8)])
    boundary_noise = np.indices((size, size)).T + boundary_displacement*boundary_noise
    boundary_noise = boundary_noise.clip(0, size-1).astype(np.uint32)

    blurred_vor_map = np.zeros_like(vor_map)

    for x in range(size):
        for y in range(size):
            j, i = boundary_noise[x, y]
            blurred_vor_map[x, y] = vor_map[i, j]

    vor_map = blurred_vor_map
    temperature_map = noise_map(size, 2, 10 + map_seed)
    precipitation_map = noise_map(size, 2, 20 + map_seed)

    uniform_temperature_map = histeq(temperature_map, alpha=0.33)
    uniform_precipitation_map = histeq(precipitation_map, alpha=0.33)
    temperature_map = uniform_temperature_map
    precipitation_map = uniform_precipitation_map

    temperature_cells = average_cells(vor_map, temperature_map)
    precipitation_cells = average_cells(vor_map, precipitation_map)

    quantize_temperature_cells = quantize(temperature_cells, n)
    quantize_precipitation_cells = quantize(precipitation_cells, n)

    quantize_temperature_map = fill_cells(vor_map, quantize_temperature_cells)
    quantize_precipitation_map = fill_cells(vor_map, quantize_precipitation_cells)

    temperature_cells = quantize_temperature_cells
    precipitation_cells = quantize_precipitation_cells

    temperature_map = quantize_temperature_map
    precipitation_map = quantize_precipitation_map

    im = np.array(Image.open("./assets/biome_image.png"))[:, :, :3]
    im = cv2.resize(im, (256, 256))
    biomes = np.zeros((256, 256))

    for i, color in enumerate(biome_colors):
        indices = np.where(np.all(im == color, axis=-1))
        biomes[indices] = i
    biomes = np.flip(biomes, axis=0).T


    n = len(temperature_cells)
    biome_cells = np.zeros(n, dtype=np.uint32)

    for i in range(n):
        temp, precip = temperature_cells[i], precipitation_cells[i]
        biome_cells[i] = biomes[temp, precip]

    biome_map = fill_cells(vor_map, biome_cells).astype(np.uint32)
    biome_color_map = color_cells(biome_map, biome_colors)
    height_map = noise_map(size, 4, 0 + map_seed, octaves=6, persistence=0.5, lacunarity=2)
    land_mask = height_map > 0
    smooth_height_map = noise_map(size, 4, 0 + map_seed, octaves=1, persistence=0.5, lacunarity=2)

    biome_height_maps = [
        # Desert
        filter_map(height_map, smooth_height_map, 0.75, 0.2, 0.95, 0.2, 0.2, 0.5),
        # Savanna
        filter_map(height_map, smooth_height_map, 0.5, 0.1, 0.95, 0.1, 0.1, 0.2),
        # Tropical Woodland
        filter_map(height_map, smooth_height_map, 0.33, 0.33, 0.95, 0.1, 0.1, 0.75),
        # Tundra
        filter_map(height_map, smooth_height_map, 0.5, 1, 0.25, 1, 1, 1),
        # Seasonal Forest
        filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.2),
        # Rainforest
        filter_map(height_map, smooth_height_map, 0.5, 0.25, 0.66, 1, 1, 0.5),
        # Temperate forest
        filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.33),
        # Temperate Rainforest
        filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.33),
        # Boreal
        filter_map(height_map, smooth_height_map, 0.8, 0.1, 0.9, 0.05, 0.05, 0.1)
    ]


    biome_count = len(biome_names)
    biome_masks = np.zeros((biome_count, size, size))

    for i in range(biome_count):
        biome_masks[i, biome_map==i] = 1
        biome_masks[i] = gaussian_filter(biome_masks[i], sigma=16)

    # Remove ocean from masks
    blurred_land_mask = land_mask
    blurred_land_mask = binary_dilation(land_mask, iterations=32).astype(np.float64)
    blurred_land_mask = gaussian_filter(blurred_land_mask, sigma=16)
    
    # biome mask - [9, size, size]
    biome_masks = biome_masks*blurred_land_mask

    adjusted_height_map = height_map.copy()

    for i in range(len(biome_height_maps)):
        adjusted_height_map = (1-biome_masks[i])*adjusted_height_map + biome_masks[i]*biome_height_maps[i]

    # add rivers
    biome_bound = get_boundary(biome_map, size, kernel=5)
    cell_bound = get_boundary(vor_map, size, kernel=2)
    river_mask = noise_map(size, 4, 4353 + map_seed, octaves=6, persistence=0.5, lacunarity=2) > 0

    new_biome_bound = biome_bound*(adjusted_height_map<0.5)*land_mask
    new_cell_bound = cell_bound*(adjusted_height_map<0.05)*land_mask

    rivers = np.logical_or(new_biome_bound, new_cell_bound)*river_mask
    loose_river_mask = binary_dilation(rivers, iterations=8)
    rivers_height = gaussian_filter(rivers.astype(np.float64), sigma=2)*loose_river_mask
    adjusted_height_river_map = adjusted_height_map*(1-rivers_height) - 0.05*rivers

    sea_color = np.array([12, 14, 255])
    river_land_mask = adjusted_height_river_map >= 0
    land_mask_color = np.repeat(river_land_mask[:, :, np.newaxis], 3, axis=-1)
    rivers_biome_color_map = land_mask_color*biome_color_map + (1-land_mask_color)*sea_color
    rivers_biome_map = river_land_mask * biome_map + (1 - river_land_mask) * biome_count # use biome count=9 as water indicator

    semantic_map = rivers_biome_map
    semantic_map_color = rivers_biome_color_map
    height_map = adjusted_height_river_map

    tree_densities = [4000, 1500, 8000, 1000, 10000, 25000, 10000, 20000, 5000]
    trees = [np.array(place_trees(river_land_mask, adjusted_height_river_map, tree_densities[i], biome_masks[i], size)) for i in range(len(biome_names))]

    canvas = np.ones((size, size)) * 255
    for k in range(len(biome_names)):
        canvas[trees[k][:, 1], trees[k][:, 0]] = k
    tree_map = canvas

    return height_map, semantic_map, tree_map, semantic_map_color

if __name__ == '__main__':
    import os
    parser = ArgumentParser()
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--nbins', type=int, default=256)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    outdir = args.outdir
    heightmap, semanticmap, treemap, colormap = PCGGen(args.size, args.nbins, args.seed)
    save_height_map(heightmap, os.path.join(outdir, 'heightmap.png'))
    cv2.imwrite(os.path.join(outdir, 'semanticmap.png'), semanticmap.astype(np.uint8))
    cv2.imwrite(os.path.join(outdir, 'colormap.png'), colormap[..., [2, 1, 0]].astype(np.uint8))
    cv2.imwrite(os.path.join(outdir, 'treemap.png'), treemap)
