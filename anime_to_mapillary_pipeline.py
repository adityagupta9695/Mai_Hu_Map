import asyncio
import aiohttp
import json
import os
import sys
import argparse
import torch
from tqdm.asyncio import tqdm
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from collections import defaultdict

# --- DEFAULT CONFIGURATION ---
PANOPTIC_MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"

# Grid and API settings
LON_DIVISIONS = 10
LAT_DIVISIONS = 10
RETRY_ATTEMPTS = 3
CONCURRENCY_LIMIT = 20
MAX_IMAGES_TO_ANALYZE = 50


def get_script_dir():
    """Get directory where script is located"""
    return os.path.dirname(os.path.abspath(__file__))


def load_tag_mapping():
    """Load the tag mapping JSON file"""
    script_dir = get_script_dir()
    tag_file = os.path.join(script_dir, "pretty_to_api_tags.json")
    
    try:
        with open(tag_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load '{tag_file}'")
        print(f"Make sure 'pretty_to_api_tags.json' is in the same folder as this script.")
        print(f"Error: {e}")
        sys.exit(1)


def run_panoptic_analysis(image_path, save_visualization=True):
    """
    Analyzes an image using Panoptic Segmentation.
    Returns point_tags and area_tags lists.
    """
    print(f"\n{'='*60}")
    print("STAGE 1: ANALYZING IMAGE WITH PANOPTIC SEGMENTATION")
    print(f"{'='*60}")
    
    # Load tag mapping
    api_tag_map = load_tag_mapping()
    
    # Load model
    print("Loading Panoptic Segmentation model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(PANOPTIC_MODEL_NAME)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(PANOPTIC_MODEL_NAME).to(device)
        print(f"Model loaded: {PANOPTIC_MODEL_NAME}")
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        sys.exit(1)

    # Load image
    try:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ERROR: Could not load image: {e}")
        sys.exit(1)

    # Run segmentation
    print("Running segmentation (this may take a moment)...")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    panoptic_result = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[image.size[::-1]]
    )[0]

    # Count detected objects
    detected_objects = defaultdict(int)
    for segment in panoptic_result["segments_info"]:
        label_id = segment["label_id"]
        pretty_name = model.config.id2label[label_id]
        detected_objects[pretty_name] += 1

    print(f"\n{'='*60}")
    print("DETECTED OBJECTS IN IMAGE")
    print(f"{'='*60}")
    for name, count in detected_objects.items():
        print(f"  • {name}: {count}")

    # Save visualization if requested
    if save_visualization:
        try:
            import numpy as np
            from matplotlib import pyplot as plt
            import matplotlib.patches as mpatches
            
            panoptic_map = panoptic_result["segmentation"].cpu().numpy()
            segment_info = panoptic_result["segments_info"]
            
            # Create colored segmentation map
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            # Create a colored overlay
            colored_mask = np.zeros((*panoptic_map.shape, 3), dtype=np.uint8)
            np.random.seed(42)
            
            for segment in segment_info:
                segment_id = segment['id']
                mask = panoptic_map == segment_id
                color = np.random.randint(0, 255, size=3)
                colored_mask[mask] = color
            
            ax.imshow(colored_mask, alpha=0.5)
            ax.axis('off')
            
            output_dir = os.path.dirname(image_path) or '.'
            output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_segmented.jpg"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            print(f"\n✓ Saved visualization to: {output_path}")
        except Exception as e:
            print(f"\nWarning: Could not save visualization: {e}")

    # Categorize tags
    point_feature_tags = set()
    segmentation_tags = set()

    print(f"\n{'='*60}")
    print("TAG MAPPING RESULTS")
    print(f"{'='*60}")

    for pretty_name in detected_objects.keys():
        mapillary_tag = api_tag_map.get(pretty_name)
        
        if not mapillary_tag:
            print(f"  ⚠ '{pretty_name}' -> NOT FOUND in mapping file")
            continue

        if (mapillary_tag.startswith("object--") or 
            mapillary_tag.startswith("marking--") or
            mapillary_tag.startswith("construction--barrier--") or
            mapillary_tag.startswith("construction--flat--crosswalk") or
            mapillary_tag.startswith("construction--flat--driveway")):
            point_feature_tags.add(mapillary_tag)
            print(f"  ✓ '{pretty_name}' -> POINT TAG: {mapillary_tag}")
        else:
            segmentation_tags.add(mapillary_tag)
            print(f"  ✓ '{pretty_name}' -> AREA TAG: {mapillary_tag}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(point_feature_tags)} point tags, {len(segmentation_tags)} area tags")
    print(f"{'='*60}")

    return list(point_feature_tags), list(segmentation_tags)


def create_grid(bbox, lon_divs, lat_divs):
    """Splits BBOX into grid cells"""
    min_lon, min_lat, max_lon, max_lat = bbox
    cell_width = (max_lon - min_lon) / lon_divs
    cell_height = (max_lat - min_lat) / lat_divs
    grid_cells = []
    for i in range(lon_divs):
        for j in range(lat_divs):
            cell_min_lon = min_lon + i * cell_width
            cell_min_lat = min_lat + j * cell_height
            cell_max_lon = cell_min_lon + cell_width
            cell_max_lat = cell_min_lat + cell_height
            grid_cells.append(f"{cell_min_lon},{cell_min_lat},{cell_max_lon},{cell_max_lat}")
    return grid_cells


async def fetch_ids_for_point_tag(session, tag, bbox_str, pbar, access_token):
    """Fetch image IDs for a point tag in a grid cell"""
    search_url = "https://graph.mapillary.com/map_features"
    params = {
        'access_token': access_token,
        'fields': 'images',
        'object_values': tag,
        'bbox': bbox_str
    }
    
    image_ids_set = set()
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with session.get(search_url, params=params, timeout=60.0) as response:
                response.raise_for_status()
                data = await response.json()
                
                for feature in data.get('data', []):
                    image_data = feature.get('images', {})
                    for image_info in image_data.get('data', []):
                        image_ids_set.add(image_info['id'])
                
                pbar.update(1)
                return image_ids_set
        except Exception:
            await asyncio.sleep(1 * (attempt + 1))
    
    pbar.update(1)
    return image_ids_set


async def fetch_all_image_ids_in_grid(session, bbox_str, pbar, access_token):
    """Fetch all image IDs in a grid cell"""
    search_url = "https://graph.mapillary.com/images"
    params = {
        'access_token': access_token,
        'fields': 'id',
        'bbox': bbox_str,
        'is_pano': 'false'
    }
    
    image_ids_set = set()
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with session.get(search_url, params=params, timeout=60.0) as response:
                response.raise_for_status()
                data = await response.json()
                for image_data in data.get('data', []):
                    image_ids_set.add(image_data['id'])
                
                pbar.update(1)
                return image_ids_set
        except Exception:
            await asyncio.sleep(1 * (attempt + 1))
    
    pbar.update(1)
    return image_ids_set


async def search_mapillary(searchable_tags, bbox, access_token):
    """Search Mapillary for candidate images"""
    print(f"\n{'='*60}")
    print("STAGE 2: SEARCHING MAPILLARY FOR CANDIDATE IMAGES")
    print(f"{'='*60}")
    
    grid_cells = create_grid(bbox, LON_DIVISIONS, LAT_DIVISIONS)
    all_found_ids = set()
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)) as session:
        if searchable_tags:
            print(f"Searching for images containing ALL of:")
            for tag in searchable_tags:
                print(f"  • {tag}")
            print(f"\nGrid cells: {len(grid_cells)} | API calls: {len(grid_cells) * len(searchable_tags)}")
            
            final_intersection_set = set()
            
            for i, tag in enumerate(searchable_tags):
                print(f"\nSearching for tag '{tag}' ({i+1}/{len(searchable_tags)})...")
                
                tag_specific_ids = set()
                tasks = []
                
                with tqdm(total=len(grid_cells), desc=f"Tag: {tag}") as pbar:
                    for bbox_str in grid_cells:
                        tasks.append(fetch_ids_for_point_tag(session, tag, bbox_str, pbar, access_token))
                    results = await asyncio.gather(*tasks)
                
                for id_set in results:
                    tag_specific_ids.update(id_set)
                
                print(f"  Found {len(tag_specific_ids)} images with '{tag}'")

                if i == 0:
                    final_intersection_set = tag_specific_ids
                else:
                    final_intersection_set = final_intersection_set.intersection(tag_specific_ids)

                print(f"  Common images so far: {len(final_intersection_set)}")
                
                if not final_intersection_set:
                    print("  No common images found. Stopping.")
                    break
            
            all_found_ids = final_intersection_set
        else:
            print("No point feature tags found. Searching for ALL images...")
            tasks = []
            with tqdm(total=len(grid_cells), desc="Searching grid") as pbar:
                for bbox_str in grid_cells:
                    tasks.append(fetch_all_image_ids_in_grid(session, bbox_str, pbar, access_token))
                results = await asyncio.gather(*tasks)
            
            for id_set in results:
                all_found_ids.update(id_set)

    print(f"\n✓ Stage 2 complete: {len(all_found_ids)} candidate images found")
    return all_found_ids


async def fetch_detections_for_image(session, image_id, sem, pbar, access_token):
    """Fetch detections for a single image"""
    async with sem:
        url = f"https://graph.mapillary.com/{image_id}/detections"
        params = {
            'access_token': access_token,
            'layers': 'segmentations',
            'fields': 'value'
        }
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                async with session.get(url, params=params, timeout=30.0) as response:
                    response.raise_for_status()
                    data = await response.json()
                    tags_found = set(item['value'] for item in data.get('data', []))
                    pbar.update(1)
                    return {"id": image_id, "tags": list(tags_found)}
            except Exception:
                await asyncio.sleep(1 * (attempt + 1))
        
        pbar.update(1)
        return {"id": image_id, "tags": []}


async def filter_by_area_tags(candidate_ids, required_area_tags, access_token):
    """Filter candidates by area/segmentation tags"""
    print(f"\n{'='*60}")
    print("STAGE 3: FILTERING BY AREA/SEGMENTATION TAGS")
    print(f"{'='*60}")
    
    if not required_area_tags:
        print("No area tags to filter by. Skipping Stage 3.")
        return candidate_ids

    print(f"Checking {len(candidate_ids)} candidates for:")
    for tag in required_area_tags:
        print(f"  • {tag}")
    
    if len(candidate_ids) > MAX_IMAGES_TO_ANALYZE:
        print(f"\nLimiting analysis to {MAX_IMAGES_TO_ANALYZE} images (set in config)")
        candidate_ids = set(list(candidate_ids)[:MAX_IMAGES_TO_ANALYZE])

    final_matched_images = []
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)) as session:
        tasks = []
        with tqdm(total=len(candidate_ids), desc="Analyzing") as pbar:
            for image_id in candidate_ids:
                tasks.append(fetch_detections_for_image(session, image_id, sem, pbar, access_token))
            image_results = await asyncio.gather(*tasks)

    required_tags_set = set(required_area_tags)
    for result in image_results:
        if not result["tags"]:
            continue
        if required_tags_set.issubset(set(result["tags"])):
            final_matched_images.append(result["id"])
    
    print(f"\n✓ Stage 3 complete: {len(final_matched_images)} images match all criteria")
    return final_matched_images


async def main_async(args):
    """Main async function"""
    # Validate image path
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    
    # Stage 1: Analyze image
    point_tags, area_tags = run_panoptic_analysis(args.image, not args.no_visualization)
    
    # Stage 2: Search Mapillary
    candidates = await search_mapillary(point_tags, args.bbox, args.token)
    
    if not candidates:
        print("\nNo candidate images found.")
        return
    
    # Stage 3: Filter by area tags
    final_results = await filter_by_area_tags(candidates, area_tags, args.token)
    
    # Display results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Found {len(final_results)} matching images\n")
    
    if final_results:
        for image_id in final_results:
            print(f"https://www.mapillary.com/app/?pKey={image_id}")
        
        # Save results
        output_file = args.output or "matched_images.json"
        with open(output_file, 'w') as f:
            json.dump(list(final_results), f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Find Mapillary images matching a reference image using panoptic segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg --bbox 77.0 28.5 77.2 28.7
  %(prog)s image.jpg --bbox 77.0 28.5 77.2 28.7 --token YOUR_TOKEN
  %(prog)s image.jpg --bbox 77.0 28.5 77.2 28.7 --output results.json
        """
    )
    
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--bbox', nargs=4, type=float, required=True,
                        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                        help='Bounding box coordinates (REQUIRED) - format: min_lon min_lat max_lon max_lat')
    parser.add_argument('--token', help='Mapillary access token (or set MAPILLARY_API_KEY env var)')
    parser.add_argument('--output', '-o', help='Output JSON file (default: matched_images.json)')
    parser.add_argument('--no-visualization', action='store_true',
                        help='Skip saving segmentation visualization')
    
    args = parser.parse_args()
    
    # Get token from env if not provided
    if not args.token:
        args.token = os.getenv('MAPILLARY_API_KEY')
    
    # Validate token exists
    if not args.token:
        print("ERROR: Mapillary API token is required!")
        print("Provide it using either:")
        print("  1. --token YOUR_TOKEN argument")
        print("  2. MAPILLARY_API_KEY environment variable")
        print("\nExample:")
        print(f"  export MAPILLARY_API_KEY='your_token_here'")
        print(f"  python {sys.argv[0]} image.jpg --bbox 77.0 28.5 77.2 28.7")
        sys.exit(1)
    
    # Run async main
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()