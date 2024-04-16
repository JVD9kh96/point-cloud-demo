from pathlib import Path
import argparse
from pointcloud_tools import PointCloudCreator, create_pcd_from_array, read_pcd, write_pcd
import tqdm 

def main(args):
    """Process RGB and depth images in the input directory, generate point clouds, and save them to the output directory."""

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    # Check if the input directories exist
    rgb_dir = base_dir / "rgb"
    depth_dir = base_dir / "depth"
    if not (rgb_dir.is_dir() and depth_dir.is_dir()):
        print("RGB or depth directory does not exist.")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a list of RGB image files in the input directory
    rgb_files = sorted(rgb_dir.glob("*.png"))

    # Init a point cloud creator with the camera configuration
    point_cloud_creator = PointCloudCreator(conf_file=base_dir / "oak-d-s2-poe_conf.json")

    # Choose method for point cloud generation
    if args.method == "open3d":
        convert_method = point_cloud_creator.convert_depth_to_pcd
    else:
        convert_method = point_cloud_creator.convert_depth_to_point_array

    # Process each RGB image
    for rgb_file in tqdm.tqdm(rgb_files):
        # Extract image name without extension
        image_name = rgb_file.stem

        # Define depth file path
        depth_file = depth_dir / f"{image_name}_depth.png"

        # Define output file path
        out_file = output_dir / f"{image_name}.{args.method}.pcd"

        # Convert depth image to point cloud or point array
        if args.method == "open3d":
            pcd_object = convert_method(rgb_file=rgb_file, depth_file=depth_file)
        else:
            points_array = convert_method(depth_file=depth_file)
            pcd_object = create_pcd_from_array(rgb_file=rgb_file, points_array=points_array)

        # Write point cloud to file
        write_pcd(pcd=pcd_object, pcd_file=out_file, down_sample=False, down_factor=9)

        # Check if point cloud was successfully written
        if out_file.is_file():
            print(f"Point cloud generated and saved for {image_name}.")
        else:
            print(f"Error generating point cloud for {image_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate point clouds from RGB and depth images.')
    parser.add_argument('--base_dir', type=str, default="data", help='Base directory containing RGB and depth images')
    parser.add_argument('--output_dir', type=str, default="data/pointclouds", help='Output directory for point clouds')
    parser.add_argument('--method', choices=['open3d', 'array'], default='open3d',
                        help='Method for point cloud generation: open3d or array')
    args = parser.parse_args()
    main(args)
