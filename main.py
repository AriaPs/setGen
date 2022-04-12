import blenderproc as bproc
import argparse
import numpy as np
import random
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

####-------------Used Costum Functions------------------####

def get_blends_paths(data_path: str) -> str:
    """ Retruns a list of .blend file paths from the given haven data set directory.

    :param data_path: A path pointing to a directory containing .blend files.
    :return The path list
    """

    if os.path.exists(data_path):
        blend_files = glob.glob(os.path.join(data_path, "*", "*.blend"))
        # this will be ensure that the call is deterministic
        blend_files.sort()
        return blend_files
    else:
        raise Exception("The data path does not exists: {}".format(data_path))

    

####---------------------------------------------####

parser = argparse.ArgumentParser()
parser.add_argument('furniture_dir', nargs='?', default="resources/furnitures", help="Path to the furniture objects")
parser.add_argument('cc_material_path', nargs='?', default="resources/cctextures", help="Path to CCTextures folder")
parser.add_argument('transmissiv_objects_dir', nargs='?', default="resources/models/transmissiv", help="Path to the transmissiv objects")
parser.add_argument('opaque_objects_dir', nargs='?', default="resources/models/opaque", help="Path to the opaque objects")
parser.add_argument('transparent_shader_path', nargs='?', default="examples/dataset_gen/resources/material", help="Path to the downloaded transparent shader")
parser.add_argument('output_dir', nargs='?', default="examples/dataset_gen/output", help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()

####-------------Random Room construct-----------####

# Load materials and objects that can be placed into the room
materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

paths = glob.glob(os.path.join(args.furniture_dir, "*.blend"))
random_path = random.choice(paths)
furniture = bproc.loader.load_blend(random_path)

# Construct random room and fill with interior_objects
room_objects = bproc.constructor.construct_random_room(used_floor_area=16, interior_objects=[],
                                                  materials=materials, amount_of_extrusions=0)


####-----------Random Scene construct----------####

# load transmissiv objects
paths = get_blends_paths(args.transmissiv_objects_dir)
interior_transmissiv_objects = []
amount_objects = int(np.random.uniform(2, 5))
for i in range(amount_objects):
    random_path = random.choice(paths)
    interior_transmissiv_objects.extend(bproc.loader.load_blend(random_path))

# load opaque objects
paths = get_blends_paths(args.opaque_objects_dir)
interior_opaque_objects = []
amount_objects = int(np.random.uniform(2, 5))
for i in range(amount_objects):
    random_path = random.choice(paths)
    interior_opaque_objects.extend(bproc.loader.load_blend(random_path))

# Set classification labels --> 0 opaque , 1 transmissiv
furniture[0].set_cp("category_id", 0)
furniture[0].enable_rigidbody(active=False, collision_shape="MESH")
for obj in room_objects:
    obj.set_cp("category_id", 0)
    obj.enable_rigidbody(active=False, collision_shape="MESH")

for inObjTrans in interior_transmissiv_objects:
    inObjTrans.set_cp("category_id", 1)
    inObjTrans.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")


mat_path = random.choice(glob.glob(os.path.join(args.transparent_shader_path ,"*.blend")))
materials = bproc.material.convert_to_materials(bproc.loader.load_blend(mat_path, data_blocks='materials')) 
for inObjOpaq in interior_opaque_objects:
    inObjOpaq.set_cp("category_id", 0)
    inObjOpaq.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
    if inObjOpaq.has_materials():
        # In 50% of all cases
        if np.random.uniform(0, 1) <= 0.5:
            inObjOpaq.set_cp("category_id", 1)
            material = random.choice(materials)
            for i in range(len(inObjOpaq.get_materials())):   
                # Replace the material with a transmissive one
                inObjOpaq.set_material(i, material)


# Define a function that samples the pose of a given object
def sample_pose(obj: bproc.types.MeshObject):
    # Sample the spheres location above the surface
    obj.set_location(bproc.sampler.upper_region(
        objects_to_sample_on=furniture,
        min_height=0.2,
        max_height=0.8,
        use_ray_trace_check=False
    ))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))

bproc.object.sample_poses_on_surface(interior_opaque_objects + interior_transmissiv_objects, furniture[0], sample_pose, max_tries=100, min_distance=0.05, max_distance=0.3)


bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=2,
    max_simulation_time=4,
    check_object_interval=1
)

####-------------Sampling n poses----------------####

# Bring light into the room
bproc.lighting.light_surface([obj for obj in room_objects if obj.get_name() == "Ceiling"], emission_strength=4.0, emission_color=[1,1,1,1])

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects(room_objects)
poses = 0
tries = 0
while tries < 10000 and poses < 3:
    furniture_loc = furniture[0].get_location()
    location = furniture_loc + np.random.uniform([-1.0, -1.0, 1.0], [1.0, 1.0, 1.5])
    
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi(furniture)
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bvh_tree):
        # Persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1
    tries += 1



####----------------Render Scene-----------------####

# activate depth rendering
bproc.renderer.enable_distance_output(activate_antialiasing=False)
bproc.renderer.set_light_bounces(max_bounces=200, diffuse_bounces=200, glossy_bounces=200, transmission_bounces=200, transparent_max_bounces=200)

# render the whole pipeline
data = bproc.renderer.render()

# Render segmentation masks (per class and per instance)
data.update(bproc.renderer.render_segmap(map_by=["class"]))


# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)



