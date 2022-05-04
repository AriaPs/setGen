import blenderproc as bproc
import argparse
import numpy as np
import random
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
parser.add_argument('furniture_dir', nargs='?', default="resources/haven_furnitures", help="Path to the furniture objects")
parser.add_argument('cc_material_path', nargs='?', default="resources/cctextures", help="Path to CCTextures folder")
parser.add_argument('objects_dir', nargs='?', default="resources/haven_objects", help="Path to the opaque objects")
parser.add_argument('transparent_shader_path', nargs='?', default="resources/material", help="Path to the downloaded transparent shader")
parser.add_argument('output_dir', nargs='?', default="./output", help="Path to where the final files, will be saved")
args = parser.parse_args()

bproc.init()

bproc.renderer.set_cpu_threads(8)

####-------------Random Room construct-----------####

# Load materials and objects that can be placed into the room
materials = bproc.loader.load_ccmaterials(args.cc_material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

paths = get_blends_paths(args.furniture_dir)
random_path_fun = random.choice(paths)
furniture = bproc.loader.load_blend(random_path_fun)

#print(random_path_fun, " has :")
#print(furniture[0].get_bound_box())

# Construct random room and fill with interior_objects
room_objects = bproc.constructor.construct_random_room(used_floor_area=16, interior_objects=[],
                                                  materials=materials, amount_of_extrusions=0)


####-----------Random Scene construct----------####

# load opaque objects
paths = get_blends_paths(args.objects_dir)
interior_objects = []
amount_objects = random.randint(3, 6)
for i in range(amount_objects):
    random_path = random.choice(paths)
    opaque_object = bproc.loader.load_blend(random_path)
    interior_objects.extend(opaque_object)

# Define a function that samples the pose of a given object
def sample_pose(obj: bproc.types.MeshObject):
    # Sample the spheres location above the surface
    obj.set_location(bproc.sampler.upper_region(
        objects_to_sample_on=furniture,
        min_height=0.1,
        max_height=0.3,
        use_ray_trace_check=False
    ))
    obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))

placed_objects = bproc.object.sample_poses_on_surface(interior_objects, furniture[0], sample_pose, max_tries=30, min_distance=0.05, max_distance=0.4)

if len(placed_objects) ==0 :
    raise Exception("Could not place any object on furniture: ", random_path_fun)

# Set classification labels --> 0 opaque , 1 transmissiv
furniture[0].set_cp("category_id", 0)
furniture[0].enable_rigidbody(active=False, collision_shape="MESH")

## Create a random procedural texture
#texture = bproc.material.create_procedural_texture("WOOD")    
for obj in room_objects:
    obj.set_cp("category_id", 0)
    obj.enable_rigidbody(active=False, collision_shape="MESH")
    ## Displace the vertices of the object based on that random texture
    #obj.add_displace_modifier(
    #    texture=texture,
    #    strength=random.gauss(0, 0.25),
    #    subdiv_level=random.randint(1, 3),
    #)

mat_path = random.choice(glob.glob(os.path.join(args.transparent_shader_path ,"*.blend")))
materials = bproc.material.convert_to_materials(bproc.loader.load_blend(mat_path, data_blocks='materials')) 
for inObjOpaq in placed_objects:
    inObjOpaq.set_cp("category_id", 0)
    inObjOpaq.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
    if inObjOpaq.has_materials():
        # In 50% of all cases
        if np.random.uniform(0, 1) <= 0.75:
            inObjOpaq.set_cp("category_id", 1)
            material = random.choice(materials)
            for i in range(len(inObjOpaq.get_materials())):   
                # Replace the material with a transmissive one
                inObjOpaq.set_material(i, material)    

bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=2,
    max_simulation_time=5,
    check_object_interval=1
)

####-------------Sampling n poses----------------####

# Bring light into the room
bproc.lighting.light_surface([obj for obj in room_objects if obj.get_name() == "Ceiling"], emission_strength=4.0, emission_color=[1,1,1,1])

# define the camera intrinsics
bproc.camera.set_resolution(512, 512)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)
# Determine point of interest in scene as the object closest to the mean of a subset of objects
poi = bproc.object.compute_poi(placed_objects)
furniture_loc = furniture[0].get_location()
poses = 0
tries = 0
while tries < 10000 and poses < 10:
    # Compute rotation based on vector going from location towards poi
    location = furniture_loc + np.random.uniform([-1.0, -1.0, 1.0], [1.0, 1.0, 1.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.4}, bvh_tree):
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

