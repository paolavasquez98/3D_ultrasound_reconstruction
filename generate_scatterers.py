import glob
import os
import numpy as np
import scipy.io
import argparse
import pyvista as pv

# GENERATE 1 EMPTY ELLIPSOID ------------------------------------------------
def generate_empty_ellipsoid_model(variation_id, save_dir):
    np.random.seed(variation_id)
    num_scatterers = 1_000_000
    a_start, b_start, c_start = 40, 30, 100
    a = a_start * np.random.uniform(0.8, 1.2)
    b = b_start * np.random.uniform(0.8, 1.2)
    c = c_start * np.random.uniform(0.8, 1.2)

    u = np.random.uniform(0, 2 * np.pi, num_scatterers)  # 0 to 2pi
    v = np.arccos(np.random.uniform(-1, 0, num_scatterers))  # arccos for a 3D even distribution
    r = np.cbrt(np.random.uniform(0, 1, num_scatterers))# Random radius factor y raiz cubica
    x_scatter = r*a * np.cos(u) * np.sin(v)
    y_scatter = r*b * np.sin(u) * np.sin(v)
    z_scatter = r*c * np.cos(v)

    # Wall thickness varies with size or randomly
    wall_thickness = np.random.uniform(4, 8)  
    # wall_thickness = np.mean([a, b, c]) * np.random.uniform(0.05, 0.15)

    # Define inner ellipsoid dimensions
    a_inner = a - wall_thickness
    b_inner = b - wall_thickness
    c_inner = c - wall_thickness

    # Outer ellipsoid points
    x_outer = a * np.cos(u) * np.sin(v)
    y_outer = b * np.sin(u) * np.sin(v)
    z_outer = c * np.cos(v)

    # Inner ellipsoid points
    x_inner = a_inner * np.cos(u) * np.sin(v)
    y_inner = b_inner * np.sin(u) * np.sin(v)
    z_inner = c_inner * np.cos(v)

    # Interpolated scatterers in shell
    x_scatter = x_inner + r * (x_outer - x_inner)
    y_scatter = y_inner + r * (y_outer - y_inner)
    z_scatter = z_inner + r * (z_outer - z_inner)

    scatterers = np.stack([x_scatter, y_scatter, z_scatter], axis=-1)
    tissue_rc = np.random.uniform(0,1,4)
    rc_values = np.random.choice(tissue_rc, size=num_scatterers, p=[0.5, 0.3, 0.15, 0.05])

    # Save
    filename = f"empty_ellipsoid_{variation_id}.mat"
    full_path = os.path.join(save_dir, filename)
    scipy.io.savemat(full_path, {
        "scatterers": scatterers,
        "rc_values": rc_values
    })
    print(f"Saved {filename} with {scatterers.shape[0]} scatterers.")


# GENERATE 2 CHAMBERS ------------------------------------------------
def generate_half_ellipsoid_shell(center, outer_radii, wall_thickness, num_points):
    """Generates scatterers in the shell (wall) of a lower-half ellipsoid."""
    # Inner ellipsoid radii
    inner_radii = np.array(outer_radii) - wall_thickness

    # Spherical coordinates restricted to lower hemisphere
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.arccos(np.random.uniform(-1, 0, num_points))  # Lower half
    r = np.cbrt(np.random.uniform(0, 1, num_points))     # Uniform volume sampling

    # Outer shell surface
    x_outer = outer_radii[0] * np.cos(u) * np.sin(v)
    y_outer = outer_radii[1] * np.sin(u) * np.sin(v)
    z_outer = outer_radii[2] * np.cos(v)

    # Inner shell surface
    x_inner = inner_radii[0] * np.cos(u) * np.sin(v)
    y_inner = inner_radii[1] * np.sin(u) * np.sin(v)
    z_inner = inner_radii[2] * np.cos(v)

    # Interpolate between inner and outer surfaces
    x = x_inner + r * (x_outer - x_inner) + center[0]
    y = y_inner + r * (y_outer - y_inner) + center[1]
    z = z_inner + r * (z_outer - z_inner) + center[2]

    return np.column_stack((x, y, z))

def generate_ventricular_model(variation_id, save_dir):
    np.random.seed(variation_id)

    # Base values (in mm)
    # d = 32  # distance between ventricles (centers) → 3.2 cm
    d = np.random.uniform(26, 38)  # e.g., vary ±4 mm around 32

    # LV
    a_LV = np.random.uniform(22, 28)  # mm
    b_LV = np.random.uniform(19, 25)
    c_LV = np.random.uniform(55, 65)
    wall_LV = np.random.uniform(6, 10)  # mm
    center_LV = [+d / 2, 0, 0]
    rc_lv = 0.8
    lv_points = generate_half_ellipsoid_shell(center_LV, [a_LV, b_LV, c_LV], wall_LV, 900000)

    # RV
    a_RV = np.random.uniform(28, 34)
    b_RV = np.random.uniform(23, 29)
    c_RV = np.random.uniform(60, 70)
    wall_RV = np.random.uniform(3, 6)
    center_RV = [-d / 2, 0, 0]
    rc_rv = 0.4
    rv_points = generate_half_ellipsoid_shell(center_RV, [a_RV, b_RV, c_RV], wall_RV, 800000)

    # Combine
    scatterers = np.vstack((lv_points, rv_points))
    rc_values = np.concatenate((
        np.full(len(lv_points), rc_lv),
        np.full(len(rv_points), rc_rv)
    ))

    # Save
    filename = f"ventricular_model_{variation_id}.mat"
    full_path = os.path.join(save_dir, filename)
    scipy.io.savemat(full_path, {
        "scatterers": scatterers,
        "rc_values": rc_values
    })
    print(f"Saved {filename} with {scatterers.shape[0]} scatterers.")


# GENERATE THREE ELLIPSOIDES ---------------------------------------------
def generate_ellipsoidal_scatterers(center, radii, num_points, surface_ratio, rc_surface, rc_inside):
    """Generate scatterers inside an ellipsoid with higher reflectivity at the surface."""
    # Generate random points inside a unit sphere
    u = np.random.uniform(0, 1, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, num_points))

    # Convert to Cartesian coordinates (unit sphere)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Scale to ellipsoid shape
    r = u ** (1/3)  # Distribute points uniformly inside the ellipsoid
    x = center[0] + radii[0] * x * r
    y = center[1] + radii[1] * y * r
    z = center[2] + radii[2] * z * r

    scatterers = np.column_stack((x, y, z))

    # Compute distance from center (normalized to [0,1])
    distances = np.sqrt(((scatterers - center) ** 2 / np.array(radii) ** 2).sum(axis=1))
    
    # Assign reflectivity based on proximity to surface
    reflectivity_values = np.where(distances > (1 - surface_ratio), rc_surface, rc_inside)
    
    return scatterers, reflectivity_values

# Function to generate multiple heart models with slight variations
def generate_heart_model(variation_id, save_dir):
    np.random.seed(variation_id)  # Set a unique seed for repeatability

    # Randomize anatomical structures (vary size, position, reflectivity)
    heart_structures = {
        "LV": {
            "center": [0, 0, 0],  
            "radii": np.random.uniform(25, 35, 3),  # Slight size variation
            "density": np.random.randint(400000, 600000),  
            "surface_ratio": 0.1, 
            "rc_surface": np.random.uniform(0.7, 0.9),  
            "rc_inside": np.random.uniform(0.2, 0.4),
        },
        "RV": {
            "center": [-np.random.uniform(10, 20), 0, 0],  
            "radii": np.random.uniform(20, 30, 3),  
            "density": np.random.randint(300000, 500000),  
            "surface_ratio": 0.1, 
            "rc_surface": np.random.uniform(0.6, 0.8),  
            "rc_inside": np.random.uniform(0.1, 0.3),
        },
        "AO": {
            "center": [0, np.random.uniform(25, 35), 0],  
            "radii": np.random.uniform(8, 12, 3),  
            "density": np.random.randint(150000, 250000),  
            "surface_ratio": 0.05, 
            "rc_surface": np.random.uniform(0.8, 1.0),  
            "rc_inside": np.random.uniform(0.3, 0.5),
        },
    }

    # Generate scatterers for each structure
    all_scatterers = []
    all_rc_values = []

    for structure, params in heart_structures.items():
        scatterers, rc_values = generate_ellipsoidal_scatterers(
            params["center"], params["radii"], params["density"],
            params["surface_ratio"], params["rc_surface"], params["rc_inside"]
        )
        all_scatterers.append(scatterers)
        all_rc_values.append(rc_values)

    # Combine all scatterers and reflectivity values
    scatterers = np.vstack(all_scatterers)
    reflectivity_values = np.concatenate(all_rc_values)

    # Save as .mat file for beamforming
    filename = f"three_ellipsoides_{variation_id}.mat"
    full_path = os.path.join(save_dir, filename)
    scipy.io.savemat(full_path, {
        "scatterers": scatterers, 
        "rc_values": reflectivity_values})

    print(f"Generated model {variation_id} with {scatterers.shape[0]} scatterers.")

# GENERATE THE MESHES ------------------------------------------------

def generate_mesh(vtk_file, variation_id, save_dir):
    """This one takes the vtk files as input"""
    mesh = pv.read(vtk_file)
    scatterers = mesh.points
    rc_values = np.random.uniform(0.2, 1.0, len(scatterers))

    # Save as .mat file for beamforming
    filename = f"heart_mesh_{variation_id}.mat"
    full_path = os.path.join(save_dir, filename)
    scipy.io.savemat(full_path, {
        "scatterers": scatterers, 
        "rc_values": rc_values})

    print(f"Generated model {variation_id} with {scatterers.shape[0]} scatterers.")

def main():

    parser = argparse.ArgumentParser(description="Generate scatterer datasets.")
    parser.add_argument("--chambers", type=int, default=50, help="Number of two chamber models.")
    parser.add_argument("--ellipsoids", type=int, default=50, help="Number of three ellipsoid models.")
    parser.add_argument("--empty_ellipsoid", type=int, default=50, help="Number of empty ellipsoid models.")
    parser.add_argument("--output_dir", type=str, default="shape_models", help="Base output directory.")
    args = parser.parse_args()

    config = {
        "chambers": {
            "n": args.chambers,
            "save_dir": os.path.join(args.output_dir, "two_chambers/scatterers")
        },
        "ellipsoids": {
            "n": args.ellipsoids,
            "save_dir": os.path.join(args.output_dir, "three_ellip/scatterers")
        },
        "empty_ellipsoid": {
            "n": args.empty_ellipsoid,
            "save_dir": os.path.join(args.output_dir, "empty_ellipsoid/scatterers")
        },
    }

    # Create directories if not exist
    for cfg in config.values():
        os.makedirs(cfg["save_dir"], exist_ok=True)

    # Generate datasets
    for i in range(config["chambers"]["n"]):
        print(f"[Chambers] Generating model {i+1}/{config['chambers']['n']}")
        generate_ventricular_model(i, config["chambers"]["save_dir"])

    for i in range(config["ellipsoids"]["n"]):
        print(f"[Ellipsoids] Generating model {i+1}/{config['ellipsoids']['n']}")
        generate_heart_model(i, config["ellipsoids"]["save_dir"])

    for i in range(config["empty_ellipsoid"]["n"]):
        print(f"[Empty Ellipsoids] Generating model {i+1}/{config['empty_ellipsoid']['n']}")
        generate_empty_ellipsoid_model(i, config["empty_ellipsoid"]["save_dir"])

    print("All scatterer datasets generated successfully.")

if __name__ == "__main__":
    main()
