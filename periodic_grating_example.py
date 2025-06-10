import torch
import matplotlib.pyplot as plt
from geometry import geometry

# Define geometry parameters
geometry_params = {
    'Lx': 1.0,  # unit cell size in x (arbitrary units)
    'Ly': 1.0   # unit cell size in y (arbitrary units)
}
grid_spacing = 0.01  # grid resolution

# Create geometry object
geom = geometry(geometry_params, grid_spacing=grid_spacing)

# Define grating parameters (in same units as Lx, Ly)
width_center = 0.2
width_side = 0.16

# Build periodic grating
er, mur = geom.build_periodic_grating(width_center, width_side)

# Plot the permittivity distribution
plt.figure(figsize=(6, 5))
plt.imshow(er.real, cmap='gray', origin='lower', extent=[0, geom.Lx, 0, geom.Ly])
plt.title('Periodic Grating (Permittivity)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Re($\epsilon_r$)')
plt.tight_layout()
plt.savefig('periodic_grating.png')
