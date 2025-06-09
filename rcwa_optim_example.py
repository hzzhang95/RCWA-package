import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rcwa import rcwa

# Constants
wavelength = 0.65  # um
theta = 0.0
phi = 0.0
TE = 1.0
TM = 0.0
Lx = 1.0
Ly = 1.0
basis = (0, 0)  # No diffraction orders for simple demonstration

# Material properties
n_air = 1.0
n_dielectric = 1.5
n_silicon = 3.5

er_air = n_air ** 2
er_dielectric = n_dielectric ** 2
er_silicon = n_silicon ** 2

# Device
device = 'cpu'

# Initialize RCWA model
model = rcwa(
    wavelength=wavelength,
    theta=theta,
    phi=phi,
    TE=TE,
    TM=TM,
    Lx=Lx,
    Ly=Ly,
    basis=basis,
    torch_device=device
)

# Add reference (input) layer: air
model.add_ref_layer(er_ref=er_air, mur_ref=1.0)

# Add dielectric layer with optimizable thickness
init_thickness = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
model.add_layer(er_layer=er_dielectric, mur_layer=1.0, thickness=init_thickness, requires_grad=True)

# Add output (transmission) layer: silicon
model.add_trs_layer(er_trs=er_silicon, mur_trs=1.0)

# Register the thickness parameter for optimization
thickness_param = model.thickness_params[0]
optimizer = torch.optim.Adam([thickness_param], lr=0.05)

# Optimization loop to maximize reflection
num_steps = 60
thickness_history = []
reflection_history = []

for step in range(num_steps):
    print(step)
    optimizer.zero_grad()
    # Update the thickness in the model
    model.set_layer_thickness(1, thickness_param)
    model.rebuild()
    reflection, transmission = model.calc_global_ref_trs()
    # We want to maximize reflection, so use negative for minimization
    loss = -reflection.real
    loss.backward()
    optimizer.step()
    # Clamp thickness to positive values
    with torch.no_grad():
        thickness_param.clamp_(min=0.01, max=2.0)
    thickness_history.append(thickness_param.item())
    reflection_history.append(reflection.item())
    if step % 10 == 0 or step == num_steps - 1:
        print(f"Step {step}: thickness = {thickness_param.item():.4f}, reflection = {reflection.item():.4f}")

# Plot reflection vs. thickness
plt.figure()
plt.plot(thickness_history, reflection_history, marker='o')
plt.xlabel('Dielectric Layer Thickness (um)')
plt.ylabel('Reflection')
plt.title('Reflection vs. Dielectric Thickness (Output: Silicon)')
plt.grid(True)
plt.show()
