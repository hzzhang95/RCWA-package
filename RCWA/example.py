import torch
import torch.optim as optim
from rcwa import RCWA

# Constants
wavelength = 1.525  # microns
n_air = 1.0
n_sio2 = 1.45
n_si = 3.48
theta = 0.0
phi = 0.0
TE = 1.0
TM = 0.0
Lx = Ly = 1.0  # arbitrary, since we use M=N=0 (no periodicity)
basis = (0, 0)  # only zeroth order

# RCWA setup
model = RCWA(
    wavelength=wavelength,
    theta=theta,
    phi=phi,
    TE=TE,
    TM=TM,
    Lx=Lx,
    Ly=Ly,
    basis=basis,
    requires_grad=True,
    dtype=torch.complex64
)

# Reference (air)
model.add_ref_layer(er_ref=n_air ** 2, mur_ref=1.0)

# Optimizable layer (SiO2)
init_thickness = 0.5 # initial guess in microns
model.add_layer(er_layer=n_sio2 ** 2 - 0.1j, mur_layer=1.0, thickness=init_thickness, optimizing='thickness')

# Transmission port (Si)
model.add_PEC_trs_layer()

# Optimizer
params = [_p for _p in model.thickness_params if _p.requires_grad]
optimizer = optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-08)

# Optimization loop
for _epoch in range(100):
    optimizer.zero_grad()
    model.rebuild()
    model.calc_global_ref_trs()
    abs = model.calc_layer_absorption(layer_number=1, grid_points=10, er_imag=None, mur_imag=None)
    loss = - abs  # maximize reflection
    loss.backward()
    optimizer.step()
    thickness = model.thickness_params[0]
    with torch.no_grad():
        thickness.clamp_(min=0.4, max=0.8)
    if _epoch % 10 == 0:
        print(f"Epoch {_epoch}: Absorption={abs.item():.4f}, Thickness={thickness.item():.4f}")

# Final result
optimal_thickness = model.thickness_params[0].item()
half_wave = wavelength / (2 * n_sio2)

print(f"Optimal thickness: {optimal_thickness:.4f} um")
print(f"Half-wave thickness: {half_wave:.4f} um")
