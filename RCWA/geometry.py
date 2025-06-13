import torch
import math

class geometry(object):
    """
    The geometry class includes some common geometries used in RCWA simulation.
    Geometries: circle, rectangle, isosceles triangle, cross, homogeneous layer, periodic grating.
    """
    def __init__(self, geometry, grid_spacing=None):
        self.Lx = geometry['Lx']
        self.Ly = geometry['Ly']
        self.grid_spacing = grid_spacing if grid_spacing is not None else self.Lx / 1000
        self.N = int(1 / self.grid_spacing)
        self.Nx = int(self.Lx * self.N)
        self.Ny = int(self.Ly * self.N)
        self.build_homogeneous_layer()

    def build_homogeneous_layer(self):
        """Create a homogeneous layer with zeros."""
        self.er = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)
        self.mur = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)

    def build_iso_triangle(self, width, height=None):
        """Build an isosceles triangle (defaults to equilateral if height not given)."""
        if height is None:
            height = width * math.sqrt(3) / 2  # Use math for scalar
        y_start = int(self.Nx - height / self.Lx * self.Ny)
        y_end = int(self.Nx + height / self.Lx * self.Ny)
        for _iy in range(y_start, y_end):
            f = (_iy - y_start) / (y_end - y_start)
            nx = int(round(f * (width / self.Lx) * self.Nx))
            nxstart = 1 + int(math.floor(self.Nx - nx))
            nxend = int(nxstart + 2 * nx) + 1
            self.er[nxstart:nxend, _iy] = 1
            self.mur[nxstart:nxend, _iy] = 1
        return self.er[::2, ::2], self.mur[::2, ::2]

    def build_circle(self, radius):
        """Build a circle of given radius."""
        radius_pix = radius * self.N
        X_mesh, Y_mesh = torch.meshgrid(
            torch.arange(-self.Nx, self.Nx), torch.arange(-self.Ny, self.Ny), indexing='ij'
        )
        mask = X_mesh ** 2 + Y_mesh ** 2 <= radius_pix ** 2
        self.er[mask] = 1
        self.mur[mask] = 1
        return self.er[::2, ::2], self.mur[::2, ::2]

    def build_rectangle(self, x_length, y_length=None, rotate_angle=0):
        """Build a rectangle (or square if y_length not given), optionally rotated."""
        if y_length is None:
            y_length = x_length
        x_length_pix = x_length * self.N
        y_length_pix = y_length * self.N
        X_mesh, Y_mesh = torch.meshgrid(
            torch.arange(-self.Nx, self.Nx), torch.arange(-self.Ny, self.Ny), indexing='ij'
        )
        cos_a = math.cos(rotate_angle)
        sin_a = math.sin(rotate_angle)
        A = torch.abs(X_mesh * cos_a + Y_mesh * sin_a) < x_length_pix
        B = torch.abs(-X_mesh * sin_a + Y_mesh * cos_a) < y_length_pix
        mask = A & B
        self.er[mask] = 1
        self.mur[mask] = 1
        return self.er[::2, ::2], self.mur[::2, ::2]

    def build_cross(self, cross_size, cross_width):
        """Build a cross shape."""
        L = cross_size * self.N
        W = cross_width * self.N
        X_mesh, Y_mesh = torch.meshgrid(
            torch.arange(-self.Nx, self.Nx), torch.arange(-self.Nx, self.Nx), indexing='ij'
        )
        Av = torch.abs(X_mesh) < L
        Bv = torch.abs(Y_mesh) < W
        Ah = torch.abs(X_mesh) < W
        Bh = torch.abs(Y_mesh) < L
        mask = (Ah & Bh) | (Av & Bv)
        self.er[mask] = 1
        self.mur[mask] = 1
        return self.er[::2, ::2], self.mur[::2, ::2]

    def build_periodic_grating(self, bar_width, bar_spacing=None):
        """Build a periodic grating structure filling the entire region, with center always filled."""
        Lc = int(bar_width * self.N)
        Ls = int(bar_spacing * self.N) if bar_spacing is not None else Lc
        period = Lc + Ls
        self.er = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)
        self.mur = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)
        # Center the grating so that a bar is always at the center
        center_idx = self.Nx
        # Find the leftmost bar start so that center is inside a bar
        offset = (center_idx % period) - (Lc // 2)
        x_start_first = -self.Nx - offset
        for x_start in range(x_start_first, self.Nx, period):
            x0 = max(x_start, -self.Nx)
            x1 = min(x_start + Lc, self.Nx)
            idx0 = x0 + self.Nx
            idx1 = x1 + self.Nx
            if idx0 < 0:
                idx0 = 0
            if idx1 > 2 * self.Nx:
                idx1 = 2 * self.Nx
            if idx0 < idx1:
                self.er[idx0:idx1, :] = 1
                self.mur[idx0:idx1, :] = 1
        return self.er[::2, ::2], self.mur[::2, ::2]

    def build_photonic_crystal(self, rod_radius, lattice_constant):
        """
        Build a 2D square lattice photonic crystal of circular rods, symmetric and filled.
        """
        self.er = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)
        self.mur = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)
        rod_radius_pix = rod_radius * self.N
        lattice_pix = lattice_constant * self.N

        X_mesh, Y_mesh = torch.meshgrid(
            torch.arange(-self.Nx, self.Nx), torch.arange(-self.Ny, self.Ny), indexing='ij'
        )

        # Find the number of rods needed to fill the region in each direction
        num_x = int(math.ceil((2 * self.Nx) / lattice_pix))
        num_y = int(math.ceil((2 * self.Ny) / lattice_pix))

        # Compute the offset so that the center rod is at (0,0)
        x_centers = torch.arange(-num_x // 2, num_x // 2 + 1) * lattice_pix
        y_centers = torch.arange(-num_y // 2, num_y // 2 + 1) * lattice_pix

        for xc in x_centers:
            for yc in y_centers:
                mask = (X_mesh - xc) ** 2 + (Y_mesh - yc) ** 2 <= rod_radius_pix ** 2
                self.er[mask] = 1
                self.mur[mask] = 1

        return self.er[::2, ::2], self.mur[::2, ::2]