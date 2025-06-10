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
        """Build a periodic grating structure."""
        Lc = bar_width * self.N
        Ls = bar_spacing * self.N if bar_spacing is not None else Lc
        period = Lc + Ls
        self.er = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)
        self.mur = torch.zeros([2 * self.Nx, 2 * self.Ny], dtype=torch.complex64)
        X_mesh, Y_mesh = torch.meshgrid(
            torch.arange(-self.Nx, self.Nx), torch.arange(-self.Ny, self.Ny), indexing='ij'
        )
        A0 = torch.abs(X_mesh) < Lc
        A1 = torch.abs(X_mesh + period) < Lc
        A_1 = torch.abs(X_mesh - period) < Lc
        A2 = torch.abs(X_mesh + 2 * period) < Lc
        A_2 = torch.abs(X_mesh - 2 *  period) < Lc
        mask = A0 | A1 | A_1 | A2 | A_2
        self.er[mask] = 1
        self.mur[mask] = 1
        return self.er[::2, ::2], self.mur[::2, ::2]
