import torch


class geometry:
    """
    The geometry class includes some common geometries used in RCWA simulation
    Currently geometry includes: circle, rectangle, isosceles triangle, cross and homogeneous layer
    """

    def __init__(self, geometry, grid_spacing=None):
        self.Lx = geometry['Lx']
        self.Ly = geometry['Ly']
        if grid_spacing is None:
            self.grid_spacing = self.Lx / 1000

    def build_iso_triangle(self, width, material, height=None):
        er_shape = material['er_shape']
        mur_shape = material['mur_shape']
        er_space = material['er_space']
        mur_space = material['mur_space']

        if height is None:
            height = width * torch.sqrt(3) / 2
            print(f'height of the isosceles triangle not given, assuming triangle is equilateral')

        # get precision
        N = int(1 / self.grid_spacing)

        y_start = int(self.Lx * N - height / self.Lx * self.Ly * N)
        y_end = int(self.Lx * N + height / self.Lx * self.Ly * N)

        Nx = int(self.Lx * N)
        Ny = int(self.Ly * N)

        ER = torch.ones([2 * Nx, 2 * Ny], dtype=complex) * er_space
        MUR = torch.ones([2 * Nx, 2 * Ny], dtype=complex) * mur_space

        for _iy in range(y_start, y_end):
            f = (_iy - y_start) / (y_end - y_start)
            nx = int(round(f * (width / self.Lx) * Nx))  # x width
            nxstart = 1 + int(torch.floor((Nx - nx)))
            nxend = int(nxstart + 2 * nx) + 1
            ER[nxstart:nxend, _iy] = er_shape
            MUR[nxstart:nxend, _iy] = mur_shape
        return torch.as_tensor(ER[::2, ::2]), torch.as_tensor(MUR[::2, ::2])

    def build_circle(self, radius, material):
        er_shape = material['er_shape']
        mur_shape = material['mur_shape']
        er_space = material['er_space']
        mur_space = material['mur_space']

        # get precision
        N = int(1 / self.grid_spacing)
        radius *= N

        Nx = int(self.Lx * N)
        Ny = int(self.Ly * N)

        ER = torch.ones([2 * Nx, 2 * Ny], dtype=complex) * er_space
        MUR = torch.ones([2 * Nx, 2 * Ny], dtype=complex) * mur_space

        X_mesh, Y_mesh = torch.meshgrid(range(-Nx, Nx), range(-Ny, Ny), indexing='ij')
        ER[X_mesh ** 2 + Y_mesh ** 2 <= radius ** 2] = er_shape
        MUR[X_mesh ** 2 + Y_mesh ** 2 <= radius ** 2] = mur_shape
        return torch.as_tensor(ER[::2, ::2]), torch.as_tensor(MUR[::2, ::2])

    def build_rectangle(self, x_length, material, y_length=None, rotate_angle=0):
        er_shape = material['er_shape']
        mur_shape = material['mur_shape']
        er_space = material['er_space']
        mur_space = material['mur_space']

        if y_length is None:
            y_length = x_length
            print(f'y_length not given, assuming shape is a square')

        N = int(1 / self.grid_spacing)
        x_length *= N
        y_length *= N

        Nx = int(self.Lx * N)
        Ny = int(self.Ly * N)

        ER = torch.ones([2 * Nx, 2 * Ny], dtype=torch.complex64) * er_space
        MUR = torch.ones([2 * Nx, 2 * Ny], dtype=torch.complex64) * mur_space

        X_mesh, Y_mesh = torch.meshgrid(range(-Nx, Nx), range(-Ny, Ny), indexing='ij')

        A = torch.abs(X_mesh * torch.cos(rotate_angle) + Y_mesh * torch.sin(rotate_angle)) < x_length
        B = torch.abs(- X_mesh * torch.sin(rotate_angle) + Y_mesh * torch.cos(rotate_angle)) < y_length

        ER[A & B] = er_shape
        MUR[A & B] = mur_shape

        return torch.as_tensor(ER[::2, ::2]), torch.as_tensor(MUR[::2, ::2])

    def build_cross(self, cross_size, cross_width, material):
        er_shape = material['er_shape']
        mur_shape = material['mur_shape']
        er_space = material['er_space']
        mur_space = material['mur_space']

        N = int(1 / self.grid_spacing)
        Nx = int(self.Lx * N)
        Ny = int(self.Ly * N)

        L = cross_size * N
        W = cross_width * N

        ER = torch.ones((2 * Nx, 2 * Ny), dtype=torch.complex64) * er_space
        MUR = torch.ones((2 * Nx, 2 * Ny), dtype=torch.complex64) * mur_space

        X_mesh, Y_mesh = torch.meshgrid(torch.arange(-Nx, Nx), torch.arange(-Nx, Nx), indexing='ij')

        Av = torch.abs(X_mesh) < L
        Bv = torch.abs(Y_mesh) < W
        Ah = torch.abs(X_mesh) < W
        Bh = torch.abs(Y_mesh) < L

        ER[(Ah & Bh) | (Av & Bv)] = er_shape
        MUR[(Ah & Bh) | (Av & Bv)] = mur_shape

        return torch.as_tensor(ER[::2, ::2]), torch.as_tensor(MUR[::2, ::2])

    def build_homogeneous_layer(self, material):
        er_space = material['er_space']
        mur_space = material['mur_space']

        N = int(1 / self.grid_spacing)
        Nx = int(self.Lx * N)
        Ny = int(self.Ly * N)

        ER = torch.ones([2 * Nx, 2 * Ny], dtype=complex) * er_space
        MUR = torch.ones([2 * Nx, 2 * Ny], dtype=complex) * mur_space

        return torch.as_tensor(ER[::2, ::2]), torch.as_tensor(MUR[::2, ::2])

    def build_periodic_grating(self, width_center, width_side, material):
        er_shape = material['er_shape']
        mur_shape = material['mur_shape']
        er_space = material['er_space']
        mur_space = material['mur_space']

        N = int(1 / self.grid_spacing)
        Nx = int(self.Lx * N)
        Ny = int(self.Ly * N)

        Lc = width_center * N
        Ls = width_side * N

        ER = torch.ones((2 * Nx, 2 * Ny), dtype=torch.complex64) * er_space
        MUR = torch.ones((2 * Nx, 2 * Ny), dtype=torch.complex64) * mur_space

        X_mesh, Y_mesh = torch.meshgrid(torch.arange(-Nx, Nx), torch.arange(-Ny, Ny), indexing='ij')

        A0 = torch.abs(X_mesh) < Lc
        A1 = torch.abs(X_mesh + 1371) < Ls
        A_1 = torch.abs(X_mesh - 1371) < Ls
        A2 = torch.abs(X_mesh + 2552) < Ls
        A_2 = torch.abs(X_mesh - 2552) < Ls

        ER[A0 | A1 | A_1 | A2 | A_2] = er_shape
        MUR[A0 | A1 | A_1 | A2 | A_2] = mur_shape

        return torch.as_tensor(ER[::2, ::2]), torch.as_tensor(MUR[::2, ::2])
