import torch
import torch.nn as nn
from safe_autograd import stable_eig


def _safe_solve(A, B=None):
    """
    Tries to solve AX = B. If A is singular or ill-conditioned, uses pseudo-inverse for stability.
    """
    if B is None:
        B = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    try:
        return torch.linalg.solve(A, B)
    except RuntimeError:
        return torch.linalg.pinv(A) @ B


class RCWA():
    def __init__(
            self, wavelength, theta, phi, TE, TM, Lx, Ly,
            basis=None, gap_material=None, dtype=torch.complex64, torch_device=None,
            requires_grad=False, stable_eig_grad=False):
        # Device selection
        if torch_device == 'cuda' or torch_device == 'cpu':
            self._torch_device = torch.device(torch_device)
            print(f"Using " + str(torch_device))
        elif torch.cuda.is_available():
            self._torch_device = torch.device('cuda')
            print(f"No device specified, defaulting to CUDA: {torch.cuda.get_device_name(self._torch_device)}")
        else:
            self._torch_device = torch.device('cpu')
            print("No device specified, defaulting to CPU.")

        self._dtype = dtype
        self.requires_grad = requires_grad
        self._stable_grad = stable_eig_grad

        # Gap material
        if gap_material is None:
            self.er_gap = torch.tensor(1, dtype=torch.float32, device=self._torch_device)
            self.mur_gap = torch.tensor(1, dtype=torch.float32, device=self._torch_device)
        else:
            self.er_gap = torch.as_tensor(gap_material['er_gap'], dtype=self._dtype, device=self._torch_device)
            self.mur_gap = torch.as_tensor(gap_material['mur_gap'], dtype=self._dtype, device=self._torch_device)

        # Source
        self.wavelength = torch.as_tensor(wavelength, dtype=self._dtype, device=self._torch_device)
        self.theta = torch.as_tensor(theta, dtype=self._dtype, device=self._torch_device)
        self.phi = torch.as_tensor(phi, dtype=self._dtype, device=self._torch_device)
        self.TE = torch.as_tensor(TE, dtype=self._dtype, device=self._torch_device)
        self.TM = torch.as_tensor(TM, dtype=self._dtype, device=self._torch_device)

        # Lattice constants
        self.Lx = torch.as_tensor(Lx, dtype=self._dtype, device=self._torch_device)
        self.Ly = torch.as_tensor(Ly, dtype=self._dtype, device=self._torch_device)

        # Basis
        print(f"Basis: {int(basis[0])} (x) x {int(basis[1])} (y)")
        self.M = torch.as_tensor(basis[0], dtype=torch.int32, device=self._torch_device)
        self.N = torch.as_tensor(basis[1], dtype=torch.int32, device=self._torch_device)
        self.MN = (2 * self.M + 1) * (2 * self.N + 1)

        self.k0 = 2 * torch.pi / self.wavelength
        self.M = torch.arange(-self.M.item(), self.M.item() + 1, dtype=torch.int32, device=self._torch_device)
        self.N = torch.arange(-self.N.item(), self.N.item() + 1, dtype=torch.int32, device=self._torch_device)

        self.I_MN = torch.eye(self.MN.item(), dtype=self._dtype, device=self._torch_device)
        self.O_MN = torch.zeros_like(self.I_MN)
        self.I_2MN = torch.eye(2 * self.MN.item(), dtype=self._dtype, device=self._torch_device)
        self.O_2MN = torch.zeros_like(self.I_2MN)

        # Global scattering matrices
        self.S_global = torch.stack([self.O_2MN.clone(), self.I_2MN.clone(), self.I_2MN.clone(), self.O_2MN.clone()])

        # Storage for simulation
        self.W_store, self.V_store, self.Lamda_store, self.S_global_store = [], [], [], []
        self.layer_configs = []
        self.layer_count = 0
        self.rebuild_counter = 0

        if self.requires_grad:
            self.thickness_params = nn.ParameterList()
            self.er_params = nn.ParameterList()
            self.mur_params = nn.ParameterList()

    def _to_tensor(self, in_data, dtype):
        """
        Converts input data to a 2D torch tensor.
        """
        if dtype is None:
            dtype = self._dtype
        if not torch.is_tensor(in_data):
            out_data = torch.as_tensor(in_data, dtype=dtype, device=self._torch_device)
        else:
            out_data = in_data.to(dtype=dtype, device=self._torch_device)
        if out_data.ndim == 0:
            return out_data.reshape(1, 1)
        elif out_data.ndim == 1:
            return out_data.unsqueeze(0)
        return out_data

    def add_ref_layer(self, er_ref=1.0, mur_ref=1.0):
        self.er_ref = torch.as_tensor(er_ref, dtype=self._dtype, device=self._torch_device)
        self.mur_ref = torch.as_tensor(mur_ref, dtype= self._dtype, device=self._torch_device)
        self.n_inc = torch.sqrt(self.er_ref * self.mur_ref)
        sin_theta = torch.sin(self.theta)
        cos_theta = torch.cos(self.theta)
        cos_phi = torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)
        self.kx_inc = self.n_inc * sin_theta * cos_phi
        self.ky_inc = self.n_inc * sin_theta * sin_phi
        self.kz_inc = self.n_inc * cos_theta
        self.k_inc = torch.stack([self.kx_inc, self.ky_inc, self.kz_inc], dim=0)
        delta_kx = 2 * torch.pi * self.M.item() / (self.k0 * self.Lx)
        delta_ky = 2 * torch.pi * self.N.item() / (self.k0 * self.Ly)
        self.kx = self.kx_inc - delta_kx
        self.ky = self.ky_inc - delta_ky
        self.Ksx, self.Ksy = torch.meshgrid(self.kx, self.ky, indexing='ij')
        self.Kx = torch.diag(self.Ksx.flatten()).to(dtype=self._dtype, device=self._torch_device)
        self.Ky = torch.diag(self.Ksy.flatten()).to(dtype=self._dtype, device=self._torch_device)
        self.kz_ref = self._find_kz(self.er_ref.item(), self.mur_ref.item())
        self._initialize_gap_medium()

    def _initialize_gap_medium(self, er_gap=1.0, mur_gap=1.0):
        kz_squared = self.I_MN.clone() * er_gap * mur_gap - self.Kx @ self.Kx - self.Ky @ self.Ky
        Kz_g = torch.sqrt(kz_squared).conj()
        self.W_g = self.I_2MN.clone()
        KxKy = self.Kx @ self.Ky
        Kx2 = self.Kx @ self.Kx
        Ky2 = self.Ky @ self.Ky
        upper_block = torch.hstack([KxKy, self.I_MN.clone() - Kx2])
        lower_block = torch.hstack([Ky2 - self.I_MN.clone(), -KxKy])
        self.Q_g = torch.vstack([upper_block, lower_block])
        inv_diag_Kz = torch.diag(-1j / torch.diag(Kz_g))
        inv_lamda_g = torch.block_diag(inv_diag_Kz, inv_diag_Kz)
        self.V_g = self.Q_g @ inv_lamda_g

    def add_layer(self, er_layer=1.0, mur_layer=1.0, thickness=0.0, optimizing=None):
        """
        Add a layer, if requires_grad=True, thickness, er, and mur will be learnable parameters.
        """
        er_layer = self._to_tensor(er_layer, dtype=self._dtype)
        mur_layer = self._to_tensor(mur_layer, dtype=self._dtype)
        thickness = self._to_tensor(thickness, dtype=torch.float32)
        if self.rebuild_counter == 0:
            if self.requires_grad:
                if optimizing == 'thickness':
                    thickness = nn.Parameter(thickness)
                    self.thickness_params.append(thickness)
                elif optimizing == 'er':
                    er_layer = nn.Parameter(er_layer)
                    self.er_params.append(er_layer)
                elif optimizing == 'mur':
                    mur_layer = nn.Parameter(mur_layer)
                    self.mur_params.append(mur_layer)
            self.layer_configs.append({'thickness': thickness, 'er': er_layer, 'mur': mur_layer, 'opt': optimizing})

        _idx = len(self.layer_configs) - 1 if self.rebuild_counter == 0 else len(self.W_store)
        er_layer = self.layer_configs[_idx]['er']
        mur_layer = self.layer_configs[_idx]['mur']
        thickness = self.layer_configs[_idx]['thickness']

        er_conv = self._convolution_matrices(er_layer)
        mur_conv = self._convolution_matrices(mur_layer)
        self.S_global = self._layer_S_matrix(thickness, self.S_global, er_conv, mur_conv)

    def add_trs_layer(self, er_trs=1.0, mur_trs=1.0):
        self.er_trs = torch.as_tensor(er_trs, dtype=self._dtype, device=self._torch_device)
        self.mur_trs = torch.as_tensor(mur_trs, dtype=self._dtype, device=self._torch_device)
        self.kz_trs = self._find_kz(self.er_trs.item(), self.mur_trs.item())
        self.S_global = self._solve_trs_region_S_matrix(self.S_global)

    def add_PEC_trs_layer(self):
        """
        Add a PEC boundary condition for the transmission region. (i.e. reflection of all modes)
        """
        self.er_trs = None
        self.mur_trs = torch.as_tensor(1.0, dtype=self._dtype, device=self._torch_device)
        smallest_normal = torch.finfo(self._dtype).smallest_normal
        self.kz_trs = torch.ones_like(self.kz_ref, dtype=self._dtype, device=self._torch_device) * smallest_normal
        self.S_global = self._solve_PEC_S_matrix(self.S_global)

    def _solve_PEC_S_matrix(self, S_global):
        """
        PEC boundary: S11 = S22 = -I, S12 = S21 = 0
        """
        S11_trs = S22_trs = - self.I_2MN.clone()
        S12_trs = S21_trs = self.O_2MN.clone()
        S_trs = torch.stack([S11_trs, S12_trs, S21_trs, S22_trs])
        S_global = self._Redheffer_star(S_global, S_trs)
        self.W_store.append(self.I_2MN.clone())
        self.V_store.append(self.I_2MN.clone())
        self.Lamda_store.append(self.O_2MN.clone())
        self.S_global_store.append(S_global)
        return S_global

    def _solve_trs_region_S_matrix(self, S_global):
        """
        Transmission region S-matrix. If gap and transmission media are identical, S11 = S22 = 0, S12 = S21 = I.
        """
        Q_trs = self._Q_matrix_half(self.er_trs, self.mur_trs)
        Lamda_trs_inv = torch.diag(torch.hstack((-1j / self.kz_trs, -1j / self.kz_trs)))
        V_trs = Q_trs @ Lamda_trs_inv
        if self.er_gap != self.er_trs or self.mur_gap != self.mur_trs:
            S11_trs, S12_trs, S21_trs, S22_trs = self._scattering_matrix_trs(V_trs)
        else:
            S11_trs = S22_trs = self.O_2MN.clone()
            S12_trs = S21_trs = self.I_2MN.clone()
        S_trs = torch.stack([S11_trs, S12_trs, S21_trs, S22_trs])
        S_global = self._Redheffer_star(S_global, S_trs)
        self.W_store.append(self.I_2MN.clone())
        self.V_store.append(V_trs)
        self.Lamda_store.append(torch.diag(torch.hstack((1j * self.kz_trs, 1j * self.kz_trs))))
        self.S_global_store.append(S_global)
        return S_global

    def _Redheffer_star(self, S_A, S_B):
        """
        Redheffer star product for cascading two scattering matrices.
        """
        S11_A, S12_A, S21_A, S22_A = S_A
        S11_B, S12_B, S21_B, S22_B = S_B
        D = self.I_2MN.clone() - S11_B @ S22_A
        F = self.I_2MN.clone() - S22_A @ S11_B
        S11_AB = S11_A + S12_A @ _safe_solve(D, S11_B @ S21_A)
        S12_AB = S12_A @ _safe_solve(D, S12_B)
        S21_AB = S21_B @ _safe_solve(F, S21_A)
        S22_AB = S22_B + S21_B @ _safe_solve(F, S22_A @ S12_B)
        S_AB = torch.stack([S11_AB, S12_AB, S21_AB, S22_AB])
        return S_AB

    def calc_global_ref_trs(self):
        # Input mode is always a plane wave with M = N = 0
        delta_mn = self.O_MN.clone()
        delta_mn[self.MN // 2] = 1
        a_z = torch.tensor([0, 0, -1], dtype=self._dtype, device=self._torch_device)
        a_TE = torch.tensor([0, 1, 0], dtype=self._dtype, device=self._torch_device)
        if self.theta != 0:
            a_TE = torch.linalg.cross(a_z, self.k_inc)
            a_TE /= torch.linalg.norm(a_TE)
        a_TM = torch.linalg.cross(self.k_inc, a_TE)
        a_TM /= torch.linalg.norm(a_TM)
        P = self.TE * a_TE + self.TM * a_TM
        P /= torch.linalg.norm(P)
        self.c_src = torch.hstack((P[0] * delta_mn, P[1] * delta_mn)).squeeze()
        self.c_ref = self.S_global[0] @ self.c_src
        self.c_trs = self.S_global[2] @ self.c_src
        rx, ry = torch.split(self.c_ref, self.MN.item())
        tx, ty = torch.split(self.c_trs, self.MN.item())
        rz = - torch.diag(1 / self.kz_ref) @ (self.Kx @ rx + self.Ky @ ry)
        tz = - torch.diag(1 / self.kz_trs) @ (self.Kx @ tx + self.Ky @ ty)
        REF = torch.real(torch.diag(self.kz_ref) / self.mur_ref) @ (
                torch.abs(rx) ** 2 + torch.abs(ry) ** 2 + torch.abs(rz) ** 2) / torch.real(
            self.kz_inc / self.mur_ref)
        TRS = torch.real(torch.diag(self.kz_trs) / self.mur_trs) @ (
                torch.abs(tx) ** 2 + torch.abs(ty) ** 2 + torch.abs(tz) ** 2) / torch.real(
            self.kz_inc / self.mur_ref)
        return REF.sum(), TRS.sum()

    def _ref_region_S_matrix(self, S_global):
        Q_ref = self._Q_matrix_half(self.er_ref, self.mur_ref)
        Lamda_ref_inv = torch.diag(torch.cat([1j / self.kz_ref, 1j / self.kz_ref]))
        V_ref = Q_ref @ Lamda_ref_inv
        if self.er_gap != self.er_ref or self.mur_gap != self.mur_ref:
            S11_ref, S12_ref, S21_ref, S22_ref = self._scattering_matrix_ref(V_ref)
        else:
            S11_ref = S22_ref = self.O_2MN.clone()
            S12_ref = S21_ref = self.I_2MN.clone()
        S_ref = torch.stack([S11_ref, S12_ref, S21_ref, S22_ref])
        S_global = self._Redheffer_star(S_ref, S_global)
        self.W_store.append(self.I_2MN.clone())
        self.V_store.append(V_ref)
        self.Lamda_store.append(torch.cat([-1j * self.kz_ref, -1j * self.kz_ref]))
        self.S_global_store.append(S_global)
        return S_global

    def _layer_S_matrix(self, layer_thickness, S_global, er_conv, mur_conv):
        P_layer = self._P_matrix(er_conv, mur_conv)
        Q_layer = self._Q_matrix(er_conv, mur_conv)
        OM2_layer = P_layer @ Q_layer

        if self._stable_grad:
            Lamda2_layer, W_layer = stable_eig.apply(OM2_layer)
        else:
            Lamda2_layer, W_layer = torch.linalg.eig(OM2_layer)

        Lamda_layer = torch.sqrt(Lamda2_layer)
        Lamda_inv = torch.diag(1 / Lamda_layer)
        V_layer = Q_layer @ W_layer @ Lamda_inv
        X_layer = torch.diag(torch.exp(-Lamda_layer * self.k0 * layer_thickness).squeeze())
        S11, S12 = self._scattering_matrix(W_layer, V_layer, X_layer)
        S_layer = torch.stack([S11, S12, S12, S11])
        S_global = self._Redheffer_star(S_global, S_layer)
        self.W_store.append(W_layer)
        self.V_store.append(V_layer)
        self.Lamda_store.append(Lamda_layer)
        self.S_global_store.append(S_global)
        return S_global

    def _find_kz(self, er=1.0, mur=1.0):
        er = self._to_tensor(er, dtype=self._dtype) if not torch.is_tensor(er) else er
        mur = self._to_tensor(mur, dtype=self._dtype) if not torch.is_tensor(mur) else mur
        erc, murc = torch.conj(er), torch.conj(mur)
        kz = torch.conj(torch.sqrt(erc * murc - self.Ksx ** 2 - self.Ksy ** 2).reshape(-1))
        return kz

    def _P_matrix(self, er_conv, mur_conv):
        assert self.Kx.shape == self.Ky.shape == er_conv.shape == mur_conv.shape, 'Kx, Ky and convolution of permittivity/permeability must have same shape'
        P_11 = self.Kx @ torch.linalg.solve(er_conv, self.Ky)
        P_12 = mur_conv - self.Kx @ torch.linalg.solve(er_conv, self.Kx)
        P_21 = self.Ky @ torch.linalg.solve(er_conv, self.Ky) - mur_conv
        P_22 = - self.Ky @ torch.linalg.solve(er_conv, self.Kx)
        P = torch.vstack([torch.hstack([P_11, P_12]), torch.hstack([P_21, P_22])])
        return P

    def _Q_matrix(self, er_conv, mur_conv):
        assert self.Kx.shape == self.Ky.shape == er_conv.shape == mur_conv.shape, 'Kx, Ky and convolution of permittivity/permeability must have same shape'
        Q_11 = self.Kx @ torch.linalg.solve(mur_conv, self.Ky)
        Q_12 = er_conv - self.Kx @ torch.linalg.solve(mur_conv, self.Kx)
        Q_21 = self.Ky @ torch.linalg.solve(mur_conv, self.Ky) - er_conv
        Q_22 = - self.Ky @ torch.linalg.solve(mur_conv, self.Kx)
        Q = torch.vstack([torch.hstack([Q_11, Q_12]), torch.hstack([Q_21, Q_22])])
        return Q

    def _Q_matrix_half(self, er, mur):
        Q_11 = self.Kx @ self.Ky / mur
        Q_12 = er * self.I_MN.clone() - self.Kx @ self.Kx / mur
        Q_21 = self.Ky @ self.Ky / mur - er * self.I_MN.clone()
        Q_22 = - self.Ky @ self.Kx / mur
        Q = torch.vstack([torch.hstack([Q_11, Q_12]), torch.hstack([Q_21, Q_22])])
        return Q

    def _scattering_matrix(self, W, V, X):
        """
        Computes S11 and S12 submatrices for a layer with a gap medium on both sides.
        """
        W_inv_Wg = _safe_solve(W, self.W_g)
        V_inv_Vg = _safe_solve(V, self.V_g)
        A = W_inv_Wg + V_inv_Vg
        B = W_inv_Wg - V_inv_Vg
        A_inv = _safe_solve(A)
        XA = X @ A
        XB = X @ B
        D = A - XB @ A_inv @ XB
        S11 = _safe_solve(D, XB @ A_inv @ XA - B)
        S12 = _safe_solve(D, X @ (A - B @ A_inv @ B))
        return S11, S12

    def _scattering_matrix_ref(self, V_ref):
        """
        Scattering matrix for the reflection region only.
        """
        V_ref = torch.as_tensor(V_ref, dtype=self._dtype, device=self._torch_device)
        V_g = self.V_g
        A = self.I_2MN.clone() + _safe_solve(V_g, V_ref)
        B = self.I_2MN.clone() - _safe_solve(V_g, V_ref)
        S11 = -_safe_solve(A, B)
        S12 = 2 * _safe_solve(A, self.I_2MN.clone())
        S21 = 0.5 * (A - B @ _safe_solve(A, B))
        S22 = _safe_solve(A, B)
        return S11, S12, S21, S22

    def _scattering_matrix_trs(self, V_trs):
        """
        Scattering matrix for the transmission region only.
        """
        V_trs = torch.as_tensor(V_trs, dtype=self._dtype, device=self._torch_device)
        V_g = self.V_g
        A = self.I_2MN.clone() + _safe_solve(V_g, V_trs)
        B = self.I_2MN.clone() - _safe_solve(V_g, V_trs)
        S11 = _safe_solve(A, B)
        S12 = 0.5 * (A - B @ _safe_solve(A, B))
        S21 = 2 * _safe_solve(A, self.I_2MN.clone())
        S22 = - _safe_solve(A, B)
        return S11, S12, S21, S22

    def _convolution_matrices(self, H):
        """
        Constructs the convolution matrix H_conv for a given real-space representation H.
        """
        H = torch.as_tensor(H, dtype=self._dtype, device=self._torch_device)
        if H.dim() == 0 or (H.dim() == 2 and (H == H[0, 0]).all()):
            return self.I_MN.clone() * H.flatten()[0]
        return self._2dconv_matrices(H)

    def _2dconv_matrices(self, H):
        """
        Builds the full 2D convolution matrix from the real-space material tensor.
        """
        x_mesh, y_mesh = torch.meshgrid(self.M, self.N, indexing='ij')
        x_flat, y_flat = x_mesh.flatten(), y_mesh.flatten()
        index = torch.arange(self.MN.item(), dtype=torch.int32, device=self._torch_device)
        x_index, y_index = torch.meshgrid(index, index, indexing='ij')
        H_fft = torch.fft.fft2(H) / (H.shape[0] * H.shape[1])
        H_conv = H_fft[x_flat[x_index] - x_flat[y_index], y_flat[x_index] - y_flat[y_index]]
        return H_conv

    def solve_xy_field(self, layer_number, z_depth):
        layer_number -= 1  # Adjust for zero-based indexing
        layer_thickness = float(self.layer_configs[layer_number]['thickness'])
        if z_depth > layer_thickness:
            raise ValueError('z depth must be less than thickness of the layer')

        er = self.layer_configs[layer_number]['er']
        mur = self.layer_configs[layer_number]['mur']
        er_conv = self._convolution_matrices(er)
        mur_conv = self._convolution_matrices(mur)

        c_ln = _safe_solve(self.S_global_store[layer_number][1],
                           self.c_ref - (self.S_global_store[layer_number][0] @ self.c_src))
        c_lp = (self.S_global_store[layer_number][2] @ self.c_src) + (
            self.S_global_store[layer_number][3] @ c_ln)

        W = self.W_store[layer_number + 1]
        V = self.V_store[layer_number + 1]
        M_int = torch.vstack([torch.hstack([W, W]), torch.hstack([-V, V])])
        M_g = torch.vstack(
            [torch.hstack([self.I_2MN.clone(), self.I_2MN.clone()]), torch.hstack([-self.V_g, self.V_g])])
        c_int = _safe_solve(M_int, M_g) @ torch.hstack([c_lp, c_ln])

        eig_val = self.Lamda_store[layer_number + 1]
        X_grid = torch.arange(er.shape[0], device=self._torch_device) / er.shape[0] * self.Lx
        Y_grid = torch.arange(er.shape[1], device=self._torch_device) / er.shape[1] * self.Ly
        X_grid, Y_grid = torch.meshgrid(X_grid, Y_grid, indexing='ij')
        phi = torch.exp(-1j * self.k0 * (self.kx_inc * X_grid + self.ky_inc * Y_grid))

        Prop = torch.diag(
            torch.hstack([
                torch.exp(-self.k0 * eig_val * z_depth),
                torch.exp(-self.k0 * eig_val * (layer_thickness - z_depth))
            ])
        )
        psi = c_int @ Prop

        sx, sy, ux, uy = torch.split(psi, self.MN.item(), dim=0)
        sz = -1j * _safe_solve(er_conv, self.Kx @ uy - self.Ky @ ux)
        uz = -1j * _safe_solve(mur_conv, self.Kx @ sy - self.Ky @ sx)

        Ex = phi * torch.fft.ifft2(sx.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), s=er.shape, norm='forward')
        Ey = phi * torch.fft.ifft2(sy.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), s=er.shape, norm='forward')
        Ez = phi * torch.fft.ifft2(sz.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), s=er.shape, norm='forward')
        Hx = phi * torch.fft.ifft2(ux.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), s=mur.shape,
                                   norm='forward')
        Hy = phi * torch.fft.ifft2(uy.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), s=mur.shape,
                                   norm='forward')
        Hz = phi * torch.fft.ifft2(uz.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), s=mur.shape,
                                   norm='forward')
        return Ex, Ey, Ez, Hx, Hy, Hz

    def solve_xz_field(self, layer_number, precision=0.01):
        layer_number -= 1  # Adjust for zero-based indexing
        layer_thickness = float(self.layer_configs[layer_number]['thickness'])
        points = int(layer_thickness / precision)
        z_points = torch.linspace(0, layer_thickness, points, device=self._torch_device)

        er = self.layer_configs[layer_number]['er']
        mur = self.layer_configs[layer_number]['mur']
        er_conv = self._convolution_matrices(er)
        mur_conv = self._convolution_matrices(mur)

        c_ln = _safe_solve(self.S_global_store[layer_number][1],
                           self.c_ref - (self.S_global_store[layer_number][0] @ self.c_src))
        c_lp = (self.S_global_store[layer_number][2] @ self.c_src) + (
            self.S_global_store[layer_number][3] @ c_ln)

        W = self.W_store[layer_number + 1]
        V = self.V_store[layer_number + 1]
        M_int = torch.vstack([torch.hstack([W, W]), torch.hstack([-V, V])])
        M_g = torch.vstack(
            [torch.hstack([self.I_2MN.clone(), self.I_2MN.clone()]), torch.hstack([-self.V_g, self.V_g])])
        c_int = _safe_solve(M_int, M_g) @ torch.hstack([c_lp, c_ln])

        eig_val = self.Lamda_store[layer_number + 1]

        # calculate the field phase phi(x,y) in the x-y plane
        X_grid = torch.arange(er.shape[0], device=self._torch_device) / er.shape[0] * self.Lx
        phi = torch.exp(-1j * self.k0 * (self.kx_inc * X_grid))

        Ex = []
        Ey = []
        Ez = []
        Hx = []
        Hy = []
        Hz = []
        for _z in z_points:
            Prop = torch.diag(
                torch.hstack([torch.exp(-self.k0 * eig_val * _z),
                              torch.exp(-self.k0 * eig_val * (layer_thickness - _z))]))

            psi = M_int @ Prop @ c_int
            sx, sy, ux, uy = torch.split(psi, self.MN.item(), dim=0)

            sz = -1j * _safe_solve(er_conv, self.Kx @ uy - self.Ky @ ux)
            uz = -1j * _safe_solve(mur_conv, self.Kx @ sy - self.Ky @ sx)

            # calculate the E and H field by inverse FFT
            Ex.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(sx.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), norm='forward')[:, 0],
                    n=er.shape[0], norm='forward')))
            Ey.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(sy.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), norm='forward')[:, 0],
                    n=er.shape[0], norm='forward')))
            Ez.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(sz.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), norm='forward')[:, 0],
                    n=er.shape[0], norm='forward')))
            Hx.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(ux.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), norm='forward')[:, 0],
                    n=er.shape[0], norm='forward')))
            Hy.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(uy.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), norm='forward')[:, 0],
                    n=er.shape[0], norm='forward')))
            Hz.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(uz.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1), norm='forward')[:, 0],
                    n=er.shape[0], norm='forward')))

        Ex = torch.stack(Ex)
        Ey = torch.stack(Ey)
        Ez = torch.stack(Ez)
        Hx = torch.stack(Hx)
        Hy = torch.stack(Hy)
        Hz = torch.stack(Hz)
        return Ex, Ey, Ez, Hx, Hy, Hz

    def solve_yz_field(self, layer_number, precision=0.01):
        layer_number -= 1  # Adjust for zero-based indexing
        layer_thickness = float(self.layer_configs[layer_number]['thickness'])
        points = int(layer_thickness / precision)
        z_points = torch.linspace(0, layer_thickness, points, device=self._torch_device)

        er = self.layer_configs[layer_number]['er']
        mur = self.layer_configs[layer_number]['mur']
        er_conv = self._convolution_matrices(er)
        mur_conv = self._convolution_matrices(mur)

        c_ln = _safe_solve(self.S_global_store[layer_number][1],
                           self.c_ref - (self.S_global_store[layer_number][0] @ self.c_src))
        c_lp = (self.S_global_store[layer_number][2] @ self.c_src) + (
            self.S_global_store[layer_number][3] @ c_ln)

        W = self.W_store[layer_number + 1]
        V = self.V_store[layer_number + 1]
        M_int = torch.vstack([torch.hstack([W, W]), torch.hstack([-V, V])])
        M_g = torch.vstack(
            [torch.hstack([self.I_2MN.clone(), self.I_2MN.clone()]), torch.hstack([-self.V_g, self.V_g])])
        c_int = _safe_solve(M_int, M_g) @ torch.hstack([c_lp, c_ln])

        eig_val = self.Lamda_store[layer_number + 1]

        # calculate the field phase phi(x,y) in the x-y plane
        X_grid = torch.arange(er.shape[0], device=self._torch_device) / er.shape[0] * self.Lx
        phi = torch.exp(-1j * self.k0 * (self.kx_inc * X_grid))

        Ex = []
        Ey = []
        Ez = []
        Hx = []
        Hy = []
        Hz = []
        for _z in z_points:
            Prop = torch.diag(
                torch.hstack([torch.exp(-self.k0 * eig_val * _z),
                              torch.exp(-self.k0 * eig_val * (layer_thickness - _z))]))

            psi = M_int @ Prop @ c_int
            sx, sy, ux, uy = torch.split(psi, self.MN.item(), dim=0)

            sz = -1j * _safe_solve(er_conv, self.Kx @ uy - self.Ky @ ux)
            uz = -1j * _safe_solve(mur_conv, self.Kx @ sy - self.Ky @ sx)

            # calculate the E and H field by inverse FFT
            Ex.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(sx.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1).T, norm='forward')[:, 0],
                    n=er.shape[1], norm='forward')))
            Ey.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(sy.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1).T, norm='forward')[:, 0],
                    n=er.shape[1], norm='forward')))
            Ez.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(sz.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1).T, norm='forward')[:, 0],
                    n=er.shape[1], norm='forward')))
            Hx.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(ux.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1).T, norm='forward')[:, 0],
                    n=er.shape[1], norm='forward')))
            Hy.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(uy.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1).T, norm='forward')[:, 0],
                    n=er.shape[1], norm='forward')))
            Hz.append(phi * torch.fft.ifftshift(
                torch.fft.ifft(
                    torch.fft.ifft(uz.reshape(2 * self.M.item() + 1, 2 * self.N.item() + 1).T, norm='forward')[:, 0],
                    n=er.shape[1], norm='forward')))

        Ex = torch.stack(Ex)
        Ey = torch.stack(Ey)
        Ez = torch.stack(Ez)
        Hx = torch.stack(Hx)
        Hy = torch.stack(Hy)
        Hz = torch.stack(Hz)
        return Ex, Ey, Ez, Hx, Hy, Hz

    def calc_layer_absorption(self, layer_number, grid_points=10, er_imag=None, mur_imag=None):
        """
        Calculate the absorption in a given layer by integrating the power loss density over the layer thickness.
        """
        layer_number -= 1  # Adjust for zero-based indexing
        er = self.layer_configs[layer_number]['er']
        mur = self.layer_configs[layer_number]['mur']
        if er_imag is None:
            er_imag = er.imag
        if mur_imag is None:
            mur_imag = mur.imag
        er_imag = torch.as_tensor(er_imag, device=self._torch_device)
        mur_imag = torch.as_tensor(mur_imag, device=self._torch_device)
        if er_imag.all() == 0 and mur_imag.all() == 0:
            return torch.tensor(0.0, device=self._torch_device)

        thickness = self.layer_configs[layer_number]['thickness'].item()
        Nz = torch.linspace(0, thickness, grid_points + 1, dtype=torch.float32, device=self._torch_device)
        dz = thickness / grid_points

        dx = self.Lx / er.shape[0]
        dy = self.Ly / er.shape[1]

        er_conv = self._convolution_matrices(er)
        mur_conv = self._convolution_matrices(mur)

        c_ln = _safe_solve(self.S_global_store[layer_number][1],
                           self.c_ref - (self.S_global_store[layer_number][0] @ self.c_src))
        c_lp = (self.S_global_store[layer_number][2] @ self.c_src) + (
            self.S_global_store[layer_number][3] @ c_ln)

        W = self.W_store[layer_number + 1]
        V = self.V_store[layer_number + 1]
        M_int = torch.vstack([torch.hstack([W, W]), torch.hstack([-V, V])])
        M_g = torch.vstack(
            [torch.hstack([self.I_2MN.clone(), self.I_2MN.clone()]), torch.hstack([-self.V_g, self.V_g])])
        c_int = _safe_solve(M_int, M_g) @ torch.hstack([c_lp, c_ln])

        eig_val = self.Lamda_store[layer_number]
        P_layer_abs = []

        for _z in Nz:
            Prop = torch.diag(
                torch.hstack([
                    torch.exp(-self.k0 * eig_val * _z),
                    torch.exp(-self.k0 * eig_val * (thickness - _z))
                ])
            )
            psi = M_int @ Prop @ c_int
            sx, sy, ux, uy = torch.split(psi, self.MN.item(), dim=0)
            sz = -1j * _safe_solve(er_conv, self.Kx @ uy - self.Ky @ ux)
            uz = -1j * _safe_solve(mur_conv, self.Kx @ sy - self.Ky @ sx)

            target_size = (2 * self.M.item() + 1, 2 * self.N.item() + 1)
            ex = torch.fft.ifftshift(torch.fft.ifft2(sx.reshape(target_size), s=er.shape))
            ey = torch.fft.ifftshift(torch.fft.ifft2(sy.reshape(target_size), s=er.shape))
            ez = torch.fft.ifftshift(torch.fft.ifft2(sz.reshape(target_size), s=er.shape))
            hx = torch.fft.ifftshift(torch.fft.ifft2(ux.reshape(target_size), s=mur.shape))
            hy = torch.fft.ifftshift(torch.fft.ifft2(uy.reshape(target_size), s=mur.shape))
            hz = torch.fft.ifftshift(torch.fft.ifft2(uz.reshape(target_size), s=mur.shape))

            E_norm2 = torch.abs(ex) ** 2 + torch.abs(ey) ** 2 + torch.abs(ez) ** 2
            H_norm2 = torch.abs(hx) ** 2 + torch.abs(hy) ** 2 + torch.abs(hz) ** 2
            P_abs = - 0.5 * E_norm2 * er_imag * dx * dy \
                    - 0.5 * H_norm2 * mur_imag * dx * dy
            P_layer_abs.append(torch.sum(P_abs).real)

        P_layer_abs = torch.stack(P_layer_abs)
        layer_abs = torch.trapz(P_layer_abs, dx=dz)
        return layer_abs

    def reset_stores(self):
        """
        Reset all internal storage lists and S_global to initial state.
        """
        self.W_store.clear()
        self.V_store.clear()
        self.Lamda_store.clear()
        self.S_global_store.clear()
        self.S_global = torch.stack([self.O_2MN.clone(), self.I_2MN.clone(), self.I_2MN.clone(), self.O_2MN.clone()])

    def rebuild(self):
        """
        Rebuild the S-matrix stack using the current thickness, er, and mur parameters.
        This should be called at each optimization step to ensure the RCWA state is up-to-date.
        """
        self.rebuild_counter += 1
        self.reset_stores()
        self.add_ref_layer(er_ref=self.er_ref.item(), mur_ref=self.mur_ref.item())
        for config in self.layer_configs:
            self.add_layer(config['er'], config['mur'], config['thickness'], config.get('optimizing', None))

        if getattr(self, 'er_trs', None) is not None and getattr(self, 'mur_trs', None) is not None:
            self.add_trs_layer(er_trs=self.er_trs.item(), mur_trs=self.mur_trs.item())
        else:
            self.add_PEC_trs_layer()
