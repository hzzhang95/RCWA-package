import torch

def _lorentzian_broadening(eigval, broadening_parameter=1.0e-12):
    tmp = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)
    return tmp / (torch.abs(tmp) ** 2 + broadening_parameter)

# Inspiration of this method is taken from:
# A. Francuz, N. Schuch, and B. Vanhecke, Stable and efficient differentiation of tensor network algorithms, Phys. Rev. Res. 7, 013237 (2025).


class stable_eig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        eigval, eigvec = torch.linalg.eig(input)
        ctx.save_for_backward(eigval, eigvec)
        return eigval, eigvec

    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvec):
        eigval, eigvec = ctx.saved_tensors

        # Efficient diagonal matrix from grad_eigval
        grad_eigval_diag = torch.diag_embed(grad_eigval)
        F = _lorentzian_broadening(eigval)
        F.fill_diagonal_(0.0)
        XH = eigvec.conj().transpose(-2, -1)
        S = F * (XH @ grad_eigvec)
        # Combine grad_eigval_diag and S
        grad_input = torch.linalg.solve(XH, (grad_eigval_diag + S) @ XH)
        return grad_input


if __name__ == "__main__":
    # This is used to test the stable_eig function
    torch.manual_seed(0)
    x_real = torch.randn(4, 4)
    x_imag = torch.randn(4, 4)
    x = torch.complex(x_real, x_imag)
    x.requires_grad_()

    eigval, eigvec = stable_eig.apply(x)

    # Define a simple loss as the sum of real parts of eigenvalues
    loss = eigval.real.sum()
    loss.backward()

    # Print gradients
    print("Input matrix x:\n", x)
    print("Eigenvalues:\n", eigval)
    print("Gradient of x:\n", x.grad)
