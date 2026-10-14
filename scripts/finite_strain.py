"""Multiplicative isotropic plasticity in the initially stressed reference body.

The referential elastic metric A = F^-1 b_e F^-T carries material history.
Hencky energy and Kirchhoff stress use the original volume; MC/Rankine
strengths retain their Cauchy-stress meaning. See B2 section 7.38.
"""
import numpy as np
import torch

from fracture_physics import _PrincipalStrengthReturn


class _SymmetricLog(torch.autograd.Function):
    """SPD logarithm with the continuous derivative at repeated eigenvalues."""
    @staticmethod
    def forward(ctx, matrix):
        values, axes = np.linalg.eigh(matrix.detach().cpu().numpy())
        if np.any(values <= 0):
            raise ArithmeticError('Elastic metric is not positive definite')
        logs = np.log(values)
        gap = values[..., :, None]-values[..., None, :]
        average = (values[..., :, None]+values[..., None, :])/2
        divided = 1/average
        np.divide(logs[..., :, None]-logs[..., None, :], gap, out=divided,
                  where=np.abs(gap) > 1.e-7*average)
        vectors = torch.as_tensor(axes, dtype=matrix.dtype, device=matrix.device)
        ctx.save_for_backward(vectors, torch.as_tensor(divided, dtype=matrix.dtype, device=matrix.device))
        return torch.as_tensor((axes*logs[..., None, :])@axes.swapaxes(-1, -2),
                               dtype=matrix.dtype, device=matrix.device)

    @staticmethod
    def backward(ctx, gradient):
        axes, divided = ctx.saved_tensors
        local = axes.transpose(-1, -2)@((gradient+gradient.transpose(-1, -2))/2)@axes
        return axes@(divided*local)@axes.transpose(-1, -2)


def symmetric_log(matrix):
    return _SymmetricLog.apply(matrix)


def material_response(deformation, metric_old, lam, mu, initial_log_volume, b):
    """Exponential principal return; evaluations leave accepted A unchanged.

    Returning tau_trial/J with the original Lamé moduli gives the same
    stress projection as using C/J: its absolute scale cancels from C G
    (N C G)^-1. Multiply the reported multipliers by J to recover the actual
    logarithmic plastic increment. Differentiation includes both J factors.
    """
    j = torch.linalg.det(deformation)
    if bool(torch.any(j <= 0)):
        raise ArithmeticError('The deformation gradient has nonpositive determinant')
    eye = torch.eye(3, dtype=deformation.dtype, device=deformation.device)
    elastic_trial = .5*symmetric_log(deformation@metric_old@deformation.transpose(-1, -2))
    trace = elastic_trial.diagonal(dim1=-2, dim2=-1).sum(-1)
    trial = 2*mu[:, None, None]*elastic_trial+lam[:, None, None]*trace[:, None, None]*eye
    stress, amounts = _PrincipalStrengthReturn.apply(trial/j[:, None, None], lam, mu,
        b.phi_deg, b.dilation_deg, b.effective_cohesion_pa, b.tensile_strength_pa)
    kirchhoff = j[:, None, None]*stress
    stress_trace = kirchhoff.diagonal(dim1=-2, dim2=-1).sum(-1)
    elastic = (kirchhoff-(lam/(3*lam+2*mu)*stress_trace)[:, None, None]*eye)/(2*mu[:, None, None])
    inverse = torch.linalg.inv(deformation)
    metric = inverse@torch.linalg.matrix_exp(2*elastic)@inverse.transpose(-1, -2)
    metric = (metric+metric.transpose(-1, -2))/2
    amounts = amounts.clamp_min(0)*j[:, None]
    plastic_j = torch.exp(initial_log_volume-.5*torch.linalg.slogdet(metric)[1])
    energy = mu*(elastic*elastic).sum((-2, -1))+.5*lam*elastic.diagonal(dim1=-2, dim2=-1).sum(-1)**2
    return {'stress': stress, 'kirchhoff_stress': kirchhoff,
            'elastic_reference_metric': metric, 'elastic_log_strain': elastic,
            'plastic_log_increment': elastic_trial-elastic, 'plastic_J': plastic_j,
            'shear_multiplier': amounts[:, 0], 'tensile_multiplier': amounts[:, 1],
            'plastic_multiplier': amounts.sum(-1), 'elastic_energy_density': energy,
            'F': deformation, 'J': j}


def swept_face_increment(position, old_position, edges):
    """Exact integral of face velocity dot normal along a linear time path.

    An open face with moving endpoints has no path-independent swept-volume
    potential. Integrate each accepted increment, including tangential motion,
    rather than applying reference face weights to horizontal displacement.
    """
    now, old = position[edges], old_position[edges]
    direction = ((now[:, 1]-now[:, 0])+(old[:, 1]-old[:, 0]))/2
    normal_measure = torch.stack([direction[:, 1], -direction[:, 0]], -1)
    return ((now-old).mean(1)*normal_measure).sum()
