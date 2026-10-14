"""Plane-strain variational PINN with shared unsaturated water inventory.

Displacement and pore pressure are neural fields trained on physical weak residuals,
without response labels. Linear-triangle quadrature differentiates the sampled
neural fields (Manav et al., 2024, section 3.2.3). An auxiliary bounded AT1 damage
solve supplies irreversible friction-dependent damage. The model is a specified, vented,
filtrate-only branch of Liu's box; no mechanically strong membrane is inserted.

SI units, tensile-positive stress, z positive upwards, crown at (0, 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import torch

ROOT = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class Problem:
    cover_m: float = .09
    box_width_m: float = .8
    box_height_m: float = .6
    shield_length_m: float = .29
    shield_diameter_m: float = .11
    porosity0: float = .8 / 1.8
    saturation0: float = .06 * 2.65 / .8
    residual_saturation: float = .05
    density_kg_m3: float = 1562.5
    phi_deg: float = 31.88
    apparent_cohesion_pa: float = 1000.
    tensile_strength_pa: float = 100.
    duncan_k: float = 276.
    duncan_n: float = .57
    duncan_kb: float = 50.
    duncan_m: float = .2
    permeability_m2: float = 1.e-11
    viscosity_pa_s: float = .001
    filter_resistance_pa_s_m: float = 2.e9
    damage_length_m: float = .015
    residual_stiffness: float = 1.e-5
    water_compressibility_pa_inv: float = 4.5e-10
    pressure_step_pa: float = 2500.
    step_time_s: float = 60.
    gravity_m_s2: float = 9.80665
    pressure_scale_pa: float = 1.e4
    displacement_scale_m: float = 1.e-3

    @property
    def k0(self):
        return 1 - np.sin(np.deg2rad(self.phi_deg))

    @property
    def pressure0(self):
        # Assumed zero effective cohesion: the reported initial apparent
        # cohesion is entirely capillary. It is not added a second time.
        return -self.apparent_cohesion_pa / (
            self.saturation0 * np.tan(np.deg2rad(self.phi_deg)))

    @property
    def retention_pressure_pa(self):
        se = (self.saturation0 - self.residual_saturation) / (1 - self.residual_saturation)
        return -self.pressure0 / np.sqrt(se**-2 - 1)


def saturation(p, b):
    return b.residual_saturation + (1 - b.residual_saturation) / torch.sqrt(
        1 + (p.clamp_max(0) / b.retention_pressure_pa)**2)


def saturation_primitive(p, b):
    """Integral S(p) dp, continuous at p=0; derivative is exactly S."""
    unsaturated = (b.residual_saturation * p + (1 - b.residual_saturation)
                   * b.retention_pressure_pa * torch.asinh(p / b.retention_pressure_pa))
    return torch.where(p < 0, unsaturated, p)


def pressure_relative_permeability(p, b):
    """Mualem n=2 law without a singular saturated-branch derivative."""
    x = -p.clamp_max(0)/b.retention_pressure_pa
    t = torch.sqrt(1+x*x)
    return 1/(t**2.5*(t+x)**2)


class KirchhoffPressure(torch.autograd.Function):
    """Integral of kr from p0 to p; positive pressure remains unbounded."""
    @staticmethod
    def forward(ctx, pressure, b):
        nodes, weights = np.polynomial.legendre.leggauss(32)
        nodes = torch.tensor((nodes+1)/2, device=pressure.device)
        weights = torch.tensor(weights/2, device=pressure.device)
        endpoint = pressure.clamp_max(0)
        interval = endpoint-b.pressure0
        samples = b.pressure0+interval[..., None]*nodes
        result = interval*(pressure_relative_permeability(samples, b)*weights).sum(-1)+pressure.clamp_min(0)
        ctx.save_for_backward(pressure)
        ctx.b = b
        return result

    @staticmethod
    def backward(ctx, gradient):
        pressure, = ctx.saved_tensors
        return gradient*pressure_relative_permeability(pressure, ctx.b), None


def segments(knots, spacing):
    return np.concatenate([np.linspace(a, c, int(np.ceil((c-a)/spacing))+1)[:-1]
                           for a, c in zip(knots[:-1], knots[1:])] + [[knots[-1]]])


class Mesh:
    def __init__(self, b, spacing, device, axes=None):
        left, right = -b.shield_length_m, b.box_width_m-b.shield_length_m
        bottom = b.cover_m-b.box_height_m
        if axes is None:
            xs = segments([left, 0., right], spacing)
            zs = segments([bottom, -b.shield_diameter_m, 0., b.cover_m], spacing)
        else:
            xs, zs = axes
        xx, zz = np.meshgrid(xs, zs)
        points = np.stack([xx.ravel(), zz.ravel()], 1)
        ids = np.arange(len(points)).reshape(xx.shape)
        triangles = []
        for j in range(len(zs)-1):
            for i in range(len(xs)-1):
                if (xs[i]+xs[i+1])/2 < 0 and -b.shield_diameter_m < (zs[j]+zs[j+1])/2 < 0:
                    continue
                a, c, e, f = ids[j, i], ids[j, i+1], ids[j+1, i], ids[j+1, i+1]
                triangles.extend([[a, c, f], [a, f, e]] if (i+j) % 2 == 0
                                 else [[a, c, e], [c, f, e]])
        used, inverse = np.unique(triangles, return_inverse=True)
        points, triangles = points[used], inverse.reshape(-1, 3)
        coordinates = points[triangles]
        twice_area = np.linalg.det(coordinates[:, 1:]-coordinates[:, :1])
        gradients = np.stack([
            coordinates[:, [1, 2, 0], 1] - coordinates[:, [2, 0, 1], 1],
            coordinates[:, [2, 0, 1], 0] - coordinates[:, [1, 2, 0], 0]], -1) / twice_area[:, None, None]
        area = twice_area / 2
        weights = np.zeros(len(points))
        np.add.at(weights, triangles.ravel(), np.repeat(area/3, 3))
        x, z = points.T
        face = np.flatnonzero(np.isclose(x, 0) & (z >= -b.shield_diameter_m-1.e-9) & (z <= 1.e-9))
        face = face[np.argsort(z[face])]
        face_edges = np.stack([face[:-1], face[1:]], 1)
        face_lengths = np.diff(z[face])
        face_weights = np.zeros(len(points))
        np.add.at(face_weights, face_edges.ravel(), np.repeat(face_lengths/2, 2))
        ux_free = ~(np.isclose(x, left) | np.isclose(x, right))
        uz_free = ~(np.isclose(z, bottom) | ((x < 1.e-9) & (
            np.isclose(z, 0) | np.isclose(z, -b.shield_diameter_m))))
        self.points_np, self.triangles_np = points, triangles
        self.area_np, self.gradients_np = area, gradients
        self.spacing, self.device = spacing, device
        self.xy = torch.tensor(points, device=device)
        self.tri = torch.tensor(triangles, device=device)
        self.area = torch.tensor(area, device=device)
        self.grad = torch.tensor(gradients, device=device)
        self.weights = torch.tensor(weights, device=device)
        self.face_weights = torch.tensor(face_weights, device=device)
        self.face_edges = torch.tensor(face_edges, device=device)
        self.face_lengths = torch.tensor(face_lengths, device=device)
        self.free = torch.tensor(np.stack([ux_free, uz_free], 1), device=device)
        self.top = np.isclose(z, b.cover_m)
        self.face = face

    def gradient(self, values):
        return torch.einsum('tni,tnj->tij', values[self.tri], self.grad) if values.ndim == 2 else torch.einsum('tn,tnj->tj', values[self.tri], self.grad)

    def mean(self, values):
        return values[self.tri].mean(1)

    def nodal(self, values):
        result = torch.zeros(len(self.xy), device=self.device)
        result.index_add_(0, self.tri.flatten(), (self.area[:, None]*values[:, None]/3).expand(-1, 3).flatten())
        return result / self.weights


def elastic_constants(mesh, b):
    depth = b.cover_m-mesh.mean(mesh.xy[:, 1])
    sv = -b.density_kg_m3*b.gravity_m_s2*depth
    sh = b.k0*sv
    pb0 = b.saturation0*b.pressure0
    conf = -sh-pb0
    young = b.duncan_k*101325.*(conf/101325.)**b.duncan_n
    bulk = b.duncan_kb*101325.*(conf/101325.)**b.duncan_m
    nu = (3*bulk-young)/(6*bulk)
    if torch.any((nu <= -1) | (nu >= .5)):
        raise ValueError('The prescribed initial Duncan moduli are not elastic-admissible')
    mu = young/(2*(1+nu))
    lam = bulk-2*mu/3
    sig0 = torch.stack([sh+pb0, sv+pb0, sh+pb0], 1)
    eps0 = (sig0 - (lam/(3*lam+2*mu))[:, None]*sig0.sum(1)[:, None])/(2*mu[:, None])
    # Match the residual Drucker-Prager cone to the reported friction angle
    # on the triaxial-compression meridian. c'=0 is the capillary-cohesion branch.
    sine = np.sin(np.deg2rad(b.phi_deg))
    gamma = 2*np.sqrt(6)*sine/(3-sine)
    # Calibrate Gc to the driving energy of THIS split at uniaxial f_t.
    trial = torch.stack([-nu, torch.ones_like(nu), -nu], 1)*b.tensile_strength_pa/young[:, None]
    positive, _ = principal_energy_split(trial, lam, mu, gamma)
    tensile_fraction = positive/(b.tensile_strength_pa**2/(2*young))
    gc = 8*b.damage_length_m*b.tensile_strength_pa**2*tensile_fraction/(3*young)
    return lam, mu, eps0, gc, gamma


def principal_energy_split(eigen, lam, mu, gamma):
    """Structured-deformation projection onto tr(eta)>=gamma*|dev(eta)|.

    The residual stress remains in the pressure-dependent friction cone;
    positive transverse strain alone is not a fracture criterion. See
    Vicentini et al., IJF (2024), section 3.4 and Appendix A.4.
    """
    trace = eigen.sum(1)
    deviator = eigen-trace[:, None]/3
    norm = torch.sqrt((deviator**2).sum(1)+1.e-30)
    bulk = lam+2*mu/3
    total = bulk/2*trace**2+mu*norm**2
    sliding = bulk*mu/(bulk*gamma**2+2*mu)*(trace-gamma*norm)**2
    persistent = torch.where(norm >= -gamma*bulk/(2*mu)*trace, sliding, total)
    persistent = torch.where(trace >= gamma*norm, torch.zeros_like(total), persistent)
    return (total-persistent).clamp_min(0), persistent


def strain_split(gu, eps0, lam, mu, gamma):
    ex, ez = gu[:, 0, 0]+eps0[:, 0], gu[:, 1, 1]+eps0[:, 1]
    ey = eps0[:, 2]
    exz = (gu[:, 0, 1]+gu[:, 1, 0])/2
    radius = torch.sqrt(((ex-ez)/2)**2+exz**2+1.e-30)
    eigen = torch.stack([(ex+ez)/2+radius, (ex+ez)/2-radius, ey], 1)
    return principal_energy_split(eigen, lam, mu, gamma)


def deformation(u, mesh, b):
    gu = mesh.gradient(u)
    # Small-strain reference volume, consistently used by pressure work/storage.
    j = 1+gu[:, 0, 0]+gu[:, 1, 1]
    return gu, j, mesh.nodal(j)-(1-b.porosity0)


def mechanics_energy(u, pressure, damage, load, mesh, b, constants):
    gu, j, _ = deformation(u, mesh, b)
    lam, mu, eps0, _, gamma = constants
    pos, neg = strain_split(gu, eps0, lam, mu, gamma)
    degradation = mesh.mean((1-b.residual_stiffness)*(1-damage)**2+b.residual_stiffness)
    pb = mesh.mean(saturation(pressure, b)*pressure)
    density = degradation*pos+neg-pb*(j-1)
    # Incremental pressure added to a balanced initial lateral traction.
    face_p = b.k0*b.density_kg_m3*b.gravity_m_s2*(b.cover_m-mesh.xy[:, 1])+load
    gravity = b.density_kg_m3*b.gravity_m_s2*(mesh.weights*u[:, 1]).sum()
    pe, ue = face_p[mesh.face_edges], u[:, 0][mesh.face_edges]
    work = (mesh.face_lengths/6*(pe[:, 0]*(2*ue[:, 0]+ue[:, 1])
                               + pe[:, 1]*(ue[:, 0]+2*ue[:, 1]))).sum()
    return (mesh.area*density).sum()+gravity-work


def water_inventory(pressure, pore_volume, b):
    return pore_volume*saturation(pressure, b) + b.porosity0*b.water_compressibility_pa_inv*(pressure-b.pressure0)



def solve_damage(u, previous, mesh, b, constants):
    """Convex bounded AT1 block; no update of accepted physical history here."""
    with torch.no_grad():
        pos, _ = strain_split(mesh.gradient(u), constants[2], constants[0], constants[1], constants[4])
    h = (1-b.residual_stiffness)*pos.cpu().numpy()
    gc = constants[3].cpu().numpy()
    tri, area, grad = mesh.triangles_np, mesh.area_np, mesh.gradients_np
    # Use nodal lumping for g(d); the gradient term uses exact triangle gradients.
    diag, linear = np.zeros(len(mesh.xy)), np.zeros(len(mesh.xy))
    np.add.at(diag, tri.ravel(), np.repeat(2*area*h/3, 3))
    np.add.at(linear, tri.ravel(), np.repeat(area*(-2*h+gc/(8/3*b.damage_length_m))/3, 3))
    local = 2*(area*gc/(8/3)*b.damage_length_m)[:, None, None]*np.einsum('tni,tmi->tnm', grad, grad)
    row = np.broadcast_to(tri[:, :, None], local.shape).ravel()
    col = np.broadcast_to(tri[:, None, :], local.shape).ravel()
    matrix = coo_matrix((local.ravel(), (row, col)), shape=(len(diag), len(diag))).tocsr()
    matrix += coo_matrix((diag, (np.arange(len(diag)), np.arange(len(diag)))), shape=matrix.shape).tocsr()
    old = previous.detach().cpu().numpy()
    d = old.copy()
    tolerance = max(np.abs(linear).max(), 1.e-20)*1.e-9
    for _ in range(len(d)+1):
        gradient = matrix@d+linear
        lower = (d <= old+1.e-12) & (gradient >= -tolerance)
        upper = (d >= 1-1.e-12) & (gradient <= tolerance)
        free = ~(lower | upper)
        if not free.any():
            break
        if np.max(np.abs(gradient[free])) <= tolerance:
            break
        trial = d.copy()
        trial[lower], trial[upper] = old[lower], 1.
        fixed = ~free
        trial[free] = spsolve(matrix[free][:, free], -linear[free]-matrix[free][:, fixed]@trial[fixed])
        d = np.clip(trial, old, 1.)
    else:
        raise ArithmeticError('Damage active set did not settle')
    return torch.tensor(d, device=mesh.device)
