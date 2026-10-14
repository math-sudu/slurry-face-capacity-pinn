"""Separate frictional plasticity, cohesive opening, and aperture transport.

Stress is tensile-positive. Trial evaluations never mutate accepted history.
The candidate crack is a specified internal interface, not a predicted path.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
import torch

from face_capacity_pinn import Problem, Mesh, segments


@dataclass(frozen=True)
class FractureProblem(Problem):
    dilation_deg: float = 5.
    fracture_energy_j_m2: float = .005
    interface_stiffness_pa_m: float = 1.e9
    crack_slope: float = .6
    water_surface_tension_n_m: float = .072
    effective_cohesion_pa: float = 1000.
    # Retain the previous hydraulic scenario independently of strength.
    initial_suction_pa: float = 7361.742864605794
    fracture_wall_resistance_pa_s_m: float = 2.e9
    # Missing fields in existing result archives retain their original model.
    interface_geometry: str = 'reference'
    bulk_kinematics: str = 'small_strain'
    fracture_retention: str = 'matrix_limit_plus_gap'

    def __post_init__(self):
        if not np.isfinite(self.initial_suction_pa) or self.initial_suction_pa <= 0:
            raise ValueError('The unsaturated initial state requires positive independent initial suction')
        if not np.isfinite(self.fracture_wall_resistance_pa_s_m) or self.fracture_wall_resistance_pa_s_m <= 0:
            raise ValueError('The finite-resistance fracture wall requires positive resistance')
        if self.interface_geometry not in ('reference', 'corotated'):
            raise ValueError('Interface geometry must be reference or corotated')
        if self.bulk_kinematics not in ('small_strain', 'finite_strain'):
            raise ValueError('Bulk kinematics must be small_strain or finite_strain')
        if self.bulk_kinematics == 'finite_strain' and self.interface_geometry != 'corotated':
            raise ValueError('Finite bulk kinematics requires a corotated interface')
        if self.fracture_retention not in ('matrix_limit_plus_gap', 'capillary_bridge'):
            raise ValueError('Unknown fracture retention model')

    @property
    def pressure0(self):
        return -self.initial_suction_pa


@lru_cache(maxsize=8)
def _strength_planes(phi_deg, dilation_deg):
    """All six MC faces and three principal-tension faces, tensile-positive."""
    sine = np.sin(np.deg2rad(phi_deg))
    sine_psi = np.sin(np.deg2rad(dilation_deg))
    nphi, npsi = (1+sine)/(1-sine), (1+sine_psi)/(1-sine_psi)
    pairs = [(0, 2), (0, 1), (1, 2), (1, 0), (2, 0), (2, 1)]
    f, g = np.zeros((9, 3)), np.zeros((9, 3))
    for k, (low, high) in enumerate(pairs):
        f[k, low], f[k, high] = -1., nphi
        g[k, low], g[k, high] = -1., npsi
    f[6:] = g[6:] = np.eye(3)
    active_sets = []
    for count in range(1, 4):
        for ids in combinations([0, 8, 7, 6, 1, 2, 3, 4, 5], count):
            ids = np.array(ids)
            if np.linalg.matrix_rank(f[ids]) == count and np.linalg.matrix_rank(g[ids]) == count:
                active_sets.append(ids)
    return nphi, f, g, active_sets


class _PrincipalStrengthReturn(torch.autograd.Function):
    """MC/Rankine active-set return and its spectral consistent derivative.

    Each accepted set satisfies all yield inequalities, nonnegative plastic
    multipliers and simultaneous consistency on its active planes. A repeated
    eigenvalue uses the spectral limit, so hydrostatic and plane-strain states
    do not acquire singular eigenvector derivatives. Material inputs are fixed.
    """
    @staticmethod
    def forward(ctx, trial, lam, mu, phi_deg, dilation_deg, cohesion, tensile):
        values, axes = np.linalg.eigh(trial.detach().cpu().numpy())
        lam_np, mu_np = lam.detach().cpu().numpy(), mu.detach().cpu().numpy()
        nphi, f, g, active_sets = _strength_planes(phi_deg, dilation_deg)
        limit = np.r_[np.full(6, 2*cohesion*np.sqrt(nphi)), np.full(3, tensile)]
        stiffness = 2*mu_np[:, None, None]*np.eye(3)+lam_np[:, None, None]*np.ones((3, 3))
        returned = values.copy()
        tangent = np.broadcast_to(np.eye(3), stiffness.shape).copy()
        multipliers = np.zeros((len(values), 9))
        # Tolerance only accounts for roundoff in the small consistency solves.
        tolerance = 128*np.finfo(values.dtype).eps*np.maximum(np.abs(values).max(1), max(cohesion, tensile, 1.))
        unresolved = (values@f.T-limit > tolerance[:, None]).any(1)
        for active in active_sets:
            ids = np.flatnonzero(unresolved)
            if not len(ids):
                break
            cg = stiffness[ids]@g[active].T
            consistency = f[active]@cg
            inverse = np.linalg.inv(consistency)
            plastic = (inverse@(values[ids]@f[active].T-limit[active])[..., None])[..., 0]
            candidate = values[ids]-(cg@plastic[..., None])[..., 0]
            valid = (plastic >= -tolerance[ids, None]/mu_np[ids, None]).all(1)
            valid &= (candidate@f.T-limit <= tolerance[ids, None]).all(1)
            accepted = ids[valid]
            returned[accepted] = candidate[valid]
            tangent[accepted] = np.eye(3)-cg[valid]@inverse[valid]@f[active]
            multipliers[np.ix_(accepted, active)] = plastic[valid]
            unresolved[accepted] = False
        if unresolved.any():
            raise ArithmeticError('No admissible MC/Rankine active set for the trial stress')
        gap = values[:, :, None]-values[:, None, :]
        returned_gap = returned[:, :, None]-returned[:, None, :]
        diagonal = np.diagonal(tangent, axis1=1, axis2=2)
        spectral = .5*(diagonal[:, :, None]+diagonal[:, None, :]-tangent-tangent.transpose(0, 2, 1))
        np.divide(returned_gap, gap, out=spectral, where=np.abs(gap) > tolerance[:, None, None])
        tensors = [torch.as_tensor(a, dtype=trial.dtype, device=trial.device) for a in [axes, tangent, spectral]]
        ctx.save_for_backward(*tensors)
        stress = torch.as_tensor((axes*returned[:, None, :])@axes.transpose(0, 2, 1), dtype=trial.dtype, device=trial.device)
        amounts = torch.as_tensor(np.c_[multipliers[:, :6].sum(1), multipliers[:, 6:].sum(1)], dtype=trial.dtype, device=trial.device)
        ctx.mark_non_differentiable(amounts)
        return stress, amounts

    @staticmethod
    def backward(ctx, gradient, unused):
        axes, tangent, spectral = ctx.saved_tensors
        local = axes.transpose(-1, -2)@((gradient+gradient.transpose(-1, -2))/2)@axes
        diagonal = torch.einsum('nji,nj->ni', tangent, local.diagonal(dim1=-2, dim2=-1))
        result = spectral*local+torch.diag_embed(diagonal)
        return axes@result@axes.transpose(-1, -2), None, None, None, None, None, None


def mohr_coulomb_rankine(strain, plastic_old, lam, mu, phi_deg, dilation_deg, cohesion_pa, tensile_pa):
    """Perfect frictional and tensile plasticity with independently given strengths.

    MC yield/potential and Rankine tension follow Itasca's Mohr-Coulomb model,
    Eqs. 5-11; intersecting surfaces are solved together. Plastic strain is
    retained on unloading. This continuum strain is not a hydraulic aperture.
    """
    eye = torch.eye(3, dtype=strain.dtype, device=strain.device)
    elastic_trial = strain-plastic_old
    trace = elastic_trial.diagonal(dim1=-2, dim2=-1).sum(-1)
    trial = 2*mu[:, None, None]*elastic_trial+lam[:, None, None]*trace[:, None, None]*eye
    stress, amounts = _PrincipalStrengthReturn.apply(trial, lam, mu, phi_deg, dilation_deg, cohesion_pa, tensile_pa)
    trace_stress = stress.diagonal(dim1=-2, dim2=-1).sum(-1)
    elastic = (stress-(lam/(3*lam+2*mu)*trace_stress)[:, None, None]*eye)/(2*mu[:, None, None])
    return stress, strain-elastic, amounts[:, 0], amounts[:, 1]


def cohesive_normal(jump, traction0, maximum_old, b):
    """Normal traction and inelastic aperture, including accepted damage history."""
    stiffness = b.interface_stiffness_pa_m
    e = jump+traction0/stiffness
    onset = b.tensile_strength_pa/stiffness
    final = 2*b.fracture_energy_j_m2/b.tensile_strength_pa
    maximum = torch.maximum(maximum_old, e.clamp_min(0))
    damage = ((1-onset/maximum.clamp_min(onset))/(1-onset/final)).clamp(0, 1)
    normal = torch.where(e > 0, (1-damage)*stiffness*e, stiffness*e)
    aperture = damage*e.clamp_min(0)
    return normal, aperture, damage, maximum


def cohesive_contact(jump, traction0, maximum_old, slip_old, b):
    """Bilinear tensile cohesion in series with elastic interface compliance.

    Inputs are local (normal,tangent) components. The inelastic normal opening
    w = e_n - t_n/K is the hydraulic aperture; compression restores contact.
    Frictional tangential slip has its own accepted history and never adds w.
    """
    normal, aperture, damage, maximum = cohesive_normal(jump[..., 0], traction0[..., 0], maximum_old, b)
    stiffness = b.interface_stiffness_pa_m
    trial_shear = traction0[..., 1]+stiffness*(jump[..., 1]-slip_old)
    capacity = ((1-damage)*b.effective_cohesion_pa
                +np.tan(np.deg2rad(b.phi_deg))*(-normal).clamp_min(0))
    shear = torch.maximum(torch.minimum(trial_shear, capacity), -capacity)
    slip = slip_old+(trial_shear-shear)/stiffness
    return torch.stack([normal, shear], -1), aperture, damage, maximum, slip


def interface_kinematics(u, coordinates, shape, corotated):
    """Material-point gaps and their exact variation on a two-node midline.

    Node order is minus-left/right, plus-left/right. Local cohesive tractions
    are nominal (per reference length), conjugate to these gap components.
    Differentiating the returned B operator retains the geometric stiffness.
    """
    position = coordinates+u if corotated else coordinates
    midline = (position[:, :2]+position[:, 2:])/2
    direction = midline[:, 1]-midline[:, 0]
    length = torch.linalg.vector_norm(direction, dim=1)
    tangent = direction/length[:, None]
    normal = torch.stack([tangent[:, 1], -tangent[:, 0]], -1)
    frame = torch.stack([normal, tangent], 1)
    jump_nodes = u[:, 2:]-u[:, :2]
    local_nodes = torch.einsum('eai,eni->ena', frame, jump_nodes)
    jump = torch.einsum('qn,ena->eqa', shape, local_nodes)
    signed_shape = torch.cat([-shape, shape], 1)
    derivative = torch.einsum('qn,eai->eqani', signed_shape, frame)
    if corotated:
        # d(theta) = -n . d(midline_right-midline_left)/length.
        # Thus dg_n = n . dg - g_t*n . dr/length, dg_t = t . dg + g_n*n . dr/length.
        turn = torch.stack([-jump[..., 1], jump[..., 0]], -1)/length[:, None, None]
        middle_difference = u.new_tensor([-.5, .5, -.5, .5])
        derivative = derivative+torch.einsum('eqa,n,ei->eqani', turn, middle_difference, normal)
    return {'jump': jump, 'jump_nodes': local_nodes, 'derivative': derivative,
            'frame': frame, 'length': length, 'midline': midline}


def fracture_retention(pressure, aperture, b):
    """Water-covered wall fraction for the added cavity, also used in storage.

    The capillary_bridge model keeps two wall-area populations. The ordinary
    capillary population has scale a0+w. The matrix-residual population, whose
    existing water remains in the bulk inventory, can occupy a new gap only as
    liquid bridges with scale w. Its occupancy is 1/sqrt(1+(s*w/(2*gamma))**2),
    not a constant cavity-water floor. At w=0 both models match the matrix law
    exactly, preserving the initial effective and total interface tractions.

    Bridges supply local retention and normal pressure area, not a separate
    along-crack flow path; the existing connected-population mobility remains.
    This is an aperture-dependent closure hypothesis, not a measured joint law.
    """
    capillary_aperture = 2*b.water_surface_tension_n_m/b.retention_pressure_pa
    x = -pressure.clamp_max(0)*(capillary_aperture+aperture)/(2*b.water_surface_tension_n_m)
    t = torch.sqrt(1+x*x)
    if b.fracture_retention == 'capillary_bridge':
        bridge_x = -pressure.clamp_max(0)*aperture/(2*b.water_surface_tension_n_m)
        bridge = torch.rsqrt(1+bridge_x*bridge_x)
    else:
        # Reproduce the accepted pre-repair archives and their causal diagnostic.
        bridge = 1.
    saturation = b.residual_saturation*bridge+(1-b.residual_saturation)/t
    relative = 1/(t**2.5*(t+x)**2)
    return saturation, relative


def fracture_water(pressure, aperture, b):
    """Vented rough crack with the matrix retention law at zero added gap.

    At positive pressure the slit is saturated. A closed interface has exactly
    zero additional storage and conductivity, including when previously broken.
    The wall fraction returned by fracture_retention multiplies both the added
    cavity volume here and the pressure traction in the interface residual.
    """
    saturation, relative = fracture_retention(pressure, aperture, b)
    storage = aperture*saturation*(1+b.water_compressibility_pa_inv*pressure)
    conductance = aperture**3/(12*b.viscosity_pa_s)*relative
    return storage, conductance, saturation


def graded_distances(length, first, maximum):
    distances = [0.]
    step = first
    while distances[-1] < length:
        distances.append(min(length, distances[-1]+step))
        step = min(1.35*step, maximum)
    return np.array(distances)


def crack_coordinates(points, b):
    """Map the vertical crown line onto the prescribed crack, fixing box walls."""
    xy = points.copy()
    left, right = -b.shield_length_m, b.box_width_m-b.shield_length_m
    shift = b.crack_slope*np.maximum(xy[:, 1], 0)
    xy[:, 0] += shift*np.where(xy[:, 0] < 0, (xy[:, 0]-left)/(-left), (right-xy[:, 0])/right)
    return xy


class CrackMesh(Mesh):
    """Quadrature grid split along x=slope*z, from the crown to the surface."""
    def __init__(self, b, spacing, normal_spacing=None, tangent_spacing=None):
        axes = None
        if normal_spacing is not None or tangent_spacing is not None:
            hn = spacing if normal_spacing is None else normal_spacing
            ht = spacing if tangent_spacing is None else tangent_spacing
            if not 0 < hn <= spacing or not 0 < ht <= spacing:
                raise ValueError('Local quadrature spacings must be positive and no larger than bulk spacing')
            left = graded_distances(b.shield_length_m, hn, spacing)
            right = graded_distances(b.box_width_m-b.shield_length_m, hn, spacing)
            xs = np.r_[-left[:0:-1], right]
            lower = graded_distances(b.box_height_m-b.cover_m-b.shield_diameter_m, ht, spacing)
            zs = np.r_[-b.shield_diameter_m-lower[:0:-1],
                       segments([-b.shield_diameter_m, 0., b.cover_m], ht)]
            axes = xs, zs
        super().__init__(b, spacing, 'cpu', axes)
        original = self.points_np.copy()
        xy = crack_coordinates(original, b)
        crack = np.flatnonzero(np.isclose(original[:, 0], 0) & (original[:, 1] >= -1.e-12))
        crack = crack[np.argsort(xy[crack, 1])]
        duplicates = np.arange(len(xy), len(xy)+len(crack))
        parent = np.r_[np.arange(len(xy)), crack]
        tri = self.triangles_np.copy()
        center = original[tri].mean(1)
        right_bank = (center[:, 0] > 0) & (center[:, 1] > 0)
        # The crown duplicate also belongs to the right-hand soil below z=0.
        right_bank |= (center[:, 0] > 0) & np.any(tri == crack[0], axis=1)
        for node, duplicate in zip(crack, duplicates):
            tri[(tri == node) & right_bank[:, None]] = duplicate
        xy = np.r_[xy, xy[crack]]
        coordinates = xy[tri]
        twice = np.linalg.det(coordinates[:, 1:]-coordinates[:, :1])
        area = twice/2
        grad = np.stack([coordinates[:, [1, 2, 0], 1]-coordinates[:, [2, 0, 1], 1],
                         coordinates[:, [2, 0, 1], 0]-coordinates[:, [1, 2, 0], 0]], -1)/twice[:, None, None]
        weights = np.zeros(len(xy))
        np.add.at(weights, tri.ravel(), np.repeat(area/3, 3))
        free = np.r_[self.free.numpy(), self.free.numpy()[crack]]
        free[duplicates, 1] = True
        face_edges = self.face_edges.numpy().copy()
        face_edges[face_edges == crack[0]] = duplicates[0]
        face_weights = np.zeros(len(xy))
        np.add.at(face_weights, face_edges.ravel(), np.repeat(self.face_lengths.numpy()/2, 2))
        self.points_np, self.triangles_np = xy, tri
        self.area_np, self.gradients_np = area, grad
        self.xy, self.tri = torch.tensor(xy), torch.tensor(tri)
        self.area, self.grad = torch.tensor(area), torch.tensor(grad)
        self.weights, self.free = torch.tensor(weights), torch.tensor(free)
        self.face_edges, self.face_weights = torch.tensor(face_edges), torch.tensor(face_weights)
        self.face = np.unique(face_edges)
        self.top = np.isclose(xy[:, 1], b.cover_m)
        self.parent = parent
        self.reference_points_np = original[parent]
        self.crack_minus, self.crack_plus = crack, duplicates
        self.crack_cells = np.stack([crack[:-1], crack[1:], duplicates[:-1], duplicates[1:]], 1)
        tangent = xy[crack[1:]]-xy[crack[:-1]]
        length = np.linalg.norm(tangent, axis=1)
        tangent /= length[:, None]
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], 1)
        self.crack_length = torch.tensor(length)
        self.crack_frame = torch.tensor(np.stack([normal, tangent], 1))
