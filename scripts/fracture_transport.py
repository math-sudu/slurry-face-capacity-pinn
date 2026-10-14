"""Continuous cohesive history and series resistance between crack pressures.

The hydraulic aperture is interpolated linearly between pressure points,
mechanical vertices and corners of the separation-history envelope. Trial
intersections with that envelope are included, so an unbroken interior cannot
be hidden by damaged endpoints. A zero aperture anywhere on a link blocks it.
"""
import numpy as np
import torch

from fracture_physics import cohesive_normal


class OpeningHistory:
    def __init__(self, length, pressure_fraction, effective_initial):
        self.length = np.asarray(length)
        self.pressure_fraction = np.asarray(pressure_fraction)
        self.effective_old = torch.tensor(effective_initial)
        self.envelope = [(np.array([0., 1.]), np.zeros(2)) for _ in length]
        self.prepare()

    def prepare(self):
        segments, fractions, maxima, edges = [], [], [], []
        order = len(self.pressure_fraction)
        for i, (x, maximum) in enumerate(self.envelope):
            points = np.unique(np.r_[x, self.pressure_fraction])
            segments.extend([i]*(len(points)-1))
            fractions.extend(np.c_[points[:-1], points[1:]])
            maxima.extend(np.c_[np.interp(points[:-1], x, maximum),
                                np.interp(points[1:], x, maximum)])
            edges.extend(i*order+np.searchsorted(self.pressure_fraction, (points[:-1]+points[1:])/2))
        self.segment = torch.tensor(segments)
        self.fraction = torch.tensor(np.asarray(fractions))
        self.maximum = torch.tensor(np.asarray(maxima))
        self.edge = torch.tensor(edges).repeat_interleave(2)
        self.nedge = len(self.length)*order+1

    def trace(self, effective):
        """Return endpoint apertures on each trial subinterval, without mutation."""
        f = self.fraction
        e = effective[self.segment]
        current = e[:, :1]*(1-f)+e[:, 1:]*f
        difference = current-self.maximum
        slope = difference[:, 1]-difference[:, 0]
        crossing = (-difference[:, 0]/torch.where(slope != 0, slope, 1.)).clamp(0, 1)
        crossing = torch.where(slope != 0, crossing, 0.)
        alpha = torch.stack([torch.zeros_like(crossing), crossing, torch.ones_like(crossing)], 1)
        fraction = f[:, :1]*(1-alpha)+f[:, 1:]*alpha
        previous_maximum = self.maximum[:, :1]*(1-alpha)+self.maximum[:, 1:]*alpha
        current = e[:, :1]*(1-fraction)+e[:, 1:]*fraction
        previous = self.effective_old[self.segment]
        previous = previous[:, :1]*(1-fraction)+previous[:, 1:]*fraction
        length = torch.diff(fraction, dim=1)*torch.tensor(self.length)[self.segment, None]
        return current, previous, previous_maximum, length

    def transmissibility(self, effective, b, current_length=None):
        current, previous, maximum, lengths = self.trace(effective)
        if current_length is not None:
            lengths = lengths*(current_length/torch.as_tensor(self.length))[self.segment, None]
        zero = torch.zeros_like(current)
        width = cohesive_normal(current, zero, maximum, b)[1]
        old = cohesive_normal(previous, zero, maximum, b)[1]
        # Each row contributes [left, crossing] and [crossing, right].
        width = torch.stack([width[:, :-1], width[:, 1:]], -1).reshape(-1, 2)
        old = torch.stack([old[:, :-1], old[:, 1:]], -1).reshape(-1, 2)
        length = lengths.flatten()
        # Time-average the whole link, interpolating only shrinking apertures;
        # opening points retain the backward-Euler value. This switches
        # continuously at each point, including mixed opening/closing links.
        # A permanently closed point stays closed at every time sample.
        theta, weights = np.polynomial.legendre.leggauss(3)
        theta = torch.tensor(np.r_[(theta+1)/2, 1.])[:, None, None]
        aperture = torch.where(width < old, (1-theta)*old+theta*width, width)
        left, right = aperture[:, :, 0], aperture[:, :, 1]
        total = left+right
        active = length > 0
        # Exact integral of 12*mu/w(s)^3 for a positive linear taper.
        local = left**2*right**2/(6*b.viscosity_pa_s
                *torch.where(active, length, 1.)*torch.where(total > 0, total, 1.))
        resistance = torch.where(active & (local > 0),
                                 1/torch.where(local > 0, local, 1.), 0.)
        resistance = torch.zeros((4, self.nedge)).index_add(1, self.edge, resistance)
        blocked = torch.zeros((4, self.nedge)).index_add(
            1, self.edge, (active & (local == 0)).to(local.dtype)) > 0
        conductance = torch.where(blocked, 0., 1/torch.where(resistance > 0, resistance, 1.))
        averaged = torch.tensor(weights/2)@conductance[:3]
        return averaged, conductance[-1]

    def updated(self, effective):
        """Exact max-of-affine accepted history, compressed to its corners."""
        result = []
        for (x, maximum), e in zip(self.envelope, np.asarray(effective)):
            difference = (1-x)*e[0]+x*e[1]-maximum
            crossing = []
            for j in range(len(x)-1):
                if difference[j]*difference[j+1] < 0:
                    crossing.append(x[j]-difference[j]*(x[j+1]-x[j])/(difference[j+1]-difference[j]))
            points = np.unique(np.r_[x, crossing])
            values = np.maximum(np.interp(points, x, maximum), (1-points)*e[0]+points*e[1])
            keep = []
            for point, value in zip(points, values):
                keep.append((point, value))
                while len(keep) >= 3:
                    (x0, y0), (x1, y1), (x2, y2) = keep[-3:]
                    cross = (y1-y0)*(x2-x1)-(y2-y1)*(x1-x0)
                    scale = max(abs(y0), abs(y1), abs(y2))*(x2-x0)
                    if abs(cross) > 16*np.finfo(float).eps*scale:
                        break
                    keep.pop(-2)
            result.append(np.array(keep).T)
        return result

    def saved(self, effective):
        envelope = self.updated(effective)
        return {'crack_history_offsets': np.r_[0, np.cumsum([len(x) for x, _ in envelope])],
                'crack_history_fraction': np.concatenate([x for x, _ in envelope]),
                'crack_history_maximum': np.concatenate([m for _, m in envelope])}

    def accept(self, saved):
        offsets = saved['crack_history_offsets']
        self.envelope = [(saved['crack_history_fraction'][left:right].copy(),
                          saved['crack_history_maximum'][left:right].copy())
                         for left, right in zip(offsets[:-1], offsets[1:])]
        self.effective_old = torch.tensor(saved['crack_effective_endpoints'])
        self.prepare()
