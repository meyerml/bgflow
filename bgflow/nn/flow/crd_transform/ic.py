import torch
import numpy as np
import warnings
from typing import Union, Optional

from ..base import Flow
from .ic_helper import (
    outer,
    dist_deriv,
    angle_deriv,
    torsion_deriv,
    orientation,
    det3x3,
    det2x2,
    init_xyz2ics,
    init_ics2xyz,
    _from_euler_angles
)
from .pca import WhitenFlow


__all__ = [
    "RelativeInternalCoordinateTransformation",
    "GlobalInternalCoordinateTransformation",
    "MixedCoordinateTransformation"
]


# noqa: C901
def ic2xyz_deriv(p1, p2, p3, d14, a124, t1234,
                 eps=1e-7,
                 enforce_boundaries=True,
                 raise_warnings=True):
    """ computes the xyz coordinates from internal coordinates
        relative to points `p1`, `p2`, `p3` together with its
        jacobian with respect to `p1`.
    """

    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2, dim=-1)
    nn = torch.cross(v1, n, dim=-1)

    n_norm = torch.norm(n, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(n_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        n_norm = n_norm.clamp_min(eps)

    n_normalized = n / n_norm

    nn_norm = torch.norm(nn, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(nn_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        nn_norm = nn_norm.clamp_min(eps)

    nn_normalized = nn / nn_norm

    n_scaled = n_normalized * -torch.sin(t1234)
    nn_scaled = nn_normalized * torch.cos(t1234)

    v3 = n_scaled + nn_scaled
    v3_norm = torch.norm(v3, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(v3_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        v3_norm = v3_norm.clamp_min(eps)

    v3_normalized = v3 / v3_norm
    v3_scaled = v3_normalized * d14 * torch.sin(a124)

    v1_norm = torch.norm(v1, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(v1_norm < eps):
            warnings.warn("singular norm in xyz reconstruction")
    if enforce_boundaries:
        v1_norm = v1_norm.clamp_min(eps)

    v1_normalized = v1 / v1_norm
    v1_scaled = v1_normalized * d14 * torch.cos(a124)

    position = p1 + v3_scaled - v1_scaled

    J_d = v3_normalized * torch.sin(a124) - v1_normalized * torch.cos(a124)
    J_a = v3_normalized * d14 * torch.cos(a124) + v1_normalized * d14 * torch.sin(a124)

    J_t1 = (d14 * torch.sin(a124))[..., None]
    J_t2 = (
        1.0
        / v3_norm[..., None]
        * (torch.eye(3)[None, :].to(p1) - outer(v3_normalized, v3_normalized))
    )

    J_n_scaled = n_normalized * -torch.cos(t1234)
    J_nn_scaled = nn_normalized * -torch.sin(t1234)
    J_t3 = (J_n_scaled + J_nn_scaled)[..., None]

    J_t = (J_t1 * J_t2) @ J_t3

    J = torch.stack([J_d, J_a, J_t[..., 0]], dim=-1)

    return position, J


def ic2xy0_deriv(p1, p2, d14, a124, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """ computes the xy coordinates (z set to 0) for the given
        internal coordinates together with the Jacobian
        with respect to `p1`.
    """

    t1234 = torch.Tensor([[0.5 * np.pi]]).to(p1)
    p3 = torch.Tensor([[0, -1, 0]]).to(p1)
    xyz, J = ic2xyz_deriv(p1, p2, p3, d14, a124, t1234, eps=eps, enforce_boundaries=enforce_boundaries, raise_warnings=raise_warnings)
    J = J[..., [0, 1, 2], :][..., [0, 1]]
    return xyz, J


def decompose_z_matrix(z_matrix, fixed):
    """Decompose the z-matrix into blocks to allow parallel (batched) reconstruction
    of cartesian coordinates starting from the fixed atoms.

    Parameters
    ----------
    z_matrix : np.ndarray
        Z-matrix definition for the internal coordinate transform.
        Each row in the z-matrix defines a (proper or improper) torsion by specifying the atom indices
        forming this torsion. Atom indices are integers >= 0.
        The shape of the z-matrix is (n_conditioned_atoms, 4).
    fixed : np.ndarray
        Fixed atoms that are used to seed the reconstruction of Cartesian from internal coordinates.

    Returns
    -------
    blocks : list of np.ndarray
        Z-matrix for each stage of the reconstruction. The shape for each block is
        (n_conditioned_atoms_in_block, 4).
    index2atom : np.ndarray
        index2atom[i] specifies the atom index of the atom that is placed by the i-th row in the original Z-matrix.
        The shape is (n_conditioned_atoms, ).
    atom2index : np.ndarray
        atom2index[i] specifies the row in the original z-matrix that is responsible for placing the i-th atom.
        The shape is (n_conditioned_atoms, ).
    index2order : np.ndarray
        order in which the reconstruction is applied, where i denotes a row in the Z-matrix.
        The shape is (n_conditioned_atoms, ).
    """
    atoms = [fixed]

    blocks = []  # blocks of Z-matrices. Each block corresponds to one stage of Cartesian reconstruction.
    given = np.sort(fixed)  # atoms that were already visited

    # filter out conditioned variables
    non_given = ~np.isin(z_matrix[:, 0], given)
    z_matrix = z_matrix[non_given]
    # prepend the torsion index to each torsion in the z matrix
    z_matrix = np.concatenate([np.arange(len(z_matrix))[:, None], z_matrix], axis=1)

    order = []  # torsion indices
    while len(z_matrix) > 0:

        can_be_placed_in_this_stage = np.all(np.isin(z_matrix[:, 2:], given), axis=-1)
        # torsions, where atoms 2-4 were already visited
        if (not np.any(can_be_placed_in_this_stage)) and len(z_matrix) > 0:
            raise ValueError(
                f"Z-matrix decomposition failed. "
                f"The following atoms were not reachable from the fixed atoms: \n{z_matrix[:,1]}"
            )

        pos = z_matrix[can_be_placed_in_this_stage, 0]
        atom = z_matrix[can_be_placed_in_this_stage, 1]

        atoms.append(atom)
        order.append(pos)

        blocks.append(z_matrix[can_be_placed_in_this_stage][:, 1:])
        given = np.union1d(given, atom)
        z_matrix = z_matrix[~can_be_placed_in_this_stage]

    index2atom = np.concatenate(atoms)
    atom2index = np.argsort(index2atom)
    index2order = np.concatenate(order)
    return blocks, index2atom, atom2index, index2order


def slice_initial_atoms(z_matrix):
    s = np.sum(z_matrix == -1, axis=-1)
    order = np.argsort(s)[::-1][:3]
    return z_matrix[:, 0][order], z_matrix[s == 0]


def normalize_torsions(torsions):
    period = 2 * np.pi
    torsions = (torsions + period / 2) / period
    dlogp = -np.log(period) * (torsions.shape[-1])
    return torsions, dlogp


def normalize_angles(angles):
    period = np.pi
    angles = angles / period
    dlogp = -np.log(period) * (angles.shape[-1])
    return angles, dlogp


def unnormalize_torsions(torsions):
    period = 2 * np.pi
    torsions = torsions * (period) - period / 2
    dlogp = np.log(period) * (torsions.shape[-1])
    return torsions, dlogp


def unnormalize_angles(angles):
    period = np.pi
    angles = angles * period
    dlogp = np.log(period) * (angles.shape[-1])
    return angles, dlogp





class ReferenceSystemTransformation(Flow):
    """
    Internal coordinate transformation of the reference frame set by the first three atoms.

    Please not that the forward transformation transforms *from* xyz coordinates *into* internal coordinates.

    By default output angles and torsions are normalized and fit into a (0, 1) interval.


    Parameters:
    ----------
    normalize_angles : bool
        bring angles and torsions into (0, 1) interval
    eps : float
        numerical epsilon used to enforce manifold boundaries
    raise_warnings : bool
        raise warnings if manifold boundaries are violated
    """

    def __init__(self, normalize_angles=True, orientation="euler", eps=1e-7, enforce_boundaries=True, raise_warnings=True):
        super().__init__()
        self._normalize_angles = normalize_angles
        self._eps = eps
        self._enforce_boundaries = enforce_boundaries
        self._raise_warnings = raise_warnings
        assert orientation in ["euler"]
        self._orientation = orientation

    def _forward(self, x0, x1, x2, *args, **kwargs):
        
        x0, d01, d12, a012, alpha, beta, gamma, dlogp = init_xyz2ics(
            x0, x1, x2,
            eps=self._eps,
            enforce_boundaries=self._enforce_boundaries,
            raise_warnings=self._raise_warnings
        )
        
        if self._normalize_angles:
            a012, dlogp_a = normalize_angles(a012)
            dlogp += dlogp_a
            
        return (x0, d01, d12, a012, alpha, beta, gamma, dlogp)

    def _inverse(self, x0, d01, d12, a012, alpha, beta, gamma, *args, **kwargs):
        
        dlogp = 0

        if self._normalize_angles:
            a012, dlogp_a = unnormalize_angles(a012)
            dlogp += dlogp_a
        
        x0, x1, x2, dlogp_b = init_ics2xyz(
            x0, d01, d12, a012, alpha, beta, gamma,
            eps=self._eps,
            enforce_boundaries=self._enforce_boundaries,
            raise_warnings=self._raise_warnings
        )
        
        dlogp += dlogp_b
        
        return (x0, x1, x2, dlogp)


class RelativeInternalCoordinateTransformation(Flow):
    """
    Internal coordinate transformation relative to a set of fixed atoms.

    Please not that the forward transformation transforms *from* xyz coordinates *into* internal coordinates.

    By default output angles and torsions are normalized and fit into a (0, 1) interval.

    Parameters:
    ----------
    z_matrix : Union[np.ndarray, torch.LongTensor]
        z matrix used for ic transformation
    fixed_atoms : np.ndarray
        atoms not affected by transformation
    normalize_angles : bool
        bring angles and torsions into (0, 1) interval
    eps : float
        numerical epsilon used to enforce manifold boundaries
    raise_warnings : bool
        raise warnings if manifold boundaries are violated

    Attributes
    ----------
    z_matrix : np.ndarray
        z matrix used for ic transformation
    fixed_atoms : np.ndarray
        atom indices that are kept as Cartesian coordinates
    dim_bonds : int
        number of bonds
    dim_angles : int
        number of angles
    dim_torsions : int
        number of torsions
    dim_fixed : int
        number of degrees of freedom for fixed atoms
    bond_indices : np.array of int
        atom ids that are connected by a bond (shape: (dim_bonds, 2))
    angle_indices : np.array of int
        atoms ids that are connected by an angle (shape: (dim_angles, 3))
    torsion_indices : np.array of int
        atoms ids that are connected by a torsion (shape: (dim_torsions, 4))
    normalize_angles : bool
        whether this transform normalizes angles and torsions to [0,1]

    """
    @property
    def z_matrix(self):
        return self._z_matrix

    @property
    def fixed_atoms(self):
        return self._fixed_atoms

    @property
    def dim_bonds(self):
        return len(self.z_matrix)

    @property
    def dim_angles(self):
        return len(self.z_matrix)

    @property
    def dim_torsions(self):
        return len(self.z_matrix)

    @property
    def dim_fixed(self):
        return 3*len(self._fixed_atoms)

    @property
    def bond_indices(self):
        return self._bond_indices

    @property
    def angle_indices(self):
        return self._angle_indices

    @property
    def torsion_indices(self):
        return self._torsion_indices

    @property
    def normalize_angles(self):
        return self._normalize_angles

    def __init__(
        self,
        z_matrix: Union[np.ndarray, torch.LongTensor],
        fixed_atoms: np.ndarray,
        normalize_angles: bool=True,
        eps: float=1e-7,
        enforce_boundaries: bool=True,
        raise_warnings: bool=True
    ):
        super().__init__()

        self._z_matrix = z_matrix

        self._fixed_atoms = fixed_atoms

        (
            self._z_blocks,
            self._index2atom,
            self._atom2index,
            self._index2order,
        ) = decompose_z_matrix(z_matrix, fixed_atoms)

        self._bond_indices = self._z_matrix[:, :2]
        self._angle_indices = self._z_matrix[:, :3]
        self._torsion_indices = self._z_matrix[:, :4]

        self._normalize_angles = normalize_angles

        self._eps = eps
        self._enforce_boundaries = enforce_boundaries
        self._raise_warnings = raise_warnings

    def _forward(self, x, with_pose=True, *args, **kwargs):

        n_batch = x.shape[0]
        x = x.view(n_batch, -1, 3)

        # compute bonds, angles, torsions
        # together with jacobians (wrt. to diagonal atom)
        bonds, jbonds = dist_deriv(
            x[:, self._z_matrix[:, 0]], x[:, self._z_matrix[:, 1]],
            eps=self._eps,
            enforce_boundaries=self._enforce_boundaries,
            raise_warnings=self._raise_warnings
        )
        angles, jangles = angle_deriv(
            x[:, self._z_matrix[:, 0]],
            x[:, self._z_matrix[:, 1]],
            x[:, self._z_matrix[:, 2]],
            eps=self._eps,
            enforce_boundaries=self._enforce_boundaries,
            raise_warnings=self._raise_warnings
        )
        torsions, jtorsions = torsion_deriv(
            x[:, self._z_matrix[:, 0]],
            x[:, self._z_matrix[:, 1]],
            x[:, self._z_matrix[:, 2]],
            x[:, self._z_matrix[:, 3]],
            eps=self._eps,
            enforce_boundaries=self._enforce_boundaries,
            raise_warnings=self._raise_warnings
        )

        # slice fixed coordinates needed to reconstruct the system
        x_fixed = x[:, self._fixed_atoms].view(n_batch, -1)

        # aggregated induced volume change
        dlogp = 0.0

        # transforms angles from [-pi, pi] to [0, 1]
        if self._normalize_angles:
            angles, dlogp_a = normalize_angles(angles)
            torsions, dlogp_t = normalize_torsions(torsions)
            dlogp += dlogp_a + dlogp_t

        # compute volume change
        j = torch.stack([jbonds, jangles, jtorsions], dim=-2)
        dlogp += det3x3(j).abs().log().sum(dim=1, keepdim=True)

        return bonds, angles, torsions, x_fixed, dlogp

    def _inverse(self, bonds, angles, torsions, x_fixed, **kwargs):

        # aggregated induced volume change
        dlogp = 0

        # transforms angles from [0, 1] to [-pi, pi]
        if self._normalize_angles:
            angles, dlogp_a = unnormalize_angles(angles)
            torsions, dlogp_t = unnormalize_torsions(torsions)
            dlogp += dlogp_a + dlogp_t

        # infer dimensions from input
        n_batch = x_fixed.shape[0]
        x_fixed = x_fixed.view(n_batch, -1, 3)
        n_fixed = x_fixed.shape[-2]
        n_conditioned = bonds.shape[-1]
        assert angles.shape[-1] == n_conditioned
        assert torsions.shape[-1] == n_conditioned

        # reconstruct points; initial points are the fixed points
        points = torch.empty((n_batch, n_fixed + n_conditioned, 3), dtype=x_fixed.dtype, device=x_fixed.device)
        points[:, :n_fixed, :] = x_fixed.view(n_batch, -1, 3)

        # blockwise reconstruction of points left
        current_index = n_fixed
        for block in self._z_blocks:

            # map atoms from z matrix
            # to indices in reconstruction order
            ref = self._atom2index[block]

            # slice three context points
            # from the already reconstructed
            # points using the indices
            context = points[:, ref[:, 1:]]
            p0 = context[:, :, 0]
            p1 = context[:, :, 1]
            p2 = context[:, :, 2]

            # obtain index of currently placed
            # point in original z-matrix
            idx = self._index2order[ref[:, 0] - len(self._fixed_atoms)]

            # get bonds, angles, torsions
            # using this z-matrix index
            b = bonds[:, idx, None]
            a = angles[:, idx, None]
            t = torsions[:, idx, None]

            # now we have three context points
            # and correct ic values to reconstruct the current point
            p, J = ic2xyz_deriv(
                p0,
                p1,
                p2,
                b,
                a,
                t,
                eps=self._eps,
                enforce_boundaries=self._enforce_boundaries,
                raise_warnings=self._raise_warnings
            )

            # compute jacobian
            dlogp += det3x3(J).abs().log().sum(-1)[:, None]

            # update list of reconstructed points
            points[:, current_index:current_index + p.shape[1], :] = p
            current_index += p.shape[1]

        # finally make sure that atoms are sorted
        # from reconstruction order to original order
        points = points[:, self._atom2index]

        return points.view(n_batch, -1), dlogp


class GlobalInternalCoordinateTransformation(Flow):
    """
    Global internal coordinate transformation.

    Please note that the forward transformation transforms *from* xyz coordinates *into* internal coordinates.

    By default output angles and torsions are normalized and fit into a (0, 1) interval.


    Parameters
    ----------
    z_matrix : Union[np.ndarray, torch.LongTensor]
        z matrix used for ic transformation
    normalize_angles : bool
        bring angles and torsions into (0, 1) interval
    eps : float
        numerical epsilon used to enforce manifold boundaries
    raise_warnings : bool
        raise warnings if manifold boundaries are violated

    Attributes
    ----------
    z_matrix : np.ndarray
        z matrix used by the underlying relative ic transformation
    fixed_atoms : np.ndarray
        empty array, just to satisfy the interface
    dim_bonds : int
        number of bonds
    dim_angles : int
        number of angles
    dim_torsions : int
        number of torsions
    dim_fixed : int
        is zero for this transform
    bond_indices : np.array of int
        atom ids that are connected by a bond (shape: (dim_bonds, 2))
    angle_indices : np.array of int
        atoms ids that are connected by an angle (shape: (dim_angles, 3))
    torsion_indices : np.array of int
        atoms ids that are connected by a torsion (shape: (dim_torsions, 4))
    normalize_angles : bool
        whether this transform normalizes angles and torsions to [0,1]
    """

    @property
    def z_matrix(self):
        return self._rel_ic.z_matrix

    @property
    def fixed_atoms(self):
        return np.array([], dtype=np.int64)

    @property
    def dim_bonds(self):
        return len(self.z_matrix) + 2

    @property
    def dim_angles(self):
        return len(self.z_matrix) + 1

    @property
    def dim_torsions(self):
        return len(self.z_matrix)

    @property
    def dim_fixed(self):
        return 0

    @property
    def bond_indices(self):
        fix = self._rel_ic.fixed_atoms
        return np.row_stack([
            np.array([
                [fix[1], fix[0]],
                [fix[2], fix[1]]
            ]),
            self._rel_ic.bond_indices,
        ])

    @property
    def angle_indices(self):
        fix = self._rel_ic.fixed_atoms
        return np.row_stack([
            np.array([
                [fix[2], fix[1], fix[0]]
            ]),
            self._rel_ic.angle_indices,
        ])

    @property
    def torsion_indices(self):
        return self._rel_ic.torsion_indices

    @property
    def normalize_angles(self):
        return self._rel_ic.normalize_angles

    def __init__(
        self,
        z_matrix,
        normalize_angles=True,
        eps: float=1e-7,
        enforce_boundaries: bool=True,
        raise_warnings: bool=True
    ):
        super().__init__()

        # find initial atoms
        initial_atoms, z_matrix = slice_initial_atoms(z_matrix)

        self._rel_ic = RelativeInternalCoordinateTransformation(
            z_matrix=z_matrix,
            fixed_atoms=initial_atoms,
            normalize_angles=normalize_angles,
            eps=eps,
            enforce_boundaries=enforce_boundaries,
            raise_warnings=raise_warnings
        )
        self._ref_ic = ReferenceSystemTransformation(
            normalize_angles=normalize_angles,
            eps=eps,
            enforce_boundaries=enforce_boundaries,
            raise_warnings=raise_warnings
        )

    def _forward(self, x, *args, **kwargs):
        """
        Parameters:
        ----------
        x: torch.Tensor
            xyz coordinates

        Returns:
        --------
        bonds: torch.Tensor
        angles: torch.Tensor
        torsions: torch.Tensor
        x0: torch.Tensor
            the systems origin point set in the first atom.
            has shape [batch, 1, 3]
        R: torch.Tensor
            the 3x3 matrix spanning the global rotation of the system
            spanned by the first three atoms. has shape [batch, 1, 3, 3]
        dlogp: torch.Tensor
            log det jacobian of the transformation
        """
        n_batch = x.shape[0]

        x = x.view(n_batch, -1, 3)

        # transform relative system wrt reference system
        bonds, angles, torsions, x_fixed, dlogp_rel = self._rel_ic(x, *args, **kwargs)

        x_fixed = x_fixed.view(n_batch, -1, 3)

        # transform reference system
        x0, d01, d12, a012, alpha, beta, gamma, dlogp_ref = self._ref_ic(
            x_fixed[:, [0]], x_fixed[:, [1]], x_fixed[:, [2]]
        )

        # gather bonds and angles
        bonds = torch.cat([d01, d12, bonds], dim=-1)

        angles = torch.cat([a012, angles], dim=-1)

        # aggregate volume change
        dlogp = dlogp_rel + dlogp_ref

        return bonds, angles, torsions, x0, alpha, beta, gamma, dlogp

    def _inverse(self, bonds, angles, torsions, x0, alpha, beta, gamma, *args, **kwargs):
        """
        Parameters:
        -----------
        bonds: torch.Tensor
        angles: torch.Tensor
        torsions: torch.Tensor
        x0: torch.Tensor
            system's origin. should have shape [batch, 1, 3]
        R: torch.Tensor
            orthogonal matrix setting the reference frame for the
            first three atoms. should have shape [batch, 1, 3, 3]

        Returns:
        --------
        x: torch.Tensor
            xyz coordinates
        dlogp: torch.Tensor
            log det jacobian of the transformation
        """

        # get ics of reference system
        d01 = bonds[:, [0]]
        d12 = bonds[:, [1]]
        a012 = angles[:, [0]]

        # transform reference system back
        x0, x1, x2, dlogp_ref = self._ref_ic(x0, d01, d12, a012, alpha, beta, gamma, inverse=True)
        x_init = torch.cat([x0, x1, x2], dim=1)

        # now transform relative system wrt reference system back
        x, dlogp_rel = self._rel_ic(
            bonds[:, 2:], angles[:, 1:], torsions, x_init, inverse=True
        )

        # aggregate volume change
        dlogp = dlogp_rel + dlogp_ref

        return x, dlogp


class MixedCoordinateTransformation(Flow):
    """
    Mixed coordinate transformation.

    This combines an relative coordinate transformation with a whitening transformation on the fixed atoms.

    Please note that the forward transformation transforms *from* xyz coordinates *into* internal coordinates.

    By default output angles and torsions are normalized and fit into a (0, 1) interval.

    Parameters
    ----------
    data : torch.Tensor
        data used to compute the whitening transformation of the fixed atoms
    z_matrix : Union[np.ndarray, torch.LongTensor]
        z matrix used for ic transformation
    fixed_atoms : torch.Tensor
        atoms not affected by transformation
    keepdims : Optional[int]
        number of dimensions kept in whitening transformation
    normalize_angles : bool
        bring angles and torsions into (0, 1) interval
    eps : float
        numerical epsilon used to enforce manifold boundaries
    raise_warnings : bool
        raise warnings if manifold boundaries are violated

    Attributes
    ----------
    z_matrix : np.ndarray
        z matrix used for ic transformation
    fixed_atoms : np.ndarray
        atom indices that are kept as Cartesian coordinates
    dim_bonds : int
        number of bonds
    dim_angles : int
        number of angles
    dim_torsions : int
        number of torsions
    dim_fixed : int
        number of learnable degrees of freedom for fixed atoms
    bond_indices : np.array of int
        atom ids that are connected by a bond (shape: (dim_bonds, 2))
    angle_indices : np.array of int
        atoms ids that are connected by an angle (shape: (dim_angles, 3))
    torsion_indices : np.array of int
        atoms ids that are connected by a torsion (shape: (dim_torsions, 4))
    normalize_angles : bool
        whether this transform normalizes angles and torsions to [0,1]
    """

    @property
    def z_matrix(self):
        return self._rel_ic.z_matrix

    @property
    def fixed_atoms(self):
        return self._rel_ic.fixed_atoms

    @property
    def dim_bonds(self):
        return len(self.z_matrix)

    @property
    def dim_angles(self):
        return len(self.z_matrix)

    @property
    def dim_torsions(self):
        return len(self.z_matrix)

    @property
    def dim_fixed(self):
        return self._whiten.keepdims

    @property
    def bond_indices(self):
        return self._rel_ic.bond_indices

    @property
    def angle_indices(self):
        return self._rel_ic.angle_indices

    @property
    def torsion_indices(self):
        return self._rel_ic.torsion_indices

    @property
    def normalize_angles(self):
        return self._rel_ic.normalize_angles

    def __init__(
        self,
        data: torch.Tensor,
        z_matrix: Union[np.ndarray, torch.Tensor],
        fixed_atoms: np.ndarray,
        keepdims: Optional[int]=None,
        normalize_angles=True,
        eps: float=1e-7,
        enforce_boundaries: bool=True,
        raise_warnings: bool=True
    ):
        super().__init__()
        self._whiten = self._setup_whitening_layer(data, fixed_atoms, keepdims=keepdims)
        self._rel_ic = RelativeInternalCoordinateTransformation(
            z_matrix=z_matrix,
            fixed_atoms=fixed_atoms,
            normalize_angles=normalize_angles,
            eps=eps,
            enforce_boundaries=enforce_boundaries,
            raise_warnings=raise_warnings
        )

    def _setup_whitening_layer(self, data, fixed_atoms, keepdims):
        n_data = data.shape[0]
        data = data.view(n_data, -1, 3)
        fixed = data[:, fixed_atoms].view(n_data, -1)
        return WhitenFlow(fixed, keepdims=keepdims, whiten_inverse=False)

    def _forward(self, x, *args, **kwargs):
        """
        Parameters:
        -----------
        x: torch.Tensor
            xyz coordinates

        Returns:
        --------
        bonds: torch.Tensor
        angles: torch.Tensor
        torsions: torch.Tensor
        z_fixed: torch.Tensor
            whitened fixed atom coordinates
        dlogp: torch.Tensor
            log det jacobian of the transformation
        """
        n_batch = x.shape[0]
        bonds, angles, torsions, x_fixed, dlogp_rel = self._rel_ic(x)
        x_fixed = x_fixed.view(n_batch, -1)
        z_fixed, dlogp_ref = self._whiten(x_fixed)
        dlogp = dlogp_rel + dlogp_ref
        return bonds, angles, torsions, z_fixed, dlogp

    def _inverse(self, bonds, angles, torsions, z_fixed, *args, **kwargs):
        """
        Parameters:
        -----------
        bonds: torch.Tensor
        angles: torch.Tensor
        torsions: torch.Tensor
        z_fixed: torch.Tensor
            whitened fixed atom coordinates

        Returns:
        --------
        x: torch.Tensor
            xyz coordinates
        dlogp: torch.Tensor
            log det jacobian of the transformation
        """
        n_batch = z_fixed.shape[0]
        x_fixed, dlogp_ref = self._whiten(z_fixed, inverse=True)
        x_fixed = x_fixed.view(n_batch, -1, 3)
        x, dlogp_rel = self._rel_ic(bonds, angles, torsions, x_fixed, inverse=True)
        dlogp = dlogp_rel + dlogp_ref
        return x, dlogp
