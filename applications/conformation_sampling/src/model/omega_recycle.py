import sys
import torch
from torch import nn
import typing
from omegafold.decode import TorsionAngleHead,StructureCycle
from omegafold.geoformer import GeoFormer
from omegafold.embedders import RecycleEmbedder
import argparse
import typing
from omegafold import modules, utils
from openfold.utils.rigid_utils import Rigid,Rotation
from openfold.np.residue_constants  import  restype_atom14_mask
from contextlib import ExitStack

class StructureModule(modules.OFModule):
    """Jumper et al. (2021) Suppl. Alg. 20 'StructureModule'"""

    def __init__(self, cfg: argparse.Namespace):
        super(StructureModule, self).__init__(cfg)
        self.node_norm = nn.LayerNorm(cfg.node_dim)
        self.edge_norm = nn.LayerNorm(cfg.edge_dim)
        self.init_proj = nn.Linear(cfg.node_dim, cfg.node_dim)

        self.cycles = nn.ModuleList(
            [StructureCycle(cfg) for _ in range(cfg.num_cycle)]
        )
        self.torsion_angle_pred = TorsionAngleHead(cfg)

    def forward(
            self,
            node_repr: torch.Tensor, edge_repr: torch.Tensor,
            mask: torch.Tensor,fasta: torch.Tensor,init_frames
    ):
        """
        Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

        Args:
            node_repr: node representation tensor of shape [num_res, dim_node]
            edge_repr: edge representation tensor of shape [num_res, dim_edge]
            fasta: the tokenized sequence of the input protein sequence
            mask

        Returns:
            node_repr: The current node representation tensor for confidence
                of shape [num_res, dim_node]
            dictionary containing:
                final_atom_positions: the final atom14 positions,
                    of shape [num_res, 14, 3]
                final_atom_mask: the final atom14 mask,
                    of shape [num_res, 14]

        """
        node_repr = self.node_norm(node_repr)
        edge_repr = self.edge_norm(edge_repr)

        init_node_repr = node_repr
        node_repr = self.init_proj(node_repr)
        # Initialize the initial frames with Black-hole Jumper et al. (2021)
        backbone_frames = utils.AAFrame.default_init(
            *node_repr.shape[:-1],
            unit='nano',
            device=self.device,
            mask=mask.bool()
        )

        for layer in self.cycles:
            node_repr, backbone_frames = layer(
                node_repr, edge_repr, backbone_frames
            )

        torsion_angles_sin_cos = self.torsion_angle_pred(
            representations_list=[node_repr, init_node_repr],
        )

        torsion_angles_mask = torch.ones_like(
            torsion_angles_sin_cos[..., 0], dtype=torch.bool
        )
        backbone_frames = backbone_frames.to_angstrom(in_place=False)

        frames8 = backbone_frames.expand_w_torsion(
            torsion_angles=torsion_angles_sin_cos,
            torsion_angles_mask=torsion_angles_mask,
            fasta=fasta
        )
        pos14, mask14 = frames8.expanded_to_pos(fasta)
        # convert to openfold form
        rots = Rotation(rot_mats=backbone_frames.rotation)
        return node_repr,{
            "final_frames": Rigid(rots=rots,trans=backbone_frames.translation).to_tensor_7(),
            "final_atom_positions": pos14,
            "angles":torsion_angles_sin_cos
        }
