from nequip.nn import SequentialGraphNetwork
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding
)
import e3nn
import nequip
import allegro
import torch
from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_Module,
    ScalarMLP,
)
from ipdb import set_trace as bp
# define the Normalized Basis that is also shifted a bit to increase stability:

class NormalizedBasis(torch.nn.Module):
    """Normalized version of a given radial basis.
    It is passed distance values from the training data so it can ensure that
     the RBFs output is normalized when passed data like this.

    Args:
        data: distance values that are exemplaric for the kind of data the RBFs will be passed, shape might be for example:
        torch.Size([40323])
        basis (constructor): callable to build the underlying basis
        basis_kwargs (dict): parameters for the underlying basis
        offset: in nm, the offset to apply to all distances to avoid large RBF values at small distance values

    """

    num_basis: int

    def __init__(
        self,
        data = None,
        original_basis=BesselBasis,
        original_basis_kwargs: dict = {},
        norm_basis_mean_shift: bool = True,
        offset = 1.
    ):
        super().__init__()
        self.offset = offset
        #### shift all entries to the right a bit.
        data += self.offset
        #### change r_max accordingly
        original_basis_kwargs["r_max"] += self.offset
        self.basis = original_basis(**original_basis_kwargs)
        self.num_basis = self.basis.num_basis
        #bp()

        with torch.no_grad():
            if data is None:
                raise ValueError("gotta pass data to inform the Basis Function")
            bs = self.basis(data)
            assert bs.ndim == 2
            if norm_basis_mean_shift:
                basis_std, basis_mean = torch.std_mean(bs, dim=0)
            else:
                basis_std = bs.square().mean().sqrt()
                basis_mean = torch.as_tensor(
                    0.0, device=basis_std.device, dtype=basis_std.dtype
                )
        self.register_buffer("_mean", basis_mean)
        self.register_buffer("_inv_std", torch.reciprocal(basis_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_o = x + self.offset
        return (self.basis(x_o) - self._mean) * self._inv_std


import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from typing import Optional, Any, Union, Callable

class CustomTransformerEncoderLayer(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)





    THe native torch.nn.TransformerENcoderLayer does not have linear Layers
    producing the  QKV vectors, but instead uses on and the same input vector for all these.

    so i added this feature.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        x = self.self_attn(q, k, v,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)






#gradient checkpointing to lower memory consumption at the cost of speed

#hyperparams
allegro_hparams = {
"r_max" : 1.2,  #1.2,
"num_types" : 10,
"num_basis" : 32, #8
"p" : 6,#48
"avg_num_neighbors" : 9, #
"num_layers" : 3,
"env_embed_multiplicity" : 32 ,#16
"latent_dim" : 32,#32, #16 #512? 32 geht
"two_body_latent_indermediate_dims" : [128, 128, 128],#,[64, 128, 256], #[64,64,64]#[64, 128, 256] #war mal 64 128 256 512 #64s
"nonscalars_include_parity" : False, #True
"irreps_edge_sh" :  '1x0e+1x1o+1x2e',# calculate only vectors and scalars :'1x0e+1x1o'
"RBF_distance_offset" : 1. ,
"GNN_output_dim" : 32, #"32x0e",
"latent_resnet": True,
"GNN_scope": "atomwise"
}


def make_allegro_config_dict(**kwargs): ###rename to make_allegro_config_dict
    r_max = kwargs["r_max"]
    num_types = kwargs["num_types"]
    num_basis = kwargs["num_basis"]
    p = kwargs["p"]
    avg_num_neighbors = kwargs["avg_num_neighbors"]
    num_layers = kwargs["num_layers"]
    env_embed_multiplicity = kwargs["env_embed_multiplicity"]
    latent_dim = kwargs["latent_dim"]
    two_body_latent_indermediate_dims = kwargs["two_body_latent_indermediate_dims"]
    nonscalars_include_parity = kwargs["nonscalars_include_parity"]
    irreps_edge_sh = kwargs["irreps_edge_sh"]
    RBF_distance_offset = kwargs["RBF_distance_offset"]
    GNN_output_dim = kwargs["GNN_output_dim"]
    latent_resnet = kwargs["latent_resnet"]
    GNN_scope = kwargs["GNN_scope"]
             
    base_dict =   {'one_hot': (nequip.nn.embedding._one_hot.OneHotAtomEncoding,
                                    {'irreps_in': None, 'set_features': True, 'num_types': num_types}),


                   'radial_basis': (nequip.nn.embedding._edge.RadialBasisEdgeEncoding,
                                    {'basis': NormalizedBasis,
                                     'cutoff': nequip.nn.cutoffs.PolynomialCutoff,
                                     'basis_kwargs': {'data': None,
                                                      'original_basis': nequip.nn.radial_basis.BesselBasis,
                                                      'original_basis_kwargs': {'num_basis': num_basis,
                                                                                'trainable': True,
                                                                                'r_max': r_max},
                                                      'norm_basis_mean_shift': True,
                                                      'offset': RBF_distance_offset},
                                     'cutoff_kwargs': {'p': p, 'r_max': r_max},
                                     'out_field': 'edge_embedding'}),


                   'spharm': (nequip.nn.embedding._edge.SphericalHarmonicEdgeAttrs,
                                    {'edge_sh_normalization': 'component',
                                     'edge_sh_normalize': True,
                                     'out_field': 'edge_attrs',
                                     'irreps_edge_sh': irreps_edge_sh}),





                   'allegro': (allegro.nn._allegro.Allegro_Module, 
                                   {'avg_num_neighbors': avg_num_neighbors,
                                     'r_start_cos_ratio': 0.8, # unused
                                     'PolynomialCutoff_p': p,
                                     'per_layer_cutoffs': None,
                                     'cutoff_type': 'polynomial',
                                     'field': 'edge_attrs',
                                     'edge_invariant_field': 'edge_embedding',
                                     'node_invariant_field': 'node_attrs',
                                     'env_embed_multiplicity': env_embed_multiplicity,
                                     'embed_initial_edge': True,
                                     'linear_after_env_embed': False,
                                     'nonscalars_include_parity': nonscalars_include_parity,
                                     'two_body_latent': allegro.nn._fc.ScalarMLPFunction,
                                     'two_body_latent_kwargs': {'mlp_nonlinearity': 'silu',
                                                                'mlp_initialization': 'uniform',
                                                                'mlp_dropout_p': 0.0,
                                                                'mlp_batchnorm': False,#False
                                                                'mlp_latent_dimensions': [*two_body_latent_indermediate_dims, latent_dim]},
                                     'env_embed': allegro.nn._fc.ScalarMLPFunction,
                                     'env_embed_kwargs': {'mlp_nonlinearity': None,
                                                          'mlp_initialization': 'uniform',
                                                          'mlp_dropout_p': 0.0,
                                                          'mlp_batchnorm': False, #False
                                                          'mlp_latent_dimensions': []},
                                     'latent': allegro.nn._fc.ScalarMLPFunction,
                                     'latent_kwargs': {'mlp_nonlinearity': 'silu',
                                                       'mlp_initialization': 'uniform',
                                                       'mlp_dropout_p': 0.0,
                                                       'mlp_batchnorm': False, #False
                                                       'mlp_latent_dimensions': [latent_dim]},
                                     'latent_resnet': latent_resnet,
                                     'latent_resnet_update_ratios': None,
                                     'latent_resnet_update_ratios_learnable': False,
                                     'latent_out_field': 'edge_features',
                                     'pad_to_alignment': 1,
                                     'sparse_mode': None,
                                     'r_max': r_max,
                                     'num_layers': num_layers,
                                     'num_types': num_types}),
                   }

    atomwise_dict = {'atomwise_gather': (allegro.nn._edgewise.EdgewiseReduce,
                                      {'field': 'edge_features',
                                       'out_field': 'atomwise_features',
                                       'avg_num_neighbors': avg_num_neighbors}
                                     ),
                  'atomwise_linear': (nequip.nn._atomwise.AtomwiseLinear,
                                      {'field': 'atomwise_features',
                                       'out_field': 'outputs',
                                       'irreps_out': f"{GNN_output_dim}x0e"}
                                      )
    }
    bondwise_dict = {'bondwise_linear': (nequip.nn._atomwise.AtomwiseLinear,  ## it is still edgewiese
                                      {'field': 'edge_features',
                                       'out_field': 'outputs',
                                       'irreps_out': f"{GNN_output_dim}x0e"}
                                      )
                    }
    if GNN_scope == "atomwise":
        dict =  base_dict|atomwise_dict
    elif GNN_scope == "bondwise":
        dict = base_dict|bondwise_dict
    else:
        raise ValueError('GNN_scope must be either "atomwise" or "bondwise"')
    return dict


allegro_config_dict = make_allegro_config_dict(**allegro_hparams)

nequip_hparams = {
    "r_max": 1.2,  # 1.2,
    "num_types": 10,
    "num_basis": 32,  # 8
    "p": 6,  # 48
    "avg_num_neighbors": 9,  #
    "num_layers": 3,
    "latent_dim": 32,  # 32, #16 #512? 32 geht
    "nonscalars_include_parity": False,  # True
    "irreps_edge_sh": '1x0e+1x1o+1x2e',  # calculate only vectors and scalars :'1x0e+1x1o'  change this to "1x0e" to reduce nequip to schnett basically
    "RBF_distance_offset": 1.,
    "GNN_output_dim": 32,  # "32x0e",

    "num_interaction_blocks": 4
}


def make_nequip_config_dict(**kwargs):  ###rename to make_nequip_config_dict
    # bp()
    r_max = kwargs["r_max"]
    num_types = kwargs["num_types"]
    num_basis = kwargs["num_basis"]
    p = kwargs["p"]
    irreps_edge_sh = kwargs["irreps_edge_sh"]
    RBF_distance_offset = kwargs["RBF_distance_offset"]
    GNN_output_dim = kwargs["GNN_output_dim"]
    num_interaction_blocks = kwargs["num_interaction_blocks"]

    base_dict =   {'one_hot': (nequip.nn.embedding._one_hot.OneHotAtomEncoding,
                                    {'irreps_in': None, 'set_features': True, 'num_types': num_types}),


                   'radial_basis': (nequip.nn.embedding._edge.RadialBasisEdgeEncoding,
                                    {'basis': NormalizedBasis,
                                     'cutoff': nequip.nn.cutoffs.PolynomialCutoff,
                                     'basis_kwargs': {'data': None,
                                                      'original_basis': nequip.nn.radial_basis.BesselBasis,
                                                      'original_basis_kwargs': {'num_basis': num_basis,
                                                                                'trainable': True,
                                                                                'r_max': r_max},
                                                      'norm_basis_mean_shift': True,
                                                      'offset': RBF_distance_offset},
                                     'cutoff_kwargs': {'p': p, 'r_max': r_max},
                                     'out_field': 'edge_embedding'}),


                   'spharm': (nequip.nn.embedding._edge.SphericalHarmonicEdgeAttrs,
                                    {'edge_sh_normalization': 'component',
                                     'edge_sh_normalize': True,
                                     'out_field': 'edge_attrs',
                                     'irreps_edge_sh': irreps_edge_sh}),

                   'chemical_embedding': (nequip.nn._atomwise.AtomwiseLinear,
                                    {'field': 'node_features',
                                     'out_field': None,
                                     'irreps_out': '32x0e'})
                   }
    conv_dict = {}

    for i in range(num_interaction_blocks):
        conv_dict_i = {f'convnet_{i}': (nequip.nn._convnetlayer.ConvNetLayer,
                       {'convolution': nequip.nn._interaction_block.InteractionBlock,
                        'convolution_kwargs': {'invariant_layers': 2,
                                               'invariant_neurons': 64,
                                               'avg_num_neighbors': None, ### have to set this from data
                                               'use_sc': True,
                                               'nonlinearity_scalars': {'e': 'silu',
                                                                        'o': 'tanh'}},
                        'num_layers': 4,  ## this is a dead end argument
                        'resnet': False,
                        'nonlinearity_type': 'gate',
                        'nonlinearity_scalars': {'e': 'silu',
                                                 'o': 'tanh'},
                        'nonlinearity_gates': {'e': 'silu',
                                               'o': 'tanh'},
                        'feature_irreps_hidden': '32x0e+32x1e+32x0o+32x1o'})
                    }
        #bp()
        conv_dict = {**conv_dict, **conv_dict_i}






    output_dict = {
    'self_interaction_0': (nequip.nn._atomwise.AtomwiseLinear,
                          {'field': 'node_features',
                           'out_field': 'outputs',
                           'irreps_out': f'{GNN_output_dim}x0e'})
                   }

    dict = base_dict | conv_dict | output_dict
    return dict


nequip_config_dict = make_nequip_config_dict(**nequip_hparams)

class nequip_wrapper(torch.nn.Module):
    def __init__(
        self,
        nequip_GNN = SequentialGraphNetwork.from_parameters,
        output_field="outputs",
        **kwargs
    ):
        super().__init__()
        if isinstance(nequip_GNN, torch.nn.Module):
            self.GNN = nequip_GNN
        elif callable(nequip_GNN):
            self.GNN = nequip_GNN(**kwargs)
        self.output_field = output_field
    def forward(self, x):
        return self.GNN.forward(x)[self.output_field]