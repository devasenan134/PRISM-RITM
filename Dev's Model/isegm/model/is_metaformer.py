import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.metaformer import SepConv,MetaFormer,PatchEmbed


class Metaformer(ISModel,MetaFormer):
    @serialize

    def __init__(self,
                 random_split=False,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.random_split=random_split
        # mF = MetaFormer()
        # self.default_cfg = mF.default_cfg
        # self.backbone=MetaFormer(**backbone_params)
        # self.head=MlpHead(**head_params)


        
    def backbone_forward(self, image, coord_features=None):
        
        features=MetaFormer.forward_features(self,x=image,additional_features=coord_features)

        return {'instances': SepConv(features[0]), 'instances_aux': None}
