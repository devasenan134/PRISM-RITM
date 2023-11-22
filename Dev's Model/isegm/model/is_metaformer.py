import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.metaformer import SepConv,MetaFormer,PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead

class Metaformer(ISModel,MetaFormer):
    @serialize

    def __init__(self,
                 random_split=False,
                 head_params={},
                 **kwargs
                ):
        
        super().__init__(**kwargs)
        self.random_split=random_split
        # mF = MetaFormer()
        # self.default_cfg = mF.default_cfg
        # self.backbone=MetaFormer(**backbone_params)
        self.head = SwinTransfomerSegHead(**head_params)


        
    def backbone_forward(self, image, coord_features=None):
        
        features=MetaFormer.forward_features(self,x=image,additional_features=coord_features)

        return {'instances': self.head(features), 'instances_aux': None}
