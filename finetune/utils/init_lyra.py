import torch.nn as nn

def add_framepack_components(dit_model):
    """添加FramePack相关组件"""
    if not hasattr(dit_model, 'clean_x_embedder'):
        inner_dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
        
        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            
            def forward(self, x, scale="1x"):
                if scale == "1x":
                    x = x.to(self.proj.weight.dtype)
                    return self.proj(x)
                elif scale == "2x":
                    x = x.to(self.proj_2x.weight.dtype)
                    return self.proj_2x(x)
                elif scale == "4x":
                    x = x.to(self.proj_4x.weight.dtype)
                    return self.proj_4x(x)
                else:
                    print(f"❌ Unsupported scale: {scale}")
                    raise ValueError(f"Unsupported scale: {scale}")
        
        dit_model.clean_x_embedder = CleanXEmbedder(inner_dim)
        model_dtype = next(dit_model.parameters()).dtype
        dit_model.clean_x_embedder = dit_model.clean_x_embedder.to(dtype=model_dtype)
        print("✅ 添加了FramePack的clean_x_embedder组件")
        
def replace_dit_model_in_manager():
    """Replace DiT model class with camera version"""
    from diffsynth.models.wan_video_dit_cam import WanModelCam
    from diffsynth.configs.model_config import model_loader_configs

    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config

        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []

            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelCam)
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)

            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)