import re
def get_layer_id(name):
    if 'embeddings' in name or 'embed' in name:
        return 0

    match = re.search(r'encoder\.layer\.(\d+)', name)
    if match:
        return int(match.group(1)) + 1
    # for some models, the layer is named as layers.0.self_attn.q_proj
    
    match = re.search(r'layers\.(\d+)', name)
    if match:
        return int(match.group(1)) + 1
    
    # for decoder layers
    match = re.search(r'decoder\.(\d+)', name)
    if match:
        return 'decoder.' + str(int(match.group(1)) + 1)

    return 999
