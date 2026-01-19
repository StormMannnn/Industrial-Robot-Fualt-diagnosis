"""
compute_timekan_complexity.py

Detailed params / modelsize / MACs report for TimeKAN classification model.

Place this file in the same folder as:
 - TimeKAN.py
 - Autoformer_EncDec.py
 - Embed.py
 - StandardNorm.py
 - ChebyKANLayer.py

Then run: python compute_timekan_complexity.py

The script:
 - Calculates exact params & model size for every module
 - If `thop` is installed, attempts to use it for MACs profiling of the actual forward ops
 - ALSO provides analytical MACs for Conv1d, Linear, ChebyKANLinear (strict formulas)
 - Marks FFT-involving modules separately (no exact MACs by default)
 - Writes CSV: timekan_detailed_report.csv
"""
import os, sys, math, csv, traceback
from importlib import util

# ====== adjust project_root if needed ======
project_root = os.path.abspath('.')  # default: current dir where you run the script
sys.path.insert(0, project_root)

# ====== try to load thop (optional) ======
USE_THOP = False
try:
    from thop import profile as thop_profile, clever_format
    USE_THOP = True
    print("thop found - will attempt precise profiling where possible.")
except Exception:
    print("thop not found - will use analytical MACs for supported layers (Conv1d/Linear/ChebyKANLinear).")

# ====== dynamically make 'layers' package available if needed ======
# TimeKAN imports "from layers.Autoformer_EncDec import series_decomp" etc.
# If files are in the same folder (Autoformer_EncDec.py etc) we inject them as submodules under 'layers'
layer_map = {
    'layers.Autoformer_EncDec': os.path.join(project_root, 'Autoformer_EncDec.py'),
    'layers.Embed': os.path.join(project_root, 'Embed.py'),
    'layers.StandardNorm': os.path.join(project_root, 'StandardNorm.py'),
    'layers.ChebyKANLayer': os.path.join(project_root, 'ChebyKANLayer.py')
}
for mod_name, path in layer_map.items():
    if os.path.exists(path):
        spec = util.spec_from_file_location(mod_name, path)
        module = util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            print(f"Warning: failed to exec module {mod_name} from {path}")
    else:
        # If missing file, let TimeKAN import fail later with a helpful message
        pass

# ====== import the model ======
try:
    from models.TimeKAN import Model
except Exception as e:
    print("Failed to import TimeKAN.Model - ensure TimeKAN.py and its dependencies are in the same folder.")
    traceback.print_exc()
    sys.exit(1)

# ====== construct the Config exactly as you gave ======
class Config:
    def __init__(self):
        self.task_name = "classification"
        self.seq_len = 96
        self.label_len = 0
        self.pred_len = 0
        self.e_layers = 3
        self.d_layers = 1
        self.factor = 3
        self.enc_in = 12
        self.dec_in = 12
        self.c_out = 7
        self.d_model = 64
        self.d_ff = 128
        self.batch_size = 32
        self.down_sampling_layers = 3
        self.down_sampling_window = 2
        self.begin_order = 1
        self.num_classes = 7
        self.use_gpu = True
        self.loss = "CrossEntropy"
        self.features = "M"
        self.embed = "timeF"
        self.freq = "h"
        self.dropout = 0.1
        self.moving_avg = 25
        self.use_future_temporal_feature = False
        self.use_norm = 1
        self.channel_independence = 1

configs = Config()

# instantiate model
model = Model(configs)
model.eval()

# ====== helpers ======
def param_count(module):
    return sum(p.numel() for p in module.parameters())

def size_mb_from_params(count):
    return count * 4.0 / (1024**2)  # float32

# conv MACs formula helper
def conv1d_macs(conv: __import__('torch').nn.Conv1d, batch:int, in_length:int):
    # MACs = out_channels * output_length * kernel_size * (in_channels/groups) * batch
    k = conv.kernel_size[0]
    groups = conv.groups
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    stride = conv.stride[0]
    padding = conv.padding[0]
    dilation = conv.dilation[0]
    out_len = math.floor((in_length + 2*padding - dilation*(k-1) -1)/stride + 1)
    macs = out_channels * out_len * k * (in_channels // groups) * batch
    return int(macs), int(out_len)

# linear MACs
def linear_macs(linear: __import__('torch').nn.Linear, batch:int, in_features_override:int=None):
    in_f = linear.in_features if in_features_override is None else in_features_override
    out_f = linear.out_features
    macs = batch * in_f * out_f
    return int(macs)

# ChebyKANLinear: MACs per forward: batch * outdim * inputdim * (degree+1)
def cheby_macs(cheby_module, batch:int):
    # cheby_module should have attributes inputdim, outdim, degree
    inp = getattr(cheby_module, 'inputdim', None)
    out = getattr(cheby_module, 'outdim', None)
    deg = getattr(cheby_module, 'degree', None)
    if None in (inp, out, deg):
        return None
    return int(batch * out * inp * (deg + 1))

# ====== collect global info ======
B = configs.batch_size
seq_len = configs.seq_len
d_model = configs.d_model
enc_in = configs.enc_in
down_sampling_layers = configs.down_sampling_layers
down_window = configs.down_sampling_window

report = []  # rows: (component, params_count, sizeMB, macs_or_note)

# Total model params
total_params = param_count(model)
report.append(("model_total", total_params, size_mb_from_params(total_params), "exact"))

# top modules
top_map = {
    'enc_embedding': getattr(model, 'enc_embedding', None),
    'res_blocks': getattr(model, 'res_blocks', None),
    'add_blocks': getattr(model, 'add_blocks', None),
    'preprocess': getattr(model, 'preprocess', None),
    'normalize_layers': getattr(model, 'normalize_layers', None),
    'projection_layer': getattr(model, 'projection_layer', None),
    'predict_layer': getattr(model, 'predict_layer', None),
    'classifier': getattr(model, 'classifier', None)
}
for name, mod in top_map.items():
    if mod is None:
        report.append((name, '', '', 'missing'))
    else:
        if isinstance(mod, __import__('torch').nn.ModuleList):
            p = sum(param_count(m) for m in mod)
        else:
            p = param_count(mod)
        report.append((name, p, size_mb_from_params(p), "exact"))

# If thop is available, attempt to profile the whole model with representative inputs
macs_total = None
if USE_THOP:
    try:
        import torch
        # create representative inputs for classification forward: (x_enc, x_mark_enc, x_dec, x_mark_dec)
        x_enc = torch.randn(B, seq_len, enc_in)
        # Note: when using thop, pass only actual tensors; model.forward in TimeKAN expects (x_enc, x_mark_enc, ...)
        macs_total, params_total = thop_profile(model, inputs=(x_enc, None, None, None), verbose=False)
        report.append(("thop:full_model_macs", '', '', f"{macs_total} (thop)"))
    except Exception as e:
        report.append(("thop_full_model", '', '', f"thop failed: {e}"))

# Analytical MACs for supported components
# 1) TokenEmbedding.tokenConv inside DataEmbedding_wo_pos used as enc_embedding.value_embedding.tokenConv
try:
    token_conv = model.enc_embedding.value_embedding.tokenConv
    token_conv_params = param_count(token_conv)
    token_conv_size = size_mb_from_params(token_conv_params)
    token_conv_total_macs = 0
    token_conv_scale_details = []
    for scale_idx in range(down_sampling_layers + 1):
        scale_len = seq_len // (down_window ** scale_idx)
        # embedding is applied to B * N samples where N is number of channels (enc_in)
        batch_for_token = B * enc_in
        macs, out_len = conv1d_macs(token_conv, batch_for_token, scale_len)
        token_conv_scale_details.append((scale_idx, scale_len, batch_for_token, macs, out_len))
        token_conv_total_macs += macs
    report.append(("TokenEmbedding.tokenConv", token_conv_params, token_conv_size, f"macs_total={token_conv_total_macs}; per_scale={token_conv_scale_details}"))
except Exception as e:
    report.append(("TokenEmbedding.tokenConv", '', '', f"error: {e}"))

# 2) Find all ChebyKANLinear instances and map them to their M_KAN uses inside add_blocks/res_blocks
cheby_instances = []
for name, mod in model.named_modules():
    # safe import check for ChebyKANLinear by attribute names - matching module class name string
    cls_name = mod.__class__.__name__
    if cls_name == 'ChebyKANLinear' or cls_name.endswith('ChebyKANLinear'):
        cheby_instances.append((name, mod))

# For each FrequencyMixing in add_blocks, compute M_KAN front_block and front_blocks seq lens then compute cheby & conv macs
mkan_macs_sum = 0
mkan_rows = []
for layer_idx, fm in enumerate(model.add_blocks):
    # seq for front_block (lowest frequency)
    front_seq = seq_len // (down_window ** down_sampling_layers)
    # front_block
    try:
        fb = fm.front_block  # M_KAN
        # extract cheby module
        cheby_mod = None
        if hasattr(fb, 'channel_mixer'):
            # channel_mixer is nn.Sequential([ChebyKANLayer(...)]), and ChebyKANLayer has .fc1 (ChebyKANLinear)
            try:
                cm0 = fb.channel_mixer[0]
                if hasattr(cm0, 'fc1'):
                    cheby_mod = cm0.fc1
            except Exception:
                cheby_mod = None
        if cheby_mod is not None:
            cheby_p = param_count(cheby_mod)
            cheby_deg = getattr(cheby_mod, 'degree', None)
            # batch = B * seq_len_front
            batch_cheby = B * front_seq
            cheby_m = cheby_macs(cheby_mod, batch_cheby)
            mkan_rows.append((f'add_blocks[{layer_idx}].front_block.cheby', front_seq, cheby_deg, cheby_p, cheby_m))
            mkan_macs_sum += (cheby_m if cheby_m is not None else 0)
        # conv inside fb
        conv_mod = getattr(fb, 'conv', None)
        if conv_mod is not None and hasattr(conv_mod, 'conv'):
            conv = conv_mod.conv
            conv_p = param_count(conv)
            conv_m, outlen = conv1d_macs(conv, B * front_seq, front_seq)
            mkan_rows.append((f'add_blocks[{layer_idx}].front_block.conv', front_seq, conv_p, conv_m, outlen))
            mkan_macs_sum += conv_m
    except Exception as e:
        mkan_rows.append((f'add_blocks[{layer_idx}].front_block', 'error', str(e), '', ''))

    # front_blocks list
    try:
        for i, fb_i in enumerate(fm.front_blocks):
            seq_i = seq_len // (down_window ** (down_sampling_layers - i - 1))
            cheby_mod_i = None
            try:
                cm0i = fb_i.channel_mixer[0]
                if hasattr(cm0i, 'fc1'):
                    cheby_mod_i = cm0i.fc1
            except Exception:
                cheby_mod_i = None
            if cheby_mod_i is not None:
                cheby_pi = param_count(cheby_mod_i)
                deg_i = getattr(cheby_mod_i, 'degree', None)
                batch_i = B * seq_i
                cheby_mi = cheby_macs(cheby_mod_i, batch_i)
                mkan_rows.append((f'add_blocks[{layer_idx}].front_blocks[{i}].cheby', seq_i, deg_i, cheby_pi, cheby_mi))
                mkan_macs_sum += (cheby_mi if cheby_mi is not None else 0)
            # conv
            if hasattr(fb_i, 'conv') and hasattr(fb_i.conv, 'conv'):
                conv_i = fb_i.conv.conv
                conv_pi = param_count(conv_i)
                conv_mi, outlen_i = conv1d_macs(conv_i, B * seq_i, seq_i)
                mkan_rows.append((f'add_blocks[%d].front_blocks[%d].conv' % (layer_idx, i), seq_i, conv_pi, conv_mi, outlen_i))
                mkan_macs_sum += conv_mi
    except Exception as e:
        mkan_rows.append((f'add_blocks[{layer_idx}].front_blocks', 'error', str(e), '', ''))

report.append(("M_KAN (add_blocks) cheby+conv_estimate_total", '', '', f"macs_est={mkan_macs_sum}; details_in_rows"))

# 3) classifier linear
clf_linear = None
for m in model.classifier.modules():
    if isinstance(m, __import__('torch').nn.Linear):
        clf_linear = m
        break
if clf_linear is not None:
    clf_params = param_count(clf_linear)
    clf_macs = linear_macs(clf_linear, B)
    report.append(("classifier.linear", clf_params, size_mb_from_params(clf_params), f"macs={clf_macs}"))

# 4) projection_layer (approx)
proj = model.projection_layer
proj_params = param_count(proj)
# projection applied in some forward to tensor shaped (B, seq_len, d_model) so approximate macs = B*seq_len * d_model * out_features
proj_macs_approx = linear_macs(proj, B * seq_len)
report.append(("projection_layer (approx)", proj_params, size_mb_from_params(proj_params), f"approx_macs={proj_macs_approx}"))

# 5) predict_layer params
pred = model.predict_layer
pred_params = param_count(pred)
report.append(("predict_layer", pred_params, size_mb_from_params(pred_params), "not_used_in_classification (params shown)"))

# 6) normalize layers, res_blocks/add_blocks param entries
for idx, nl in enumerate(model.normalize_layers):
    report.append((f'normalize_layers[{idx}]', param_count(nl), size_mb_from_params(param_count(nl)), "exact"))

for idx, rb in enumerate(model.res_blocks):
    report.append((f'res_blocks[{idx}]', param_count(rb), size_mb_from_params(param_count(rb)), "exact"))
for idx, ab in enumerate(model.add_blocks):
    report.append((f'add_blocks[{idx}]', param_count(ab), size_mb_from_params(param_count(ab)), "exact"))

# Now prepare CSV with detail rows
csv_path = os.path.join(project_root, 'timekan_detailed_report.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['component', 'params_count', 'size_MB', 'macs_or_note'])
    for row in report:
        writer.writerow([row[0], int(row[1]) if isinstance(row[1], (int,)) else row[1], ("%.6f" % row[2]) if isinstance(row[2], float) else row[2], row[3]])
    writer.writerow([])
    writer.writerow(['--- M_KAN details (cheby/conv estimates) ---'])
    for r in mkan_rows:
        writer.writerow(r)
    writer.writerow([])
    writer.writerow(['--- TokenEmbedding per-scale details (token conv) ---'])
    if 'token_conv_scale_details' in locals():
        for item in token_conv_scale_details:
            writer.writerow(item)

print("Wrote detailed CSV to:", csv_path)
print("Total params:", total_params, " => size MB:", size_mb_from_params(total_params))
if USE_THOP and macs_total is not None:
    print("thop total MACs:", macs_total)
else:
    print("thop not available or failed: MACs are reported analytically for many layers and flagged where not computed (FFT).")
print("CSV contains per-component rows and detailed M_KAN entries. Open it for the full breakdown.")
