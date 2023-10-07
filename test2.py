from graphviz import Digraph
import torch
from torch.autograd import Variable, Function
from model_3d import UNet3D
import nibabel as nib

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
                return True
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

import os
if __name__ == '__main__':

    mri = nib.load('t1_ai_msles2_1mm_pn3_rf20.mnc');
    d = mri.get_fdata();
    nib.save(mri, 'b1.nii.gz');


    os.environ["PATH"] += os.pathsep + 'C:\\Graphviz\\bin'
    sample = torch.rand((1,4,64,64,64)).to('cuda');

    net = UNet3D(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        ).to('cuda')
    
    total_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad);
    print(f'total parameters: {total_parameters}')
    

    out = net(sample, sample);
    get_dot = register_hooks(out)
    out = torch.mean(out);
    out.retain_grad();
    out.backward()
    dot = get_dot()
    dot.render('temp.dot', view=True);