import os
import numpy as np
import tensorflow as tf
import shutil

def get_tensor_at(tensor, msk, i):
    return tf.ragged.boolean_mask(tensor, msk[i])


def list_var_name(list_of_tensors):
    """Gets list of tensor's name
    """
    return [var.name for var in list_of_tensors]


def get_var(list_of_tensors, prefix_name=None):
    """Returns the the list of name, tensors with prefix name
    """
    if prefix_name is None:
        return list_var_name(list_of_tensors), list_of_tensors
    else:
        specific_tensor = []
        specific_tensor_name = []
        for var in list_of_tensors:
            if var.name.startswith(prefix_name):
                specific_tensor.append(var)
                specific_tensor_name.append(var.name)
        return specific_tensor_name, specific_tensor

# class Cutout(object):
#     def __init__(self, length):
#         self.length = length

#     def __call__(self, img):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)

#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)

#         mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         return img


# def count_parameters_in_MB(model):
#     return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


# def save_checkpoint(state, is_best, save):
#     filename = os.path.join(save, 'checkpoint.pth.tar')
#     torch.save(state, filename)
#     if is_best:
#         best_filename = os.path.join(save, 'model_best.pth.tar')
#         shutil.copyfile(filename, best_filename)


# def save(model, model_path):
#     torch.save(model, model_path)

# def load(model, model_path):
#     torch.load(model_path)


# def drop_path(x, drop_prob):
#     if drop_prob > 0.:
#         keep_prob = 1.-drop_prob
#         mask = Variable(torch.cuda.FloatTensor(
#             x.size(0), 1, 1, 1).bernoulli_(keep_prob))
#         x.div_(keep_prob)
#         x.mul_(mask)
#     return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def write_genotype(genotype, file_name="final_genotype"):
    cwd = os.getcwd()
    path = os.path.join('./final_model')
    
    if(not os.path.exists(path)):
        os.mkdir(path)
    
    comment = '# This file is auto-generated. This file contains the model searched through train_search'
    
    with open(os.path.join(path, '{}.py'.format(file_name)), 'w+') as f:
        f.write("{}\n\nfrom collections import namedtuple\n\nGenotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')\n\ngenotype={}".format(comment, genotype))