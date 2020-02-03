from SinGAN.training import train
from SinGAN import functions
from SinGAN.manipulate import SinGAN_generate
from SinGAN.config import get_arguments
import torch

def singan_augment(img_tensor_CHW, n_aug, opt=None):
    """
    Given a pytorch image tensor (scaled 0-1) on CUDA device return a list of n_aug augmented versions of it.
    Augmentation is carried out by SinGAN accordings to opt.
    """
    if opt is None:  # Use default options
        parser = get_arguments()
        opt = parser.parse_args([])
        opt.mode = 'train'
        opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
        opt.niter_init = opt.niter
        opt.noise_amp_init = opt.noise_amp
        opt.nfc_init = opt.nfc
        opt.min_nfc_init = opt.min_nfc
        opt.scale_factor_init = opt.scale_factor
        opt = get_default_opts()

    Gs, Zs, reals, NoiseAmp = train(img_tensor_CHW, opt)
    def gen_aug_image():
        img = SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
        return functions.move_to_cpu(functions.denorm(img)[0]).clamp(0, 1)
    images_aug = [gen_aug_image() for _ in range(n_aug)]
    return images_aug

def toimage(tensor):
    """ Convert CUDA tensor CHW to PIL """
    a = tensor.cpu().detach().numpy()[0]
    a -= a.min()
    a /= a.max()
    return Image.fromarray(np.moveaxis(np.uint8(a * 255.), 0, -1))

def totensor(img):
    """ Convert PIL to CUDA tensor CHW """
    img = torch.tensor(array(img).transpose((2, 0, 1))/255)
    img = functions.move_to_gpu(img)
    img = img.type(torch.cuda.FloatTensor)
    img = img[0:3,:,:]
    return img