import argparse
from skyboxengine import *
import utils
import torch
import time

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='SkyTransfer')
parser.add_argument('--path', type=str, default='./config/config-my_photo-seaSunSet.json', metavar='str',
                    help='configurations')


class SkyFilter:

    def __init__(self, params):
        self.ckptdir = params.ckptdir
        self.datadir = params.datadir
        self.in_size_w, self.in_size_h = params.in_size_w, params.in_size_h
        self.out_size_w, self.out_size_h = params.out_size_w, params.out_size_h
        self.skyboxengine = SkyBox(params)

        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG=params.net_G).to(device)
        self.load_model()

        if params.save_jpgs and os.path.exists(params.output_dir) is False:
            os.mkdir(params.output_dir)

        self.save_jpgs = params.save_jpgs

    def load_model(self):
        print('loading the best checkpoint...')
        checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'),
                                map_location=None if torch.cuda.is_available() else device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()

    def synthesize(self, img_HD, img_HD_prev):

        h, w, c = img_HD.shape

        img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))

        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            G_pred = self.net_G(img.to(device))
            G_pred = torch.nn.functional.interpolate(G_pred, (h, w), mode='bicubic', align_corners=False)
            G_pred = G_pred[0, :].permute([1, 2, 0])
            G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)
            G_pred = np.array(G_pred.detach().cpu())
            G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

        skymask = self.skyboxengine.skymask_refinement(G_pred, img_HD)

        syneth = self.skyboxengine.skyblend(img_HD, img_HD_prev, skymask)

        return syneth, G_pred, skymask

    def cvtcolor_and_resize(self, img_HD):

        img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
        img_HD = np.array(img_HD / 255., dtype=np.float32)
        img_HD = cv2.resize(img_HD, (self.out_size_w, self.out_size_h))

        return img_HD

    def run_imgseq(self):

        print('running evaluation...')
        img_names = os.listdir(self.datadir)
        img_HD_prev = None

        for idx in range(len(img_names)):

            this_dir = os.path.join(self.datadir, img_names[idx])
            img_HD = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_HD = self.cvtcolor_and_resize(img_HD)

            if img_HD_prev is None:
                img_HD_prev = img_HD

            syneth, G_pred, skymask = self.synthesize(img_HD, img_HD_prev)

            if self.save_jpgs:
                tempPath = args.output_dir
                tempPath += "/"
                tempPath += img_names[idx][:-4]
                tempPath += "_out_"
                tempPath += str(time.time())
                os.mkdir(tempPath)
                fpath = os.path.join(tempPath, img_names[idx])
                # plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                # plt.imsave(fpath[:-4] + 'coarse_skymask.jpg', G_pred)
                # plt.imsave(fpath[:-4] + 'refined_skymask.jpg', skymask)
                plt.imsave(fpath[:-4] + '_syneth.jpg', syneth.clip(min=0, max=1))

            print('processing: %d / %d ...' % (idx, len(img_names)))

            img_HD_prev = img_HD

    def run(self):
        self.run_imgseq()


if __name__ == '__main__':
    config_path = parser.parse_args().path
    args = utils.parse_config(config_path)
    sf = SkyFilter(args)
    sf.run()
