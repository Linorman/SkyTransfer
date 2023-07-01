import argparse
from pathlib import Path

from PIL import Image

from skyboxengine import *
import utils
import json
import torch
import datetime
import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='SkyTransfer')
parser.add_argument('--path', type=str, default='./config/config-my_photo-seaSunSet.json', metavar='str',
                    help='configurations')
parser.add_argument('--on-server', type=bool, default=False, metavar='bool',
                    help='path to checkpoints')


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

    def run(self, timeNow):
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
                tempPath += timeNow
                os.mkdir(tempPath)
                fpath = os.path.join(tempPath, img_names[idx])
                # plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                # plt.imsave(fpath[:-4] + 'coarse_skymask.jpg', G_pred)
                # plt.imsave(fpath[:-4] + 'refined_skymask.jpg', skymask)
                plt.imsave(fpath[:-4] + '_syneth.jpg', syneth.clip(min=0, max=1))

            print('processing: %d / %d ...' % (idx, len(img_names)))

            img_HD_prev = img_HD

    def run_server(self, timeNow):
        print('running evaluation...')
        img_names = os.listdir(self.datadir)
        img_HD_prev = None
        results = []

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
                tempPath += timeNow
                os.mkdir(tempPath)
                fpath = os.path.join(tempPath, img_names[idx])
                # plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                # plt.imsave(fpath[:-4] + 'coarse_skymask.jpg', G_pred)
                # plt.imsave(fpath[:-4] + 'refined_skymask.jpg', skymask)
                plt.imsave(fpath[:-4] + '_syneth.jpg', syneth.clip(min=0, max=1))

                # 返回文件路径
                results.append(fpath[:-4] + '_syneth.jpg')

            print('processing: %d / %d ...' % (idx, len(img_names)))

            img_HD_prev = img_HD

        return results


if __name__ == '__main__':
    if parser.parse_args().on_server is False:
        config_path = parser.parse_args().path
        args = utils.parse_config(config_path)
        sf = SkyFilter(args)
        sf.run(str(datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")))
    else:
        app = FastAPI()

        @app.post("/api/sky-transfer/")
        async def create_upload_file(file: UploadFile = File(...), maskId: int = Form(...)):
            # 搜索文件夹下与maskId匹配的文件
            mask_files = list(Path("./skybox").glob(f"{maskId}.jpg"))

            if len(mask_files) > 0:
                mask_name = mask_files[0].name
            else:
                return {
                    "code": "400",
                    "message": f"No mask image found for maskId={maskId}"
                }

            # 检查上传的文件后缀名，如果不是jpg则转换成jpg格式
            if file.filename.endswith(".jpeg") or file.filename.endswith(".png"):
                image = Image.open(file.file)
                converted_image = image.convert("RGB")
                new_filename = file.filename[:file.filename.rfind(".")] + ".jpg"
                new_file_path = f"./imageinput/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/{new_filename}"
                new_file_dir = Path(new_file_path).parent
                new_file_dir.mkdir(parents=True, exist_ok=True)
                converted_image.save(new_file_path)
            else:
                new_filename = file.filename
                new_file_path = f"./imageinput/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/{new_filename}"
                new_file_dir = Path(new_file_path).parent
                new_file_dir.mkdir(parents=True, exist_ok=True)
                file.file.seek(0)
                contents = await file.read()
                with open(new_file_path, "wb") as f:
                    f.write(contents)

            # 生成配置字典
            config = {
                "net_G": "coord_resnet50",
                "ckptdir": "./checkpoints_G_coord_resnet50",
                "datadir": new_file_dir,
                "skybox": mask_name,
                "in_size_w": 384,
                "in_size_h": 384,
                "out_size_w": 845,
                "out_size_h": 480,
                "skybox_center_crop": 0.5,
                "auto_light_matching": True,
                "relighting_factor": 0.8,
                "recoloring_factor": 0.5,
                "halo_effect": True,
                "output_dir": "./output",
                "save_jpgs": True
            }

            # 生成配置文件名
            file_name = f"config-{datetime.datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}.json"

            # 保存配置文件
            config_dir = Path("./config")
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_dir / file_name, 'w') as f:
                json.dump(config, f)

            # 生成配置文件路径
            params = utils.parse_config(config_dir / file_name)
            server = SkyFilter(params)
            path = server.run_server(str(datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")))

            return FileResponse(path)
