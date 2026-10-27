import os

import sys
from PIL import Image
import glob
import argparse
import pickle
from torchvision import transforms
from opt import PoisonGeneration


def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


def main():
    poison_generator = PoisonGeneration(
        target_concept=args.target_name,
        device="cuda",
        eps=args.eps,
        constraint=args.constraint,
        lpips_lambda=args.lpips_lambda,
        lpips_net=args.lpips_net,
        steps=args.steps,
    )
    # all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    # all_data_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) # sort
    # all_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_data_paths]
    # all_imgs = [pickle.load(open(f, "rb"))['img'] for f in all_data_paths]
    # all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    # all_imgs = [Image.fromarray(img) for img in all_imgs]

    all_data_paths = sorted(glob.glob(os.path.join(args.directory, "*.png")))
    all_imgs = [Image.open(f).convert('RGB') for f in all_data_paths]

    all_result_imgs = poison_generator.generate_all(all_imgs, args.target_name)
    os.makedirs(args.outdir, exist_ok=True)

    # for idx, cur_img in enumerate(all_result_imgs):
    #     cur_data = {"text": all_texts[idx], "img": cur_img}
    #     original_id = all_ids[idx]  # 원본 번호 가져오기
    #     pickle.dump(cur_data, open(os.path.join(args.outdir, "{}.p".format(original_id)), "wb"))

    for idx, cur_img in enumerate(all_result_imgs):
        save_path = os.path.join(args.outdir, f"poisoned_{idx:04d}.png")
        cur_img.save(save_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str,
                        help="", default='')
    parser.add_argument('-od', '--outdir', type=str,
                        help="", default='')
    parser.add_argument('-e', '--eps', type=float, default=0.04)
    parser.add_argument('--constraint', type=str, default='linf', choices=['linf', 'lpips'],
                        help="Perturbation budget type. In lpips mode, --eps is treated as the LPIPS budget.")
    parser.add_argument('--lpips_lambda', type=float, default=10.0,
                        help="Penalty strength used only when --constraint lpips.")
    parser.add_argument('--lpips_net', type=str, default='vgg', choices=['alex', 'vgg', 'squeeze'],
                        help="LPIPS backbone used only when --constraint lpips.")
    parser.add_argument('--steps', type=int, default=500,
                        help="Optimization steps for both linf and lpips modes.")
    parser.add_argument('-t', '--target_name', type=str, default="cat")
    return parser.parse_args(argv)


if __name__ == '__main__':
    import time

    args = parse_arguments(sys.argv[1:])
    main()
