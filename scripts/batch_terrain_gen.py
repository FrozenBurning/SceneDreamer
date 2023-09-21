import os
import time
import subprocess
from tqdm import tqdm
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--nbins', type=int, default=256)
parser.add_argument('--seed', type=int, default=762345)
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()


def par_job(command):
    if args.dry_run:
        print(command)
    else:
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    t0 = time.time()

    cmd_list = []
    for idx in tqdm(range(args.bs)):
        current_dir = os.path.join(args.outdir, 'world_{:04d}'.format(idx))
        cmd = 'python scripts/single_terrain_gen.py --size {} --seed {} --outdir {}'.format(args.size, args.seed + idx, current_dir)
        if not args.parallel:
            if args.dry_run:
                print(cmd)
            else:
                subprocess.call(cmd, shell=True)
        cmd_list.append(cmd)

    if args.parallel:
        with Pool(processes=args.num_workers) as pool:
            with tqdm(total=len(cmd_list)) as pbar:
                for _ in tqdm(pool.imap_unordered(par_job, cmd_list)):
                    pbar.update()
    t1 = time.time()
    print("Finished in %.4f seconds" % (t1 - t0))
    os.system("stty sane")