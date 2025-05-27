import numpy as np
from scipy.spatial.transform import Rotation as R
import mdtraj
import argparse
import os,tqdm
from multiprocessing import Pool,cpu_count
import time

def calculate_aligned_rmsd(A_centered, B_centered):
    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    A_rotated = A_centered @ R.T
    rmsd = np.sqrt(np.mean(np.linalg.norm(A_rotated - B_centered, axis=1)**2))

    return rmsd


def farthest_point_sampling(proteins, M):
    F, N, _ = proteins.shape
    selected_indices = [np.random.randint(F)]
    selected_proteins = [proteins[selected_indices[-1]]]

    for _ in range(M - 1):
        max_dist = -np.inf
        farthest_idx = None
        for i in range(F):
            if i in selected_indices:
                continue
            min_dist = np.inf
            for sp in selected_proteins:
                dist = calculate_aligned_rmsd(proteins[i], sp)
                if dist < min_dist:
                    min_dist = dist
            if min_dist > max_dist:
                max_dist = min_dist
                farthest_idx = i
        selected_indices.append(farthest_idx)
        selected_proteins.append(proteins[farthest_idx])

    return np.array(selected_indices).reshape(-1, 1)

def load_traj(pdb_name,atlas_dir,traj_id=1):
    # load traj
    print(f'{atlas_dir}/{pdb_name}/{pdb_name}_prod_R{traj_id}_fit.xtc')
    traj = mdtraj.load(f'{atlas_dir}/{pdb_name}/{pdb_name}_prod_R{traj_id}_fit.xtc', top=f'{atlas_dir}/{pdb_name}/{pdb_name}.pdb')
    # get ca index
    ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
    # get ca_position
    ca_positions = traj.atom_slice(ca_indices).xyz
    # convert nm to angs
    return ca_positions*10


def do_job(pdb_name):
    path = f"{args.outdir}/{pdb_name}.npz.npy"
    start_time = time.time()
    matrix_ori = np.load(path)
    matrix = matrix_ori.astype(np.float32)
    np.save(path.replace('ca_rmsd_matrix6', 'ca_rmsd_matrix6_compress'), matrix)
    end_time = time.time()

    print("seconds:", end_time - start_time, '=====', np.sum(np.abs(matrix-matrix_ori)))
    return matrix

# multi-processors
def main():
    jobs = []

    for pdb_name in os.listdir(args.atlas_dir):
        jobs.append(pdb_name)

    jobs.sort()

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='test_data.csv')
parser.add_argument('--atlas_dir', type=str, default='data/atlas')
parser.add_argument('--outdir', type=str, default='data/dist_matrix')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument("--rank_idx",type=int, help="Batch index to process, you may pass machine rank as it like $RANK")
parser.add_argument("--world_size", type=int, help="num of batches to create like $WORLD_SIZE", required=False)

args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

if __name__ == '__main__':
    main()
