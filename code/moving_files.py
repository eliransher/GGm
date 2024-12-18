import shutil
import os


path  = '/scratch/eliransc/n_servers_single10'

files = os.listdir(path)
batch_size = 10000

num_folds = int(len(files)/batch_size)

dst_path = '/scratch/eliransc/new_n_servers'

for fold in range(num_folds):

    for within_batch in range(batch_size):
        ind = batch_size * fold + within_batch
        src = os.path.join(path, files[ind])

        curr_dst = os.path.join(dst_path, 'n_servers_' + str(fold))
        if not os.path.exists(curr_dst):
            os.mkdir(curr_dst)


        dst = os.path.join(curr_dst, files[ind])
        shutil.move(src, dst)
