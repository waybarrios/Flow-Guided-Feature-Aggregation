import argparse
import os

from joblib import Parallel, delayed
import pandas as pd


def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

def dump_frames(filename, output_folder, basename_format=None):
    """Dump frames of a video-file into a folder
    Parameters
    ----------
    filename : string
        Fullpath of video-file
    output_folder : string
        Fullpath of folder to place frames
    basename_format: (None, string)
        String format used to save video frames. If None, the
        format is assigned according the length of the video
    Outputs
    -------
    success : bool
    Note: this function makes use of ffmpeg and its results depends on it.
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    fbasename = '%06d.jpg'

    output_format = os.path.join(output_folder, fbasename)
    cmd = ['ffmpeg', '-v', 'error', '-i', filename, '-qscale:v', '2', 
           '-threads', '1', '-f', 'image2', output_format]
    print ' '.join(cmd)
    ret = os.system(' '.join(cmd))
    if ret:
        return False
    return True


def dump_wrapper(filename, output_folder, baseformat, fullpath, video_path):
    filename_noext = os.path.splitext(filename)[0]
    if fullpath:
        filename_noext = os.path.basename(filename_noext)
    if video_path:
        filename = os.path.join(video_path, filename)
    output_dir = os.path.join(output_folder, filename_noext)
    return dump_frames(filename, output_dir, baseformat)


def input_parse():
    description = ('Extract frames of a bucnh of videos. The file-system '
                   'organization is preserved if relative-path are used.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_file',
                   help='CSV file with list of videos to process')
    p.add_argument('output_folder', help='Folder to allocate frames')
    p.add_argument('-bf', '--baseformat', default=None,
                   help='Format used for naming frames e.g. %%06d.jpg')
    p.add_argument('-n', '--n_jobs', default=1, type=int,
                   help='Number of CPUs')
    p.add_argument('-ff', '--fullpath', action='store_true',
                   help='Input-file uses fullpath instead of rel-path')
    p.add_argument('-vpath', '--video_path', default=None,
                   help='Path where the videos are located.')
    p.add_argument('--jobid', default=None, type=int,
                   help='Slurm job identifier.')
    p.add_argument('--nr_jobs_slurm', default=None, type=int,
                   help='Number of jobs slurm.')
    args = p.parse_args()
    return args

def main(input_file, output_folder, baseformat, n_jobs, fullpath, 
         video_path, jobid, nr_jobs_slurm):
    video_lst = pd.read_csv(input_file,
                            sep=' ', header=None).values.flatten().tolist()
    if jobid is not None:
        if not nr_jobs_slurm:
            raise IOError('Number of slurm jobs is required.')
        video_lst = partition(video_lst, nr_jobs_slurm)[jobid]
    Parallel(n_jobs=n_jobs)(delayed(dump_wrapper)(i, output_folder,
                                                  baseformat, fullpath,
                                                  video_path)
                            for i in video_lst)
    return None


if __name__ == '__main__':
    args = input_parse()
    main(**vars(args))
