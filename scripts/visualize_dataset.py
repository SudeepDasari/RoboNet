import h5py
import cv2


def summarize_file(filename):
    with h5py.File(filename, 'r') as f:
        meta_data, env, policy, misc = [f.get(x) for x in ['metadata', 'env', 'policy', 'misc']]
        
        n_cams = env.attrs.get('n_cams', 0)
        cam_videos = []
        for n in range(n_cams):
            cam_group = env.get('cam{}_video'.format(n))
            T = len(cam_group)
            
            im_buf = []
            for t in range(T):
                frame = cam_group.get('frame{}'.format(t))[:]
                im_buf.append(cv2.imdecode(frame, cv2.IMREAD_COLOR)[:, :, ::-1].copy())
            cam_videos.append(im_buf)
        
        attributes = [meta_data.attrs[x] for x in ['background', 'policy_desc', 'action_space']]
        return attributes + [cam_videos]


def _format_img_row(title, path_list, height=128):
    row_template = "  <tr>\n    <td> <b> {} </b> </td> \n".format(title)
    for path in path_list:
        row_template += "    <td> <img src=\"{}\" height=\"{}\"> </td>\n".format(path, height)
    row_template += "  </tr>\n"
    return row_template


def _format_dataset_segment(dataset_name, policy_description, background, action_space, row_string):
    base_template = """
    <h2>Dataset: {0}</h2>
    <p> <b> Policy Description: </b> {1} </p>
    <p> <b> Background: </b> {2} </p>
    <p> <b> Action Space: </b> {3} </p>

    <table>
    {4}
    </table>
    """

    return base_template.format(dataset_name, policy_description, background, action_space, row_string)  


def _proc_img(frame, factor):
    if factor == 1:
        return frame
    elif factor < 1:
        return cv2.resize(frame,None,fx=factor,fy=factor, interpolation=cv2.INTER_AREA)
    return cv2.resize(frame,None,fx=factor,fy=factor, interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    doc_template = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    table, th, td {{
    border: 1px solid black;
    border-collapse: collapse;
    }}
    th, td {{
    padding: 5px;
    }}

    td {{
    text-align: middle;
    }}
    </style>
    </head>
    <body>

    {0}

    </body>
    </html>
    """

    import argparse
    import imageio as io
    import glob
    import os
    import random

    parser = argparse.ArgumentParser(description="creates a summary html of given datasets")
    parser.add_argument('input_folder', type=str, help='folder of datasets')
    parser.add_argument('save_dir', type=str, help='where to save html summary')
    parser.add_argument('--fps', type=int, default=4, help="fps for mp4 saves")
    parser.add_argument('--save_factor', type=float, default=0.5, help="factor to resize the image before saving gif")
    parser.add_argument('--disp_height', type=int, default=128, help="height for html display")
    parser.add_argument('--samps_per_set', type=int, default=5, help="maximum number of examples from dataset")
    parser.add_argument('--no_gif_save', action='store_true', help="will not save gifs if flag is present (but html will save)")
    args = parser.parse_args()

    assert os.path.exists(args.input_folder) and os.path.isdir(args.input_folder), "Please pass path to datasets!"
    if not os.path.exists(args.save_dir + "/assets"):
        os.makedirs(args.save_dir + "/assets")
    
    datasets = [x for x in glob.glob(args.input_folder + '/*') if os.path.isdir(x)]
    if len(datasets) == 0:
        raise ValueError("No datasets found!")
    
    s_ctr = 0
    datasets_string = ""
    for d in datasets:
        dataset_name = d.split('/')[-1]
        
        samples = glob.glob(d+"/*.hdf5")
        random.shuffle(samples)
        samples = samples[:args.samps_per_set]
        
        all_rows = ""
        background, policy_desc, action_space = None, None, None

        for i, s in enumerate(samples):
            background, policy_desc, action_space, cam_videos = summarize_file(s)
            cam_paths = []
            for n, v in enumerate(cam_videos):
                path = 'assets/samp{}_cam_{}.gif'.format(s_ctr, n)
                cam_paths.append(path)
                
                if not args.no_gif_save:
                    writer = io.get_writer('{}/{}'.format(args.save_dir, path), fps=args.fps)
                    [writer.append_data(_proc_img(f, args.save_factor)) for f in v]
                    writer.close()
            
            if len(cam_paths):
                all_rows += _format_img_row("samp_{}".format(i), cam_paths, height=128) + "\n"
            s_ctr += 1
        
        datasets_string += _format_dataset_segment(dataset_name, policy_desc, background, action_space, all_rows)

    final_html = doc_template.format(datasets_string)
    with open('{}/index.html'.format(args.save_dir), 'w') as f:
        f.write(final_html)
        f.write('\n')
