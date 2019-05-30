import io
import imageio
from flask import Flask, render_template, url_for, redirect, abort, send_file
import argparse
app = Flask(__name__)


args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Web based utility to visualize RoboNet trajectoriers (in hdf5 format). I don't even pretend like this is secure. Don't serve on a public website!")
    parser.add_argument('input_dir', type=str, help='path to stored hdf5 files')
    args = parser.parse_args()
    f = open('scripts/test.mp4', 'rb')
    vid = f.read()
    f.close()


@app.route('/')
def index():
    filter_names = ['filt0', 'filt1']

    traj0 = {'url': '/traj/0', 'text': 'here lies traj 0'}
    traj10 = {'url': '/traj/10', 'text': 'here lies traj 10'}
    filters = [[traj0], [traj0, traj10]]

    return render_template('index_template.html', filter_names=filter_names, filters=filters)


@app.route('/traj/<int:traj_id>')
def traj_page(traj_id):
    if traj_id != 0 and traj_id != 10:      # page not found if traj id not valid
        abort(404)
    
    attr_list = [{'name': 'robot', 'value': 'sawyer'}, {'name': 'te', 'value': 'st'}]

    vid_url = '/traj/{}/cam{}.mp4'.format(traj_id, 0)
    name_list = ['cam0', 'cam1', 'cam2']
    video_list = [{'url': vid_url, 'type':'video/mp4'}, {'url': vid_url, 'type':'video/mp4'}, {'url': vid_url, 'type':'video/mp4'}]
    return render_template('traj_template.html', traj_name=str(traj_id), videos=video_list, video_names=name_list, attributes=attr_list)


@app.route('/traj/<int:traj_id>/cam<int:cam_id>.mp4')
def get_mp4(traj_id, cam_id):
    if traj_id != 0 and traj_id != 10:      # page not found if traj id not valid
        abort(404)
    
    if not 0 <= cam_id < 5:                         # page not found if camera id is invalid
        abort(404)

    return send_file(
            io.BytesIO(vid),
            mimetype='video/mp4',
            as_attachment=True,
            attachment_filename='cam{}.mp4'.format(cam_id))


@app.after_request
def add_header(r):
    """
    Source: https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    # disable caching trick 2 (same source as above)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run()
