import imageio
import cv2
import glob

files = glob.glob('frs*.gif')
DIMS = (640, 480)

for f in files:
    name = f.split('.')[0]
    reader = imageio.get_reader(f)
    imgs = [cv2.resize(im, DIMS, interpolation=cv2.INTER_CUBIC) for im in reader]
    reader.close()
    writer = imageio.get_writer(name + '_large.gif')
    [writer.append_data(img) for img in imgs]
    writer.close()
