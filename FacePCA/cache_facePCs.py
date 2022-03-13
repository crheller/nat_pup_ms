from sklearn.decomposition import PCA, IncrementalPCA
from nems_lbhb.baphy_experiment import BAPHYExperiment
import av
import numpy as np
import nems_lbhb.pup_py.utils as ut
from nems_lbhb.baphy_experiment import BAPHYExperiment
import pickle 

import os
import sys
import nems.db as nd
import logging 
import nems

log = logging.getLogger(__name__)

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick

else:
    queueid = 0

if queueid:
    log.info("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)


# Get sys args
site = sys.argv[1]  
batch = sys.argv[2]
modelname = sys.argv[3]

manager = BAPHYExperiment(cellid=site, batch=int(batch))
parmfiles = manager.parmfile

videos = [str(p).replace(".m", ".avi") for p in parmfiles]
nframes = 0
suffix = []
for video in videos:
    if os.path.isfile(video):
        container = av.open(video)
        video_stream = [s for s in container.streams][0]
        suffix.append(None)
    elif os.path.isfile(video.replace(".avi", ".mj2")):
        video = video.replace(".avi", ".mj2")
        container = av.open(video)
        video_stream = [s for s in container.streams][0]
        suffix.append(".mj2")
    elif os.path.isfile(video.replace(".avi", ".mj2.avi")):
        video = video.replace(".avi", ".mj2.avi")
        container = av.open(video)  
        video_stream = [s for s in container.streams][0]
        suffix.append(".mj2.avi")
    else:
        raise ValueError(f"cound't find pupil files for {video}")
    for packet in container.demux(video_stream):
        nframes+=1

log.info("Decoding video")
data = np.zeros((224*224, nframes))
idx = 0
for i, video in enumerate(videos):
    if suffix[i] is not None:
        video = video.replace(".avi", suffix[i])
    log.info(video)
    container = av.open(video)
    video_stream = [s for s in container.streams][0]
    for i, packet in enumerate(container.demux(video_stream)):
        try:
            frame = packet.decode()[0]

            frame_ = np.asarray(frame.to_image().convert('LA'))
            frame_ = frame_[:, :-10, :]
            frame_ = frame_ - np.min(frame_)

            size = (224, 224)
            sf, im = ut.resize(frame_, size=size)

            data[:, idx] = im[:, :, 0].flatten()
            if i % 1000 == 0:
                log.info("frame: {0}/{1}...".format(i, nframes))
        except:
            log.info(f"couldn't decode frame {i}")
        
        idx+=1

log.info("Perform incremental PCA")
pca = IncrementalPCA(batch_size=200)
pca.fit(data.T)

log.info("Save results")
# save top 10 components, save variance explained, save top 10 projections
projection = data.T.dot(pca.components_[0:10, :].T)
components = pca.components_[0:10, :]
var_explained = pca.explained_variance_ratio_
results = {
    "proejction": projection,
    "components": components,
    "var_explained": var_explained
}


def save(d, path):
    with open(path+f'/{modelname}.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

path = "/auto/users/hellerc/results/nat_pupil_ms/face_pca/"
if os.path.isdir(os.path.join(path, str(batch), site)):
   pass
elif os.path.isdir(os.path.join(path, str(batch))):
    os.mkdir(os.path.join(path, str(batch), site))
else:
    os.mkdir(os.path.join(path, str(batch)))
    os.mkdir(os.path.join(path, str(batch), site))

save(results, os.path.join(path, str(batch), site))

if queueid:
    nd.update_job_complete(queueid)
