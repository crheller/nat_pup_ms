from sklearn.decomposition import PCA, IncrementalPCA
from nems_lbhb.baphy_experiment import BAPHYExperiment
import av
import numpy as np
import nems_lbhb.pup_py.utils as ut
from nems_lbhb.baphy_experiment import BAPHYExperiment
import h5py

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
animal = str(parmfiles[0]).split("/")[4]
pen = str(parmfiles[0]).split("/")[5]

outputfile =  f"/auto/data/daq/{animal}/{pen}/sorted/facePCs.h5"

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
data = np.zeros((224*224, nframes), dtype=np.float32)
idx = 0
per_vid_ends = []
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
    per_vid_ends.append(idx) # keep track of per parmfile data

per_vid_starts = [0] + per_vid_ends

log.info("Perform incremental PCA")
pca = IncrementalPCA(batch_size=200)
pca.fit(data.T)

log.info("Save results")
# save top 10 components, save variance explained, save top 10 projections
projection = data.T.dot(pca.components_[0:10, :].T)
components = pca.components_[0:10, :]
var_explained = pca.explained_variance_ratio_

def save_hdf5(d, filename):
    fh = h5py.File(filename, 'w')
    for k in d.keys():
        fh.create_dataset(k, data=d[k])
    fh.close()
    return None

for i, p in enumerate(parmfiles):
    thisfile = os.path.basename(p).strip(".m")
    results = {
        "projection": projection[per_vid_starts[i]:per_vid_ends[i], :],
        "components": components,
        "var_explained": var_explained
    }
    savefile = outputfile.replace("facePCs.h5", f"{thisfile}.facePCs.h5")
    save_hdf5(results, savefile)

if queueid:
    nd.update_job_complete(queueid)
