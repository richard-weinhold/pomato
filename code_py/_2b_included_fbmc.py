import numpy as np
import pandas as pd
from pathlib import Path
import imageio
import datetime as dt


wdir = Path('C:\\Users\\riw\\tubCloud\\Uni\\Market_Tool\\pomato')

timesteps = ['t'+ "{0:0>4}".format(x) for x in range(1, 25)]

start_date = dt.datetime(2018, 5, 18, 0)

timesteps_datetime = [start_date + dt.timedelta(hours=x) for x in range(0, 24)]

#timesteps_selection = ['t'+ "{0:0>4}".format(int((x - start_date).total_seconds()/3600 + 1))
#                       for x in timesteps_datetime if x.hour==19]

timesteps_selection = timesteps_datetime

folder = wdir.joinpath("domains_basecase")
save_folder = wdir.joinpath("domains_basecase")
if not folder.is_dir():
    folder.mkdir()
#gsk_strategy = "flat"

t = timesteps_datetime[0]
t1 = t - start_date

#with imageio.get_writer(str(wdir.joinpath(f"FBMC_quad.gif")), mode='I', duration=1) as writer:
##with imageio.get_writer(str(wdir.joinpath(f"FBMC_combined.mp4")), mode='I', fps=1) as writer:
#    for t in timesteps:
##        filenames = [f"FBMC_{t}_{gsk_strategy}.png" for gsk_strategy in ["flat", "G", "g_max", "g_max_G_flat", "jao"]]
#        filenames = [f"FBMC_{t}_{gsk_strategy}.png" for gsk_strategy in ["flat", "G", "g_max", "g_max_G_flat"]] # ["flat", "G", "g_max", "g_max_G_flat", "jao"]]
#        imgs = [imageio.imread(str(folder.joinpath(filename))) for filename in filenames]
#        min_shape = sorted( [(np.sum(i.shape), i.shape ) for i in imgs])[0][1]
##        imgs_comb_domain = np.hstack(np.asarray(i) for i in imgs)
##        imgs_comb_domain = np.hstack((np.resize(i, min_shape)) for i in imgs)
##        img_net_position = imageio.imread(str(save_folder.joinpath(f"net_position/NP_{t}.png")))
#
##        imgs_comb = np.vstack((imgs_comb_domain, img_net_position))
##        imgs_comb = np.vstack((np.hstack((np.asarray(imgs[0].resize(min_shape)), np.asarray(imgs[1].resize(min_shape)))),
##                               np.hstack((np.asarray(imgs[2].resize(min_shape)), np.asarray(imgs[3].resize(min_shape))))))
#        imgs_comb = np.vstack((np.hstack((imgs[0], imgs[1])), np.hstack((imgs[2], imgs[3]))))
#        ### save sinle images
#        writer.append_data(imgs_comb)

with imageio.get_writer(str(wdir.joinpath(f"FBMC_domain.gif")), mode='I', duration=1) as writer:
#with imageio.get_writer(str(wdir.joinpath(f"FBMC_combined.mp4")), mode='I', fps=1) as writer:
    for t in timesteps:
#        filenames = [f"FBMC_{t}_{gsk_strategy}.png" for gsk_strategy in ["flat", "G", "g_max", "g_max_G_flat", "jao"]]
        filename = f"FBMC_{t}_gmax.png" 
        img = imageio.imread(str(folder.joinpath(filename)))
        ### save sinle images
        writer.append_data(img)
