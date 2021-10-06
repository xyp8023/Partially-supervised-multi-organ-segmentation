# how to load the dataset

## to use it in rpl cluster
first install jupyter lab:
`conda create --name wasp-p3 python=3.7.5`

`conda activate wasp-p3`

`conda install -c conda-forge jupyterlab`

then allocate a node and start the jupyter lab

`salloc --constrain=<node_name>`

`conda activate wasp-p3`

`jupyter lab --no-browser --port=90001 --ip <node_name>.csc.kth.se`


## load coco format json

### install `kwcoc`

https://kwcoco.readthedocs.io/en/release/autoapi/kwcoco/index.html

`conda activate wasp-p3`

`pip install kwcoco`

### usage

`
import kwcoco
import json
from kwcoco import CocoDataset


dataset = CocoDataset("1715_all_annotations.json")
cid=1
print(dataset.cats[cid])
# {'supercategory': '', 'id': 1, 'name': 'thistle'}

aid = 1
print(dataset.anns[aid])

"""
{'segmentation': [[3010.77294921875,
   1111.3174285888672,
   3031.9967041015625,
   1108.6644592285156,
   3029.343719482422,
   1099.3790588378906,
   3014.089141845703,
   1100.0423126220703,
   3010.77294921875,
   1111.3174285888672]],
 'area': 186.95350167853758,
 'bbox': [3010.77294921875,
  1099.3790588378906,
  21.2237548828125,
  11.938369750976562],
 'id': 1,
 'attributes': {'occluded': False},
 'iscrowd': 0,
 'image_id': 56,
 'category_id': 1}
 """
`