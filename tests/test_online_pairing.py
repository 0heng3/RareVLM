import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from online_aug_dataset import OnlineAugmentedTrainDataset


def dataset(tmp_path,seed=0):
    rows=[]
    for i in range(4):
        path=tmp_path/'train'/'n00000000'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True)
        y,x=np.indices((192,288));pixels=np.stack([(x+i*19)%256,(y+i*37)%256,(x+y+i*11)%256],axis=-1).astype(np.uint8)
        Image.fromarray(pixels).save(path)
        rows.append({'path':str(path.relative_to(tmp_path)),'label':0})
    manifest={'schema_version':'rarevlm.imagenet_lt.v1','manifest_sha256':'synthetic-pairing','splits':{'train':rows}}
    return OnlineAugmentedTrainDataset(manifest,tmp_path,seed)


def test_same_seed_epoch_sample_is_independent_of_global_rng(tmp_path):
    ce=dataset(tmp_path);la=dataset(tmp_path);ce.set_epoch(3);la.set_epoch(3)
    torch.manual_seed(999);torch.rand(200)
    first=[ce[i][0] for i in range(4)]
    torch.manual_seed(42);torch.rand(170)
    second=[la[i][0] for i in range(4)]
    assert all(torch.equal(a,b) for a,b in zip(first,second))
    assert ce.audit_manifest(3,[3,1,2,0])==la.audit_manifest(3,[3,1,2,0])


def test_actual_views_change_across_epochs(tmp_path):
    d=dataset(tmp_path);d.set_epoch(1);first=[d[i][0] for i in range(4)]
    d.set_epoch(2);second=[d[i][0] for i in range(4)]
    assert any(not torch.equal(a,b) for a,b in zip(first,second))


def test_persistent_workers_match_single_process_and_receive_epoch(tmp_path):
    d=dataset(tmp_path);d.set_epoch(1)
    loader=DataLoader(d,batch_size=2,num_workers=2,persistent_workers=True,shuffle=False)
    try:
        for epoch in (1,2):
            d.set_epoch(epoch)
            worker=torch.cat([batch[0] for batch in loader])
            direct=torch.stack([d[i][0] for i in range(4)])
            assert torch.equal(worker,direct)
    finally:
        if loader._iterator is not None:loader._iterator._shutdown_workers()
