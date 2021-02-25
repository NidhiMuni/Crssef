from mammogram_dataset import MammogramDataset
import torchvision.transforms as T
import PIL

def test_MammogramDataset_NoTranforms():
    ds = MammogramDataset('Mini_DDSM_Upload', 'train')
    ds.print_summary()
    ds.print_datapoint(2)

def test_MammogramDataset_WithTranforms():
    transform = T.Compose([
                T.ToPILImage(),
                T.Resize(int(1024/0.9) , interpolation=PIL.Image.BICUBIC),
                T.RandomCrop(1024),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor()
            ])

    ds = MammogramDataset('Mini_DDSM_Upload', 'train', transform = transform)
    ds.print_summary()
    ds.print_datapoint(2)

# Main
#test_MammogramDataset_NoTranforms()
test_MammogramDataset_WithTranforms()