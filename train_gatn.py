import argparse

import polars as pl
from torch.utils.data import Dataset, random_split

from GATN.coco import *
from GATN.engine import *
from GATN.models import *
from GATN.util import *


class YelpImageDataset(Dataset):
    def __init__(
        self,
        model,
        image_dir="data/yelp_photos/photos",
        embeddings_file="data/embedding/yelp_embeddings.pkl",
        categories_file="data/processed/unique_categories.csv",
        images_labels="data/processed/photos.parquet",
        root="data/",
    ):
        images_df = pl.read_parquet(images_labels)
        categories_df = pl.read_csv(categories_file)
        self.cat2idx = {
            c: i for c, i in zip(categories_df["category"], categories_df["idx"])
        }
        self.img_list = [
            {"filename": f + ".jpg", "label": l}
            for f, l in zip(images_df["photo_id"], images_df["idx"])
        ]
        self.inp = pickle.load(open(embeddings_file, "rb"))
        self.num_classes = self.inp.shape[0]
        self.image_dir = image_dir
        self.root = root
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                MultiScaleCrop(
                    (224, 224),
                    scales=(1.0, 0.875, 0.75, 0.66, 0.5),
                    max_distort=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=model.image_normalization_mean,
                    std=model.image_normalization_std,
                ),
            ]
        )

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item["filename"]
        labels = sorted(item["label"])
        image_path = os.path.join(self.image_dir, filename)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        target[labels] = 1
        return (img, filename, self.inp), target


parser = argparse.ArgumentParser(description="GATN Training")
parser.add_argument("data", metavar="DIR", help="path to dataset (e.g. data/")
parser.add_argument(
    "--image-size",
    "-i",
    default=448,
    type=int,
    metavar="N",
    help="image size (default: 224)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--epoch_step",
    default=[30],
    type=int,
    nargs="+",
    help="number of epochs to change learning rate",
)
parser.add_argument(
    "--device_ids",
    default=[0],
    type=int,
    nargs="+",
    help="number of epochs to change learning rate",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=16,
    type=int,
    metavar="N",
    help="mini-batch size (default: 16)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--lrd",
    "--learning-rate-decay",
    default=0.1,
    type=float,
    metavar="LRD",
    help="learning rate decay",
)
parser.add_argument(
    "--lrp",
    "--learning-rate-pretrained",
    default=0.1,
    type=float,
    metavar="LR",
    help="learning rate for pre-trained layers",
)
parser.add_argument(
    "--lrt",
    "--learning-rate-transformer",
    default=0.001,
    type=float,
    metavar="LR",
    help="learning rate for pre-trained layers",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=0,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--embedding",
    default="model/embedding/coco_glove_word2vec_80x300.pkl",
    type=str,
    metavar="EMB",
    help="path to embedding (default: glove)",
)
parser.add_argument(
    "--embedding-length",
    default=300,
    type=int,
    metavar="EMB",
    help="embedding length (default: 300)",
)
parser.add_argument(
    "--adj-file",
    default="model/topology/coco_adj.pkl",
    type=str,
    metavar="ADJ",
    help="Adj file (default: model/topology/coco_adj.pkl",
)
parser.add_argument(
    "--t1",
    default=0.2,
    type=float,
    metavar="ADJTS",
    help="Adj strong threshold  (default: 0.4)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--exp-name",
    dest="exp_name",
    default="coco",
    type=str,
    metavar="COCO2014",
    help="Name of experiment to have different location to save checkpoints",
)


def main_coco(
    data,
    num_classes,
    image_size=448,
    workers=8,
    epochs=200,
    epoch_step=[30],
    device_ids=[0],
    start_epoch=0,
    batch_size=16,
    lr=0.03,
    lrd=0.1,
    lrp=0.1,
    lrt=0.001,
    momentum=0.9,
    weight_decay=1e-4,
    print_freq=0,
    embedding="model/embedding/coco_glove_word2vec_80x300.pkl",
    embedding_length=300,
    adj_file="model/topology/coco_adj.pkl",
    t1=0.2,
    resume="",
    evaluate=False,
    exp_name="coco",
):

    global args, best_prec1, use_gpu
    if __name__ == "__main__":
        args = parser.parse_args()
    else:
        args = argparse.Namespace(
            data=data,
            image_size=image_size,
            workers=workers,
            epochs=epochs,
            epoch_step=epoch_step,
            device_ids=device_ids,
            start_epoch=start_epoch,
            batch_size=batch_size,
            lr=lr,
            lrd=lrd,
            lrp=lrp,
            lrt=lrt,
            momentum=momentum,
            weight_decay=weight_decay,
            print_freq=print_freq,
            embedding=embedding,
            embedding_length=embedding_length,
            adj_file=adj_file,
            t1=t1,
            resume=resume,
            evaluate=evaluate,
            exp_name=exp_name,
        )

    use_gpu = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False

    print("Embedding:", args.embedding, "(", args.embedding_length, ")")
    print("Adjacency file:", args.adj_file)
    print("Adjacency t1:", args.t1)

    model = gatn_resnet(
        num_classes=num_classes,
        t1=args.t1,
        adj_file=args.adj_file,
        in_channel=args.embedding_length,
    )

    whole_dataset = YelpImageDataset(model)
    train_dataset, val_dataset = random_split(whole_dataset, [0.8, 0.2])

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(
        model.get_config_optim(args.lr, args.lrp, args.lrt),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    model_path = "checkpoint/coco/%s" % exp_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    state = {
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "max_epochs": args.epochs,
        "evaluate": args.evaluate,
        "resume": args.resume,
        "num_classes": num_classes,
        "difficult_examples": True,
        "save_model_path": model_path,
        "workers": args.workers,
        "epoch_step": args.epoch_step,
        "lr": args.lr,
        "lr_decay": args.lrd,
        "device_ids": args.device_ids,
        "evaluate": args.evaluate,
    }

    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == "__main__":
    main_coco()
