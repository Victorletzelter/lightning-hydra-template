import pytorch_lightning as pl
from src.data.data_handlers.tut_sound_events import TUTSoundEvents
from torch.utils.data import DataLoader

class TUTSoundEventsDataModule(pl.LightningDataModule):
    def __init__(self, root: str,
                 tmp_dir: str = './tmp',
                 test_fold_idx: int = 1,
                 sequence_duration: float = 30.,
                 chunk_length: float = 2.,
                 frame_length: float = 0.04,
                 num_fft_bins: int = 2048,
                 max_num_sources: int = 5,
                 num_overlapping_sources: int = None,
                 batch_size: int = 32):
        super().__init__()
        self.root = root
        self.tmp_dir = tmp_dir
        self.test_fold_idx = test_fold_idx
        self.sequence_duration = sequence_duration
        self.chunk_length = chunk_length
        self.frame_length = frame_length
        self.num_fft_bins = num_fft_bins
        self.max_num_sources = max_num_sources
        self.num_overlapping_sources = num_overlapping_sources
        self.batch_size = batch_size

    def prepare_data(self):
        # Download data if needed
        pass

    def setup(self, stage=None):
        # Define dataset split for train/val/test
        self.train_dataset = TUTSoundEvents(self.root, split='train', test_fold_idx=self.test_fold_idx,
                                            sequence_duration=self.sequence_duration, chunk_length=self.chunk_length,
                                            frame_length=self.frame_length, num_fft_bins=self.num_fft_bins,
                                            max_num_sources=self.max_num_sources,
                                            num_overlapping_sources=self.num_overlapping_sources)
        self.val_dataset = TUTSoundEvents(self.root, split='valid', test_fold_idx=self.test_fold_idx,
                                          sequence_duration=self.sequence_duration, chunk_length=self.chunk_length,
                                          frame_length=self.frame_length, num_fft_bins=self.num_fft_bins,
                                          max_num_sources=self.max_num_sources,
                                          num_overlapping_sources=self.num_overlapping_sources)
        self.test_dataset = TUTSoundEvents(self.root, split='test', test_fold_idx=self.test_fold_idx,
                                           sequence_duration=self.sequence_duration, chunk_length=self.chunk_length,
                                           frame_length=self.frame_length, num_fft_bins=self.num_fft_bins,
                                           max_num_sources=self.max_num_sources,
                                           num_overlapping_sources=self.num_overlapping_sources)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def test_dataloader(self) :
        # During testing, a whole sequence is packed into one batch. The batch size set for training and validation
        # is ignored in this case.
        num_chunks_per_sequence = int(self.hparams['sequence_duration'] / self.hparams['chunk_length'])

        test_loaders = []

        for num_overlapping_sources in range(1, 4):
            test_dataset = TUTSoundEvents(self.root, test_fold_idx=self.test_fold_idx,
                                        sequence_duration=self.sequence_duration, chunk_length=self.chunk_length,
                                        frame_length=self.frame_length, num_fft_bins=self.num_fft_bins,
                                        max_num_sources=self.max_num_sources,
                                        num_overlapping_sources=num_overlapping_sources)

            test_loaders.append(DataLoader(test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False))

        return test_loaders
