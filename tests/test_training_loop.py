from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.common.config import load_config
from tests.helpers import write_fixture

ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        (ROOT / 'runs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.config = load_config()
        self.config['training'].update(epochs=2, batch_size=2)
        self.path, _ = write_fixture(self.root / 'data', self.config)
        self.config['runtime'].update(backend='cpu', data_root=str(self.root / 'data'),
                                      manifest=str(self.path), output_dir=str(self.root / 'first'))

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.temporal.train'),
                             'shared training loop not implemented')
        from training.temporal import train
        return train

    def test_training_checkpoint_and_resumption_match_uninterrupted_run(self):
        api = self.api()
        result = api.run_training(self.config)
        self.assertEqual(result['completed_epochs'], 2)
        from training.common.checkpoint import load_checkpoint
        first = load_checkpoint(Path(result['last_checkpoint']))
        self.assertEqual(first['epoch'], 2)
        self.assertEqual(first['global_step'], 4)
        self.assertTrue(all(t.device.type == 'cpu' for t in first['model_state'].values()))
        resume_config = deepcopy(self.config)
        resume_config['training']['epochs'] = 3
        resume_config['runtime']['output_dir'] = str(self.root / 'resumed')
        resumed = api.run_training(resume_config, resume=result['last_checkpoint'])
        continuous_config = deepcopy(resume_config)
        continuous_config['runtime']['output_dir'] = str(self.root / 'continuous')
        continuous = api.run_training(continuous_config)
        a = load_checkpoint(resumed['last_checkpoint'])
        b = load_checkpoint(continuous['last_checkpoint'])
        self.assertEqual(a['global_step'], 6)
        for key in a['model_state']:
            torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
        self.assertEqual(a['history'], b['history'])
        best = load_checkpoint(resumed['best_checkpoint'])
        self.assertEqual(best['kind'], 'weights')
        self.assertEqual(best['epoch'], a['best_epoch'])

    def test_resume_rejects_hyperparameter_and_dataset_changes(self):
        api = self.api()
        result = api.run_training(self.config)
        changed = deepcopy(self.config)
        changed['training'].update(epochs=3, learning_rate=0.02)
        changed['runtime']['output_dir'] = str(self.root / 'rejected')
        with self.assertRaisesRegex(ValueError, 'config|configuration'):
            api.run_training(changed, resume=result['last_checkpoint'])
        changed['training']['learning_rate'] = self.config['training']['learning_rate']
        values_path = self.root / 'data/r0_values.npy'
        values = np.load(values_path)
        values[0, 0] += 1
        np.save(values_path, values)
        with self.assertRaisesRegex(ValueError, 'data|fingerprint'):
            api.run_training(changed, resume=result['last_checkpoint'])

    def test_existing_output_directory_never_overwritten(self):
        api = self.api()
        output = Path(self.config['runtime']['output_dir'])
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('user data', encoding='utf-8')
        with self.assertRaises((ValueError, FileExistsError)):
            api.run_training(self.config)
        self.assertEqual(marker.read_text(), 'user data')

    def test_unsupervised_epoch_fails_without_parameter_update(self):
        api = self.api()
        from training.temporal.model import TemporalTCN
        model = TemporalTCN()
        x = torch.zeros(2, 100, 24)
        x[:, :, 12:] = 1
        target = torch.zeros(2, 4)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        loader = DataLoader(TensorDataset(x, target, mask), batch_size=1)
        before = deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
        with self.assertRaisesRegex(ValueError, 'supervision|updates'):
            api.run_epoch(model, loader, torch.device('cpu'), torch.ones(4, dtype=torch.bool), optimizer, 1.)
        for key in before:
            torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
        self.assertEqual(len(optimizer.state), 0)

    def test_validation_loss_is_invariant_to_batch_partition(self):
        api = self.api()
        from training.temporal.model import TemporalTCN
        torch.manual_seed(4)
        model = TemporalTCN()
        x = torch.randn(7, 100, 24)
        x[:, :, 12:] = 1
        targets = torch.zeros(7, 4)
        targets[::2] = 1
        masks = torch.ones(7, 4, dtype=torch.bool)
        masks[:2, 2:] = False
        targets[~masks] = 0
        data = TensorDataset(x, targets, masks)
        reports = [api.run_epoch(model, DataLoader(data, batch_size=n), torch.device('cpu'),
                                torch.ones(4, dtype=torch.bool)) for n in (2, 7)]
        self.assertAlmostEqual(reports[0]['loss'], reports[1]['loss'], places=6)
        self.assertEqual(reports[0]['metrics'], reports[1]['metrics'])

    def test_unknown_batches_count_coverage_without_forward_or_optimizer_step(self):
        api = self.api()
        from training.temporal.model import TemporalTCN
        model = TemporalTCN()
        x = torch.zeros(3, 100, 24)
        x[:, :, 12:] = 1
        targets = torch.zeros(3, 4)
        targets[-1] = 1
        masks = torch.ones(3, 4, dtype=torch.bool)
        masks[1] = False
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        forwarded = []
        hook = model.register_forward_hook(lambda module, args, output: forwarded.append(len(output)))
        try:
            result = api.run_epoch(model, DataLoader(TensorDataset(x, targets, masks), batch_size=1),
                                   torch.device('cpu'), torch.ones(4, dtype=torch.bool), optimizer)
        finally:
            hook.remove()
        self.assertEqual(forwarded, [1, 1])
        self.assertEqual(result['updates'], 2)
        self.assertEqual(result['skipped_batches'], 1)
        for task in result['metrics']['per_task'].values():
            self.assertEqual(task['unknown'], 1)
            self.assertEqual(task['positive'], 1)
            self.assertEqual(task['negative'], 1)

    def test_checkpoint_checksum_and_weights_only_contract(self):
        api = self.api()
        result = api.run_training(self.config)
        from training.common.checkpoint import load_checkpoint
        path = Path(result['last_checkpoint'])
        payload = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(payload['format_version'], 1)
        with path.open('ab') as handle:
            handle.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
