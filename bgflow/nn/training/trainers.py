#from urllib.request import _DataType
import torch
import numpy as np
#from tqdm import tqdm, tqdm_notebook
import warnings
#from tqdm.notebook import tqdm_notebook as tqdmf

from ipdb import set_trace as bp

from bgflow.utils.types import assert_numpy
from bgflow.distribution.sampling import DataSetSampler


__all__ = ["LossReporter", "KLTrainer", "KLTrainerAcc"]


class LossReporter:
    """
    Simple reporter use for reporting losses and plotting them.
    """

    def __init__(self, *labels):
        self._labels = list(labels)
        self._n_reported = len(labels)
        self._raw = [[] for _ in range(self._n_reported)]

    def report(self, *losses):
        assert len(losses) == self._n_reported
        for i in range(self._n_reported):
            self._raw[i].append(assert_numpy(losses[i]))

    def print(self, *losses):
        iter = len(self._raw[0])
        report_str = "{0}\t".format(iter)
        for i in range(self._n_reported):
            report_str += "{0}: {1:.4f}\t".format(self._labels[i], self._raw[i][-1])
        print(report_str)

    def losses(self, n_smooth=1):
        x = np.arange(n_smooth, len(self._raw[0]) + 1)
        ys = []
        for (label, raw) in zip(self._labels, self._raw):
            raw = assert_numpy(raw).reshape(-1)
            kernel = np.ones(shape=(n_smooth,)) / n_smooth
            ys.append(np.convolve(raw, kernel, mode="valid"))
        return self._labels, x, ys

    def recent(self, n_recent=1):
        return np.array([raw[-n_recent:] for raw in self._raw])

    def add_loss(self,lossname):
        self._labels.append(lossname)
        self._n_reported +=1


class KLTrainer(object):
    def __init__(
        self, bg, optim=None, train_likelihood=True, train_energy=True, custom_loss=None, test_likelihood=False,
    ):
        """Trainer for minimizing the forward or reverse

        Trains in either of two modes, or a mixture of them:
        1. Forward KL divergence / energy based training. Minimize KL divergence between
           generation probability of flow and target distribution
        2. Reverse KL divergence / maximum likelihood training. Minimize reverse KL divergence between
           data mapped back to latent space and prior distribution.

        """
        self.bg = bg

        if optim is None:
            optim = torch.optim.Adam(bg.parameters(), lr=5e-3)
        self.optim = optim

        loss_names = []
        self.train_likelihood = train_likelihood
        self.w_likelihood = 0.0
        self.train_energy = train_energy
        self.w_energy = 0.0
        self.test_likelihood = test_likelihood
        if train_energy:
            loss_names.append("KLL")
            self.w_energy = 1.0
        if train_likelihood:
            loss_names.append("NLL")
            self.w_likelihood = 1.0
        if test_likelihood: 
            loss_names.append("NLL(Test)")
        self.custom_loss = custom_loss
        if self.custom_loss:
            loss_names.append("custom_loss")
        self.reporter = LossReporter(*loss_names)
        

    def train(
        self,
        n_iter,
        data=None,
        testdata=None,
        batchsize=128,
        w_likelihood=None,
        w_energy=None,
        w_custom=None,
        custom_loss_kwargs={},
        n_print=0,
        temperature=1.0,
        schedulers=(),
        clip_forces=None,
        progress_bar=lambda x:x,
        device = "cpu"
    ):
        """
        Train the network.

        Parameters
        ----------
        n_iter : int
            Number of training iterations.
        data : torch.Tensor or Sampler
            Training data
        testdata : torch.Tensor or Sampler
            Test data
        batchsize : int
            Batchsize
        w_likelihood : float or None
            Weight for backward KL divergence during training;
            if specified, this argument overrides self.w_likelihood
        w_energy : float or None
            Weight for forward KL divergence divergence during training;
            if specified, this argument overrides self.w_energy
        n_print : int
            Print interval
        temperature : float
            Temperature at which the training is performed
        schedulers : iterable
            A list of pairs (int, scheduler), where the integer specifies the number of iterations between
            steps of the scheduler. Scheduler steps are invoked before the optimization step.
        progress_bar : callable
            To show a progress bar, pass `progress_bar = tqdm.auto.tqdm`

        Returns
        -------
        """
        if w_likelihood is None:
            w_likelihood = self.w_likelihood
        if w_energy is None:
            w_energy = self.w_energy
        if clip_forces is not None:
            warnings.warn(
                "clip_forces is deprecated and will be ignored. "
                "Use GradientClippedEnergy instances instead",
                DeprecationWarning
            )

        if isinstance(data, torch.Tensor):
            data = DataSetSampler(data, device=device)
        if isinstance(testdata, torch.Tensor):
            testdata = DataSetSampler(testdata, device=device)

        for iter in progress_bar(range(n_iter)):
            # invoke schedulers
            for interval, scheduler in schedulers:
                if iter % interval == 0:
                    scheduler.step()
            self.optim.zero_grad()
            reports = []

            if self.train_energy:
                # kl divergence to the target
                kll = self.bg.kldiv(batchsize, temperature=temperature).mean()
                reports.append(kll)
                # aggregate weighted gradient
                if w_energy > 0:
                    l = w_energy / (w_likelihood + w_energy)
                    (l * kll).backward(retain_graph=True)

            if self.train_likelihood:
                batch = data.sample(batchsize)
                if isinstance(batch, torch.Tensor):
                    batch = (batch,)
                # negative log-likelihood of the batch is equal to the energy of the BG
                nll = self.bg.energy(*batch, temperature=temperature).mean()
                reports.append(nll)
                # aggregate weighted gradient
                if w_likelihood > 0:
                    l = w_likelihood / (w_likelihood + w_energy)
                    (l * nll).backward(retain_graph=True)
                
            # compute NLL over test data 
            if self.test_likelihood:
                testnll = torch.zeros_like(nll)
                if testdata is not None:
                    testbatch = testdata.sample(batchsize)
                    if isinstance(testbatch, torch.Tensor):
                        testbatch = (testbatch,)
                    with torch.no_grad():
                        testnll = self.bg.energy(*testbatch, temperature=temperature).mean()
                reports.append(testnll)

            if w_custom is not None:
                cl = self.custom_loss(**custom_loss_kwargs)
                (w_custom * cl).backward(retain_graph=True)
                reports.append(cl)

            self.reporter.report(*reports)
            if n_print > 0:
                if iter % n_print == 0:
                    self.reporter.print(*reports)
            
            if any(torch.any(torch.isnan(p.grad)) for p in self.bg.parameters() if p.grad is not None):
                print("found nan in grad; skipping optimization step")
            else:
                self.optim.step()


    def losses(self, n_smooth=1):
        return self.reporter.losses(n_smooth=n_smooth)


class KLTrainerAcc(object):
    def __init__(
        self, bg, optim=None, train_likelihood=True, train_buffer_likelihood=True, train_energy=True, custom_loss=None, test_likelihood=False,
    ):
        """Trainer for minimizing the forward or reverse

        Trains in either of two modes, or a mixture of them:
        1. Forward KL divergence / energy based training. Minimize KL divergence between
           generation probability of flow and target distribution
        2. Reverse KL divergence / maximum likelihood training. Minimize reverse KL divergence between
           data mapped back to latent space and prior distribution.

        """
        self.bg = bg

        if optim is None:
            optim = torch.optim.Adam(bg.parameters(), lr=5e-3)
        self.optim = optim

        loss_names = []
        self.train_likelihood = train_likelihood
        self.w_likelihood = 0.0
        self.train_buffer_likelihood = train_buffer_likelihood
        self.w_buffer_likelihood = 0.0
        self.train_energy = train_energy
        self.w_energy = 0.0
        self.test_likelihood = test_likelihood
        if train_energy:
            loss_names.append("KLL")
            self.w_energy = 1.0
        if train_likelihood:
            loss_names.append("NLL")
            self.w_likelihood = 1.0
        if train_buffer_likelihood:
            loss_names.append("NLL_buffer")
            self.w_likelihood = 1.0
        if test_likelihood: 
            loss_names.append("NLL(Test)")
        self.custom_loss = custom_loss
        if self.custom_loss:
            loss_names.append("custom_loss")
        self.reporter = LossReporter(*loss_names)
        self.replay_buffer=torch.tensor([])
        

    def train(
        self,
        n_iter,
        data=None,
        testdata=None,
        buffer_size=1000,
        batchsize=128,
        w_likelihood=None,
        w_buffer_likelihood=1.,
        w_energy=None,
        w_custom=None,
        custom_loss_kwargs={},
        n_print=0,
        temperature=1.0,
        schedulers=(),
        clip_forces=None,
        progress_bar=lambda x:x
    ):
        """
        Train the network.

        Parameters
        ----------
        n_iter : int
            Number of training iterations.
        data : torch.Tensor or Sampler
            Training data
        testdata : torch.Tensor or Sampler
            Test data
        batchsize : int
            Batchsize
        w_likelihood : float or None
            Weight for backward KL divergence during training;
            if specified, this argument overrides self.w_likelihood
        w_energy : float or None
            Weight for forward KL divergence divergence during training;
            if specified, this argument overrides self.w_energy
        n_print : int
            Print interval
        temperature : float
            Temperature at which the training is performed
        schedulers : iterable
            A list of pairs (int, scheduler), where the integer specifies the number of iterations between
            steps of the scheduler. Scheduler steps are invoked before the optimization step.
        progress_bar : callable
            To show a progress bar, pass `progress_bar = tqdm.auto.tqdm`

        Returns
        -------
        """
        if w_likelihood is None:
            w_likelihood = self.w_likelihood
        if w_energy is None:
            w_energy = self.w_energy
        if clip_forces is not None:
            warnings.warn(
                "clip_forces is deprecated and will be ignored. "
                "Use GradientClippedEnergy instances instead",
                DeprecationWarning
            )

        if isinstance(data, torch.Tensor):
            data = DataSetSampler(data)
        if isinstance(testdata, torch.Tensor):
            testdata = DataSetSampler(testdata)

        for iter in progress_bar(range(n_iter)):
            # invoke schedulers
            for interval, scheduler in schedulers:
                if iter % interval == 0:
                    scheduler.step()
            
            reports = []
                
            if self.train_buffer_likelihood:
                tried_samples = 0
                if self.replay_buffer.shape[0] >= buffer_size:
                    nlls=[]
                    with progress_bar(total=buffer_size, desc = "NLL") as pbar:
                        while tried_samples < buffer_size:
                            batch = self.replay_buffer_sampler.sample(batchsize)
                            tried_samples += batchsize
                            if isinstance(batch, torch.Tensor):
                                batch = (batch,)
                            # negative log-likelihood of the batch is equal to the energy of the BG
                            nll = self.bg.energy(*batch, temperature=temperature).mean() * batchsize/buffer_size
                            if w_buffer_likelihood > 0:
                                l = w_buffer_likelihood / (w_buffer_likelihood + w_custom)
                                (l * nll).backward(retain_graph=True)
                            nlls.append(nll.detach())
                            pbar.update(batchsize)
                    to_report= np.mean([n.item() for n in nlls])
                    #bp()
                    reports.append(torch.tensor(to_report))
                    #reports.append(torch.cat(nll))
                    # aggregate weighted gradient

                else:
                    reports.append(np.array(0))
                

            if w_custom is not None:
                total_successful_samples = 0
                dcl=[]
                self.xs = []
                with progress_bar(total=buffer_size, desc="KLL") as pbar:
                    while total_successful_samples < buffer_size:                
                        cl, successful_samples, x = self.custom_loss(**custom_loss_kwargs, buffer_size=buffer_size, batchsize=batchsize)
                        l = w_custom / (w_buffer_likelihood + w_custom)
                        (l*cl).backward(retain_graph=True)
                        total_successful_samples += successful_samples
                        pbar.update(successful_samples)
                        #print(total_successful_samples)
                        dcl.append(cl.detach())
                        self.xs.append(x)
                #dcl=torch.cat(dcl)
                self.update_replay_buffer(torch.cat(self.xs))
                to_report= np.mean([d.item() for d in dcl])
                #bp()
                reports.append(torch.tensor(to_report))

            self.reporter.report(*reports)
            if n_print > 0:
                if iter % n_print == 0:
                    self.reporter.print(*reports)
            
            if any(torch.any(torch.isnan(p.grad)) for p in self.bg.parameters() if p.grad is not None):
                bp()
                print("found nan in grad; skipping optimization step")
            else:
                self.optim.step()
                self.optim.zero_grad()


    def losses(self, n_smooth=1):
        return self.reporter.losses(n_smooth=n_smooth)
    
    def update_replay_buffer(self, tensor):
        self.replay_buffer = tensor
        self.replay_buffer_sampler = DataSetSampler(self.replay_buffer)
