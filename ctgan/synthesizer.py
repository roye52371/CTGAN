import numpy as np
import torch
from packaging import version
from torch import optim
from torch.nn import functional
import torch.nn as nn


from ctgan.conditional import ConditionalGenerator
from ctgan.models import Discriminator, Generator
from ctgan.sampler import Sampler
from ctgan.transformer import DataTransformer

#git try to commit changes
#remember to add confidence level as input to CTGAN Synthesizer

class CTGANSynthesizer(object):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        gen_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        dis_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        l2scale (float):
            Weight Decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        blackbox_model:
            Model that implements fit, predict, predict_proba
    """

    def __init__(
        self,
        embedding_dim=128,
        gen_dim=(256, 256),
        dis_dim=(256, 256),
        l2scale=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        blackbox_model=None,
        preprocessing_pipeline=None,
        bb_loss="logloss",
        confidence_levels = [], #default value if no confidence given
    ):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.log_frequency = log_frequency
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trained_epoches = 0
        self.discriminator_steps = discriminator_steps
        self.blackbox_model = blackbox_model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.confidence_levels = confidence_levels #set here not in fit
        self.bb_loss = bb_loss
        #print("self.confidence_levels")

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """

        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(
                    logits, tau=tau, hard=hard, eps=eps, dim=dim
                )
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == "tanh":
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == "softmax":
                ed = st + item[0]
                transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        skip = False
        for item in self.transformer.output_info:
            if item[1] == "tanh":
                st += item[0]
                skip = True

            elif item[1] == "softmax":
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction="none",
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                assert 0

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def fit(
        self,
        train_data,
        discrete_columns=tuple(),
        epochs=300,
        verbose=True,
        gen_lr=2e-4,
    ):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a
                pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of training epochs. Defaults to 300.
        """
        """
        self.confidence_level = confidence_level
        loss_other_name = "loss_bb" if confidence_level != -1 else "loss_d"
        history = {"loss_g": [], loss_other_name: []}
        """
        # Eli: add Mode-specific Normalization
        if not hasattr(self, "transformer"):
            self.transformer = DataTransformer()
            self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions

        if not hasattr(self, "cond_generator"):
            self.cond_generator = ConditionalGenerator(
                train_data, self.transformer.output_info, self.log_frequency
            )

        if not hasattr(self, "generator"):
            self.generator = Generator(
                self.embedding_dim + self.cond_generator.n_opt, self.gen_dim, data_dim
            ).to(self.device)

        if not hasattr(self, "discriminator"):
            self.discriminator = Discriminator(
                data_dim + self.cond_generator.n_opt, self.dis_dim
            ).to(self.device)

        if not hasattr(self, "optimizerG"):
            self.optimizerG = optim.Adam(
                self.generator.parameters(),
                lr=gen_lr,
                betas=(0.5, 0.9),
                weight_decay=self.l2scale,
            )

        if not hasattr(self, "optimizerD"):
            self.optimizerD = optim.Adam(
                self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9)
            )

        assert self.batch_size % 2 == 0

        # init mean to zero and std to one
        #keep one spot to confidence level, which will be add after normal dis will
        mean = torch.zeros(((self.batch_size * self.embedding_dim) - 1), device=self.device)
        std = mean + 1

        # steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        steps_per_epoch = 10  # magic number decided with Gilad. feel free to change it

        loss_other_name = "loss_bb" if self.confidence_levels != [] else "loss_d"
        allhist = {"confidence_levels_history": []}
        # Eli: start training loop
        for current_conf_level in self.confidence_levels:
            #need to change, if non confidence give, aka []) no loop will run
            #so if no conf, need to jump over conf loop
            history = {"confidence_level": current_conf_level, "loss_g": [], loss_other_name: []}
            for i in range(epochs):
                self.trained_epoches += 1
                for id_ in range(steps_per_epoch):
                    #we examine confidence lelvel so no need parts which they does not show up
                    """
                    if self.confidence_levels == []:
                        # discriminator loop
                        for n in range(self.discriminator_steps):
                            fakez = torch.normal(mean=mean, std=std)

                            condvec = self.cond_generator.sample(self.batch_size)
                            if condvec is None:
                                c1, m1, col, opt = None, None, None, None
                                real = data_sampler.sample(self.batch_size, col, opt)
                            else:
                                c1, m1, col, opt = condvec
                                c1 = torch.from_numpy(c1).to(self.device)
                                m1 = torch.from_numpy(m1).to(self.device)
                                fakez = torch.cat([fakez, c1], dim=1)

                                perm = np.arange(self.batch_size)
                                np.random.shuffle(perm)
                                real = data_sampler.sample(
                                    self.batch_size, col[perm], opt[perm]
                                )
                                c2 = c1[perm]

                            fake = self.generator(fakez)
                            fakeact = self._apply_activate(fake)

                            real = torch.from_numpy(real.astype("float32")).to(self.device)

                            if c1 is not None:
                                fake_cat = torch.cat([fakeact, c1], dim=1)
                                real_cat = torch.cat([real, c2], dim=1)
                            else:
                                real_cat = real
                                fake_cat = fake

                            y_fake = self.discriminator(fake_cat)
                            y_real = self.discriminator(real_cat)

                            pen = self.discriminator.calc_gradient_penalty(
                                real_cat, fake_cat, self.device
                            )
                            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                            if self.confidence_levels == []:  # without bb loss
                                self.optimizerD.zero_grad()
                                pen.backward(retain_graph=True)
                                loss_d.backward()
                                self.optimizerD.step()
                    """
                    # we examine confidence lelvel so no need parts which they does not show up
                    #as above(no confidence and uses discriminator, no need)

                    fakez = torch.normal(mean=mean, std=std)
                    #added here the part of adding conf to sample
                    #not sure why we need this part in the code but debug shows it this usses this lines
                    #ask gilad eli
                    confi = current_conf_level.astype(
                        np.float32)  # generator excpect float
                    conf = torch.tensor(
                        [confi]).to(self.device)  # change conf to conf input that will sent!!
                    fakez = torch.cat([fakez, conf], dim=0)
                    fakez = torch.reshape(fakez, (self.batch_size, self.embedding_dim))
                    #end added here the part of adding conf to sample

                    condvec = self.cond_generator.sample(self.batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device)
                        m1 = torch.from_numpy(m1).to(self.device)
                        fakez = torch.cat([fakez, c1], dim=1)

                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                        y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                    else:
                        y_fake = self.discriminator(fakeact)

                    if condvec is None:
                        cross_entropy = 0
                    else:
                        cross_entropy = self._cond_loss(fake, c1, m1)

                    if self.confidence_levels != []:
                        # generate `batch_size` samples
                        gen_out = self.sample(self.batch_size,current_conf_level)
                        loss_bb = self._calc_bb_confidence_loss(gen_out,current_conf_level) #send specific confidence to loss computation
                        loss_g = loss_bb + cross_entropy
                    else:  # original loss
                        loss_g = -torch.mean(y_fake) + cross_entropy

                    self.optimizerG.zero_grad()
                    loss_g.backward()
                    self.optimizerG.step()

                loss_g_val = loss_g.detach().cpu()
                loss_other_val = locals()[loss_other_name].detach().cpu()
                history["loss_g"].append(loss_g.item())
                history[loss_other_name].append(loss_other_val.item())

                if verbose:
                    print(
                        f"Epoch {self.trained_epoches}, Loss G: {loss_g_val}, {loss_other_name}: {loss_other_val}",
                        flush=True,
                    )
            allhist["confidence_levels_history"].append(history)

        return allhist

    def sample(self, n,confidence_level, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.covert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = (
                self.cond_generator.generate_cond_from_condition_column_info(
                    condition_info, self.batch_size
                )
            )
        else:
            global_condition_vec = None

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            #check if it is okay decrease noise by one for adding conf
            #add conf ass input to sample function, as number
            mean = torch.zeros(((self.batch_size*self.embedding_dim)-1))
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)
            #fakez = torch.reshape(fakez,(-1,))
            confidence_level = confidence_level.astype(np.float32)#generator excpect float
            conf = torch.tensor([confidence_level]).to(self.device)#change conf to conf input that will sent!!
            fakez = torch.cat([fakez, conf], dim=0)
            fakez = torch.reshape(fakez,(self.batch_size,self.embedding_dim))

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.cond_generator.sample_zero(self.batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            #conftens = torch.tensor([0.5]);
            #fakez = torch.cat([fakez, conftens], dim=1)#check to delelte!!
            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        # return data
        return self.transformer.inverse_transform(data, None)

    def save(self, path):
        assert hasattr(self, "generator")
        assert hasattr(self, "discriminator")
        assert hasattr(self, "transformer")

        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.generator.to(model.device)
        model.discriminator.to(model.device)
        return model

    def _calc_bb_confidence_loss(self, gen_out,conf_level):
        #added specific conf level as input arguement
        y_prob = self.blackbox_model.predict_proba(gen_out)
        y_conf_gen = y_prob[:, 0]  # confidence scores

        # create vector with the same size of y_confidence filled with `confidence_level` values
        if isinstance(conf_level, list):
            conf = np.random.choice(conf_level)
        else:
            conf = conf_level
        y_conf_wanted = np.full(len(y_conf_gen), conf)

        # to tensor
        y_conf_gen = torch.tensor(y_conf_gen, requires_grad=True).to(self.device)
        y_conf_wanted = torch.tensor(y_conf_wanted).to(self.device)

        # loss
        bb_loss = self._get_loss_by_name(self.bb_loss)
        bb_loss_val = bb_loss(y_conf_gen, y_conf_wanted)

        return bb_loss_val

    @staticmethod
    def _get_loss_by_name(loss_name):
        if loss_name == "log":
            return torch.nn.BCELoss()
        elif loss_name == "l1":
            return torch.nn.L1Loss()
        elif loss_name == "l2":
            return torch.nn.L1Loss()
        elif loss_name == "focal":
            return WeightedFocalLoss()
        else:
            raise ValueError(f"Unknown loss name '{loss_name}'")


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
