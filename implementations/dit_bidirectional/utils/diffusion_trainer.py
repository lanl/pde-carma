import torch

from utils.diffschedule import DiffusionSchedule


class Trainer:

    @staticmethod
    def _diffusion_loss(
        model,
        x_cond,
        x1,
        c,
        device,
        num_diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        """
        Conditional DDPM epsilon-prediction loss.

        x_cond : (B, C, H, W)
        x1     : (B, C, H, W), clean target frame
        c      : (B, 2), [1,0] forward or [0,1] backward
        """

        x_cond = x_cond.to(device)
        x1 = x1.to(device)
        c = c.to(device)

        B = x1.shape[0]

        schedule = DiffusionSchedule(
            num_timesteps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            dtype=x1.dtype,
        )

        t_idx = torch.randint(
            low=0,
            high=schedule.num_timesteps,
            size=(B,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(x1)
        x_t = schedule.q_sample(x1, t_idx, noise)

        x_in = torch.cat([x_cond, x_t], dim=1)
        t = schedule.normalized_time(t_idx)

        noise_pred = model(x_in, t, c)

        loss = ((noise_pred - noise) ** 2).mean()

        return loss

    @staticmethod
    @torch.no_grad()
    def _sample_next_step(
        model,
        x_cond,
        c,
        samp_diff_steps,
        device,
        num_diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        ddim_eta=0.0,
    ):
        """
        Generate a target frame with DDIM sampling.

        samp_diff_steps means reverse diffusion sampling steps.
        """

        x_cond = x_cond.to(device)
        c = c.to(device)

        B = x_cond.shape[0]

        schedule = DiffusionSchedule(
            num_timesteps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            dtype=x_cond.dtype,
        )

        if samp_diff_steps is None:
            samp_diff_steps = schedule.num_timesteps

        timesteps = schedule.ddim_timesteps(samp_diff_steps)

        x = torch.randn_like(x_cond)

        for step_idx, timestep in enumerate(timesteps):
            is_last_step = step_idx == len(timesteps) - 1

            t_idx = torch.full(
                (B,),
                fill_value=int(timestep.item()),
                device=device,
                dtype=torch.long,
            )

            t = schedule.normalized_time(t_idx)
            x_in = torch.cat([x_cond, x], dim=1)

            noise_pred = model(x_in, t, c)

            alpha_bar_t = schedule.alpha_bars[timestep]
            x0_pred = (
                x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
            ) / torch.sqrt(alpha_bar_t)

            if is_last_step:
                alpha_bar_prev = torch.ones((), device=device, dtype=x.dtype)
            else:
                alpha_bar_prev = schedule.alpha_bars[timesteps[step_idx + 1]]

            if ddim_eta > 0.0 and not is_last_step:
                sigma = ddim_eta * torch.sqrt(
                    torch.clamp(
                        (1.0 - alpha_bar_prev)
                        / (1.0 - alpha_bar_t)
                        * (1.0 - alpha_bar_t / alpha_bar_prev),
                        min=0.0,
                    )
                )
                z = torch.randn_like(x)
            else:
                sigma = torch.zeros((), device=device, dtype=x.dtype)
                z = 0.0

            direction_scale = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)
            )

            x = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + direction_scale * noise_pred
                + sigma * z
            )

        return x

    @staticmethod
    def train_epoch(
        model,
        dataloader,
        optimizer,
        device=torch.device("cuda:0"),
        num_diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        model.train()

        running_loss = 0.0
        num_batches = 0

        for x_cond, x1, c in dataloader:

            loss = Trainer._diffusion_loss(
                model=model,
                x_cond=x_cond,
                x1=x1,
                c=c,
                device=device,
                num_diffusion_steps=num_diffusion_steps,
                beta_start=beta_start,
                beta_end=beta_end,
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = running_loss / num_batches

        return avg_epoch_loss

    @staticmethod
    def val_epoch(
        model,
        dataloader,
        device=torch.device("cuda:0"),
        num_diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():

            for x_cond, x1, c in dataloader:

                loss = Trainer._diffusion_loss(
                    model=model,
                    x_cond=x_cond,
                    x1=x1,
                    c=c,
                    device=device,
                    num_diffusion_steps=num_diffusion_steps,
                    beta_start=beta_start,
                    beta_end=beta_end,
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        return avg_loss

    @staticmethod
    @torch.no_grad()
    def nsp_epoch(
        model,
        dataloader,
        samp_diff_steps,
        device=torch.device("cuda:0"),
        max_batches=1,
        num_diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        ddim_eta=0.0,
    ):
        """
        Next-step prediction MSE using reverse diffusion sampling.
        """

        model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x_cond, x1, c) in enumerate(dataloader):

            if max_batches is not None and batch_idx >= max_batches:
                break

            x1 = x1.to(device)

            x_pred = Trainer._sample_next_step(
                model=model,
                x_cond=x_cond,
                c=c,
                samp_diff_steps=samp_diff_steps,
                device=device,
                num_diffusion_steps=num_diffusion_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                ddim_eta=ddim_eta,
            )

            loss = ((x_pred - x1) ** 2).mean()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = (
            total_loss / num_batches
            if num_batches > 0
            else float("inf")
        )

        return avg_loss
