import torch


class Trainer:
    @staticmethod
    def _flow_matching_loss(model, x_cond, x1, device):
        """
        x_cond: (B, C, H, W)
        x1:     (B, C, H, W)
        """
        x_cond = x_cond.to(device)
        x1 = x1.to(device)

        B, C, H, W = x1.shape

        # noise for target frame
        x0 = torch.randn_like(x1)

        # time
        t = torch.rand(B, 1, device=device, dtype=x1.dtype)
        t_broadcast = t[:, :, None, None]

        # interpolation between noise and target
        x_t = (1.0 - t_broadcast) * x0 + t_broadcast * x1

        # condition + noisy target
        x_in = torch.cat([x_cond, x_t], dim=1)  # (B, 2C, H, W)

        # target velocity
        v_target = x1 - x0  # (B, C, H, W)

        # model prediction
        v_out = model(x_in, t.squeeze(1))

        # predicted velocity 
        v_pred = v_out

        loss = ((v_pred - v_target) ** 2).mean()

        return loss
    
    @staticmethod
    @torch.no_grad()
    def _sample_next_step(model, x_cond, num_ode_steps, device):
        """
        x_cond: [B, C, H, W]
        returns x_pred: [B, C, H, W]
        """
        x_cond = x_cond.to(device)

        B, C, H, W = x_cond.shape

        x = torch.randn_like(x_cond)
        dt = 1.0 / num_ode_steps

        for i in range(num_ode_steps):
            t = torch.full(
                (B,),
                fill_value=i / num_ode_steps,
                device=device,
                dtype=x_cond.dtype,
            )

            x_in = torch.cat([x_cond, x], dim=1)

            v = model(x_in, t)

            if v.shape != x.shape:
                raise RuntimeError(
                    f"Velocity shape mismatch: v has shape {v.shape}, "
                    f"but x has shape {x.shape}."
                )

            x = x + dt * v

        return x

    @staticmethod
    @torch.no_grad()
    def nsp_epoch(
        model,
        dataloader,
        num_ode_steps,
        device=torch.device("cuda:0"),
        max_batches=1,
    ):
        """
        Next-step prediction MSE.

        Uses:
            true x_cond -> generated x_pred
            true x1     -> only for MSE
        """
        model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (x_cond, x1) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x1 = x1.to(device)

            x_pred = Trainer._sample_next_step(
                model=model,
                x_cond=x_cond,
                num_ode_steps=num_ode_steps,
                device=device,
            )

            loss = ((x_pred - x1) ** 2).mean()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

        return avg_loss

    @staticmethod
    def train_epoch(model, dataloader, optimizer, device=torch.device("cuda:0")):
        model.train()

        running_loss = 0.0
        num_batches = 0

        for x_cond, x1 in dataloader:
            loss = Trainer._flow_matching_loss(
                model=model,
                x_cond=x_cond,
                x1=x1,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = running_loss / num_batches
        return avg_epoch_loss

    @staticmethod
    def val_epoch(model, dataloader, device=torch.device("cuda:0")):
        model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x_cond, x1 in dataloader:
                loss = Trainer._flow_matching_loss(
                    model=model,
                    x_cond=x_cond,
                    x1=x1,
                    device=device,
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss