import torch


class FrameMetrics2D:
    """
    Metrics for one scalar frame.

    Input:
        pred: (H, W)
        true: (H, W)

    Output:
        dictionary of scalar metrics
    """

    metric_names = [
        "mse",
        "rmse",
        "nrmse",
        "vrmse",
        "frmse_low",
        "frmse_med",
        "frmse_high",
        "energy_spectrum_error",
    ]

    def __init__(self, eps=1e-8, low_cut=1.0 / 3.0, high_cut=2.0 / 3.0):
        self.eps = eps
        self.low_cut = low_cut
        self.high_cut = high_cut

    def _radius(self, H, W, device):
        """
        Normalized radial frequency:

            r_kl = sqrt(k^2 + l^2) / max_{k,l} sqrt(k^2 + l^2)
        """
        fy = torch.fft.fftfreq(H, device=device)
        fx = torch.fft.fftfreq(W, device=device)

        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)

        return r / (r.max() + self.eps)

    def _band_energy(self, x_fft, mask):
        """
        Band energy:

            E_B(u) = sum_{(k,l) in B} |FFT2(u)_kl|^2
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_fft.device)

        return torch.sum(torch.abs(x_fft[mask]) ** 2)

    def __call__(self, pred, true):
        # Keep both tensors on the same device.
        pred = pred.detach().float()
        true = true.detach().float().to(pred.device)

        H, W = true.shape
        err = pred - true

        # ---------------- basic metrics ----------------
        mse = torch.mean(err**2)
        rmse = torch.sqrt(mse)

        """
        Error:
            err = pred - true

        MSE:
            MSE = (1 / HW) * sum_i sum_j (pred_ij - true_ij)^2

        RMSE:
            RMSE = sqrt(MSE)
        """

        true_rms = torch.sqrt(torch.mean(true**2))
        nrmse = rmse / (true_rms + self.eps)

        """
        NRMSE:
            RMS(true) = sqrt((1 / HW) * sum_i sum_j true_ij^2)

            NRMSE = RMSE / (RMS(true) + eps)

        This normalizes error by the overall magnitude of the true field.
        """

        true_mean = torch.mean(true)
        true_std = torch.sqrt(torch.mean((true - true_mean) ** 2))
        vrmse = rmse / (true_std + self.eps)

        """
        VRMSE:
            mean(true) = (1 / HW) * sum_i sum_j true_ij

            Var(true) = (1 / HW) * sum_i sum_j (true_ij - mean(true))^2

            Std(true) = sqrt(Var(true))

            VRMSE = RMSE / (Std(true) + eps)

        This normalizes error by the spatial variation of the true field.

        Note:
            If mean(true) is close to 0, RMS(true) and Std(true)
            can be very similar, so NRMSE and VRMSE can also be similar.
            If mean(true) is not close to 0, they are different.
        """

        # ---------------- spectral metrics ----------------
        """
        Fourier transform of error:
            err_fft = FFT2(err)

        Normalized radial frequency:
            r_kl = sqrt(k^2 + l^2) / max_{k,l} sqrt(k^2 + l^2)

        Frequency bands:
            B_low  = {(k,l): r_kl < 1/3}
            B_med  = {(k,l): 1/3 <= r_kl < 2/3}
            B_high = {(k,l): r_kl >= 2/3}

        Fourier RMSE in one band:
            fRMSE_B = sqrt((1 / |B|) * sum_{(k,l) in B} |err_fft_kl|^2)
        """

        r = self._radius(H, W, pred.device)

        low_mask = r < self.low_cut
        med_mask = (r >= self.low_cut) & (r < self.high_cut)
        high_mask = r >= self.high_cut

        err_fft = torch.fft.fft2(err, norm="ortho")

        frmse_low = torch.sqrt(torch.mean(torch.abs(err_fft[low_mask]) ** 2))
        frmse_med = torch.sqrt(torch.mean(torch.abs(err_fft[med_mask]) ** 2))
        frmse_high = torch.sqrt(torch.mean(torch.abs(err_fft[high_mask]) ** 2))

        # ---------------- energy spectrum error ----------------
        """
        Fourier energy in one band:
            E_B(u) = sum_{(k,l) in B} |FFT2(u)_kl|^2

        Predicted energy vector:
            E(pred) = [
                E_B_low(pred),
                E_B_med(pred),
                E_B_high(pred)
            ]

        True energy vector:
            E(true) = [
                E_B_low(true),
                E_B_med(true),
                E_B_high(true)
            ]

        Energy spectrum error:
            ESE = ||E(pred) - E(true)||_2 / (||E(true)||_2 + eps)
        """

        pred_fft = torch.fft.fft2(pred, norm="ortho")
        true_fft = torch.fft.fft2(true, norm="ortho")

        e_pred = torch.stack([
            self._band_energy(pred_fft, low_mask),
            self._band_energy(pred_fft, med_mask),
            self._band_energy(pred_fft, high_mask),
        ])

        e_true = torch.stack([
            self._band_energy(true_fft, low_mask),
            self._band_energy(true_fft, med_mask),
            self._band_energy(true_fft, high_mask),
        ])

        energy_spectrum_error = (
            torch.linalg.vector_norm(e_pred - e_true)
            / (torch.linalg.vector_norm(e_true) + self.eps)
        )

        return {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "nrmse": nrmse.item(),
            "vrmse": vrmse.item(),
            "frmse_low": frmse_low.item(),
            "frmse_med": frmse_med.item(),
            "frmse_high": frmse_high.item(),
            "energy_spectrum_error": energy_spectrum_error.item(),
        }