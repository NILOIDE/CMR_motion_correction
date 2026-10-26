    def ncc_lines(self, theta_deg, t, use_gradient: bool, prof=None) -> torch.Tensor:
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        return ncc(a, b, self.W[:, None, :], use_gradient).mean(1)             # (M,)

    def mi_lines(self, theta_deg, t, bins: int, sigma_bins: float, prof=None) -> torch.Tensor:
        """Normalized mutual information per line, samples pooled over frames -> (M,)."""
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "img")
        w = self.W[:, None, :].expand_as(a)
        return soft_nmi(a.reshape(self.M, -1), b.reshape(self.M, -1), w.reshape(self.M, -1), bins, sigma_bins)

    def similarity_lines(self, theta_deg, t, args, prof=None) -> torch.Tensor:
        """The optimized similarity per line (NCC, gradient NCC or normalized MI)."""
        if args.metric == "mi":
            return self.mi_lines(theta_deg, t, args.mi_bins, args.mi_sigma, prof)
        return self.ncc_lines(theta_deg, t, args.metric == "ncc_grad", prof)

    def dice_lines(self, theta_deg, t, labels: list[int], labels_sax: list[int], prof=None) -> torch.Tensor:
        """Mean Dice over the counted labels (see counted_labels) along each line -> (M,).

        A counted label that the two profiles place at non-overlapping positions -- including the case where only
        one profile hits it -- scores 0. Per frame: mean over counted labels; then mean over frames with at least
        one counted label. NaN if no label is counted on any frame. `labels` applies to LAX-LAX lines,
        `labels_sax` to SAX-LAX lines (the SAX model lacks some classes).
        """
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "seg")
        total = torch.zeros(a.shape[:2], device=self.device)
        count = torch.zeros(a.shape[:2], device=self.device)
        for lab, counted, ma, mb in self.counted_labels(a, b, labels, labels_sax):
            inter = (ma & mb).sum(-1).float()
            tot = (ma.sum(-1) + mb.sum(-1)).float()
            dice = 2 * inter / tot.clamp_min(1)
            total += torch.where(counted, dice, torch.zeros_like(dice))
            count += counted.float()
        valid = count > 0
        per_frame = total / count.clamp_min(1)
        out = (per_frame * valid).sum(1) / valid.sum(1).clamp_min(1)
        return torch.where(valid.any(1), out, torch.full_like(out, float("nan")))

    def hd_lines(self, theta_deg, t, labels: list[int], labels_sax: list[int], miss_mm: float = 100.0, prof=None) -> torch.Tensor:
        """1D Hausdorff distance (mm) between the two views' labels along each line -> (M,).

        Counted labels as in dice_lines. If only one profile hits a counted label the distance is undefined
        (infinite); it is reported as `miss_mm`. Per frame: the worst (largest) value over the counted labels;
        then the mean over frames with at least one counted label. NaN if no label is counted on any frame.
        """
        a, b = prof if prof is not None else self.profiles(theta_deg, t, "seg")
        if getattr(self, "_dist", None) is None:                                 # (M, 1, L, L) mm along the line
            self._dist = (self.T[:, :, None] - self.T[:, None, :]).abs()[:, None]
        dist = self._dist
        inf = torch.tensor(float("inf"), device=self.device)
        worst = torch.full(a.shape[:2], -1.0, device=self.device)               # (M, F)
        for lab, counted, ma, mb in self.counted_labels(a, b, labels, labels_sax):
            both = ma.any(-1) & mb.any(-1)
            d_ab = torch.where(mb[:, :, None, :], dist, inf).amin(-1)            # each A point to nearest B point
            d_ba = torch.where(ma[:, :, :, None], dist, inf).amin(-2)            # each B point to nearest A point
            h = torch.maximum(torch.where(ma, d_ab, -inf).amax(-1), torch.where(mb, d_ba, -inf).amax(-1))
            h = torch.where(both, h, torch.full_like(h, miss_mm))
            worst = torch.where(counted, torch.maximum(worst, h), worst)
        ok = worst >= 0
        out = (worst.clamp_min(0) * ok).sum(1) / ok.sum(1).clamp_min(1)
        return torch.where(ok.any(1), out, torch.full_like(out, float("nan")))

    def per_plane(self, values: torch.Tensor) -> list[float]:
        """Line values (M,) -> npts-weighted mean per plane, ignoring NaN (one host transfer)."""
        ok = torch.isfinite(values)
        v = torch.where(ok, values, torch.zeros_like(values)) * self.npts
        w = self.npts * ok
        num = torch.zeros(self.N, device=self.device).index_add_(0, self.IA, v).index_add_(0, self.IB, v)
        den = torch.zeros(self.N, device=self.device).index_add_(0, self.IA, w).index_add_(0, self.IB, w)
        return torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(num, float("nan"))).tolist()

    def per_plane_max(self, values: torch.Tensor) -> list[float]:
        """Line values (M,) -> worst (largest) value per plane, ignoring NaN (one host transfer)."""
        v = torch.where(torch.isfinite(values), values, torch.full_like(values, -float("inf")))
        out = torch.full((self.N,), -float("inf"), device=self.device)
        out = out.scatter_reduce(0, self.IA, v, "amax").scatter_reduce(0, self.IB, v, "amax")
        return torch.where(torch.isfinite(out), out, torch.full_like(out, float("nan"))).tolist()

    @torch.no_grad()
    def evaluate(self, theta_deg, t, args, labels, labels_sax, has_seg: bool) -> dict:
        """All reported per-line metrics at the given poses, as tensors of shape (M,).

        The image and segmentation profiles are sampled once each and shared by the metrics.
        """
        nan = torch.full((self.M,), float("nan"), device=self.device)
        img = self.profiles(theta_deg, t, "img")
        seg = self.profiles(theta_deg, t, "seg") if has_seg else None
        out = dict(ncc=self.ncc_lines(theta_deg, t, args.metric == "ncc_grad", img),
                   mi=self.mi_lines(theta_deg, t, args.mi_bins, args.mi_sigma, img),
                   dice=self.dice_lines(theta_deg, t, labels, labels_sax, seg) if has_seg else nan,
                   hd=self.hd_lines(theta_deg, t, labels, labels_sax, getattr(args, "hd_miss_mm", 100.0), seg) if has_seg else nan)
        return {k: v.float() for k, v in out.items()}


