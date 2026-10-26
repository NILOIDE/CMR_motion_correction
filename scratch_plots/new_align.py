def align(final_sampler: LineSampler, args, stages: list[tuple[int, LineSampler]] | None = None) -> dict:
    """Jointly optimize a 6-DOF rigid pose per plane from the acquired geometry.

    `stages` optionally lists (start_step, sampler) for the annealed line/mask geometry used early in the run, ending
    with (anneal_end, final_sampler); `final_sampler` -- the nominal geometry -- is used from anneal_end on (or always,
    if no stages are given). The best pose is only tracked once every annealed quantity (geometry, blur, PSF width) is
    at its nominal value, on the full (all lines) objective.

    Each optimization step (objective, backward, best-pose bookkeeping, Adam update) is captured once per geometry
    stage as a CUDA graph and replayed, which removes the per-step Python/kernel-launch overhead; all per-step inputs
    (learning rates, blur, PSF scale, line-dropping mask, step counter) live in device buffers that are updated in
    place before each replay. --no-cuda-graph runs the identical step eagerly.
    """
    device, N = final_sampler.device, final_sampler.N
    stages = sorted(stages or [(0, final_sampler)], key=lambda s: s[0])
    anneal_end = stages[-1][0]

    def sampler_at(step):
        active = stages[0][1]
        for start, smp in stages:
            if step >= start:
                active = smp
        return active

    f32 = dict(dtype=torch.float32, device=device)
    rotate = args.dof == 6 and args.rot_lr > 0
    theta_deg = torch.zeros(N, 3, **f32).requires_grad_(rotate)
    t = torch.zeros(N, 3, **f32).requires_grad_(True)

    def objective(theta_deg, t, keep=None, smp=final_sampler):
        c = smp.similarity_lines(theta_deg, t, args)
        w = smp.npts if keep is None else smp.npts * keep
        score = (c * w).sum() / w.sum()
        gauge = args.gauge * (t.mean(0).pow(2).sum() + theta_deg.mean(0).pow(2).sum())
        dz = (t * smp.Nrm).sum(1)
        prior = args.prior * (theta_deg.pow(2).sum(1) + dz.pow(2)).mean()
        return score - gauge - prior

    # stochastic line-pair dropping: each step keeps a random subset of the intersection lines (at least one)
    keep_frac = getattr(args, "pair_keep", 1.0)
    stochastic = keep_frac < 1.0
    gen = torch.Generator(device=device)
    gen.manual_seed(getattr(args, "seed", 0))
    eval_every = max(1, getattr(args, "eval_every", 10))

    def draw_keep(smp):
        r = torch.rand(smp.M, generator=gen, device=device)
        return ((r < keep_frac) | (r == r.min())).float()

    # learning rates as device tensors (updated in place each step; captured graphs read them)
    lr_t = torch.tensor(float(args.lr), **f32)
    lr_r = torch.tensor(float(args.rot_lr), **f32)
    param_groups = [{"params": [t], "lr": lr_t}] + ([{"params": [theta_deg], "lr": lr_r}] if rotate else [])
    opt = torch.optim.Adam(param_groups, capturable=device.type == "cuda")

    def lr_at(step, start, end):
        if args.lr_schedule == "constant" or start <= 0:
            return start
        return start + (end - start) * min(1.0, step / max(1, args.steps - 1))        # linear start -> end

    blur_sigma = getattr(args, "blur_sigma", 0.0)
    blur_steps = int(round(getattr(args, "blur_frac", 0.6) * args.steps)) if blur_sigma > 0 else 0
    blur_quantum = 0.2                                     # mm; the blur is only recomputed when sigma moves this much
    psf_s0 = getattr(args, "psf_scale_start", 1.0)
    psf_steps = int(round(getattr(args, "psf_anneal_frac", 0.6) * args.steps)) if psf_s0 > 1.0 else 0
    track_from = max(anneal_end, blur_steps, psf_steps)    # from here on the objective is the nominal one

    # device-side bookkeeping: best pose so far (no host sync per step)
    best_obj = torch.tensor(-float("inf"), **f32)
    best_th, best_t = theta_deg.detach().clone(), t.detach().clone()
    best_step = torch.zeros((), **f32)
    step_buf, track_buf = torch.zeros((), **f32), torch.zeros((), **f32)
    keep_bufs = {id(smp): torch.ones(smp.M, **f32) for _, smp in stages}

    def update_best(o):
        better = (o > best_obj) & (track_buf > 0)
        best_obj.copy_(torch.where(better, o, best_obj))
        best_th.copy_(torch.where(better, theta_deg.detach(), best_th))
        best_t.copy_(torch.where(better, t.detach(), best_t))
        best_step.copy_(torch.where(better, step_buf, best_step))

    def body(smp):
        opt.zero_grad(set_to_none=False)
        obj = objective(theta_deg, t, keep_bufs[id(smp)], smp)
        (-obj).backward()
        update_best(obj.detach())                          # pose *before* this update, as before
        opt.step()

    use_graph = device.type == "cuda" and not getattr(args, "no_cuda_graph", False)
    graphs = {}

    def capture(smp):
        """Capture body(smp) as a CUDA graph. Warm-up iterations run for real, so all state is restored after."""
        params = [p for g in param_groups for p in g["params"]]
        saved_p = [p.detach().clone() for p in params]
        saved_state = {i: {k: v.clone() for k, v in opt.state[p].items()} for i, p in enumerate(params) if p in opt.state}
        saved_book = [x.clone() for x in (best_obj, best_th, best_t, best_step)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                body(smp)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body(smp)
        with torch.no_grad():
            for p, v in zip(params, saved_p):
                p.copy_(v)
            for i, p in enumerate(params):
                for k, v in opt.state[p].items():
                    if i in saved_state:
                        v.copy_(saved_state[i][k])
                    else:
                        v.zero_()
            for x, v in zip((best_obj, best_th, best_t, best_step), saved_book):
                x.copy_(v)
        return graph

    def run_step(smp):
        nonlocal use_graph
        if use_graph:
            graph = graphs.get(id(smp))
            if graph is None:
                try:
                    graph = graphs[id(smp)] = capture(smp)
                except Exception as e:  # noqa: BLE001
                    print(f"CUDA graph capture failed ({type(e).__name__}: {e}); continuing without graphs")
                    use_graph = False
            if use_graph:
                graph.replay()
                return
        body(smp)

    every = getattr(args, "history_every", 0.0)
    snap_steps = sorted({int(round(k * every * args.steps)) for k in range(int(round(1 / every)) + 1)}) if every > 0 else []
    snapshots = []
    with torch.enable_grad():
        for step in range(args.steps):
            if step in snap_steps:
                snapshots.append((step, theta_deg.detach().clone(), t.detach().clone()))
            smp = sampler_at(step)
            sigma = blur_sigma * max(0.0, 1.0 - step / blur_steps) if step < blur_steps else 0.0
            smp.set_blur(math.ceil(sigma / blur_quantum) * blur_quantum if sigma > 0 else 0.0)
            smp.set_psf_scale(psf_s0 + (1.0 - psf_s0) * step / psf_steps if step < psf_steps else 1.0)
            lr_t.fill_(lr_at(step, args.lr, args.lr_end))
            lr_r.fill_(lr_at(step, args.rot_lr, args.rot_lr_end))
            nominal = step >= track_from
            step_buf.fill_(float(step))
            if stochastic:
                keep_bufs[id(smp)].copy_(draw_keep(smp))
                # the dropped-line objective is not comparable between steps: track the best periodically on the
                # full objective (eagerly, before the update), and never inside the graph
                if nominal and (step % eval_every == 0 or step == args.steps - 1):
                    track_buf.fill_(1.0)
                    with torch.no_grad():
                        update_best(objective(theta_deg, t))
                track_buf.fill_(0.0)
            else:
                track_buf.fill_(1.0 if nominal else 0.0)
            run_step(smp)
    for _, smp in stages:
        smp.set_blur(0.0)
        smp.set_psf_scale(1.0)
    if args.steps in snap_steps:
        snapshots.append((args.steps, theta_deg.detach().clone(), t.detach().clone()))
    with torch.no_grad():
        if stochastic and track_from <= args.steps:        # the final parameters are also a candidate
            step_buf.fill_(float(args.steps))
            track_buf.fill_(1.0)
            update_best(objective(theta_deg, t))
    return dict(theta_deg=best_th.clone(), t=best_t.clone(), best_step=int(best_step.item()), snapshots=snapshots)


