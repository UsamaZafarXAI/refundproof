def _extract_features_worker(args):
    aid, gen, real_path, edit_path = args
    try:
        import cv2, numpy as np
        from PIL import Image, ImageOps
        from skimage.feature import local_binary_pattern as _lbp
        import pandas as pd
        RESIZE = (384, 384)
        def _load(p): return np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB").resize(RESIZE, Image.LANCZOS), dtype=np.uint8)
        def _gray(a): return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        def _blur_var(g): return float(cv2.Laplacian(g.astype(np.float64), cv2.CV_64F).var())
        def _bright(g): return float(g.mean())
        def _contrast(g): return float(g.std())
        def _noise_sigma(g):
            lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
            return float(np.median(np.abs(lap - np.median(lap))) / 0.6745)
        def _lbp_entropy(g):
            lbp = _lbp(g, P=8, R=1, method="uniform")
            h, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True); h = h + 1e-12
            return float(-(h * np.log(h)).sum())
        def _noise_residual_var(g):
            k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) / 8.0
            return float(cv2.filter2D(g.astype(np.float32), -1, k).var())
        def _dct_skewness(g):
            H, W = g.shape; H8, W8 = (H//8)*8, (W//8)*8
            g8 = g[:H8, :W8].astype(np.float32)
            blks = g8.reshape(H8//8, 8, W8//8, 8).transpose(0, 2, 1, 3)
            vals = [cv2.dct(blks[i, j]).flatten()
                    for i in range(blks.shape[0]) for j in range(blks.shape[1])]
            return float(pd.Series(np.concatenate(vals)).skew())
        def _dct_stats(g):
            H, W = g.shape; H8, W8 = (H//8)*8, (W//8)*8
            g8 = g[:H8, :W8].astype(np.float32)
            blks = g8.reshape(H8//8, 8, W8//8, 8).transpose(0, 2, 1, 3)
            db = np.zeros_like(blks)
            for i in range(blks.shape[0]):
                for j in range(blks.shape[1]): db[i, j] = cv2.dct(blks[i, j])
            dc = db[:,:,0,0].flatten(); ac = db[:,:,1:,1:].flatten(); eps = 1e-8
            return {"dct_ac_energy": float(np.mean(ac**2)), "dct_dc_std": float(dc.std()),
                    "dct_hf_kurtosis": float(pd.Series(ac).kurt()),
                    "dct_energy_ratio": float(np.mean(ac**2) / (np.mean(dc**2) + eps))}
        def _fft_stats(g):
            f = np.fft.fftshift(np.fft.fft2(g.astype(np.float32))); mag = np.abs(f) + 1e-8
            H, W = mag.shape; cy, cx = H//2, W//2; Y, X = np.ogrid[:H, :W]
            R = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(int); max_r = min(cx, cy)
            rad = np.array([mag[R==r].mean() for r in range(1, max_r)])
            log_r = np.log(np.arange(1, len(rad)+1) + 1e-8); log_m = np.log(rad + 1e-8)
            lo = max(2, len(log_r)//6); hi = int(len(log_r)*0.75)
            slope, _ = np.polyfit(log_r[lo:hi], log_m[lo:hi], 1)
            return {"fft_slope": float(slope), "fft_hf_ratio": float(rad[hi:].mean() / (rad[lo:hi].mean() + 1e-8))}
        def _color_coh(arr):
            ccs = []
            for ch in range(3):
                chi = arr[:,:,ch]
                _, binary = cv2.threshold(chi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                nlabels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                if nlabels < 2: ccs.append(0.5); continue
                areas = stats[1:, cv2.CC_STAT_AREA]
                ccs.append(float(areas[areas > 50].sum() / (areas.sum() + 1e-8)))
            return float(np.mean(ccs))
        def _grad_stats(g):
            gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            gm = np.sqrt(gx**2 + gy**2); h, _ = np.histogram(gm, bins=64, density=True); h = h + 1e-12
            return {"grad_mean": float(gm.mean()), "grad_entropy": float(-(h * np.log(h)).sum())}
        def _ssim(gr, ge):
            from skimage.metrics import structural_similarity
            return float(structural_similarity(gr, ge, data_range=255))
        def _edge_pair(gr, ge):
            def _e(g): return cv2.Canny(cv2.GaussianBlur(g, (5,5), 0), 80, 160) > 0
            er, ee = _e(gr), _e(ge); inter = (er & ee).sum(); union = (er | ee).sum()
            return (float(inter/union) if union > 0 else 1., float(er.mean()), float(ee.mean()))

        ar = _load(real_path); ae = _load(edit_path)
        gr, ge = _gray(ar), _gray(ae)
        dr = _dct_stats(gr); fr = _fft_stats(gr); gr_s = _grad_stats(gr)
        de = _dct_stats(ge); fe = _fft_stats(ge); ge_s = _grad_stats(ge)
        jac, ed_r, ed_e = _edge_pair(gr, ge)
        return {"ok": True, "artifact_id": aid, "generator": gen,
                "real_blur_var": _blur_var(gr), "real_brightness": _bright(gr),
                "real_contrast": _contrast(gr), "real_noise_sigma": _noise_sigma(gr),
                "real_lbp_entropy": _lbp_entropy(gr),
                "real_dct_ac_energy": dr["dct_ac_energy"], "real_dct_dc_std": dr["dct_dc_std"],
                "real_dct_hf_kurtosis": dr["dct_hf_kurtosis"], "real_dct_energy_ratio": dr["dct_energy_ratio"],
                "real_fft_slope": fr["fft_slope"], "real_fft_hf_ratio": fr["fft_hf_ratio"],
                "real_cc_mean": _color_coh(ar), "real_grad_mean": gr_s["grad_mean"],
                "real_grad_entropy": gr_s["grad_entropy"],
                "real_noise_residual_var": _noise_residual_var(gr),
                "real_dct_skewness": _dct_skewness(gr),
                "edit_blur_var": _blur_var(ge), "edit_brightness": _bright(ge),
                "edit_contrast": _contrast(ge), "edit_noise_sigma": _noise_sigma(ge),
                "edit_lbp_entropy": _lbp_entropy(ge),
                "edit_dct_ac_energy": de["dct_ac_energy"], "edit_dct_dc_std": de["dct_dc_std"],
                "edit_dct_hf_kurtosis": de["dct_hf_kurtosis"], "edit_dct_energy_ratio": de["dct_energy_ratio"],
                "edit_fft_slope": fe["fft_slope"], "edit_fft_hf_ratio": fe["fft_hf_ratio"],
                "edit_cc_mean": _color_coh(ae), "edit_grad_mean": ge_s["grad_mean"],
                "edit_grad_entropy": ge_s["grad_entropy"],
                "edit_noise_residual_var": _noise_residual_var(ge),
                "edit_dct_skewness": _dct_skewness(ge),
                "ssim": _ssim(gr, ge), "edge_jaccard": jac,
                "edge_density_real": ed_r, "edge_density_edit": ed_e,
                "edge_density_delta": float(ed_e - ed_r), "clip_sim": 0.0}
    except Exception as exc:
        return {"ok": False, "artifact_id": aid, "generator": gen, "error": str(exc)[:150]}
