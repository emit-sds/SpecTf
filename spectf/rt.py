#!/usr/bin/env python3
#
# author : william.r.keely@jpl.nasa.gov
#
import h5py
import numpy as np
from typing import Tuple
from isofit.core.common import VectorInterpolator


def spectral_response_function(response_range, mu, sigma):
    u = (response_range - mu) / abs(sigma)
    y = (1.0 / (np.sqrt(2.0 * np.pi) * abs(sigma))) * np.exp(-u * u / 2.0)
    return y / y.sum()


def build_H(wl1, wl2, fwhm2):
    H = np.array(
        [spectral_response_function(wl1, wi, fwhmi / 2.355)
         for wi, fwhmi in zip(wl2, fwhm2)],
        dtype=np.float32,
    )
    H[np.isfinite(H) == False] = 0
    return H

"""Precomputing H reduces mat ops during pixel injection"""
def resample_with_H(x, H):
    x[np.isfinite(x) == False] = 0
    if x.ndim > 1:
        return np.dot(H, x.T).T
    return np.dot(H, x.reshape(-1, 1)).ravel() 



class RadiativeTransferEngine:
    """
    Inject CH4 into background radiance, with optional 0.1 nm wavelength shift.
    """

    def __init__(
        self,
        lut_path: str  = "/store/wkeely/methane/luts/0-500/lut_0-500.nc",
        emit_wl_path: str = "/store/wkeely/methane/wl_emit.npz",
        *,
        ppmm_range: Tuple[float, float] = (0.0, 3000.0),   # ppmm
        version: str = "mlg",
        sigma_nm: float = 0.05,    # 1-Sigma 0.05 shift in nm
        clip_nm:  float = 0.1,
        shift: bool=True,    
    ):
        with h5py.File(lut_path, "r") as lut:
            self.wl_lut = lut["wl"][:]

            self.transm_down = np.moveaxis(lut["transm_down_dif"][:], 0, -1)
            self.transm_up = np.moveaxis(lut["transm_up_dir"][:], 0, -1)
            self.rhoatm = np.moveaxis(lut["rhoatm"][:], 0, -1)
            self.solar_irr = lut["solar_irr"][:]

            grid_input = [
                lut["AERFRAC_2"][:],
                lut["CH4"][:],
                lut["H2OSTR"][:],
                lut["observer_zenith"][:],
                lut["surface_elevation_km"][:],
            ]

        self.interp_down = VectorInterpolator(grid_input, self.transm_down, version=version)
        self.interp_up  = VectorInterpolator(grid_input, self.transm_up,   version=version)
        self.interp_rho  = VectorInterpolator(grid_input, self.rhoatm,      version=version)

        wl_dat = np.load(emit_wl_path)
        self.wl_emit = wl_dat["wl_emit"]
        self.fwhm_emit  = wl_dat["fwhm_emit"]

        self.H = build_H(self.wl_lut, self.wl_emit, self.fwhm_emit).astype(np.float32)
        self.solar_irr_resamp = resample_with_H(self.solar_irr, self.H).astype(np.float32)

        self.ppmm_lo, self.ppmm_hi = ppmm_range[0] / 1000.0, ppmm_range[1] / 1000.0
        self.sigma_nm = sigma_nm
        self.clip_nm = clip_nm
        self.rng = np.random.default_rng()
        self.shift = shift

    def _shift_batch(self, spectra: np.ndarray, shift_nm: np.ndarray) -> np.ndarray:
        """
        Interpolates each rdn onto emi wl +/- delta_shift.
        """
        B, _ = spectra.shape
        out  = np.empty_like(spectra, dtype=np.float32)
        for i in range(B):
            wl_target = self.wl_emit - shift_nm[i]
            out[i] = np.interp(wl_target,
                               self.wl_emit,
                               spectra[i],
                               left=spectra[i, 0],
                               right=spectra[i, -1])
        return out.astype(np.float32)
    
    def _shift_batch_pt_one(self, spectra: np.ndarray):
        """
        Interpolates each rdn onto emi wl +/-0.1 nm or 0 nm.
        """
        B, _ = spectra.shape
        shift_nm = self.rng.choice(np.array([-0.1, 0.0, 0.1], dtype=np.float32), size=B)
        out = np.empty_like(spectra, dtype=np.float32)
        for i in range(B):
            wl_target = self.wl_emit - shift_nm[i]
            out[i] = np.interp(
                wl_target,
                self.wl_emit,
                spectra[i],
                left=spectra[i, 0],
                right=spectra[i, -1],
            )
        return out.astype(np.float32), shift_nm

    def inject(
        self,
        rdn: np.ndarray,   # (B, 285)
        atm: np.ndarray,   # (B, 3)
        loc: np.ndarray,   # (B, 3)
        obs: np.ndarray,   # (B, 5)
    ):
        """
        Inject CH4 into background radiances and convert to TOA reflectance.
        Optional wl shift of 0.10 nm (2Sigma)

        Returns: inj_rdn, rdn, inj_toa, bg_toa, ch4_col, shift_nm
        """
        B, _ = rdn.shape

        if self.shift:
            shift_nm = self.rng.normal(0.0, self.sigma_nm, size=B).astype(np.float32)
            np.clip(shift_nm, -self.clip_nm, self.clip_nm, out=shift_nm)

            rdn = self._shift_batch(rdn, shift_nm)          # shifted spectra
        # if self.shift:
        #     rdn, shift_nm = self._shift_batch_pt_one(rdn)

        # ch4_col = np.full(B, 1.5, dtype=np.float32)      # shape (B,)
        ch4_col = self.rng.uniform(self.ppmm_lo, self.ppmm_hi, size=B).astype(np.float32)


        AERFRAC_2       = atm[:, 0]
        H2OSTR          = atm[:, 2]
        elev_km         = loc[:, 2] / 1_000.0
        observer_zen    = 180.0 - obs[:, 2]
        solzen          = obs[:, 4]
        cosz            = np.cos(np.deg2rad(solzen))

        inj_rdn = rdn.astype(np.float32, copy=True)
        inj_toa = np.empty_like(rdn, dtype=np.float32)

        for i in range(B):
            if cosz[i] <= 0.0 or np.isnan(cosz[i]):
                ch4_col[i] = 0.0
                inj_toa[i] = (inj_rdn[i] * np.pi / self.solar_irr_resamp) / np.maximum(cosz[i], 1e-6)
                continue

            bg = np.array([AERFRAC_2[i], 0.0,       H2OSTR[i],
                           observer_zen[i], elev_km[i]], dtype=float)
            fg = bg.copy(); fg[1] = ch4_col[i]

            Lp_bg = self.interp_rho(bg) * self.solar_irr * cosz[i] / np.pi
            T_bg = self.interp_down(bg) * self.interp_up(bg)
            Lp_bg = resample_with_H(Lp_bg, self.H)
            T_bg = resample_with_H(T_bg, self.H)
            L_s = (rdn[i] - Lp_bg) / T_bg

            Lp_fg = self.interp_rho(fg) * self.solar_irr * cosz[i] / np.pi
            T_fg = self.interp_down(fg) * self.interp_up(fg)
            Lp_fg = resample_with_H(Lp_fg, self.H)
            T_fg = resample_with_H(T_fg,self.H)

            inj_rdn[i] = Lp_fg + L_s * T_fg
            inj_toa[i] = (inj_rdn[i] * np.pi / self.solar_irr_resamp) / cosz[i]
            bg_toa = (rdn[i] * np.pi / self.solar_irr_resamp) / cosz[i]

        return inj_rdn, rdn, inj_toa, bg_toa, ch4_col, shift_nm

    __call__ = inject
