from __future__ import annotations

import numpy as np

from seoct_uncertainty import single_reflector_localization_crlb, two_reflector_separation_crlb


def test_localization_improves_with_noise_and_bandwidth() -> None:
    wide=np.linspace(10,11,64); narrow=np.linspace(10.45,10.55,64)
    high=single_reflector_localization_crlb(wide,1+.2j,.7,np.eye(64)*1e-2)
    low=single_reflector_localization_crlb(wide,1+.2j,.7,np.eye(64)*1e-3)
    narrow_result=single_reflector_localization_crlb(narrow,1+.2j,.7,np.eye(64)*1e-3)
    assert low.z_standard_deviation < high.z_standard_deviation
    assert low.z_standard_deviation < narrow_result.z_standard_deviation


def test_two_reflector_separation_is_ill_conditioned_when_close() -> None:
    k=np.linspace(10,11,64); covariance=np.eye(64)*1e-3
    close=two_reflector_separation_crlb(k,1,.8*np.exp(.3j),.5,.02,covariance)
    separated=two_reflector_separation_crlb(k,1,.8*np.exp(.3j),.5,.8,covariance)
    assert close.condition_number > separated.condition_number
    assert close.separation_standard_deviation > separated.separation_standard_deviation
