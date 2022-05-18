import torch as pt


from ..constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE
from ..mps import MPS
from .tn_contractions import contr_3, contr_4, contr_5
from ..rbm_wave_function import RBMWaveFunction


# The main function used to calculate quantum chemical energies
def energy_5(ham_mpo=None,
             wf: MPS = None,
             alpha_mpo=None,
             beta_mpo=None,
             hf_mps=None,
             hf_ham_mps=None,
             work_device=None,
             cpu_device=None):
    assert work_device is not None
    assert cpu_device is not None

    if hf_mps is None:
        assert hf_ham_mps is None
        num = contr_5(bra=wf.conj(),
                      top=alpha_mpo,
                      mid=ham_mpo,
                      bot=beta_mpo,
                      ket=wf,
                      work_device=work_device,
                      cpu_device=cpu_device)
        denom = contr_4(bra=wf.conj(),
                        ham=alpha_mpo,
                        sym=beta_mpo,
                        ket=wf,
                        work_device=work_device,
                        cpu_device=cpu_device)
    else:
        #####################################################
        # Step 1, renormalise the MPS
        assert hf_ham_mps is not None
        wf = wf.clone()
        wf_norm = wf.norm()
        for idx in range(wf.visible_num):
            wf.tensors[idx] = wf.tensors[idx] / pt.pow(wf_norm, 1 / wf.visible_num)

        #####################################################
        # Step 2, calculate the numerator consisting of three terms
        num_hf = contr_3(bra=hf_mps.conj(),
                         mpo=ham_mpo,
                         ket=hf_mps,
                         work_device=work_device,
                         cpu_device=cpu_device)

        num_overlap = contr_4(ket=hf_ham_mps,
                              ham=alpha_mpo,
                              sym=beta_mpo,
                              bra=wf.conj(),
                              work_device=work_device,
                              cpu_device=cpu_device)

        num_anti_hf = contr_5(bra=wf.conj(),
                              top=alpha_mpo,
                              mid=ham_mpo,
                              bot=beta_mpo,
                              ket=wf,
                              work_device=work_device,
                              cpu_device=cpu_device)

        #####################################################
        # Step 3, calculate the denominator also consisting of three terms
        denom_hf = hf_mps.norm()

        denom_overlap = contr_4(bra=wf.conj(),
                                ham=alpha_mpo,
                                sym=beta_mpo,
                                ket=hf_mps,
                                work_device=work_device,
                                cpu_device=cpu_device)

        denom_anti_hf = contr_4(bra=wf.conj(),
                                ham=alpha_mpo,
                                sym=beta_mpo,
                                ket=wf,
                                work_device=work_device,
                                cpu_device=cpu_device)

        num = num_hf + num_overlap + num_overlap.conj() + num_anti_hf
        denom = denom_hf + denom_overlap + denom_overlap.conj() + denom_anti_hf

    loss = num / denom

    if wf.dtype == BASE_REAL_TYPE:
        return loss
    else:
        return loss.real


# An analogue of the previous function relying instead on a 4-layer contraction (because
# so far it is impossible to use energy_5 for RBMs)
def energy_4(ham_mpo=None,
             wf: MPS = None,
             sym_mpo=None,
             hf_mps=None,
             hf_ham_mps=None,
             work_device=None,
             cpu_device=None):
    assert work_device is not None
    assert cpu_device is not None

    if hf_mps is None:
        num = contr_4(bra=wf.conj(),
                      ham=ham_mpo,
                      sym=sym_mpo,
                      ket=wf,
                      work_device=work_device,
                      cpu_device=cpu_device)
        denom = contr_3(bra=wf.conj(),
                        mpo=sym_mpo,
                        ket=wf,
                        work_device=work_device,
                        cpu_device=cpu_device)
    else:
        #####################################################
        # Step 1, renormalise the MPS
        assert hf_ham_mps is not None
        wf = wf.clone()
        wf_norm = wf.norm()
        for idx in range(wf.visible_num):
            wf.tensors[idx] = wf.tensors[idx] / pt.pow(wf_norm, 1 / wf.visible_num)

        #####################################################
        # Step 2, calculate the numerator consisting of three terms
        num_hf = contr_3(bra=hf_mps.conj(),
                         mpo=ham_mpo,
                         ket=hf_mps,
                         work_device=work_device,
                         cpu_device=cpu_device)

        num_overlap = contr_3(ket=hf_ham_mps,
                              mpo=sym_mpo,
                              bra=wf.conj(),
                              work_device=work_device,
                              cpu_device=cpu_device)

        num_anti_hf = contr_4(bra=wf.conj(),
                              ham=ham_mpo,
                              sym=sym_mpo,
                              ket=wf,
                              work_device=work_device,
                              cpu_device=cpu_device)

        #####################################################
        # Step 3, calculate the denominator also consisting of three terms
        denom_hf = hf_mps.norm()

        denom_overlap = contr_3(bra=wf.conj(),
                                mpo=sym_mpo,
                                ket=hf_mps,
                                work_device=work_device,
                                cpu_device=cpu_device)

        denom_anti_hf = contr_3(bra=wf.conj(),
                                mpo=sym_mpo,
                                ket=wf,
                                work_device=work_device,
                                cpu_device=cpu_device)

        num = num_hf + num_overlap + num_overlap.conj() + num_anti_hf
        denom = denom_hf + denom_overlap + denom_overlap.conj() + denom_anti_hf

    loss = num / denom

    if wf.dtype == BASE_REAL_TYPE:
        return loss
    else:
        return loss.real


def split_energy_5(ham_mpos=None,
                   wf: MPS = None,
                   alpha_mpo=None,
                   beta_mpo=None,
                   hf_mps=None,
                   hf_ham_mpses=None,
                   work_device=None,
                   cpu_device=None):
    result = 0.0
    for split_idx in range(len(ham_mpos)):
        loss = energy_5(ham_mpo=ham_mpos[split_idx],
                        wf=wf,
                        alpha_mpo=alpha_mpo,
                        beta_mpo=beta_mpo,
                        hf_mps=hf_mps,
                        hf_ham_mps=hf_ham_mpses[split_idx] if hf_mps is not None else None,
                        work_device=work_device,
                        cpu_device=cpu_device)
        if loss.requires_grad:
            loss.backward()

        result += loss.detach()

    return result


# Legacy functions which can still be used to calculate energies (no support for anti-HF bias though)
# (even though I could spend some time to make one anti-HF oriented functions, just taking underlying
# contractor lambda as as parameter
def energy_3(ham_mpo=None,
             wf: MPS = None,
             work_device=None,
             cpu_device=None):
    num = contr_3(bra=wf.conj(),
                  mpo=ham_mpo,
                  ket=wf,
                  work_device=work_device,
                  cpu_device=cpu_device)
    denom = MPS.overlap(bra=wf.conj(), ket=wf)

    loss = num / denom

    if wf.dtype == BASE_REAL_TYPE:
        return loss
    else:
        return loss.real


def bf_energy(ham_pt: pt.Tensor = None,
              space_idx: pt.Tensor = None,
              wf: RBMWaveFunction = None,
              anti_hf: bool = False,
              hf_idx: int = None,
              renorm: bool = False):
    wf_amps = wf.amplitude(space_idx)
    if anti_hf is False:
        num = pt.einsum('i,ij,j',
                        pt.conj(wf_amps),
                        ham_pt,
                        wf_amps)
        denom = pt.dot(pt.conj(wf_amps), wf_amps)
    else:
        assert hf_idx is not None
        hf_amps = pt.zeros_like(wf_amps)

        hf_pos = (space_idx == hf_idx)
        hf_amps[hf_pos] = 1.0
        wf_amps[hf_pos] = 0.0
        wf_norm = wf.norm_bf()
        wf_amps = wf_amps / wf_norm

        num_hf = pt.squeeze(ham_pt[hf_pos, hf_pos])
        num_cross = pt.dot(pt.squeeze(ham_pt[hf_pos, :]), wf_amps)
        num_wf = pt.einsum('i,ij,j',
                           pt.conj(wf_amps),
                           ham_pt,
                           wf_amps)
        num = num_hf + num_cross + pt.conj(num_cross) + num_wf

        denom_hf = pt.dot(pt.conj(hf_amps), hf_amps)
        denom_cross = pt.dot(pt.conj(hf_amps), wf_amps)
        denom_wf = pt.dot(pt.conj(wf_amps), wf_amps)
        denom = denom_hf + denom_cross + pt.conj(denom_cross) + denom_wf

    loss = num / denom
    
    if wf.dtype == BASE_REAL_TYPE:
        return loss
    else:
        return loss.real
