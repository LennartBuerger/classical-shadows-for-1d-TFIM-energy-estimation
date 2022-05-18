import torch as pt


from ..mps import MPS
from ..svd import robust_qr


# A function to contract ket, bra an an MPO — it is more efficient
# than simply merging MPO into ket, and then contracting two MPSes
def contr_3(*,
            bra: MPS = None,
            mpo=None,
            ket: MPS = None,
            work_device=None,
            cpu_device=None):
    assert ket.visible_num == bra.visible_num
    assert ket.visible_num == len(mpo)

    assert work_device is not None
    assert cpu_device is not None

    top = [tensor for tensor in ket.tensors]
    mid = [tensor for tensor in mpo]
    bot = [tensor for tensor in bra.tensors]

    top[0] = pt.squeeze(top[0], dim=0)
    mid[0] = pt.squeeze(mid[0], dim=0)
    bot[0] = pt.squeeze(bot[0], dim=0)

    for idx in range(ket.visible_num - 1):
        mid[idx] = pt.einsum('ai,bja->bji',
                             top[idx],
                             mid[idx].to(work_device))
        top[idx + 1] = pt.reshape(top[idx + 1], (-1, top[idx + 1].shape[-1]))

        mid[idx + 1] = pt.einsum('bji,jakc->bakic',
                                 mid[idx],
                                 mid[idx + 1].to(work_device))
        shape = mid[idx + 1].shape
        mid[idx + 1] = pt.reshape(mid[idx + 1],
                                  (*shape[:3], -1))

        mid[idx + 1] = pt.einsum('ai,abjc->ibjc',
                                 bot[idx],
                                 mid[idx + 1])

        shape = mid[idx + 1].shape
        mid[idx + 1] = pt.reshape(mid[idx + 1],
                                  (-1, *shape[2:]))
        bot[idx + 1] = pt.reshape(bot[idx + 1],
                                  (-1, bot[idx + 1].shape[-1]))

    return pt.squeeze(pt.einsum('ai,bja,bk->ijk',
                                top[-1],
                                mid[-1],
                                bot[-1]))


# A function to contract a sandwich of ket, bra and two MPOs
def contr_4(*,
            ket: MPS = None,
            ham=None,
            sym=None,
            bra: MPS = None,
            work_device=None,
            cpu_device=None):
    assert ket.visible_num == bra.visible_num
    assert ket.visible_num == len(ham)
    assert ket.visible_num == len(sym)

    assert work_device is not None
    assert cpu_device is not None

    visible_num = ket.visible_num

    ket = [tensor for tensor in ket.tensors]
    ham = [tensor for tensor in ham]
    sym = [tensor for tensor in sym]
    bra = [tensor for tensor in bra.tensors]

    ket[0] = pt.squeeze(ket[0], dim=0)
    ham[0] = pt.squeeze(ham[0], dim=0)
    sym[0] = pt.squeeze(sym[0], dim=0)
    bra[0] = pt.squeeze(bra[0], dim=0)

    for idx in range(visible_num - 1):
        #####################################################
        # Step 1
        ham[idx] = pt.einsum('ai,bja->bji',
                             ket[idx].to(work_device),
                             ham[idx].to(work_device))

        #####################################################
        # Step 2
        sym[idx] = pt.einsum('ckb,cl->lkb',
                             sym[idx].to(work_device),
                             bra[idx].to(work_device))

        #####################################################
        # Step 3
        side_tensor = pt.einsum('bji,lkb->lkji',
                                ham[idx],
                                sym[idx])

        #####################################################
        # Step 4
        ham[idx + 1] = pt.einsum('lkji,jend->leknid',
                                 side_tensor,
                                 ham[idx + 1].to(work_device))

        #####################################################
        # Step 5
        shape = ket[idx + 1].shape
        ket[idx + 1] = pt.reshape(ket[idx + 1],
                                  (-1, shape[-1]))
        q, r = pt.linalg.qr(ket[idx + 1].to(cpu_device))
        q = pt.reshape(q.to(work_device), (shape[0], shape[1], -1))
        ket[idx + 1] = r.to(work_device)

        #####################################################
        # Step 6
        ham[idx + 1] = pt.einsum('idq,leknid->leknq',
                                 q,
                                 ham[idx + 1])

        #####################################################
        # Step 7
        shape = bra[idx + 1].shape
        bra[idx + 1] = pt.reshape(bra[idx + 1],
                                  (-1, shape[-1]))
        q, r = pt.linalg.qr(bra[idx + 1].to(cpu_device))
        q = pt.reshape(q.to(work_device), (shape[0], shape[1], -1))
        bra[idx + 1] = r.to(work_device)

        #####################################################
        # Step 8
        sym[idx + 1] = pt.permute(sym[idx + 1], (1, 2, 3, 0))
        sym[idx + 1] = pt.einsum('foek,lfr->rolek',
                                 sym[idx + 1],
                                 q)

        #####################################################
        # Step 9
        shape = ham[idx + 1].shape
        ham[idx + 1] = pt.reshape(ham[idx + 1],
                                  (-1, shape[3], shape[4]))
        shape = sym[idx + 1].shape
        sym[idx + 1] = pt.reshape(sym[idx + 1],
                                  (shape[0], shape[1], -1))

        #####################################################
        # Step 10 (optional QR to reduce the middle bond dimension)
        shape = sym[idx + 1].shape
        if shape[0] * shape[1] < shape[2]:
            sym[idx + 1] = pt.reshape(sym[idx + 1],
                                      (-1, shape[-1]))
            q, r = pt.linalg.qr(sym[idx + 1].T.to(cpu_device))
            sym[idx + 1] = pt.reshape(r.T.to(work_device),
                                      (shape[0], shape[1], -1))
            ham[idx + 1] = pt.einsum('gnq,hg->hnq',
                                     ham[idx + 1],
                                     q.T.to(work_device))

    return pt.squeeze(pt.einsum('ai,bja,ckb,cl->ijkl',
                                ket[-1].to(work_device),
                                ham[-1].to(work_device),
                                sym[-1].to(work_device),
                                bra[-1].to(work_device)))


# A function to contract a sandwich of ket, bra and three MPOs
def contr_5(*,
            ket: MPS = None,
            top=None,
            mid=None,
            bot=None,
            bra: MPS = None,
            work_device=None,
            cpu_device=None):
    assert ket.visible_num == len(top)
    assert ket.visible_num == len(mid)
    assert ket.visible_num == len(bot)
    assert ket.visible_num == bra.visible_num

    assert work_device is not None
    assert cpu_device is not None

    visible_num = ket.visible_num

    ket = [tensor for tensor in ket.tensors]
    top = [tensor for tensor in top]
    mid = [tensor for tensor in mid]
    bot = [tensor for tensor in bot]
    bra = [tensor for tensor in bra.tensors]

    ket[0] = pt.squeeze(ket[0], dim=0)
    top[0] = pt.squeeze(top[0], dim=0)
    mid[0] = pt.squeeze(mid[0], dim=0)
    bot[0] = pt.squeeze(bot[0], dim=0)
    bra[0] = pt.squeeze(bra[0], dim=0)

    times = {step_idx: 0.0 for step_idx in range(4)}

    for idx in range(visible_num - 1):
        # print(f'Iteration #{idx}')
        #####################################################
        # Step 1, ket to top
        top[idx] = pt.einsum('ai,bja->bji',
                             ket[idx].to(work_device),
                             top[idx].to(work_device))

        # ----------------------------------------------------
        # Step 1, bra to bottom
        bot[idx] = pt.einsum('dlc,dm->mlc',
                             bot[idx].to(work_device),
                             bra[idx].to(work_device))

        #####################################################
        # Step 2, top to mid
        mid[idx] = pt.einsum('bji,ckb->ckji',
                             top[idx],
                             mid[idx].to(work_device))
        # ----------------------------------------------------
        # Step 2, bottom to mid
        mid[idx] = pt.einsum('ckji,mlc->mlkji',
                             mid[idx],
                             bot[idx])

        #####################################################
        # Step 3, mid to mid
        mid[idx + 1] = pt.einsum('mlkji,kgpf->mlgpfji',
                                 mid[idx],
                                 mid[idx + 1].to(work_device))

        #####################################################
        # Step 4, ket QR
        shape = ket[idx + 1].shape
        ket[idx + 1] = pt.reshape(ket[idx + 1].to(work_device), (-1, shape[-1]))

        # theor_rank = min(ket[idx + 1].shape)
        # exp_rank = pt.linalg.matrix_rank(ket[idx + 1])
        # print(f'Expected rank: {theor_rank}, real rank: {exp_rank}')
        q_t, r_t = pt.linalg.qr(ket[idx + 1].T.to(cpu_device))
        ket[idx + 1] = q_t.T.to(work_device)
        ket_remdr = r_t.T.to(work_device)
        ket_remdr = pt.reshape(ket_remdr, (shape[0], shape[1], -1))

        # ----------------------------------------------------
        # Step 4, bra QR
        shape = bra[idx + 1].shape
        bra[idx + 1] = pt.reshape(bra[idx + 1].to(work_device), (-1, shape[-1]))

        # theor_rank = min(bra[idx + 1].shape)
        # exp_rank = pt.linalg.matrix_rank(bra[idx + 1])
        # print(f'Expected rank: {theor_rank}, real rank: {exp_rank}')
        q_t, r_t = pt.linalg.qr(bra[idx + 1].T.to(cpu_device))
        bra[idx + 1] = q_t.T.to(work_device)
        bra_remdr = r_t.T.to(work_device)
        bra_remdr = pt.reshape(bra_remdr, (shape[0], shape[1], -1))

        #####################################################
        # Step 5, ket QR remainder to top
        top[idx + 1] = pt.einsum('ies,jfoe->ijfos',
                                 ket_remdr,
                                 top[idx + 1].to(work_device))
        # ----------------------------------------------------
        # Step 5, bra QR remainder to bottom
        bot[idx + 1] = pt.einsum('mht,lhqg->tqglm',
                                 bra_remdr,
                                 bot[idx + 1].to(work_device))

        #####################################################
        # Step 6, top QR
        shape = top[idx + 1].shape
        top[idx + 1] = pt.reshape(top[idx + 1],
                                  (shape[0] * shape[1] * shape[2], shape[3] * shape[4]))

        # theor_rank = min(top[idx + 1].shape)
        # exp_rank = pt.linalg.matrix_rank(top[idx + 1])
        # print(f'Expected rank: {theor_rank}, real rank: {exp_rank}')
        q, r = pt.linalg.qr(top[idx + 1].to(cpu_device))
        #q, r = robust_qr(matrix=top[idx + 1])
        top[idx + 1] = pt.reshape(r.to(work_device), (-1, shape[3], shape[4]))
        top_remdr = pt.reshape(q.to(work_device), (shape[0], shape[1], shape[2], -1))

        # ----------------------------------------------------
        # Step 6, bot QR
        shape = bot[idx + 1].shape
        bot[idx + 1] = pt.reshape(bot[idx + 1],
                                  (shape[0] * shape[1], shape[2] * shape[3] * shape[4]))

        # theor_rank = min(bot[idx + 1].shape)
        # exp_rank = pt.linalg.matrix_rank(bot[idx + 1])
        # print(f'Expected rank: {theor_rank}, real rank: {exp_rank}')
        q_t, r_t = pt.linalg.qr(bot[idx + 1].T.to(cpu_device))
        #q_t, r_t = robust_qr(matrix=bot[idx + 1].T.to(cpu_device))
        bot[idx + 1] = pt.reshape(r_t.T.to(work_device), (shape[0], shape[1], -1))
        bot_remdr = pt.reshape(q_t.T.to(work_device), (-1, shape[2], shape[3], shape[4]))

        #####################################################
        # Step 7, top to mid
        mid[idx + 1] = pt.einsum('ijfu,mlgpfji->mlgpu',
                                 top_remdr,
                                 mid[idx + 1])

        # ----------------------------------------------------
        # Step 7, bot to mid
        mid[idx + 1] = pt.einsum('mlgpu,vglm->vpu',
                                 mid[idx + 1],
                                 bot_remdr)
        #print()

    return pt.squeeze(pt.einsum('ai,bja,ckb,dlc,dm->ijklm',
                                ket[-1].to(work_device),
                                top[-1].to(work_device),
                                mid[-1].to(work_device),
                                bot[-1].to(work_device),
                                bra[-1].to(work_device)))
