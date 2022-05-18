import torch as pt


def _e_num_symmetry(electron_num: int = None,
                    dtype=None,
                    device=None):
    if electron_num is None:
        return None, None, None
    else:
        left_cap = pt.zeros((2, electron_num + 1),
                            dtype=dtype,
                            device=device)
        left_cap[0, 0] = 1
        left_cap[1, 1] = 1
        left_cap = pt.unsqueeze(left_cap, dim=0)

        right_cap = pt.zeros((electron_num + 1, 2),
                             dtype=dtype,
                             device=device)
        right_cap[-1, 0] = 1
        right_cap[-2, 1] = 1
        right_cap = pt.unsqueeze(right_cap, dim=-1)

        ones_diag = pt.ones(electron_num + 1,
                            dtype=dtype,
                            device=device)

        site_tensor = pt.stack([pt.diag(ones_diag),
                                pt.diag(ones_diag[:-1], diagonal=1)],
                               dim=0)
        site_tensor = site_tensor.transpose(0, 1)

        return left_cap, site_tensor, right_cap


def _spin_symmetry(visible_num: int = None,
                   spin: int = None,
                   dtype=None,
                   device=None):
    if spin is None:
        return None, None, None, None
    else:
        assert spin <= (visible_num // 2)
        left_cap = pt.zeros((2, visible_num + 1), dtype=dtype, device=device)
        left_cap[0, visible_num // 2] = 1
        left_cap[1, visible_num // 2 + 1] = 1
        left_cap = pt.unsqueeze(left_cap, dim=0)

        right_cap = pt.zeros((visible_num + 1, 2), dtype=dtype, device=device)
        right_cap[visible_num // 2 + spin, 0] = 1
        right_cap[visible_num // 2 + 1 + spin, 1] = 1
        right_cap = pt.unsqueeze(right_cap, dim=-1)

        ones_diag = pt.ones(visible_num + 1,
                            dtype=dtype,
                            device=device)

        odd_site_tensor = pt.stack([pt.diag(ones_diag),
                                    pt.diag(ones_diag[:-1], diagonal=-1)],
                                   dim=0)
        odd_site_tensor = odd_site_tensor.transpose(0, 1)

        even_site_tensor = pt.stack([pt.diag(ones_diag),
                                     pt.diag(ones_diag[:-1], diagonal=1)],
                                    dim=0)
        even_site_tensor = even_site_tensor.transpose(0, 1)

        return left_cap, odd_site_tensor, even_site_tensor, right_cap
