def estimate_exp(full_measurement, one_observable):
    sum_product, cnt_match = 0, 0

    for single_measurement in full_measurement:
        not_match = 0
        product = 1

        for pauli_XYZ, position in one_observable:
            if pauli_XYZ != single_measurement[position][0]:
                not_match = 1
                break
            product *= int(single_measurement[position][1]) # here the outcome of the measurement is
            # stored if the measurement has been performed in the same basis
        if not_match == 1:
            continue
        # we jump straight in the next iteration if the measurement was not in the same basis
        # otherwise the measurement outcome is added to sum_product and the number of matches goes one up
        sum_product += product
        cnt_match += 1
    # the expectation value is just the average over all measurement outcomes
    # in which the measurement basis was the same
    return sum_product, cnt_match

