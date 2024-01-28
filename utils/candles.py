from tinkoff.invest.utils import Quotation


def quotation_to_float(q: Quotation) -> float:
    return q.units + (q.nano / 1000000000)