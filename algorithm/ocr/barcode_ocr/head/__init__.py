
__all__ = ["build_head"]
from .DBHead import DBHead
from .ConvHead import ConvHead

support_head = ["ConvHead", "DBHead"]


def build_head(head_name, **kwargs):
    assert head_name in support_head, f"all support head is {support_head}"
    head = eval(head_name)(**kwargs)
    return head
