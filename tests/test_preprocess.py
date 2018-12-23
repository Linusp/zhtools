import pytest

from zhtools.preprocess import to_halfwidth


@pytest.mark.parametrize(
    'text, cleaned',
    [
        ('第１条', '第1条'),
        ('Ａｐｐｌｅ', 'Apple'),
        ('Ａｐｐｌｅ　ｐｅｎｃｉｌ', 'Apple pencil'),
    ]
)
def test_to_halfwidth(text, cleaned):
    assert to_halfwidth(text) == cleaned
