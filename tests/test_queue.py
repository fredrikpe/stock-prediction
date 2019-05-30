import collections as c


def test_order():
    q = c.deque(maxlen=3)
    q.append(1)
    q.append(2)
    q.append(3)

    assert q.pop() == 3
    assert q.pop() == 2
    assert q.pop() == 1


def test_replaces():
    q = c.deque(maxlen=3)
    q.append(1)
    q.append(2)
    q.append(3)
    q.append(4)

    assert q.pop() == 4
    assert q.pop() == 3
    assert q.pop() == 2


def test_average():
    q = c.deque(maxlen=3)
    q.append(1)
    q.append(2)

    assert sum(q) / 2 == 1.5
