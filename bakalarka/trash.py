

class A:
    def __init__(self):
        pass

    def p(self):
        self._a()

    def _a(self):
        print("a")


class B(A):
    def __init__(self):
        A.__init__(self)

    def _a(self):
        print("b")


w = A()
q = B()


