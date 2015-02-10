import theano
import theano.tensor as T

a = T.scalar()
b = T.scalar()

y = a * b

f = theano.function([a, b], y)

print f(1, 2) # 2
print f(3, 3) # 9