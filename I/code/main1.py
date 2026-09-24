import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
data = scipy.io.loadmat("../data/mnist_train_test.mat", squeeze_me=True, struct_as_record=False)
'''t(data.keys())
print(type(data["train"]))
print(data["train"].shape)
print(type(data["test"]))
print(data["test"].shape)
print(data["train"].dtype.names)
print(data["test"].dtype.names)
'''
train, test=data["train"],data["test"]
X_train=train.X
Y_train=train.y
X_test=test.X
Y_test=test.y
#print(X_train.shape)#
#print(Y_train.shape)#
l=0.005
Y_train=Y_train.astype(float)
Y_test=Y_test.astype(float)

def dot(x,theta):
    return x@theta

def phi(z):
    if z<=-1:
        return 0.0
    elif z<=0:
        return (z+1)**2/2
    else:
        return 1/2+z

def phi_prime(z):
    if z<=-1:
        return 0.0
    elif z<=0:
        return (z+1)
    else:
        return 1.0

theta=np.zeros(785)
def f_lambda(theta):
    f=0.0
    t=np.ones(Y_train.shape[0])
    s=dot(X_train.T, theta)
    p=s*(1-2*Y_train)
    f=np.sum(np.vectorize(phi)(p))
    f+=l/2*np.linalg.norm(theta)**2
    return f

def grad_f(theta):
    one=np.ones(Y_train.shape[0])
    s=dot(X_train.T, theta)
    p=s*(1-2*Y_train)
    t=np.vectorize(phi_prime)(p)
    u=(1-2*Y_train)*X_train
    f=(t*u)@one
    f+=l*theta
    return f

t= np.logspace(-8.0, 0.0, num = 101)

np.random.seed(1)
v=np.random.randn(785)
v=v/np.linalg.norm(v)
#print(v.shape)#
theta=np.random.randn(785)

err=[]
for i in range(101):
    value=abs(f_lambda(theta+t[i]*v)-f_lambda(theta)-t[i]*dot(v,grad_f(theta)))
    err.append(value)
err=np.array(err)
#print((np.log(err[70])-np.log(err[69]))/(np.log(t[70])-np.log(t[69])))#
#plt.loglog(t,err)#
#plt.show()#



np.random.seed(40)

def algo(step):
    l1, l2=[], []
    theta_0=np.random.randn(785)
    x=theta_0
    start=time.perf_counter()   
    g=grad_f(x)
    print("initial:", np.linalg.norm(g))
    print("final:", 10**(-3)*np.linalg.norm(grad_f(theta_0)))
    while np.linalg.norm(g)>10**(-3)*np.linalg.norm(grad_f(theta_0)):
        x=x-step*g
        g=grad_f(x)
        print(np.linalg.norm(g))
        if time.perf_counter()-start>=180:
            break
        l1.append(f_lambda(x))
        l2.append(np.linalg.norm(grad_f(x)))
    plt.plot(l1, label='objective')
    plt.show()
    plt.plot(l2, label='gradient norm')
    plt.show()
    plt.plot(np.log(l1), label='objective')
    plt.show()
    plt.plot(np.log(l2))
    plt.show()
    plt.legend()
    return x

print(algo(0.1))
'''print("firstttt")
print(algo(1))
print("secondddd")
print(algo(0.01))
print("thirddddd")
print(algo(10))'''

"""
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
"""
