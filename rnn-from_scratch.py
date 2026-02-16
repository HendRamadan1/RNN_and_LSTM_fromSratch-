import numpy as np 

def softmax(x):
    e_X=np.exp(x-np.max(x))
    return e_X/e_X.sum(axis=0)


def smooth(loss,curve_loss):
    return loss*0.999+curve_loss*0.001 # EMA Exponential Moving Average

def print_sample(sample_ix,ix_to_char):
    txt=''.join(ix_to_char[ix]for ix in sample_ix)
    print(f'.......\n %s \n......%{txt,}')


def get_intial_loss(vocab_szie,seq_length):
    return -np.log(1.0/vocab_szie)*seq_length

def intialization_parameters(n_a,n_x,n_y):

    np.random.seed(1)
    Wax=np.random.randn(n_a,n_x)*0.01
    waa=np.random.randn(n_a,n_a)*0.01
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1))
    parameters = {"Wax": Wax, "Waa": waa, "Wya": Wya, "b": b,"by": by}

    return parameters

def rnn_step_forward(parameters,a_prev,x):
    wax,waa,wya,by,b=parameters['Wax'],parameters['Waa'],parameters['Wya'],parameters['by'],parameters['b']
    a_next=np.tanh(np.dot(wax,x)+np.dot(waa,a_prev)+b)
    p_t=softmax(np.dot(wya,a_next)+by)


def rnn_step_backward(dy,gradient,paramters,x,a,a_prev):
   gradient['dwya'] +=np.dot(dy,a.T)
   gradient['dby'] +=dy
   da=np.dot(paramters['Wya'].T,dy)+gradient['da_next']
   daraw=(1-a*a)*da # tanh
   gradient['db']+=daraw
   gradient['dwax']+=np.dot(daraw,x.T)
   gradient['dwaa']+=np.dot(daraw,a_prev.T)
   gradient['da_next']=np.dot(paramters['waa'].T,daraw)




def Update_parameter(parameters,gradients,lr):
  parameters['Wax']+=lr*gradients['dwax']
  parameters['Waa']+=lr*gradients['dwaa']
  parameters['Way']+=lr*gradients['dwya']
  parameters['b']+=lr*gradients['db']
  parameters['by']+=lr*gradients['dby']



def run_forward(X,Y,a0,parameter,vocab_size=80):
  x,a,y_hat={},{},{}
  a[-1]=np.copy(a0)
  loss=0
  for t  in range(len(X)):
      x[t]=np.zeros(vocab_size,1)
      x[t][X[t]]=1
      a[t],y_hat[t]=rnn_step_forward(parameter,a[t-1],x[t])
      loss -=np.log(y_hat[t][Y[t],0])
      cache=(y_hat,a,x)
      return cache,loss


def run_backword(X,Y,cache,parameters):
    gradients={}
    (y_hat,a,x)=cache
    wax,waa,wya,by,b=parameters['Wax'],parameters['Waa'],parameters['Wya'],parameters['by'],parameters['b']
    gradients['dwax'],gradients['dwaa'],gradients['dwya'],gradients['db'],gradients['dby']=np.zeros_like(wax),np.zeros_like(waa),np.zeros_like(wya),np.zeros_like(b),np.zeros_like(by)
    gradients['da_next']=np.zeros_like(a[0])
    for t in reversed(range(len(X))):
        dy=np.copy(y_hat[t])
        dy[Y[t]]-=1
        gradients=rnn_step_backward(dy,gradients,parameters,x[t],a[t],a[t-1])
        return gradients,a








