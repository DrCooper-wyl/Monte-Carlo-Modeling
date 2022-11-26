from scipy import stats, special
import numpy
def monte_carlo(bitcoin_price:list,n:int):
    past_bitcoin_price=bitcoin_price[:n]
    bitcoin_lm = stats.boxcox_normmax(past_bitcoin_price)
    bitcoin_nor = stats.boxcox(past_bitcoin_price, lmbda=bitcoin_lm)
    mu=(bitcoin_nor[-1]/bitcoin_nor[1])**(365/len(bitcoin_nor))-1
    t_array= []
    for i in range(1,len(bitcoin_nor)) :
        t_array=bitcoin_nor[i]/bitcoin_nor[i-1]-1
        vol=numpy.array(t_array).std()*numpy.sqrt(365)
        pe=bitcoin_price[n-1]
        res= []
    for i in range(10000):
        res.append(numpy.random.normal(mu/len(bitcoin_nor),
            vol/numpy.sqrt(len(bitcoin_nor)),len(bitcoin_nor)))
    res=numpy.array(res)
    res=res.mean()
    res=special.inv_boxcox(res,bitcoin_lm)
    return(res) *pe