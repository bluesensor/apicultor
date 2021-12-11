import numpy as np

# decrease peaks at fourier size to reduce noise
lstmw = np.array([-0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128587, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128595, -0.13706497639128573, -0.1370649763912857, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128587, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128587, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128573, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128568, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128573, -0.1370649763912857, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128568, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128587, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128568, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128568, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128587, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -
                 0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.13706497639128587, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128587, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.13706497639128587, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128587, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.13706497639128595, -0.13706497639128595, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128587, -0.1370649763912857, -0.13706497639128568, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128595, -0.13706497639128568, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128587, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128595, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128573, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128595, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.13706497639128587, -0.1370649763912857, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128595, -0.13706497639128573, -0.13706497639128595, -0.13706497639128593, -0.13706497639128593, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.13706497639128595, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128573, -0.13706497639128593, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857, -0.1370649763912857, -0.1370649763912857, -0.13706497639128593, -0.13706497639128593, -0.1370649763912857])