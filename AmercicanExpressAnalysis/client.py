import requests
import numpy as np

if __name__ == '__main__':
    
    r = requests.post('http://localhost:5000/predict', json=list(np.linspace(-0.9, 0.9, 174)))
    print('Status code: {}'.format(r.status_code))

    if r.status_code == 200:
        
        print('Prediction: {}'.format(r.json()['prediction']))
        
    else:
        
        print(r.text)