"""Deformation models for worm registration"""

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class DeformationModel(ABC):

    @abstractmethod
    def fit(self, x, y, weights):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass
    

class Affine(DeformationModel):
    
    def __init__(self):
        
        self._model = Pipeline([
            ('poly', PolynomialFeatures(degree=1, include_bias=True)),
            ('linear', LinearRegression(fit_intercept=False))
        ])
        
    @property
    def beta(self):
        
        return self._model.named_steps['linear'].coef_
        
    def fit(self, x, y, weights):
        
        self._model.fit(x, y, linear__sample_weight=weights)
        
        return self
        
    def predict(self, x):
        
        return self._model.predict(x)
    
    
class Quadratic(DeformationModel):
    
    def __init__(self):
        
        self._model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=True)),
            ('linear', LinearRegression(fit_intercept=False))
        ])
        
    @property
    def beta(self):
        
        return self._model.named_steps['linear'].coef_
        
    def fit(self, x, y, weights):
        
        self._model.fit(x, y, linear__sample_weight=weights)
        
        return self
        
    def predict(self, x):
        
        return self._model.predict(x)
        
    def _compute_jac(self, x):
    
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]

        d_phi = np.array([
            [0,         0,         0        ],
            [1,         0,         0        ],
            [0,         1,         0        ], 
            [0,         0,         1        ],
            [2 * x0,    0,         0        ],
            [x1,        x0,        0        ],
            [x2,        0,         x0       ],
            [0,         2 * x1,    0        ],
            [0,         x2,        x1       ],
            [0,         0,         2 * x2   ],
        ])
        
        return self.beta @ d_phi

    def det_jac(self, x):
        
        det_vals = [np.linalg.det(self._compute_jac(x_i)) for x_i in x]
        
        return np.array(det_vals).reshape(-1, 1)

    
class Cubic(DeformationModel):
    
    def __init__(self):
        
        self._model = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=True)),
            ('linear', LinearRegression(fit_intercept=False))
        ])
        
    @property
    def beta(self):
        
        return self._model.named_steps['linear'].coef_
        
    def fit(self, x, y, weights):
        
        self._model.fit(x, y, linear__sample_weight=weights)
        return self
        
    def predict(self, x):
        
        return self._model.predict(x)
        
    def _compute_jac(self, x):
        
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
    
        x0_2 = x0 ** 2
        x1_2 = x1 ** 2
        x2_2 = x2 ** 2
    
        x0_x1 = x0 * x1
        x1_x2 = x1 * x2
        x0_x2 = x0 * x2
    
        d_phi = np.array([
            [0,         0,         0        ],
            [1,         0,         0        ],
            [0,         1,         0        ], 
            [0,         0,         1        ],
            [2 * x0,    0,         0        ],
            [x1,        x0,        0        ],
            [x2,        0,         x0       ],
            [0,         2 * x1,    0        ],
            [0,         x2,        x1       ],
            [0,         0,         2 * x2   ],
            [3 * x0_2,  0,         0        ],
            [2 * x0_x1, x0_2,      0        ],
            [2 * x0_x2, 0,         x0_2     ],
            [x1_2,      2 * x0_x1, 0        ],
            [x1_x2,     x0_x2,     x0_x1    ],
            [x2_2,      0,         2 * x0_x2],
            [0,         3 * x1_2,  0        ],
            [0,         2 * x1_x2, x1_2     ],
            [0,         x2_2,      2 * x1_x2],
            [0,         0,         3 * x2_2 ],
        ])
    
        return beta @ d_phi

    def det_jac(self, x):
        
        det_vals = [np.linalg.det(self._compute_jac(x_i)) for x_i in x]
        
        return np.array(det_vals).reshape(-1, 1)
