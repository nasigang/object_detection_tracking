import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag
from typing import List

class Tracker():
    def __init__(self, id: int=0, box: List[int]=[], hits: int=0, no_losses: int=0, label: int=None, score: float=None,
                 x_state: List=[], dt: float=1.0, F: np.ndarray=None, H: np.ndarray=None, L: float=10.0, 
                 Q_comp_mat: np.ndarray=None, R_scalar: float=1.0):
        """
        The Kalman Filter class.

        Args:
            id (int): ID of tracker object. Defaults to 0.
            box (List[int]): Bounding box of tracker object. Defaults to [].
            hits (int): The number of showing up of a tracker. Defaults to 0.
            no_losses (int): The number of show up of a tracker without a matched detection box. Defaults to 0.
            label (int): Label of the tracked object. Defaults to None.
            score (float): Confidence score of tracker object. Defaults to None.
            x_state (List): The x state for Kalman filter. Defaults to None.
            dt (float): Time step. Defaults to 1.0.
            F (np.ndarray): The process matrix specifying change of x_state. Defaults to None.
            H (np.ndarray): The measurement matrix that converts state variables into measurement. Defaults to None.
            L (float): The state covariance value that show initial value of state variable's uncertainty. Defaults to 10.0.
            P (float): The state covariance value that show current state variable's uncertainty. Defaults to 10.0.
            Q_comp_mat (np.ndarray): Process covariance unit, which shows the change rate of x_state. Defaults to None.
            R_scalar (float): The measurement covariance unit that represents uncertainty of a measurement. Defaults to 1.0.
        """
        self.id = id
        self.box = box
        self.hits = hits
        self.no_losses = no_losses 
        self.label = label
        self.score = score

        self.x_state = x_state  
        self.dt = dt
        
        if F is None:   # no. state: 4, no.state variable: 8
            self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],  # x1(t) + v_x1(t)*Δt
                            [0, 1,  0,  0,  0,  0,  0, 0],          # v_x1(t)
                            [0, 0,  1,  self.dt, 0,  0,  0, 0],     # x2(t) + v_x2(t)*Δt
                            [0, 0,  0,  1,  0,  0,  0, 0],          # v_x2(t)
                            [0, 0,  0,  0,  1,  self.dt, 0, 0],
                            [0, 0,  0,  0,  0,  1,  0, 0],
                            [0, 0,  0,  0,  0,  0,  1, self.dt],
                            [0, 0,  0,  0,  0,  0,  0,  1]])
        
        if H is None:
            self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],    # only consider location
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 1, 0]])
        

        self.L = L
        self.P = np.diag(self.L*np.ones(8))
        
        if Q_comp_mat is None:
            self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                        [self.dt**3/2., self.dt**2]])
        
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, 
                            self.Q_comp_mat, self.Q_comp_mat)       # state noise
        
        self.R_scaler = R_scalar
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)     # sensor noise
        
        
    def update_R(self):
        """
        Updates the measurement covariance, R.
        """
           
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)
        
        
    def prediction_and_update(self, z: List[int]): 
        """
        Implements prediction and update phases of the Kalman Filter.

        Args:
            z (List[int]): Ground truth. The bounding box comes from the detector.
        """
        
        self.x_state = dot(self.F, self.x_state)    # predidct x_state
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q     # F=A


        S = dot(self.H, self.P).dot(self.H.T) + self.R  # 
        K = dot(self.P, self.H.T).dot(inv(S))       # Kalman gain
        y = z - dot(self.H, self.x_state)   # z ������ ������ ��
        self.x_state += dot(K, y)   # current state prediction; like Low pass filter
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = self.x_state.astype(int)
        
        
    def predict_only(self):  
        """
        Implements the prediction phase of the Kalman Filter.
        """
        
        self.x_state = dot(self.F, self.x_state)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = self.x_state.astype(int)