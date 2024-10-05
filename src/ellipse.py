import  numpy as np
import matplotlib.pyplot as plt


""" Visualize the Covariance Ellipse """
def GetEllipse(x, P, sigma):
    R, D, V = np.linalg.svd(P[0:2,0:2], full_matrices=False)
    # Square root of the eigenvalues to get the standard deviations along the principal axes
    S = np.sqrt(D)
    # Generate points on a unit circle
    alpha = np.linspace(0, 2 * np.pi, 200)
    unit_circle_points = np.array([np.cos(alpha), np.sin(alpha)])
    x = x.reshape(2,1)  # Transpose to make it a (2, 1) column vector
    # Scale and rotate the unit circle points to create the ellipse
    ellipse_points = sigma * R @ np.diag(S) @ unit_circle_points + x  # Add x to each column
    return ellipse_points