import numpy as np
import plotly.graph_objects as go
import streamlit as st
import transforms3d as tr
from numpy.linalg import inv

from utils.plot import init_figure, plot_camera, plot_points, plot_world_coordinates

EXERCISE = 4
TASKS = 5
TITLE = f"Exercise {EXERCISE}"
st.set_page_config(page_title=TITLE, page_icon="🗺️", layout="wide")
st.sidebar.header(TITLE)
task = 0

task += 1
task += 1
st.write(f"## 7.1.{EXERCISE}.{task}")

W = tr.affines.compose(np.zeros(3), np.identity(3), np.ones(3))

fig1 = init_figure()
plot_world_coordinates(fig1, xrange=[-3, 3], yrange=[-3, 3], zrange=[0, 4])

plot_camera(fig1, W, name="World", color="black")

T1 = [-0.5, 0, 1]
R1 = np.identity(3)
K1 = tr.affines.compose(T1, R1, np.ones(3))
st.code(f"K1=\n{K1}")
plot_camera(fig1, K1, name="K1", color="red")

T2 = [0.5, 0, 1]
R2 = np.identity(3)
K2 = tr.affines.compose(T2, R2, np.ones(3))
st.code(f"K2=\n{K2}")
plot_camera(fig1, K2, name="K2", color="blue")

samples = (np.random.rand(80) - 0.5) * 8
p0 = np.array(list(map(lambda s: [s, s, 3], samples)))
plot_points(fig1, p0, "green", name="random_samples")

T = inv(K1) @ K2
st.code(f"T = inv(K1) @ K2 =\n {T}")

task += 1
st.write(f"## 7.1.{EXERCISE}.{task}")
st.plotly_chart(fig1)

st.write(f"## 7.1.{EXERCISE}.{task+1}")

p1 = (p0 - T1) @ inv(R1)
p2 = (p0 - T2) @ inv(R2)
# plot_points(fig, p1, "red")
# plot_points(fig, p2, "blue")

f1, f2, f3 = st.columns(3)
f1.latex(r" x_u = \frac{X_c}{Z_c}")
f2.latex(r" y_u = \frac{Y_c}{Z_c}")
f3.text("Slide 33, Formula (69)")
b1 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], p1)))
b2 = np.array(list(map(lambda p: [p[0] / p[2], p[1] / p[2], 1], p2)))
# plot_points(fig, b1, "red")
# plot_points(fig, b2, "blue")

FOCAL_LENGTH = 0.002
CAMERA_DIST = 1

g1, g2, g3 = st.columns(3)
g1.latex(r" Z = \frac{f_k d_b}{x_l-x_r}")
g2.latex(r" X = \frac{x_l Z}{f_k}")
g2.latex(r" Y = \frac{y_l Z}{f_k}")
g3.text("Slide 41, Formula (80),(82)")
zr = np.array([FOCAL_LENGTH * CAMERA_DIST / (b1[i][0] - b2[i][0]) for i in range(80)])
zr = zr / FOCAL_LENGTH
xr = np.multiply(zr, b1[:, 0])
yr = np.multiply(zr, b1[:, 1])

fig2 = go.Figure(fig1)
points_reconstructed = np.stack((xr, yr, zr), axis=1)
# We still need to go back to world coordinates
points_reconstructed = points_reconstructed @ R1 + T1
plot_points(fig2, points_reconstructed, "purple", name="reconstructed points")
st.plotly_chart(fig2)
st.write("The reconstucted purple points perfectly cover the original green points.")

fig3 = go.Figure(fig1)
noisy1 = b1 + np.random.rand(80, 3) * 0.1
noisy2 = b2 + np.random.rand(80, 3) * 0.1
noisy_z = np.array(
    [FOCAL_LENGTH * CAMERA_DIST / (noisy1[i][0] - noisy2[i][0]) for i in range(80)]
)
noisy_z = noisy_z / FOCAL_LENGTH
noisy_x = np.multiply(noisy_z, noisy1[:, 0])
noisy_y = np.multiply(noisy_z, noisy1[:, 1])
noisy_reconstructed = np.stack((noisy_x, noisy_y, noisy_z), axis=1)
# We still need to go back to world coordinates
noisy_reconstructed = noisy_reconstructed @ R1 + T1
plot_points(fig3, noisy_reconstructed, "red", name="reconstructed noisy points")


st.write(
    "With random noise (±0.1) added to all coordiantes, the reconstruction still works reasonably well."
)
rmse = np.sqrt(np.mean(np.sum((p0 - noisy_reconstructed) ** 2, axis=1)))
st.write(f"The Root Mean Squared Error is {rmse}.")

st.plotly_chart(fig3)
