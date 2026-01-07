import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="MAT201 Gradient Visualizer", layout="wide")

# --- TITLE AND INTRODUCTION ---
st.title("⛰️ MAT201: Gradient & Steepest Ascent Visualizer")
st.markdown("""
This interactive application visualizes the concept of **Gradient Vectors** and the **Direction of Steepest Ascent** for functions of two variables $f(x, y)$.
It allows you to input any standard mathematical function and observe how the gradient points in the direction of maximum increase.
""")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("1. Define Function & Point")

# Helper function to process standard math notation to Python
def clean_input(eq):
    eq = eq.replace('^', '**') # Allow users to use ^ for power
    eq = eq.replace('e^', 'exp') # Handle exponential
    return eq

# User Input for Function
func_input = st.sidebar.text_input("Enter f(x, y):", value="10 - x^2 - y^2")
st.sidebar.caption("Use standard notation: e.g., `x^2 + y^2`, `sin(x)*cos(y)`, `10 - x^2 - y^2`")

# User Input for Point (a, b)
st.sidebar.subheader("Select Point (a, b)")
a_val = st.sidebar.slider("Value for x (a):", -5.0, 5.0, 1.0, 0.1)
b_val = st.sidebar.slider("Value for y (b):", -5.0, 5.0, 1.0, 0.1)

# Range settings for plotting
st.sidebar.subheader("Plotting Range")
range_val = st.sidebar.slider("Range +/-", 2.0, 10.0, 5.0)

# --- MATHEMATICAL CALCULATION (SYMPY) ---
try:
    x, y = sp.symbols('x y')
    
    # Parse the function string
    expr_str = clean_input(func_input)
    f = sp.sympify(expr_str)
    
    # Calculate Partial Derivatives
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    
    # Calculate values at point (a, b)
    f_val = float(f.subs({x: a_val, y: b_val}))
    fx_val = float(fx.subs({x: a_val, y: b_val}))
    fy_val = float(fy.subs({x: a_val, y: b_val}))
    
    # Gradient Magnitude
    grad_mag = np.sqrt(fx_val**2 + fy_val**2)

    # --- DISPLAY CALCULATIONS ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2. Mathematical Results")
        st.latex(rf"f(x, y) = {sp.latex(f)}")
        st.write("Partial Derivatives:")
        st.latex(rf"f_x = {sp.latex(fx)}")
        st.latex(rf"f_y = {sp.latex(fy)}")
        
    with col2:
        st.subheader("At Point P({}, {})".format(a_val, b_val))
        st.write("The Gradient Vector (Steepest Ascent):")
        st.latex(rf"\nabla f({a_val}, {b_val}) = \langle {fx_val:.2f}, {fy_val:.2f} \rangle")
        st.write(f"Magnitude (Rate of Ascent): **{grad_mag:.4f}**")
        st.info("The arrow in the plots below represents this vector.")

    # --- VISUALIZATION DATA PREP ---
    # Create grid for plotting
    num_points = 50
    x_range = np.linspace(-range_val, range_val, num_points)
    y_range = np.linspace(-range_val, range_val, num_points)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Lambdify function for fast numerical calculation
    f_func = sp.lambdify((x, y), f, "numpy")
    Z = f_func(X, Y)

    # --- PLOTTING ---
    st.divider()
    st.header("3. Visualization")
    
    tab1, tab2 = st.tabs(["3D Surface & Gradient", "2D Contour & Steepest Ascent"])

    # Plot 1: 3D Surface
    with tab1:
        fig_3d = go.Figure()
        
        # Surface
        fig_3d.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Surface'))
        
        # Point P
        fig_3d.add_trace(go.Scatter3d(
            x=[a_val], y=[b_val], z=[f_val],
            mode='markers', marker=dict(size=5, color='red'), name='Point P'
        ))

        # Gradient Vector in 3D (Projected on Tangent Plane approximately)
        # We scale the vector for visibility
        scale = 1.0 
        fig_3d.add_trace(go.Scatter3d(
            x=[a_val, a_val + fx_val*scale],
            y=[b_val, b_val + fy_val*scale],
            z=[f_val, f_val + (fx_val**2 + fy_val**2)*scale], # Logic: dz = fx*dx + fy*dy
            mode='lines+markers',
            line=dict(color='red', width=5),
            name='Gradient Direction'
        ))

        fig_3d.update_layout(title='3D Surface with Gradient Direction', autosize=True, height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("The red line indicates the direction you would walk to climb the hill fastest.")

    # Plot 2: 2D Contour
    with tab2:
        fig_2d = go.Figure()
        
        # Contour
        fig_2d.add_trace(go.Contour(
            z=Z, x=x_range, y=y_range,
            colorscale='Viridis', contours=dict(start=np.min(Z), end=np.max(Z), size=(np.max(Z)-np.min(Z))/20)
        ))
        
        # Point P
        fig_2d.add_trace(go.Scatter(
            x=[a_val], y=[b_val],
            mode='markers', marker=dict(size=10, color='red'), name='Point P'
        ))

        # Gradient Arrow (Quiver)
        fig_2d.add_annotation(
            x=a_val + fx_val*0.5,  # Scaling for visibility
            y=b_val + fy_val*0.5,
            ax=a_val, ay=b_val,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
        )

        fig_2d.update_layout(
            title='2D Contour Map (Top-Down View)',
            xaxis_title='x', yaxis_title='y',
            autosize=True, height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1) # Keep aspect ratio 1:1
        )
        st.plotly_chart(fig_2d, use_container_width=True)
        st.caption("Note: The gradient vector (Red Arrow) is always perpendicular to the contour lines.")

    # --- REAL WORLD APPLICATION ---
    st.divider()
    st.header("4. Real World Application")
    
    col_app1, col_app2 = st.columns([1, 2])
    
    with col_app1:
        st.info("### Concept: Hill Climbing")
        st.write("""
        Imagine being on a foggy mountain (the 3D surface) where you can only see the ground directly beneath your feet. 
        
        To reach the peak as fast as possible, you check the slope at your current position and take a step in the steepest direction.
        
        This is exactly what the **Gradient Vector** $\\nabla f$ calculates.
        """)
    
    with col_app2:
        st.success("### Application: Heat Seeking Drones / Missiles")
        st.write("""
        **Context:** In engineering, gradient based methods are used in navigation and optimization.
        
        **Scenario:** Consider a drone looking for a heat source (like a forest fire) to extinguish it.
        1. Let $T(x, y)$ represent the temperature at coordinates $(x, y)$.
        2. The drone has sensors that measure the temperature change in small distances.
        3. By calculating the gradient $\\nabla T$, the drone finds the direction of the "steepest ascent" in temperature.
        4. By continuously flying in the direction of $\\nabla T$, the drone automatically guides itself to the center of the fire (the maximum point).
        
        **In this App:** If you set $f(x, y) = 100 - x^2 - y^2$ (representing a heat distribution), the red arrow shows the drone exactly which way to fly to find the hottest point.
        """)

except Exception as e:
    st.error(f"Error parsing function. Please ensure you use standard notation (e.g., 3*x^2 + y). Details: {e}")
