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
This interactive application visualizes the concept of **Gradient Vectors** and the **Direction of Steepest Ascent** for functions of several variables.
It satisfies the requirement to use an AI application builder to explain topics related to functions of several variables.
""")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("1. Define Function & Point")

# --- KEY UPDATE: CLEAN INPUT FUNCTION (More Robust) ---
def clean_input(eq):
    # 1. Replace Unicode superscripts with Python syntax
    superscripts = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    for sup, val in superscripts.items():
        eq = eq.replace(sup, f"**{val}")
    
    # 2. Handle Chinese/Full-width punctuation just in case
    # This prevents errors if you accidentally use Chinese input method
    eq = eq.replace('＋', '+')
    eq = eq.replace('－', '-')
    eq = eq.replace('（', '(')
    eq = eq.replace('）', ')')
    
    # 3. Standard replacements
    eq = eq.replace('^', '**') 
    eq = eq.replace('e**', 'exp')
    
    return eq

# User Input for Function (Updated default to x² + y²)
func_input = st.sidebar.text_input("Enter f(x, y):", value="x² + y²")
st.sidebar.caption("Supported formats: `x² + y²`, `sin(x)*cos(y)`, `10 - x²`")

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
    
    # Parse the function string using the new cleaner
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
        # Display the input function nicely
        st.latex(rf"f(x, y) = {sp.latex(f)}")
        st.write("Partial Derivatives (Rates of Change):")
        st.latex(rf"f_x = \frac{{\partial f}}{{\partial x}} = {sp.latex(fx)}")
        st.latex(rf"f_y = \frac{{\partial f}}{{\partial y}} = {sp.latex(fy)}")
        
    with col2:
        st.subheader(f"At Point P({a_val}, {b_val})")
        st.write("The Gradient Vector (Direction of Steepest Ascent):")
        st.latex(rf"\nabla f({a_val}, {b_val}) = \langle {fx_val:.2f}, {fy_val:.2f} \rangle")
        st.write(f"Magnitude (Steepness): **{grad_mag:.4f}**")
        st.info("The red arrow in the visualization points in this direction.")

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
    st.header("3. Visualization (Interactive)")
    st.markdown("Use your mouse to rotate the 3D graph or zoom into the 2D contour map.")
    
    tab1, tab2 = st.tabs(["3D Surface & Gradient", "2D Contour Map"])

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

        # Gradient Vector in 3D (Projected logic)
        scale = 1.0 
        fig_3d.add_trace(go.Scatter3d(
            x=[a_val, a_val + fx_val*scale],
            y=[b_val, b_val + fy_val*scale],
            # To visualize gradient on surface, we approximate the tangent rise
            z=[f_val, f_val + (fx_val**2 + fy_val**2)*scale], 
            mode='lines+markers',
            line=dict(color='red', width=5),
            name='Steepest Ascent'
        ))

        fig_3d.update_layout(
            title='3D Surface: The Red Line shows the path up', 
            autosize=True, height=600,
            scene=dict(aspectmode='cube')
        )
        st.plotly_chart(fig_3d, use_container_width=True)

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

        # Gradient Arrow
        fig_2d.add_annotation(
            x=a_val + fx_val*0.5,
            y=b_val + fy_val*0.5,
            ax=a_val, ay=b_val,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor="red"
        )

        fig_2d.update_layout(
            title='2D Contour: The Gradient is perpendicular to level curves',
            xaxis_title='x', yaxis_title='y',
            autosize=True, height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1)
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    # --- REAL WORLD APPLICATION ---
    st.divider()
    st.header("4. Real World Application: Gradient Ascent")
    
    col_app1, col_app2 = st.columns([1, 2])
    
    with col_app1:
        st.info("### Concept: Climbing Out")
        st.write("""
        Imagine you are at the bottom of a valley (like the shape $x^2 + y^2$).
        
        To climb out as quickly as possible, you need to find the steepest path upwards.
        
        The **Gradient Vector** $\\nabla f$ points exactly in that direction—straight up the wall of the valley.
        """)
    
    with col_app2:
        st.success("### Application: AI & Machine Learning")
        st.write("""
        **How ChatGPT works (Conceptually):**
        Most modern AI is trained using **Gradient Descent** (going downhill to minimize error) or **Gradient Ascent** (going uphill to maximize reward).
        
        1. We define an objective function (like a score).
        2. The computer calculates the gradient of this function.
        3. It adjusts its parameters in the direction of the gradient to increase the score.
        
        **Relevance:** This app visualizes the core mathematical engine used to train models like Gemini and ChatGPT.
        """)

except Exception as e:
    st.error(f"Input Error: {e}")
    st.warning("Tip: Use standard notation. For powers, you can use x² or x^2. For multiplication, use *. Example: 3*x² + y")
