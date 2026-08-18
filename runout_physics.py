import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, griddata
import plotly.graph_objects as go

def calculate_sliding_volume(dem_df, rupture_df, friction_angle_deg):
    """
    Calculates the sliding volume and returns the 3D surface grid 
    for the basal failure plane to be used in rendering.
    """
    points = dem_df[['x', 'y']].values
    values = dem_df['z'].values
    topo_interp = LinearNDInterpolator(points, values)
    
    rx = rupture_df['x'].values
    ry = rupture_df['y'].values
    rz = rupture_df['z'].values
    num_points = len(rx)
    
    tan_phi = np.tan(np.radians(friction_angle_deg))
    cross_section_areas = np.zeros(num_points)
    cumulative_distance = np.zeros(num_points)
    
    # Transverse search array
    s_vals = np.linspace(-200, 200, 800)
    
    # Initialize arrays to store the 3D surface of the failure plane.
    # We fill them with NaNs so Plotly knows not to draw outside the daylight bounds.
    X_surf = np.full((num_points, len(s_vals)), np.nan)
    Y_surf = np.full((num_points, len(s_vals)), np.nan)
    Z_surf = np.full((num_points, len(s_vals)), np.nan)
    
    for i in range(num_points):
        if i > 0:
            dist = np.sqrt((rx[i]-rx[i-1])**2 + (ry[i]-ry[i-1])**2)
            cumulative_distance[i] = cumulative_distance[i-1] + dist
            
        if i < num_points - 1:
            dx = rx[i+1] - rx[i]
            dy = ry[i+1] - ry[i]
        else:
            dx = rx[i] - rx[i-1]
            dy = ry[i] - ry[i-1]
            
        length = np.sqrt(dx**2 + dy**2)
        if length == 0: continue
        
        nx = -dy / length
        ny = dx / length
        
        z_top_center = topo_interp(rx[i], ry[i])
        if np.isnan(z_top_center): continue
            
        H = z_top_center - rz[i]
        if H <= 0: continue
            
        transverse_x = rx[i] + s_vals * nx
        transverse_y = ry[i] + s_vals * ny
        z_top = topo_interp(transverse_x, transverse_y)
        
        z_base = np.zeros_like(s_vals)
        for idx, s in enumerate(s_vals):
            if abs(s) <= H:
                z_base[idx] = rz[i]
            else:
                z_base[idx] = rz[i] + (abs(s) - H) * tan_phi
                
        thickness = z_top - z_base
        
        center_idx = len(s_vals) // 2
        if np.isnan(thickness[center_idx]) or thickness[center_idx] <= 0:
            continue
            
        left_idx = center_idx
        while left_idx > 0 and thickness[left_idx] > 0 and not np.isnan(thickness[left_idx]):
            left_idx -= 1
            
        right_idx = center_idx
        while right_idx < len(s_vals) - 1 and thickness[right_idx] > 0 and not np.isnan(thickness[right_idx]):
            right_idx += 1
            
        valid_s = s_vals[left_idx:right_idx]
        valid_thickness = thickness[left_idx:right_idx]
        
        if len(valid_s) > 1:
            cross_section_areas[i] = np.trapz(valid_thickness, valid_s)
            
            # Store the 3D coordinates of the valid sliding base for this cross-section
            X_surf[i, left_idx:right_idx] = transverse_x[left_idx:right_idx]
            Y_surf[i, left_idx:right_idx] = transverse_y[left_idx:right_idx]
            Z_surf[i, left_idx:right_idx] = z_base[left_idx:right_idx]

    total_volume = np.trapz(cross_section_areas, cumulative_distance)
    
    return total_volume, X_surf, Y_surf, Z_surf

def render_topography_and_mass(dem_df, rupture_df, X_surf, Y_surf, Z_surf, volume):
    """
    Renders the DEM, the 2D rupture plane, and the resulting 3D failure surface.
    """
    points = dem_df[['x', 'y']].values
    values = dem_df['z'].values
    
    grid_x, grid_y = np.mgrid[
        min(points[:, 0]):max(points[:, 0]):100j,
        min(points[:, 1]):max(points[:, 1]):100j
    ]
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    fig = go.Figure()

    # 1. Base Topography (Earth colors, slightly transparent)
    fig.add_trace(go.Surface(
        x=grid_x, y=grid_y, z=grid_z, 
        colorscale='Earth', 
        name='Topography',
        opacity=0.6,
        showscale=False
    ))

    # 2. The Calculated 3D Sliding Mass Base (Red colors, solid)
    fig.add_trace(go.Surface(
        x=X_surf, y=Y_surf, z=Z_surf,
        colorscale='Reds',
        name='Failure Surface',
        opacity=0.9,
        showscale=False
    ))

    # 3. The Initial 2D Rupture Section Line
    fig.add_trace(go.Scatter3d(
        x=rupture_df['x'], y=rupture_df['y'], z=rupture_df['z'],
        mode='lines+markers',
        line=dict(color='black', width=6),
        marker=dict(size=4, color='black'),
        name='Rupture Section'
    ))

    fig.update_layout(
        title=f'Initial State Validation (Computed Sliding Volume: {volume:.2f} m³)',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Elevation (Z)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

# --- Pipeline Execution ---
# volume, X_surf, Y_surf, Z_surf = calculate_sliding_volume(dem_data, rupture_data, friction_angle_deg=30)
# fig = render_topography_and_mass(dem_data, rupture_data, X_surf, Y_surf, Z_surf, volume)
# fig.show()