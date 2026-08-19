import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import plotly.graph_objects as go
from scipy.optimize import brentq

# 1. Define a unified discretization resolution (e.g., 0.5 meters)
# This controls both the calculation nodes and the visual mesh density
GRID_RES = 0.5 

def calculate_sliding_volume(dem_df, rupture_df, friction_angle_deg, grid_res=GRID_RES):
    """
    Calculates the sliding volume and returns the 3D surface grid.
    This version mathematically resamples the rupture line to match the grid resolution.
    """
    points = dem_df[['x', 'y']].values
    values = dem_df['z'].values
    topo_interp = LinearNDInterpolator(points, values)
    
    # 1. Extract the raw input coordinates
    rx_orig = rupture_df['x'].values
    ry_orig = rupture_df['y'].values
    rz_orig = rupture_df['z'].values
    
    # 2. Calculate the cumulative distance of the raw input line
    orig_dist = np.zeros(len(rx_orig))
    for i in range(1, len(rx_orig)):
        orig_dist[i] = orig_dist[i-1] + np.sqrt((rx_orig[i]-rx_orig[i-1])**2 + (ry_orig[i]-ry_orig[i-1])**2)
        
    # 3. Create a high-resolution distance array using the exact GRID_RES step
    # We add a tiny buffer (grid_res/10) to ensure the very last point is included
    resampled_dist = np.arange(0, orig_dist[-1] + grid_res/10, grid_res)
    
    # 4. Interpolate the original points onto this new high-resolution line
    rx = np.interp(resampled_dist, orig_dist, rx_orig)
    ry = np.interp(resampled_dist, orig_dist, ry_orig)
    rz = np.interp(resampled_dist, orig_dist, rz_orig)
    
    num_points = len(rx)
    tan_phi = np.tan(np.radians(friction_angle_deg))
    cross_section_areas = np.zeros(num_points)
    cumulative_distance = resampled_dist # We already have our high-res distances!
    
    # Transverse calculation array using the exact discretization step
    s_vals = np.arange(-200, 200 + grid_res/2, grid_res)
    
    X_surf = np.full((num_points, len(s_vals)), np.nan)
    Y_surf = np.full((num_points, len(s_vals)), np.nan)
    Z_surf = np.full((num_points, len(s_vals)), np.nan)
    
    for i in range(num_points):
        # Calculate the normal vector using adjacent points on the high-res line
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
            cross_section_areas[i] = np.trapezoid(valid_thickness, valid_s)
            
            X_surf[i, left_idx:right_idx] = transverse_x[left_idx:right_idx]
            Y_surf[i, left_idx:right_idx] = transverse_y[left_idx:right_idx]
            Z_surf[i, left_idx:right_idx] = z_base[left_idx:right_idx]

    total_volume = np.trapezoid(cross_section_areas, cumulative_distance)
    
    return total_volume, X_surf, Y_surf, Z_surf

def render_topography_and_mass(dem_df, rupture_df, X_surf, Y_surf, Z_surf, X_dep, Y_dep, Z_dep, grid_res=GRID_RES):
    points = dem_df[['x', 'y']].values
    values = dem_df['z'].values
    topo_interp = LinearNDInterpolator(points, values)
    
    x_min, x_max = min(points[:, 0]), max(points[:, 0])
    y_min, y_max = min(points[:, 1]), max(points[:, 1])
    
    grid_x_1d = np.arange(x_min, x_max + grid_res/2, grid_res)
    grid_y_1d = np.arange(y_min, y_max + grid_res/2, grid_res)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)
    
    grid_z = topo_interp(grid_x, grid_y)
    
    fig = go.Figure()

    # Base Topography
    fig.add_trace(go.Surface(
        x=grid_x, y=grid_y, z=grid_z, 
        colorscale='Earth', name='Topography', opacity=0.75, showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.5, specular=0.2)
    ))

    # Initial Failure Surface (Red)
    fig.add_trace(go.Surface(
        x=X_surf, y=Y_surf, z=Z_surf,
        colorscale='Reds', name='Failure Surface', opacity=0.95, showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.6, roughness=0.4, specular=0.1)
    ))
    
    # Final Deposited Mass (Blue/Teal)
    fig.add_trace(go.Surface(
        x=X_dep, y=Y_dep, z=Z_dep,
        colorscale='Teal', name='Deposited Mass', opacity=0.95, showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.6, roughness=0.4, specular=0.1)
    ))

    # Rupture Axis Line
    fig.add_trace(go.Scatter3d(
        x=rupture_df['x'], y=rupture_df['y'], z=rupture_df['z'],
        mode='lines+markers', line=dict(color='black', width=6),
        marker=dict(size=4, color='black'), name='Main Axis'
    ))

    fig.update_layout(
        title=f'Runout Propagation (Grid cell: {grid_res}m)',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Elevation (Z)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, y=1.05, showactive=True,
            buttons=list([
                dict(label="Show All", method="update", args=[{"visible": [True, True, True, True]}]),
                dict(label="Hide Results (Topo Only)", method="update", args=[{"visible": [True, False, False, True]}]),
                dict(label="Failure Mass Only", method="update", args=[{"visible": [True, True, False, True]}]),
                dict(label="Deposit Only", method="update", args=[{"visible": [True, False, True, True]}])
            ])
        )]
    )

    return fig

def calculate_deposit_surface(dem_df, rupture_df, final_distance, target_volume, angle_of_repose, grid_res=GRID_RES):
    """
    Calculates the 3D surface of the deposited mass.
    Anchors the toe of the deposit at 'final_distance' and searches backward 
    for the peak location that perfectly conserves the initial volume.
    """
    points = dem_df[['x', 'y']].values
    values = dem_df['z'].values
    topo_interp = LinearNDInterpolator(points, values)
    
    rx = rupture_df['x'].values
    ry = rupture_df['y'].values
    
    orig_dist = np.zeros(len(rx))
    for i in range(1, len(rx)):
        orig_dist[i] = orig_dist[i-1] + np.sqrt((rx[i]-rx[i-1])**2 + (ry[i]-ry[i-1])**2)
        
    # Helper to find X,Y coordinates anywhere along the axis (even extrapolated backwards or forwards)
    def get_xy(s):
        if s < 0:
            dx, dy = rx[1] - rx[0], ry[1] - ry[0]
            ds = orig_dist[1] - orig_dist[0]
            return rx[0] + (dx/ds) * s, ry[0] + (dy/ds) * s
        elif s <= orig_dist[-1]:
            return np.interp(s, orig_dist, rx), np.interp(s, orig_dist, ry)
        else:
            dx, dy = rx[-1] - rx[-2], ry[-1] - ry[-2]
            ds = orig_dist[-1] - orig_dist[-2]
            return rx[-1] + (dx/ds) * (s - orig_dist[-1]), ry[-1] + (dy/ds) * (s - orig_dist[-1])

    # 1. Anchor the toe at the final_distance
    s_toe = final_distance
    toe_x, toe_y = get_xy(s_toe)
    z_toe = topo_interp(toe_x, toe_y)
    
    if np.isnan(z_toe):
        raise ValueError("The target toe distance falls outside the boundaries of the Topography DEM.")
        
    tan_repose = np.tan(np.radians(angle_of_repose))
    
    # 2. Setup a local grid large enough to contain the deposit searching backwards
    buffer = 300 
    x_vals = np.arange(min(np.min(rx), toe_x) - buffer, max(np.max(rx), toe_x) + buffer, grid_res)
    y_vals = np.arange(min(np.min(ry), toe_y) - buffer, max(np.max(ry), toe_y) + buffer, grid_res)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_topo = topo_interp(X_grid, Y_grid)
    
    # 3. Objective function: Guess a peak position (s_peak) behind the toe, and calculate volume
    def volume_difference(s_peak):
        peak_x, peak_y = get_xy(s_peak)
        
        # The peak height is strictly defined so that the cone edge exactly touches the toe elevation
        z_peak = z_toe + (s_toe - s_peak) * tan_repose
        
        dist_from_peak = np.sqrt((X_grid - peak_x)**2 + (Y_grid - peak_y)**2)
        Z_cone = z_peak - dist_from_peak * tan_repose
        
        thickness = Z_cone - Z_topo
        thickness = np.where(np.isnan(thickness), 0, thickness)
        thickness[thickness < 0] = 0
        
        vol = np.sum(thickness) * (grid_res**2)
        return vol - target_volume
        
    # 4. Search backwards from the toe (s_toe) up to 1000m behind the start line (-1000)
    try:
        s_peak_opt = brentq(volume_difference, -1000, s_toe)
    except ValueError:
        raise ValueError("Could not find a valid deposit geometry. The volume may be too large or the angle of repose too shallow.")
    
    # 5. Generate the final true deposit arrays based on the exact found peak
    peak_x, peak_y = get_xy(s_peak_opt)
    z_peak_opt = z_toe + (s_toe - s_peak_opt) * tan_repose
    
    dist_from_peak = np.sqrt((X_grid - peak_x)**2 + (Y_grid - peak_y)**2)
    Z_cone_final = z_peak_opt - dist_from_peak * tan_repose
    
    deposit_thickness = Z_cone_final - Z_topo
    Z_deposit_surf = np.where(deposit_thickness > 0, Z_cone_final, np.nan)
    
    # Note we are now returning the optimized peak position (s_peak_opt) as well
    return X_grid, Y_grid, Z_deposit_surf, z_peak_opt, s_peak_opt

def render_cross_section(dem_df, rupture_df, final_distance, angle_repose, z_peak_opt, s_peak_opt, grid_res=GRID_RES):
    """Generates a 2D longitudinal profile along the central axis."""
    points = dem_df[['x', 'y']].values
    values = dem_df['z'].values
    topo_interp = LinearNDInterpolator(points, values)

    rx_orig = rupture_df['x'].values
    ry_orig = rupture_df['y'].values
    rz_orig = rupture_df['z'].values

    orig_dist = np.zeros(len(rx_orig))
    for i in range(1, len(rx_orig)):
        orig_dist[i] = orig_dist[i-1] + np.sqrt((rx_orig[i]-rx_orig[i-1])**2 + (ry_orig[i]-ry_orig[i-1])**2)

    # Make the 2D plot extend slightly past the toe
    max_dist = max(final_distance, orig_dist[-1]) + 50
    s_1d = np.arange(-50, max_dist, grid_res) # Start a bit behind 0 to see backward spill

    x_1d, y_1d = np.zeros_like(s_1d), np.zeros_like(s_1d)
    dx_last, dy_last = rx_orig[-1] - rx_orig[-2], ry_orig[-1] - ry_orig[-2]
    ds_last = orig_dist[-1] - orig_dist[-2]
    dx_first, dy_first = rx_orig[1] - rx_orig[0], ry_orig[1] - ry_orig[0]
    ds_first = orig_dist[1] - orig_dist[0]
    
    for i, s in enumerate(s_1d):
        if s < 0:
            x_1d[i] = rx_orig[0] + (dx_first/ds_first) * s
            y_1d[i] = ry_orig[0] + (dy_first/ds_first) * s
        elif s <= orig_dist[-1]:
            x_1d[i] = np.interp(s, orig_dist, rx_orig)
            y_1d[i] = np.interp(s, orig_dist, ry_orig)
        else:
            x_1d[i] = rx_orig[-1] + (dx_last/ds_last) * (s - orig_dist[-1])
            y_1d[i] = ry_orig[-1] + (dy_last/ds_last) * (s - orig_dist[-1])

    z_topo_1d = topo_interp(x_1d, y_1d)
    z_fail_1d = np.interp(s_1d, orig_dist, rz_orig, right=np.nan, left=np.nan)
    z_fail_1d = np.where(z_fail_1d <= z_topo_1d, z_fail_1d, np.nan)
    
    # 3. Deposit Top Line using the newly found s_peak_opt
    tan_repose = np.tan(np.radians(angle_repose))
    dist_from_peak = np.abs(s_1d - s_peak_opt)
    z_dep_1d = z_peak_opt - dist_from_peak * tan_repose
    z_dep_1d = np.where(z_dep_1d >= z_topo_1d, z_dep_1d, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_1d, y=z_topo_1d, mode='lines', name='Topography', line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=s_1d, y=z_fail_1d, mode='lines', name='Failure Base', line=dict(color='red', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=s_1d, y=z_dep_1d, mode='lines', name='Deposit Top', line=dict(color='teal', width=3)))

    # Add a vertical marker to explicitly show the target runout toe
    fig.add_vline(x=final_distance, line_width=1, line_dash="dot", line_color="teal", annotation_text="Target Runout Reach")

    fig.update_layout(
        title='2D Longitudinal Cross-Section',
        xaxis_title='Distance along main axis (m)', yaxis_title='Elevation (Z)',
        template='plotly_white', hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig