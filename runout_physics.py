import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

import ezdxf
import io
import zipfile

GRID_RES_SOURCE = 0.5

# ==========================================
# STEPS 1 & 2: Read and Remesh Topography
# ==========================================
def build_structured_topo(dem_df, max_points=25000):
    x_min, x_max = dem_df['x'].min(), dem_df['x'].max()
    y_min, y_max = dem_df['y'].min(), dem_df['y'].max()
    
    area = (x_max - x_min) * (y_max - y_min)
    topo_res = np.sqrt(area / max_points)
    topo_res = max(topo_res, 1.0) 
    
    x_1d = np.arange(x_min, x_max + topo_res, topo_res)
    y_1d = np.arange(y_min, y_max + topo_res, topo_res)
    
    grid_x, grid_y = np.meshgrid(x_1d, y_1d, indexing='ij')
    
    points = dem_df[['x', 'y']].values
    values = dem_df['z'].values
    
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        grid_z_nearest = griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        grid_z[nan_mask] = grid_z_nearest
        
    topo_interp = RegularGridInterpolator((x_1d, y_1d), grid_z, bounds_error=False, fill_value=np.nan)
    
    return x_1d, y_1d, grid_x, grid_y, grid_z, topo_interp, topo_res

# ==========================================
# STEPS 3 & 4: Read and Discretize Failure Plane
# ==========================================
def discretize_failure_plane(rupture_df, topo_interp, friction_angle_deg, source_res=GRID_RES_SOURCE):
    rx = rupture_df['x'].values
    ry = rupture_df['y'].values
    rz = rupture_df['z'].values
    num_points = len(rx)
    
    tan_phi = np.tan(np.radians(friction_angle_deg))
    
    orig_dist = np.zeros(num_points)
    for i in range(1, num_points):
        orig_dist[i] = orig_dist[i-1] + np.sqrt((rx[i]-rx[i-1])**2 + (ry[i]-ry[i-1])**2)
        
    resampled_dist = np.arange(0, orig_dist[-1] + source_res/10, source_res)
    rx_rs = np.interp(resampled_dist, orig_dist, rx)
    ry_rs = np.interp(resampled_dist, orig_dist, ry)
    rz_rs = np.interp(resampled_dist, orig_dist, rz)
    
    num_rs = len(rx_rs)
    cross_section_areas = np.zeros(num_rs) 
    
    s_vals = np.arange(-200, 200 + source_res/2, source_res)
    
    X_surf = np.full((num_rs, len(s_vals)), np.nan)
    Y_surf = np.full((num_rs, len(s_vals)), np.nan)
    Z_surf = np.full((num_rs, len(s_vals)), np.nan)
    
    for i in range(num_rs):
        if i < num_rs - 1:
            dx, dy = rx_rs[i+1] - rx_rs[i], ry_rs[i+1] - ry_rs[i]
        else:
            dx, dy = rx_rs[i] - rx_rs[i-1], ry_rs[i] - ry_rs[i-1]
            
        length = np.sqrt(dx**2 + dy**2)
        if length == 0: continue
        
        nx, ny = -dy / length, dx / length
        z_top_center = topo_interp(np.array([[rx_rs[i], ry_rs[i]]]))[0]
        
        if np.isnan(z_top_center): continue
        H = z_top_center - rz_rs[i]
        if H <= 0: continue
            
        transverse_x = rx_rs[i] + s_vals * nx
        transverse_y = ry_rs[i] + s_vals * ny
        
        pts = np.stack((transverse_x, transverse_y), axis=-1)
        z_top = topo_interp(pts)
        
        z_base = np.where(np.abs(s_vals) <= H, rz_rs[i], rz_rs[i] + (np.abs(s_vals) - H) * tan_phi)
        thickness = z_top - z_base
        
        center_idx = len(s_vals) // 2
        if np.isnan(thickness[center_idx]) or thickness[center_idx] <= 0: continue
            
        left_idx, right_idx = center_idx, center_idx
        while left_idx > 0 and thickness[left_idx] > 0 and not np.isnan(thickness[left_idx]): left_idx -= 1
        while right_idx < len(s_vals) - 1 and thickness[right_idx] > 0 and not np.isnan(thickness[right_idx]): right_idx += 1
            
        valid_s = s_vals[left_idx:right_idx]
        valid_thickness = thickness[left_idx:right_idx]
        
        if len(valid_s) > 1:
            cross_section_areas[i] = np.trapezoid(valid_thickness, valid_s)
            X_surf[i, left_idx:right_idx] = transverse_x[left_idx:right_idx]
            Y_surf[i, left_idx:right_idx] = transverse_y[left_idx:right_idx]
            Z_surf[i, left_idx:right_idx] = z_base[left_idx:right_idx]

    total_volume = np.trapezoid(cross_section_areas, resampled_dist)
    return total_volume, X_surf, Y_surf, Z_surf

# ==========================================
# STEP 5: Angle of Reach Footprint Envelope
# ==========================================
def calculate_runout_surface(X_grid, Y_grid, grid_z, topo_interp, rupture_df, angle_of_reach, X_surf, Y_surf, Z_surf, x_1d, y_1d, topo_res):
    """
    Identifies the failure footprint and traces continuous flow paths via steepest descent.
    Calculates the convex envelope of the final stopping points.
    """
    rx = rupture_df['x'].values
    ry = rupture_df['y'].values
    
    dx_dir = rx[-1] - rx[0]
    dy_dir = ry[-1] - ry[0]
    len_dir = np.sqrt(dx_dir**2 + dy_dir**2)
    dir_x = dx_dir / len_dir if len_dir > 0 else 0
    dir_y = dy_dir / len_dir if len_dir > 0 else 1
    
    valid_mask = ~np.isnan(Z_surf)
    total_valid = np.sum(valid_mask)
    
    target_pts = 150
    stride = 1
    if total_valid > target_pts:
        stride = max(1, int(np.sqrt(total_valid / target_pts)))
        
    Xs_sample, Ys_sample = [], []
    for i in range(0, X_surf.shape[0], stride):
        for j in range(0, X_surf.shape[1], stride):
            if valid_mask[i, j]:
                Xs_sample.append(X_surf[i, j])
                Ys_sample.append(Y_surf[i, j])
                
    Xs_sample = np.array(Xs_sample)
    Ys_sample = np.array(Ys_sample)
    
    pts_sample = np.stack((Xs_sample, Ys_sample), axis=-1)
    Zs_sample = topo_interp(pts_sample)
    
    # Pre-calculate Topographical Gradients
    grad_x = np.gradient(grid_z, topo_res, axis=0)
    grad_y = np.gradient(grid_z, topo_res, axis=1)
    interp_dx = RegularGridInterpolator((x_1d, y_1d), grad_x, bounds_error=False, fill_value=0)
    interp_dy = RegularGridInterpolator((x_1d, y_1d), grad_y, bounds_error=False, fill_value=0)
    
    Z_reach_max = np.full_like(grid_z, -np.inf)
    tan_reach = np.tan(np.radians(angle_of_reach))
    
    check_lines_x, check_lines_y, check_lines_z = [], [], []
    end_x, end_y, end_z = [], [], [] # Storage for the final stopping points
    
    ds = max(topo_res / 2.0, 0.5) 
    paint_radius = max(2.0, 1.5 * topo_res) 
    
    for xs, ys, zs in zip(Xs_sample, Ys_sample, Zs_sample):
        if np.isnan(zs): continue
        
        x_curr, y_curr = xs, ys
        L = 0
        path_x, path_y, path_z = [xs], [ys], [zs]
        
        prev_dir_x, prev_dir_y = dir_x, dir_y
        
        for _ in range(int(2500 / ds)):
            pts = np.array([[x_curr, y_curr]])
            gx = interp_dx(pts)[0]
            gy = interp_dy(pts)[0]
            g_mag = np.sqrt(gx**2 + gy**2)
            
            if g_mag < 1e-4:
                curr_dir_x, curr_dir_y = prev_dir_x, prev_dir_y
            else:
                raw_dir_x, raw_dir_y = -gx/g_mag, -gy/g_mag
                curr_dir_x = 0.6 * raw_dir_x + 0.4 * prev_dir_x
                curr_dir_y = 0.6 * raw_dir_y + 0.4 * prev_dir_y
                m = np.sqrt(curr_dir_x**2 + curr_dir_y**2)
                curr_dir_x /= m
                curr_dir_y /= m
                
            prev_dir_x, prev_dir_y = curr_dir_x, curr_dir_y
            
            x_next = x_curr + ds * curr_dir_x
            y_next = y_curr + ds * curr_dir_y
            L += ds
            
            z_next = topo_interp(np.array([[x_next, y_next]]))[0]
            if np.isnan(z_next):
                break
                
            z_reach = zs - L * tan_reach
            
            path_x.append(x_next)
            path_y.append(y_next)
            path_z.append(z_reach)
            
            ix = int(np.round((x_curr - x_1d[0]) / topo_res))
            iy = int(np.round((y_curr - y_1d[0]) / topo_res))
            
            b = max(2, int(paint_radius / topo_res) + 1)
            for di in range(-b, b+1):
                for dj in range(-b, b+1):
                    ni, nj = ix + di, iy + dj
                    if 0 <= ni < len(x_1d) and 0 <= nj < len(y_1d):
                        dist = np.sqrt((x_1d[ni] - x_curr)**2 + (y_1d[nj] - y_curr)**2)
                        if dist <= paint_radius:
                            Z_reach_max[ni, nj] = max(Z_reach_max[ni, nj], z_reach)
            
            # Record the final point where the element stops[cite: 1]
            if z_next > z_reach and L > 2.0:
                end_x.append(x_next)
                end_y.append(y_next)
                end_z.append(z_next) # Anchor to actual topography Z for the line
                break
                
            x_curr, y_curr = x_next, y_next
            
        check_lines_x.extend(path_x + [None])
        check_lines_y.extend(path_y + [None])
        check_lines_z.extend(path_z + [None])

    # Calculate Convex Hull Envelope for the final points
    if len(end_x) >= 3:
        pts_end = np.column_stack((end_x, end_y))
        hull = ConvexHull(pts_end)
        # Append the first vertex to the end to close the ring
        hull_indices = np.append(hull.vertices, hull.vertices[0])
        env_x = np.array(end_x)[hull_indices]
        env_y = np.array(end_y)[hull_indices]
        env_z = np.array(end_z)[hull_indices] + 1.0 # Slight offset so it hovers visibly above ground
    else:
        env_x, env_y, env_z = [], [], []

    runout_mask = grid_z <= Z_reach_max
    Z_runout_surf = np.where(runout_mask, Z_reach_max, np.nan)
    
    return Z_runout_surf, check_lines_x, check_lines_y, check_lines_z, Xs_sample, Ys_sample, Zs_sample, env_x, env_y, env_z

def render_cross_section(x_1d, y_1d, topo_interp, rupture_df, angle_of_reach, Z_runout_surf, grid_res=0.5):
    """
    Generates the 2D longitudinal profile strictly along the original axis. 
    Uses nearest-neighbor interpolation on the mask to prevent plunging edge artifacts,
    and truncates the global H/L reference line at the topographic intersection.
    """
    import plotly.graph_objects as go
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    rx_orig = rupture_df['x'].values
    ry_orig = rupture_df['y'].values
    rz_orig = rupture_df['z'].values
    
    dx_dir = rx_orig[-1] - rx_orig[0]
    dy_dir = ry_orig[-1] - ry_orig[0]
    len_dir = np.sqrt(dx_dir**2 + dy_dir**2)
    
    max_idx = np.argmax(rz_orig)
    x_max, y_max, z_max = rx_orig[max_idx], ry_orig[max_idx], rz_orig[max_idx]

    orig_dist = np.zeros(len(rx_orig))
    for i in range(1, len(rx_orig)):
        orig_dist[i] = orig_dist[i-1] + np.sqrt((rx_orig[i]-rx_orig[i-1])**2 + (ry_orig[i]-ry_orig[i-1])**2)

    s_1d = np.arange(-50, 2500, grid_res) 
    x_prof, y_prof = np.zeros_like(s_1d), np.zeros_like(s_1d)
    
    dx_last, dy_last = rx_orig[-1] - rx_orig[-2], ry_orig[-1] - ry_orig[-2]
    ds_last = orig_dist[-1] - orig_dist[-2]
    dx_first, dy_first = rx_orig[1] - rx_orig[0], ry_orig[1] - ry_orig[0]
    ds_first = orig_dist[1] - orig_dist[0]
    
    for i, s in enumerate(s_1d):
        if s < 0:
            x_prof[i] = rx_orig[0] + (dx_first/ds_first) * s
            y_prof[i] = ry_orig[0] + (dy_first/ds_first) * s
        elif s <= orig_dist[-1]:
            x_prof[i] = np.interp(s, orig_dist, rx_orig)
            y_prof[i] = np.interp(s, orig_dist, ry_orig)
        else:
            x_prof[i] = rx_orig[-1] + (dx_last/ds_last) * (s - orig_dist[-1])
            y_prof[i] = ry_orig[-1] + (dy_last/ds_last) * (s - orig_dist[-1])

    pts = np.stack((x_prof, y_prof), axis=-1)
    z_topo_1d = topo_interp(pts)
    
    z_fail_1d = np.interp(s_1d, orig_dist, rz_orig, right=np.nan, left=np.nan)
    z_fail_1d = np.where(z_fail_1d <= z_topo_1d, z_fail_1d, np.nan)
    
    # === FIX: Truncate Global H/L Reference Line ===
    dist_L = np.sqrt((x_prof - x_max)**2 + (y_prof - y_max)**2)
    tan_reach = np.tan(np.radians(angle_of_reach))
    proj_1d = (x_prof - x_max) * (dx_last/ds_last) + (y_prof - y_max) * (dy_last/ds_last)
    
    z_reach_limit_1d = z_max - dist_L * tan_reach
    z_reach_limit_1d = np.where(proj_1d > -5, z_reach_limit_1d, np.nan) # Start at crest
    
    # Find where the mathematical reference line hits the topography
    diff_1d = z_reach_limit_1d - z_topo_1d
    hits_1d = np.where((diff_1d < 0) & (proj_1d > 2.0))[0]
    if len(hits_1d) > 0:
        stop_idx_1d = hits_1d[0]
        z_reach_limit_1d[stop_idx_1d + 1:] = np.nan # Nullify everything past the intersection
    
    # === Nearest-Neighbor Masking for Composite Footprint ===
    mask_grid = Z_runout_surf.copy()
    mask_grid[np.isnan(mask_grid)] = -99999
    
    interp_mask = RegularGridInterpolator(
        (x_1d, y_1d), mask_grid, method='nearest', bounds_error=False, fill_value=-99999
    )
    z_reach_contour_1d = interp_mask(pts)
    z_deposit_1d = np.where(z_reach_contour_1d > -50000, z_reach_contour_1d, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_1d, y=z_topo_1d, mode='lines', name='Topography', line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=s_1d, y=z_fail_1d, mode='lines', name='Failure Base', line=dict(color='red', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=s_1d, y=z_reach_limit_1d, mode='lines', name='Global H/L Reference', line=dict(color='teal', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=s_1d, y=z_deposit_1d, mode='lines', name='Composite Footprint Envelope', line=dict(color='teal', width=4)))

    fig.update_layout(
        title='2D Longitudinal Cross-Section (Dual Resolution Model)',
        xaxis_title='Distance along main axis (m)', yaxis_title='Elevation (Z)',
        template='plotly_white', hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig

def render_topography_and_mass(X_grid, Y_grid, grid_z, rupture_df, X_surf, Y_surf, Z_surf, Z_runout_surf, check_x, check_y, check_z, pt_x, pt_y, pt_z, env_x, env_y, env_z):
    fig = go.Figure()

    # Trace 0: Topography
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=grid_z, colorscale='Earth', name='Topography',
        opacity=0.75, showscale=False, lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.5, specular=0.2)
    ))

    # Trace 1: Failure Surface
    fig.add_trace(go.Surface(
        x=X_surf, y=Y_surf, z=Z_surf, colorscale='Reds', name='Failure Surface',
        opacity=0.95, showscale=False, lighting=dict(ambient=0.7, diffuse=0.6, roughness=0.4, specular=0.1)
    ))
    
    # Trace 2: Traced Runout
    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=Z_runout_surf, colorscale='Teal', name='Traced Runout',
        opacity=0.95, showscale=False, lighting=dict(ambient=0.7, diffuse=0.6, roughness=0.4, specular=0.1)
    ))

    # Trace 3: Main Axis
    fig.add_trace(go.Scatter3d(
        x=rupture_df['x'], y=rupture_df['y'], z=rupture_df['z'],
        mode='lines+markers', line=dict(color='black', width=6),
        marker=dict(size=4, color='black'), name='Main Axis'
    ))

    # Trace 4: Check Points (Discretized Source Centers)
    fig.add_trace(go.Scatter3d(
        x=pt_x, y=pt_y, z=pt_z, mode='markers',
        marker=dict(size=3, color='blue', opacity=0.8), name='Discretized Cells', visible=False
    ))

    # Trace 5: Check Lines (Source to Toe projection paths)
    fig.add_trace(go.Scatter3d(
        x=check_x, y=check_y, z=check_z, mode='lines',
        line=dict(color='orange', width=2), name='Reach Projection Paths', visible=False
    ))

    # Trace 6: Envelope Limit (Final Stopping Points Contour)
    fig.add_trace(go.Scatter3d(
        x=env_x, y=env_y, z=env_z, mode='lines+markers',
        line=dict(color='yellow', width=5), marker=dict(size=4, color='red'), name='Final Points Envelope', visible=True
    ))

    # Update menus with 7 trace booleans
    fig.update_layout(
        title=dict(text='3D Geometric Surface Model', x=0.02, xanchor='left', y=0.96, yanchor='top'),
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Elevation (Z)',
            aspectmode='manual',
            aspectratio=dict(x=4.0, y=4.0, z=1.0),
            camera=dict(
                projection=dict(type='orthographic'),
                eye=dict(x=1.35, y=1.35, z=0.85)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=55), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
# ==========================================
# EXPORT WORKFLOWS
# ==========================================

def export_madflow_asc(x_1d, y_1d, X_grid, Y_grid, grid_z, X_surf, Y_surf, Z_surf, topo_res):
    """
    Generates ESRI ASCII (.asc) strings for MADFlow.
    Interpolates the local failure surface onto the global grid to carve out the volume.
    """
    valid_mask = ~np.isnan(Z_surf)
    xs_valid = X_surf[valid_mask]
    ys_valid = Y_surf[valid_mask]
    zs_valid = Z_surf[valid_mask]
    
    z_after_failure = np.copy(grid_z)
    
    if len(xs_valid) > 0:
        # Interpolate the 0.5m failure surface elevations onto the global topographic grid
        z_surf_global = griddata((xs_valid, ys_valid), zs_valid, (X_grid, Y_grid), method='linear')
        
        # Where the failure surface exists, take the lower elevation (carving the volume)
        valid_interp = ~np.isnan(z_surf_global)
        z_after_failure[valid_interp] = np.minimum(grid_z[valid_interp], z_surf_global[valid_interp])
    
    def build_asc_string(z_matrix):
        ncols = len(x_1d)
        nrows = len(y_1d)
        xllcorner = x_1d[0]
        yllcorner = y_1d[0]
        nodata = -9999
        
        header = f"ncols         {ncols}\n"
        header += f"nrows         {nrows}\n"
        header += f"xllcorner     {xllcorner}\n"
        header += f"yllcorner     {yllcorner}\n"
        header += f"cellsize      {topo_res}\n"
        header += f"NODATA_value  {nodata}\n"
        
        # ASC format writes rows from Top-Left (Max Y, Min X) downwards
        data_lines = []
        for j in range(nrows - 1, -1, -1):
            row_data = z_matrix[:, j]
            row_clean = np.where(np.isnan(row_data), nodata, row_data)
            line = " ".join([f"{val:.3f}" for val in row_clean])
            data_lines.append(line)
            
        return header + "\n".join(data_lines)

    base_asc = build_asc_string(grid_z)
    after_failure_asc = build_asc_string(z_after_failure)
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('base.asc', base_asc)
        zf.writestr('after_failure.asc', after_failure_asc)
    memory_file.seek(0)
    
    return memory_file

def export_to_dxf(X_grid, Y_grid, grid_z, X_surf, Y_surf, Z_surf, Z_runout_surf):
    """Exports to DXF, correctly pairing the distinct grid resolutions."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    doc.layers.add(name='TOPOGRAPHY', color=7)
    doc.layers.add(name='FAILURE_SURFACE', color=1)
    doc.layers.add(name='RUNOUT_FOOTPRINT', color=3)
    
    def add_grid_to_dxf(xs, ys, zs, layer_name):
        valid = ~np.isnan(zs)
        xv, yv, zv = xs[valid], ys[valid], zs[valid]
        for x, y, z in zip(xv, yv, zv):
            msp.add_point((x, y, z), dxfattribs={'layer': layer_name})

    add_grid_to_dxf(X_grid, Y_grid, grid_z, 'TOPOGRAPHY')
    add_grid_to_dxf(X_surf, Y_surf, Z_surf, 'FAILURE_SURFACE')
    add_grid_to_dxf(X_grid, Y_grid, Z_runout_surf, 'RUNOUT_FOOTPRINT')
    
    memory_file = io.StringIO()
    doc.write(memory_file)
    memory_file.seek(0)
    
    bytes_io = io.BytesIO(memory_file.getvalue().encode('utf-8'))
    return bytes_io

def export_to_csv(surface_type, X_grid, Y_grid, grid_z, X_surf, Y_surf, Z_surf, Z_runout_surf):
    """Exports CSV, pairing the correct X, Y vectors to the requested Z surface."""
    if surface_type == 'topography':
        tx, ty, tz = X_grid, Y_grid, grid_z
    elif surface_type == 'failure':
        tx, ty, tz = X_surf, Y_surf, Z_surf
    elif surface_type == 'runout':
        tx, ty, tz = X_grid, Y_grid, Z_runout_surf
    else:
        raise ValueError("Invalid surface type selected.")
        
    valid = ~np.isnan(tz)
    df = pd.DataFrame({
        'x': tx[valid],
        'y': ty[valid],
        'z': tz[valid]
    })
    
    memory_file = io.BytesIO()
    df.to_csv(memory_file, index=False)
    memory_file.seek(0)
    
    return memory_file