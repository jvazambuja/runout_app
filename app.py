import os
import sys
import threading
import webbrowser
from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import traceback
import runout_physics 

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller. """
    # PyInstaller extracts bundled files to _MEIPASS. When running from source,
    # resolve resources relative to this file rather than the shell's directory.
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Initialize Flask with absolute paths for the bundled templates
app = Flask(__name__, template_folder=resource_path('templates'))
app.secret_key = 'super_secret_key_for_flash_messages'

sim_cache = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        angle_of_reach = 20.0
        friction_angle = 30.0
        
        # Resolve paths for the default files inside the executable
        default_dem = resource_path('dem_mesh.txt')
        default_rupture = resource_path('rupture_section.txt')
        
        if request.method == 'POST':
            angle_of_reach = float(request.form.get('angle_of_reach', 20.0))
            friction_angle = float(request.form.get('friction_angle', 30.0))
            
            dem_file = request.files.get('dem_file')
            rupture_file = request.files.get('rupture_file')
            
            if dem_file and dem_file.filename != '':
                dem_df = pd.read_csv(dem_file)
            else:
                dem_df = pd.read_csv(default_dem)
                
            if rupture_file and rupture_file.filename != '':
                rupture_df = pd.read_csv(rupture_file)
            else:
                rupture_df = pd.read_csv(default_rupture)
                
        else:
            dem_df = pd.read_csv(default_dem)
            rupture_df = pd.read_csv(default_rupture)

        x_1d, y_1d, X_grid, Y_grid, grid_z, topo_interp, topo_res = runout_physics.build_structured_topo(dem_df)

        volume, X_surf, Y_surf, Z_surf = runout_physics.discretize_failure_plane(
            rupture_df, topo_interp, friction_angle_deg=friction_angle
        )
        
        Z_runout_surf, check_x, check_y, check_z, pt_x, pt_y, pt_z, env_x, env_y, env_z = runout_physics.calculate_runout_surface(
            X_grid, Y_grid, grid_z, topo_interp, rupture_df, angle_of_reach, X_surf, Y_surf, Z_surf, x_1d, y_1d, topo_res
        )
        
        sim_cache.clear()
        sim_cache.update({
            'x_1d': x_1d, 'y_1d': y_1d, 'X_grid': X_grid, 'Y_grid': Y_grid,
            'grid_z': grid_z, 'X_surf': X_surf, 'Y_surf': Y_surf, 'Z_surf': Z_surf, 
            'Z_runout_surf': Z_runout_surf, 'topo_res': topo_res
        })

        fig_3d = runout_physics.render_topography_and_mass(
            X_grid, Y_grid, grid_z, rupture_df, X_surf, Y_surf, Z_surf, Z_runout_surf, 
            check_x, check_y, check_z, pt_x, pt_y, pt_z, env_x, env_y, env_z
        )
        plot_3d_html = fig_3d.to_html(
            full_html=False,
            default_width='100%',
            default_height='100%',
            config={'responsive': True}
        )
        
        fig_2d = runout_physics.render_cross_section(
            x_1d, y_1d, topo_interp, rupture_df, angle_of_reach, Z_runout_surf
        )
        plot_2d_html = fig_2d.to_html(
            full_html=False,
            default_width='100%',
            default_height='100%',
            config={'responsive': True}
        )
        
    except Exception as e:
        return f"An error occurred processing the data: {e} <br><br><b>Stack Trace:</b><br><pre>{traceback.format_exc()}</pre>"

    return render_template(
        'index.html', 
        plot_html=plot_3d_html, plot_2d_html=plot_2d_html, 
        volume=volume, friction_angle=friction_angle, angle_of_reach=angle_of_reach,
        has_results=bool(sim_cache) 
    )

@app.route('/export/madflow')
def export_madflow():
    if not sim_cache: return redirect(url_for('index'))
    zip_io = runout_physics.export_madflow_asc(
        sim_cache['x_1d'], sim_cache['y_1d'], sim_cache['X_grid'], sim_cache['Y_grid'],
        sim_cache['grid_z'], sim_cache['X_surf'], sim_cache['Y_surf'], sim_cache['Z_surf'], 
        sim_cache['topo_res']
    )
    return send_file(zip_io, mimetype='application/zip', as_attachment=True, download_name='MADFlow_Inputs.zip')

@app.route('/export/dxf')
def export_dxf():
    if not sim_cache: return redirect(url_for('index'))
    dxf_io = runout_physics.export_to_dxf(
        sim_cache['X_grid'], sim_cache['Y_grid'], sim_cache['grid_z'], 
        sim_cache['X_surf'], sim_cache['Y_surf'], sim_cache['Z_surf'], 
        sim_cache['Z_runout_surf']
    )
    return send_file(dxf_io, mimetype='application/dxf', as_attachment=True, download_name='runout_results.dxf')

@app.route('/export/csv', methods=['POST'])
def export_csv():
    if not sim_cache: return redirect(url_for('index'))
    surface_type = request.form.get('surface_type')
    csv_io = runout_physics.export_to_csv(
        surface_type, sim_cache['X_grid'], sim_cache['Y_grid'], sim_cache['grid_z'], 
        sim_cache['X_surf'], sim_cache['Y_surf'], sim_cache['Z_surf'], 
        sim_cache['Z_runout_surf']
    )
    filename = f"{surface_type}_surface.csv"
    return send_file(csv_io, mimetype='text/csv', as_attachment=True, download_name=filename)

def open_browser():
    """ Automatically opens the default web browser after a short delay. """
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Launch the browser in a separate thread so it doesn't block the Flask server
    threading.Timer(1.25, open_browser).start()
    
    # Run the server (turn debug=False for production)
    app.run(host='127.0.0.1', port=5000, debug=False)