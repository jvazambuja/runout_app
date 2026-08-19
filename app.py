from flask import Flask, render_template, request
import pandas as pd
import runout_physics 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    friction_angle = 30.0
    runout_distance = 60.0
    angle_repose = 20.0
    
    try:
        if request.method == 'POST':
            friction_angle = float(request.form.get('friction_angle', 30.0))
            runout_distance = float(request.form.get('runout_distance', 60.0))
            angle_repose = float(request.form.get('angle_repose', 20.0))
            
            dem_file = request.files.get('dem_file')
            rupture_file = request.files.get('rupture_file')
            
            if dem_file and dem_file.filename != '':
                dem_df = pd.read_csv(dem_file)
            else:
                dem_df = pd.read_csv('dem_mesh.txt')
                
            if rupture_file and rupture_file.filename != '':
                rupture_df = pd.read_csv(rupture_file)
            else:
                rupture_df = pd.read_csv('rupture_section.txt')
        else:
            dem_df = pd.read_csv('dem_mesh.txt')
            rupture_df = pd.read_csv('rupture_section.txt')

            
        # 1. Calculate Initial Mass
        volume, X_surf, Y_surf, Z_surf = runout_physics.calculate_sliding_volume(
            dem_df, rupture_df, friction_angle_deg=friction_angle
        )
        
        # 2. Propagate and Calculate Deposit
        X_dep, Y_dep, Z_dep, z_peak_opt, s_peak_opt = runout_physics.calculate_deposit_surface(
            dem_df, rupture_df, runout_distance, volume, angle_repose
        )
        
        # 3. Render 3D and 2D figures
        fig_3d = runout_physics.render_topography_and_mass(
            dem_df, rupture_df, X_surf, Y_surf, Z_surf, X_dep, Y_dep, Z_dep
        )
        plot_3d_html = fig_3d.to_html(full_html=False)
        
        # Notice s_peak_opt is added to the end of this function call
        fig_2d = runout_physics.render_cross_section(
            dem_df, rupture_df, runout_distance, angle_repose, z_peak_opt, s_peak_opt
        )
        plot_2d_html = fig_2d.to_html(full_html=False)
        
    except Exception as e:
        return f"An error occurred processing the data: {e}"

    return render_template(
        'index.html', plot_html=plot_3d_html, plot_2d_html=plot_2d_html, 
        volume=volume, friction_angle=friction_angle, 
        runout_distance=runout_distance, angle_repose=angle_repose
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)