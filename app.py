from flask import Flask, render_template, request
import pandas as pd
import traceback
import runout_physics 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        angle_of_reach = 20.0
        friction_angle = 30.0
        
        if request.method == 'POST':
            angle_of_reach = float(request.form.get('angle_of_reach', 20.0))
            friction_angle = float(request.form.get('friction_angle', 30.0))
            
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

        # STEPS 1 & 2: Read Topography and Remesh
        x_1d, y_1d, X_grid, Y_grid, grid_z, topo_interp, topo_res = runout_physics.build_structured_topo(dem_df)

        # STEPS 3 & 4: Discretize the Failure Plane
        volume, X_surf, Y_surf, Z_surf = runout_physics.discretize_failure_plane(
            rupture_df, topo_interp, friction_angle_deg=friction_angle
        )
        
        # STEP 5: Angle of Reach footprint mapping and diagnostic trace evaluation
        Z_runout_surf, check_x, check_y, check_z, pt_x, pt_y, pt_z = runout_physics.calculate_runout_surface(
            X_grid, Y_grid, grid_z, topo_interp, rupture_df, angle_of_reach, X_surf, Y_surf, Z_surf, x_1d, y_1d, topo_res
        )
        
        # Render 3D scene including the "Check Discretization" arrays
        fig_3d = runout_physics.render_topography_and_mass(
            X_grid, Y_grid, grid_z, rupture_df, X_surf, Y_surf, Z_surf, Z_runout_surf, 
            check_x, check_y, check_z, pt_x, pt_y, pt_z
        )
        plot_3d_html = fig_3d.to_html(full_html=False)
        
        # Render 2D cross-section
        fig_2d = runout_physics.render_cross_section(
            x_1d, y_1d, topo_interp, rupture_df, angle_of_reach, Z_runout_surf
        )
        plot_2d_html = fig_2d.to_html(full_html=False)
        
    except Exception as e:
        return f"An error occurred processing the data: {e} <br><br><b>Stack Trace:</b><br><pre>{traceback.format_exc()}</pre>"

    return render_template(
        'index.html', 
        plot_html=plot_3d_html, 
        plot_2d_html=plot_2d_html, 
        volume=volume, 
        friction_angle=friction_angle, 
        angle_of_reach=angle_of_reach
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)