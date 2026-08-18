from flask import Flask, render_template
import pandas as pd
import runout_physics 

app = Flask(__name__)

@app.route('/')
def index():
    # Load the example files we just created
    dem_df = pd.read_csv('dem_mesh.txt')
    rupture_df = pd.read_csv('rupture_section.txt')
    
    # Calculate the 3D volume using a 30-degree friction angle
    volume, X_surf, Y_surf, Z_surf = runout_physics.calculate_sliding_volume(
        dem_df, rupture_df, friction_angle_deg=30
    )
    
    # Generate the interactive Plotly figure
    fig = runout_physics.render_topography_and_mass(
        dem_df, rupture_df, X_surf, Y_surf, Z_surf, volume
    )
    
    # Extract the HTML representation of the plot to send to the browser
    plot_html = fig.to_html(full_html=False)
    
    return render_template('index.html', plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True, port=5000)