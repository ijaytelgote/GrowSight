from GeneratedData import GeneratedData
from Visualization import VisualizationDashboard

if __name__ == "__main__":
    data = GeneratedData.generate_fake_data(rows=100)
    dashboard = VisualizationDashboard(data)
    app = dashboard.app  # Accessing the Dash app instance from VisualizationDashboard
    app.layout = dashboard.default_layout  # Set the layout
    dashboard.run_server()
