from GeneratedData import GeneratedData
from Visualization import VisualizationDashboard

if __name__ == "__main__":
    data = GeneratedData.generate_fake_data(rows=100)
    dashboard = VisualizationDashboard(data)
    dashboard.run_server()
