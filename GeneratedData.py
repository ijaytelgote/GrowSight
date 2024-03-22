import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

class DataGenerator:
    @staticmethod
    def generate_fake_data(rows=100):
        data = []

        for _ in range(rows):
            row = {
                 'Timestamp': fake.date_time_between(start_date='-30d', end_date='now'),
                 'comment': fake.text(),
                "Monthly Revenue": random.randint(1000, 10000),
                "Opportunity Amount": random.randint(10000, 100000),
                "Support Tickets Open": random.randint(0, 10),
                "Support Tickets Closed": random.randint(0, 10),
                "Lead Score": random.randint(0, 100),
                "Age": random.randint(18, 90),
                "Size": random.uniform(5, 30),
                "Continent": random.choice(["Asia", "Europe", "Africa", "Americas"]),
                "Contract Type": random.choice(["One-time", "Recurring"]),
                "Gender": random.choice(["Male", "Female"]),
                "Lead Status": random.choice(["Qualified", "Unqualified", "Contacted", "Not Contacted"]),
                "Country": fake.country(),
                "Population": random.randint(1000000, 1000000000),
                "Area (sq km)": random.randint(100000, 10000000),
                "GDP (USD)": random.randint(1000000, 10000000000),
                'Last Email Sent Date': (pd.date_range(start='2023-01-01', periods=rows, freq='M')).strftime('%B %Y'),
                'Last Interaction Date': (pd.date_range(start='2020-10-01', periods=rows, freq='M')).strftime('%B %Y'),
                'Last Meeting Date': (pd.date_range(start='2022-12-01', periods=rows, freq='M')).strftime('%B %Y'),
                'Last Phone Call Date': (pd.date_range(start='2003-01-06', periods=rows, freq='M')).strftime('%B %Y'),
                'Probability of Close': random.randint(0, 100)
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df
